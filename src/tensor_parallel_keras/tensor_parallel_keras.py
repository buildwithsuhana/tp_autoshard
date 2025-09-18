from typing import Collection, Optional, Sequence, Union
import re
import numpy as np
import keras
import tensorflow as tf
from keras import device

from .autoconfig_keras import get_default_config_keras
from .parameter_sharding import make_parameter_sharded_model
from .sharding_keras import ShardedKeras
from .coordinated_optimizer import TensorParallelOptimizer

class TensorParallelKeras(keras.Model):  
    def __init__(self, model, world_size=None, device_ids=None, distributed_backend="auto", **kwargs):
        super().__init__()
        if world_size is None:
            world_size, device_ids = self._auto_detect_parallelism()
        elif device_ids is None:
            device_ids = self._auto_configure_devices(world_size, distributed_backend)
        
        self.world_size = world_size
        self.device_ids = device_ids
        self.sharding_strategy = "auto"
        self.distributed_backend = distributed_backend
        
        self.tensor_parallel_config = None
        self.distributed = True
        
        self.original_model = model
        self.sharded_models = [self.original_model]
        original_params = 0
        for p in model.weights:
            if hasattr(p, 'shape') and hasattr(p.shape, 'num_elements'):
                original_params += p.shape.num_elements()
            elif hasattr(p, 'shape') and hasattr(p.shape, '__iter__'):
                original_params += np.prod(p.shape)
            else:
                try:
                    original_params += np.prod(p.shape)
                except:
                    original_params += 1
        
        device_ids = list(self.check_device_ids(device_ids))  # Convert to list for modification
        
        if not device_ids:
            device_ids = self._auto_configure_devices(world_size, distributed_backend)

        if keras.backend.backend() == 'jax' or distributed_backend == 'jax':
            try:
                import jax
                all_devices = jax.devices()
                accel_devices = [d for d in all_devices if d.platform == 'tpu']
                if not accel_devices:
                    accel_devices = [d for d in all_devices if d.platform == 'gpu']
                if not accel_devices:
                    accel_devices = [d for d in all_devices if d.platform == 'cpu']
                
                
                if len(accel_devices) >= world_size:
                    device_ids = accel_devices[:world_size]
                else:
                    world_size = len(accel_devices)
                    device_ids = accel_devices[:world_size]
                    
            except Exception as e:
                device_ids = [f"cpu:{i}" for i in range(world_size)]
                
        if len(device_ids) != world_size:
            device_ids = self._adjust_device_list(device_ids, world_size)
            
        self.devices = device_ids
        self.world_size = world_size
        self.sharding_manager = None
        
        if self.world_size <= 1:
            self.model_shards = [model]
            self.distributed = False
            if len(self.devices) == 1:
                with device(self.devices[0]):
                    self.model_shards[0] = model
            super().__init__(**kwargs)
            return
            
        if self.tensor_parallel_config is None:
            device_names = [str(d) for d in self.devices]
            self.tensor_parallel_config = get_default_config_keras(model, device_names)
        config_with_ops = self.tensor_parallel_config.create_collective_ops(self.devices, self.distributed)
        
        
        self._is_multi_layer_model = len(model.layers) > 2  # More than just Input + Output
        
        self.model_shards = []
        self.modified_parameters_names = set()
        
        if keras.backend.backend() == 'jax':
            import jax
        
        for rank, device_id in enumerate(self.devices):
            shard, modified_parameters_names = make_parameter_sharded_model(
                model, config_with_ops, rank=rank, world_size=self.world_size
            )
            self.model_shards.append(shard)
            self.modified_parameters_names.update(modified_parameters_names)
                    
        params_per_shard = []
        for i, shard in enumerate(self.model_shards):
            total_params = 0
            for p in shard.weights:
                if hasattr(p, 'num_elements'):
                    total_params += p.num_elements()
                elif hasattr(p, 'numel'):
                    total_params += p.numel()
                elif hasattr(p.shape, 'num_elements'):
                    total_params += p.shape.num_elements()
                else:
                    try:
                        total_params += np.prod(p.shape)
                    except:
                        total_params += 1
            
            params_per_shard.append(int(total_params))
        
        self.distributed_backend_name = distributed_backend

        try:
            from .distributed_backend import get_distributed_backend
            self.distributed_backend = get_distributed_backend(distributed_backend, self.world_size, rank=0)
        except Exception as e:
            self.distributed_backend = None

        super().__init__(**kwargs)
        self.built = True
    
    def _auto_detect_parallelism(self):
        try:
            from .distribution_lib import list_devices, get_best_devices
            
            available_devices = list_devices()
            world_size = len(available_devices)
            
            device_ids = get_best_devices(world_size)
            
            return world_size, device_ids
            
        except Exception as e:
            world_size = 1
            device_ids = ['cpu:0']
            return world_size, device_ids
        
    def _adjust_device_list(self, device_ids, target_world_size):
        current_size = len(device_ids)
        
        if current_size < target_world_size:
            if device_ids:
                base_device = device_ids[0]
                if isinstance(base_device, str) and ':' in base_device:
                    device_type, base_index = base_device.rsplit(':', 1)
                    try:
                        base_index = int(base_index)
                        additional_devices = [f"{device_type}:{base_index + i + 1}" for i in range(target_world_size - current_size)]
                        return device_ids + additional_devices
                    except ValueError:
                        additional_devices = [f"cpu:{i}" for i in range(current_size, target_world_size)]
                        return device_ids + additional_devices
                else:
                    additional_devices = [f"cpu:{i}" for i in range(current_size, target_world_size)]
                    return device_ids + additional_devices
            else:
                return [f"cpu:{i}" for i in range(target_world_size)]
        elif current_size > target_world_size:
            return device_ids[:target_world_size]
        else:
            return device_ids
        
    def _auto_configure_devices(self, world_size, distributed_backend):
        try:
            from .distribution_lib import list_devices
            available_devices = list_devices()
            
            if available_devices:
                devices = available_devices[:world_size]
                return devices
            else:
                return ['cpu:0']
                
        except Exception as e:
            return ['cpu:0']
        
    def check_device_ids(self, device_ids: Optional[Sequence[str]]) -> Sequence[str]:
        if device_ids is None:
            device_ids = self._get_all_device_indices()
            
        device_ids = list(device_ids)
        
        canonical_ids = []
        for device_id in device_ids:
            if isinstance(device_id, str):
                canonical_ids.append(self.canonicalize_device(device_id))
            else:
                canonical_ids.append(device_id)
        
        return tuple(canonical_ids)
        
    def _get_all_device_indices(self) -> Sequence[str]:
        try:
            from .distribution_lib import list_devices
            devices = list_devices()
            return devices
        except ImportError:
            devices = []
            
            try:
                tpu_devices = keras.config.list_physical_devices('TPU')
                if tpu_devices:
                    for i, device in enumerate(tpu_devices):
                        devices.append(f"tpu:{i}")
            except Exception as e:
                return
            
            try:
                gpu_devices = keras.config.list_physical_devices('GPU')
                if gpu_devices:
                    for i, device in enumerate(gpu_devices):
                        devices.append(f"gpu:{i}")
            except Exception as e:
                pass
            try:
                cpu_devices = keras.config.list_physical_devices('CPU')
                if cpu_devices:
                    for i, device in enumerate(cpu_devices):
                        devices.append(f"cpu:{i}")
            except Exception as e:
                pass
            if not devices:
                devices.append("cpu:0")
            
            return devices
        
    def build_assembled_model(self):

        if not self.distributed:
            return self.original_model
        input_layers = {
            inp.name.split(':')[0]: keras.Input(
                shape=inp.shape[1:], dtype=inp.dtype, name=inp.name.split(':')[0]
            )
            for inp in self.original_model.inputs
        }

        # --- FIX START ---
        # Check if the shard models expect a single tensor or a dict/list
        if len(input_layers) == 1:
            # If there's only one input, pass the single tensor directly.
            # This matches the shard's expectation of a single tensor.
            shard_inputs = list(input_layers.values())[0]
        else:
            # Otherwise, pass the full dictionary for multi-input models.
            shard_inputs = input_layers
        
        partial_outputs = [model(shard_inputs) for model in self.sharded_models]
        # --- FIX END ---

        final_layer = self.original_model.layers[-1]
        sharding_type = "unknown"
        final_kernel_name = f"{final_layer.name}.kernel"
        if hasattr(self.original_model, 'name') and self.original_model.name:
                final_kernel_name = f"{self.original_model.name}.{final_kernel_name}"
        
        for pattern, action in self.tensor_parallel_config.state_rules.items():
            if re.search(pattern, final_kernel_name):
                if hasattr(action, 'sharding_type'):
                    sharding_type = action.sharding_type
                break

        if sharding_type == "column":
            final_output = keras.ops.concatenate(partial_outputs, axis=-1)
            original_output_dim = self.original_model.output_shape[-1]
            if final_output.shape[-1] != original_output_dim:
                final_output = keras.layers.Lambda(
                    lambda x: x[..., :original_output_dim]
                )(final_output)
        elif sharding_type == "row":
            if len(partial_outputs) > 1:
                summed_output = keras.layers.Add()(partial_outputs)
            else:
                summed_output = partial_outputs[0]

            if final_layer.use_bias:
                bias = final_layer.bias
                final_output = keras.layers.Lambda(
                    lambda x: x - bias * (self.world_size - 1)
                )(summed_output)
            else:
                final_output = summed_output
        else:
            final_output = partial_outputs[0]

        assembled_model = keras.Model(inputs=list(input_layers.values()), outputs=final_output)        
        return assembled_model   
    
    def _get_device_index(self, device_spec: str) -> int:
        if isinstance(device_spec, str):
            if device_spec == "cpu":
                return -1
            elif device_spec.startswith("gpu:"):
                return int(device_spec.split(":")[1])
            else:
                return 0
        return 0
            
    def canonicalize_device(self, device_spec: Union[str, int]) -> str:
        if isinstance(device_spec, int):
            if device_spec == -1:
                return "cpu"
            else:
                return f"gpu:{device_spec}"
        elif isinstance(device_spec, str):
            if device_spec == "cpu":
                return "cpu"
            elif device_spec.startswith("gpu:"):
                return device_spec
            elif device_spec.startswith("cuda:"):
                return f"gpu:{device_spec.split(':')[1]}"
            else:
                return device_spec
        else:
            return "cpu"
            
    def apply_sharding(self, replicated_param_names: Optional[Collection[str]] = None):
        if replicated_param_names is None:
            replicated_param_names = self.modified_parameters_names
            
        self.sharding_manager = ShardedKeras(
            self.model_shards,
            replicated_param_names,
            self.tensor_parallel_config,
            self.devices,
            0
        )
    
    def call(self, inputs, training=None, **kwargs):
        if not self.distributed:
            return self.original_model(inputs, training=training, **kwargs)

        partial_outputs = []
        for shard in self.model_shards:
            with device(shard.device): 
                output = shard(inputs, training=training, **kwargs)
                partial_outputs.append(output)

        if not partial_outputs:
            return None

        final_layer = self.original_model.layers[-1]
        sharding_type = "unknown"
        
        final_kernel_name = f"{final_layer.name}.kernel"
        if hasattr(self.original_model, 'name') and self.original_model.name:
            final_kernel_name = f"{self.original_model.name}.{final_kernel_name}"
        
        if hasattr(self, 'tensor_parallel_config') and self.tensor_parallel_config:
            for pattern, action in self.tensor_parallel_config.state_rules.items():
                if re.search(pattern, final_kernel_name):
                    if hasattr(action, 'sharding_type'):
                        sharding_type = action.sharding_type
                    break
        

        if sharding_type == "column":
            final_output = keras.ops.concatenate(partial_outputs, axis=-1)
            original_output_dim = self.original_model.output_shape[-1]
            if final_output.shape[-1] > original_output_dim:
                final_output = final_output[..., :original_output_dim]
        elif sharding_type == "row":
            final_output = keras.ops.sum(keras.ops.stack(partial_outputs), axis=0)
            if final_layer.use_bias:
                bias = final_layer.bias
                bias_shape = [1] * (len(final_output.shape) - 1) + [-1]
                reshaped_bias = keras.ops.reshape(bias, bias_shape)
                final_output -= reshaped_bias * (self.world_size - 1)
        else:
            final_output = partial_outputs[0]

        return final_output

    def _reconstruct_full_model_from_shards(self):
        try:
            
            model_config = self.original_model.get_config()
            reconstructed_model = keras.Model.from_config(model_config)
            reconstructed_model.build(self.original_model.input_shape)
            
            self._reconstruct_weights_from_shards(reconstructed_model)
            
            return reconstructed_model
            
        except Exception as e:
            return self.original_model
    
    def _reconstruct_weights_from_shards(self, reconstructed_model):
        try:
            
            state_rules = self.tensor_parallel_config.state_rules
            
            for layer in reconstructed_model.layers:
                for weight in layer.weights:
                    weight_name = f"{layer.name}.{weight.name.split('/')[-1].split(':')[0]}"
                    
                    sharding_rule = self._find_sharding_rule_for_weight(weight_name, state_rules)
                    
                    if sharding_rule:
                        full_weight = self._gather_weight_shards(weight_name, sharding_rule)
                        if full_weight is not None:
                            weight.assign(full_weight)
                    else:
                        shard_weight = self._get_weight_from_shard(weight_name, 0)
                        if shard_weight is not None:
                            weight.assign(shard_weight)
            
            
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    def _find_sharding_rule_for_weight(self, weight_name, state_rules):
        for pattern, rule in state_rules.items():
            if self._pattern_matches(weight_name, pattern):
                return rule
        return None
    
    def _gather_weight_shards(self, weight_name, sharding_rule):
        try:
            weight_shards = []
            for i, shard in enumerate(self.model_shards):
                shard_weight = self._get_weight_from_shard(weight_name, i)
                if shard_weight is not None:
                    weight_shards.append(shard_weight)
            
            if not weight_shards:
                return None
            
            if hasattr(sharding_rule, 'undo'):
                torch_shards = []
                for shard in weight_shards:
                    import torch
                    torch_shard = torch.from_numpy(shard.numpy())
                    torch_shards.append(torch_shard)
                
                full_torch_weight = sharding_rule.undo(torch_shards)
                
                import tensorflow as tf
                full_weight = tf.convert_to_tensor(full_torch_weight.numpy())
                return full_weight
            else:
                import tensorflow as tf
                return tf.concat(weight_shards, axis=-1)
                
        except Exception as e:
            return None
    
    def _get_weight_from_shard(self, weight_name, shard_index):
        try:
            if shard_index >= len(self.model_shards):
                return None
                
            shard = self.model_shards[shard_index]
            
            for layer in shard.layers:
                for weight in layer.weights:
                    shard_weight_name = f"{layer.name}.{weight.name.split('/')[-1].split(':')[0]}"
                    if shard_weight_name == weight_name:
                        return weight
            
            return None
            
        except Exception as e:
            return None
    
    def _combine_tensor_parallel_outputs(self, shard_outputs):
        try:
            
            shapes = [output.shape for output in shard_outputs]
            
            outputs_np = []
            for output in shard_outputs:
                if hasattr(output, 'numpy'):
                    outputs_np.append(output.numpy())
                else:
                    outputs_np.append(np.array(output))
            
            if len(set(str(shape) for shape in shapes)) == 1:
                combined_np = np.sum(outputs_np, axis=0)
                
            else:
                
                shape0 = shapes[0]
                concat_dim = -1
                
                combined_np = np.concatenate(outputs_np, axis=concat_dim)
            
            import tensorflow as tf
            combined_output = tf.convert_to_tensor(combined_np)
            
            return combined_output
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return shard_outputs[0]
    
    def _apply_allreduce(self, output, backend):
        try:
            
            if hasattr(output, 'numpy'):
                output_np = output.numpy()
            else:
                output_np = output
            
            return output
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return output
    
    def _apply_allgather(self, output, backend, dim=-1):
        try:
            
            if hasattr(output, 'numpy'):
                output_np = output.numpy()
            else:
                output_np = output
            
            outputs_list = [output_np, output_np]
            
            gathered_outputs = backend.all_gather(outputs_list, dim=dim)
            
            if hasattr(output, 'numpy'):
                import tensorflow as tf
                result = tf.convert_to_tensor(gathered_outputs[0])
                return result
            else:
                result = gathered_outputs[0]
                return result
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return output
    
    def _pattern_matches(self, layer_name, pattern):
        import re
        
        clean_pattern = pattern.strip('^$')
        
        if clean_pattern == layer_name:
            return True
        
        if re.match(pattern, layer_name):
            return True
        
        if clean_pattern in layer_name:
            return True
        
        return False
    
    def _apply_output_rule(self, output, rule):
        if isinstance(rule, dict):
            rule_str = str(rule)
        else:
            rule_str = str(rule)
        
        if 'gather' in rule_str.lower():
            return output
        elif 'allreduce' in rule_str.lower():
            return output
        else:
            return output
    
    def _apply_forward_communication(self, inputs, training=None, **kwargs):
        if not hasattr(self, 'tensor_parallel_config') or self.tensor_parallel_config is None:
            return self.shard_outputs[0]
        
        try:
            output_rules = self.tensor_parallel_config.output_rules
            
            if not output_rules:
                return self.shard_outputs[0]
            
            from .communications_keras import TensorParallelCommunicator
            communicator = TensorParallelCommunicator(self.world_size, rank=0)
            
            if hasattr(self, '_is_mlp_model') and self._is_mlp_model:
                return self._handle_mlp_forward_communication(communicator)
            else:
                return self._handle_single_layer_forward_communication(communicator, output_rules)
                
        except Exception as e:
            return self.shard_outputs[0]
    
    def _handle_mlp_forward_communication(self, communicator):
        try:
            up_outputs = []
            down_outputs = []
            
            for i in range(self.world_size):
                if i in self.shard_outputs:
                    up_outputs.append(self.shard_outputs[i])
                    down_outputs.append(self.shard_outputs[i])
            
            final_up, final_down = communicator.handle_mlp_handshake(up_outputs, down_outputs)
            
            return final_down[0] if isinstance(final_down, list) else final_down
            
        except Exception as e:
            return self.shard_outputs[0]
    
    def _handle_single_layer_forward_communication(self, communicator, output_rules):
        try:
            first_output = self.shard_outputs[0]
            if hasattr(first_output, 'shape') and len(first_output.shape) >= 2:
                if hasattr(self, '_is_multi_layer_model') and self._is_multi_layer_model:
                    return first_output
                
                
                partial_outputs = []
                for i in range(self.world_size):
                    if i in self.shard_outputs:
                        partial_outputs.append(self.shard_outputs[i])
                
                return first_output
            
            return self.shard_outputs[0]
            
        except Exception as e:
            return self.shard_outputs[0]
    
    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        if len(self.model_shards) > 1 and optimizer is not None:
            backend_name = getattr(self, 'distributed_backend_name', 'auto')
            
            self.coordinated_optimizer = TensorParallelOptimizer(
                optimizer, 
                self.world_size, 
                distributed_backend=backend_name,
                tensor_parallel_config=self.tensor_parallel_config
            )
            
            super().compile(optimizer=self.coordinated_optimizer, loss=loss, metrics=metrics, **kwargs)
            
            try:
                for shard in self.model_shards:
                    shard.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
            except Exception as e:
                pass
        else:
            super().compile(optimizer, loss, metrics, **kwargs)

    def train_step(self, data, state=None, **kwargs):
        if isinstance(data, tuple):
            x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        else:
            x, y, sample_weight = keras.utils.unpack_x_y_sample_weight((data,))

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True, **kwargs) 
            
            loss = self.compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply(gradients, trainable_vars)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight)
        
        return {m.name: m.result() for m in self.metrics}
        
    def _apply_backward_communication(self, gradients, layer_type="unknown"):
        if len(self.model_shards) <= 1:
            return gradients
        
        try:
            from .communications_keras import TensorParallelCommunicator
            communicator = TensorParallelCommunicator(self.world_size, rank=0)
            
            if "column" in layer_type.lower() or "up_projection" in layer_type.lower():
                return communicator.backward_column_parallel(gradients, op="sum")
            elif "row" in layer_type.lower() or "down_projection" in layer_type.lower():
                gathered = communicator.backward_row_parallel(gradients, dim=-1)
                return [gathered] * self.world_size
            else:
                return gradients
                
        except Exception as e:
            return gradients
    
    def _slice_upstream_gradients_for_backward(self, full_gradients, sharding_type="unknown"):
        if len(self.model_shards) <= 1:
            return [full_gradients]
        
        try:
            from .communications_keras import TensorParallelCommunicator
            communicator = TensorParallelCommunicator(self.world_size, rank=0)
            
            sliced_gradients = []
            
            for rank in range(self.world_size):
                if sharding_type == "column_parallel":
                    sliced_grad = communicator.slice_upstream_gradient_for_column_parallel(
                        full_gradients, rank, self.world_size, dim=-1
                    )
                elif sharding_type == "row_parallel":
                    sliced_grad = communicator.slice_upstream_gradient_for_row_parallel(
                        full_gradients, rank, self.world_size, dim=0
                    )
                else:
                    sliced_grad = full_gradients
                
                sliced_gradients.append(sliced_grad)
            
            return sliced_gradients
            
        except Exception as e:
            return [full_gradients] * self.world_size
    
    def _compute_shard_gradients_with_sliced_upstream(self, shard, sliced_upstream_grad, inputs, training=True):
        try:
            with tf.GradientTape() as tape:
                shard_output = shard(inputs, training=training)
                loss = self._compute_shard_loss(shard_output, sliced_upstream_grad)
            
            gradients = tape.gradient(loss, shard.trainable_variables)
            return gradients
            
        except Exception as e:
            return [tf.zeros_like(v) for v in shard.trainable_variables]
    
    def _compute_shard_loss(self, shard_output, sliced_upstream_grad):
        try:
            if hasattr(sliced_upstream_grad, 'shape') and hasattr(shard_output, 'shape'):
                target = sliced_upstream_grad
                loss = tf.reduce_mean(tf.square(shard_output - target))
                return loss
            else:
                return tf.reduce_mean(tf.square(shard_output))
                
        except Exception as e:
            return tf.reduce_mean(tf.square(shard_output))
    
    def _detect_layer_sharding_type(self):
        try:
            if not hasattr(self, 'tensor_parallel_config') or self.tensor_parallel_config is None:
                return "unknown"
            
            output_rules = self.tensor_parallel_config.output_rules
            if not output_rules:
                return "unknown"
            
            first_rule = list(output_rules.values())[0] if output_rules else None
            if first_rule:
                if "gather" in str(first_rule).lower():
                    return "column_parallel"
                elif "allreduce" in str(first_rule).lower():
                    return "row_parallel"
            
            if hasattr(self, 'original_model') and self.original_model is not None:
                if hasattr(self.original_model, 'layers') and self.original_model.layers:
                    layer_names = [layer.name.lower() for layer in self.original_model.layers]
                    if any("up" in name for name in layer_names) and any("down" in name for name in layer_names):
                        return "mlp_handshake"
            
            return "unknown"
            
        except Exception as e:
            return "unknown"
    
    def fit(self, x=None, y=None, **kwargs):
        if len(self.model_shards) > 1:
            self._synchronize_gradients()
            return super().fit(x, y, **kwargs)
        else:
            return super().fit(x, y, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "model": self.original_model,
            "device_ids": self.devices,
            "output_device_index": 0,
            "sharded": hasattr(self, 'sharding_manager') and self.sharding_manager is not None
        })
        return config 

    def auto_detect_parallelism(self):
        try:
            from .distribution_lib import list_devices, get_best_devices
            
            all_devices = list_devices()
            
            optimal_world_size = len(all_devices)
            if optimal_world_size != self.world_size:
                self.world_size = optimal_world_size
            
            optimal_devices = get_best_devices(self.world_size)
            if optimal_devices != self.device_ids:
                self.device_ids = optimal_devices
            
            return True
            
        except Exception as e:
            return False

    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True, return_dict=False):        
        try:
            return super().train_on_batch(
                x, y, 
                sample_weight=sample_weight, 
                class_weight=class_weight, 
                reset_metrics=reset_metrics, 
                return_dict=return_dict
            )
        except TypeError:
            return super().train_on_batch(
                x, y, 
                sample_weight=sample_weight, 
                class_weight=class_weight
            )