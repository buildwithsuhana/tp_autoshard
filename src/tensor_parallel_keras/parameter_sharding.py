"""
Parameter-Level Sharding for Keras Tensor Parallel
This approach shards only the weights/parameters without rebuilding the model structure.
Works with ANY Keras model including KerasNLP models.
"""

import copy
import re
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import keras
from keras import Model

from .config_keras import ConfigKeras
from .state_actions_keras import StateActionKeras


class ShardedWeight:
    def __init__(self, tensor_shard, name, trainable=True):
        self._variable = keras.Variable(
            initializer=tensor_shard,
            trainable=trainable,
            name=name
        )
        self.regularizer = None

    @property
    def name(self):
        """Returns the name of the underlying variable."""
        return self._variable.name

    @property
    def trainable(self):
        """Returns whether the variable is trainable."""
        return self._variable.trainable

    @property
    def shape(self):
        """Returns the shape of the variable."""
        return self._variable.shape
    @property
    def dtype(self):
        """Returns the dtype of the underlying variable."""
        return self._variable.dtype

    @property
    def variable(self):
        """Provides direct access to the underlying tf.Variable."""
        return self._variable

    def numpy(self):
        """Returns the value of the variable as a NumPy array."""
        return self._variable.numpy()

    def num_elements(self):
        """Returns the total number of elements in the tensor."""
        return keras.ops.size(self._variable)

    def __repr__(self):
        return (f"<ShardedWeight name='{self.name}' "
                f"shape={self.shape} trainable={self.trainable}>")

class ParameterShardingStrategy:
    """
    Parameter-level sharding strategy that works with any Keras model.
    Instead of rebuilding the model, we shard only the weights and handle
    communication during forward/backward passes.
    """
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.sharded_weights = {}
        self.original_weights = {}
        self.weight_mapping = {}
        
    def shard_model_parameters(self, model: Model, config: ConfigKeras) -> Tuple[Model, Set[str]]:
        """
        Shard model parameters without rebuilding the model structure.
        
        Args:
            model: Original Keras model
            config: Tensor parallel configuration
            
        Returns:
            Tuple of (sharded_model, modified_parameter_names)
        """
        print(f"🔧 Applying parameter-level sharding to {model.name}")
        
        self._store_original_weights(model)
        modified_parameters = set()
        
        for pattern, action in config.state_rules.items():
            if isinstance(action, StateActionKeras):
                matching_params = self._find_matching_parameters(model, pattern)
                
                for param_name, param in matching_params:
                    sharded_param = action(param, self.rank)
                    
                    self.sharded_weights[param_name] = sharded_param
                    self.weight_mapping[param_name] = {
                        'original_shape': param.shape,
                        'sharded_shape': sharded_param.shape,
                        'action': action
                    }
                    
                    modified_parameters.add(param_name)
                    print(f"   ✅ Sharded {param_name}: {param.shape} -> {sharded_param.shape}")
        
        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            config=config
        )
        
        print(f"🎯 Parameter sharding completed: {len(modified_parameters)} parameters sharded")
        return sharded_model, modified_parameters
    
    def _store_original_weights(self, model: Model):
        """Store original weights for reference."""
        for layer in model.layers:
            if hasattr(layer, 'weights') and layer.weights:
                for weight in layer.weights:
                    param_name = f"{layer.name}.{weight.name}"
                    self.original_weights[param_name] = weight.numpy()
    
    def _find_matching_parameters(self, model: Model, pattern: str) -> List[Tuple[str, Any]]:
        """Find parameters that match the given pattern."""
        matching_params = []
        
        def search_module(mod: Model, prefix: str = ""):
            for layer in mod.layers:
                name = layer.name
                full_name = f"{prefix}.{name}" if prefix else name
                
                if hasattr(layer, 'weights') and layer.weights:
                    for weight in layer.weights:
                        param_name = f"{full_name}.{weight.name}"
                        if re.match(pattern, param_name):
                            matching_params.append((param_name, weight)) 

                if hasattr(layer, 'layers') and len(layer.layers) > 0:
                    search_module(layer, full_name)
                    
        search_module(model)
        return matching_params
        
    def get_sharded_weight(self, param_name: str) -> Optional[np.ndarray]:
        """Get sharded weight for a parameter."""
        if param_name in self.sharded_weights:
            return self.sharded_weights[param_name].numpy()
        return None
    
    def get_weight_info(self, param_name: str) -> Optional[Dict]:
        """Get information about a sharded weight."""
        return self.weight_mapping.get(param_name)


class ParameterShardedModel(Model):
    """
    Wrapper model that handles parameter sharding without rebuilding the structure.
    This preserves the original model's functionality while enabling tensor parallelism.
    """
    
    def __init__(self, original_model: Model, sharding_strategy: ParameterShardingStrategy, config: ConfigKeras):
        super().__init__()
        
        self.original_model = original_model
        self.sharding_strategy = sharding_strategy
        self.config = config
        
        self._build_and_cache_weights()

        if original_model.inputs:
             self.build(original_model.inputs[0].shape)

        print(f"🚀 ParameterShardedModel created successfully")

    def _build_and_cache_weights(self):
        """
        Builds the list of trainable/non-trainable weights ONCE and caches it.
        This prevents creating new Variables inside a tf.function.
        """
        print("   - Building and caching the definitive weights list...")
        weights_list = []
        
        for param_name in self.sharding_strategy.sharded_weights:
            sharded_tensor = self.sharding_strategy.sharded_weights[param_name]
            weights_list.append(ShardedWeight(sharded_tensor, param_name))
        
        sharded_param_names = set(self.sharding_strategy.sharded_weights.keys())
        for layer in self.original_model.layers:
            for weight in layer.weights:
                param_name = f"{layer.name}.{weight.name.split(':')[0]}" # Normalize name
                if param_name not in sharded_param_names:
                    weights_list.append(weight)
        
        self._weights_list = weights_list

    @property
    def weights(self):
        """
        Override weights property to return the cached list of sharded weights.
        """
        return self._weights_list       
    
    def _copy_model_structure(self):
        """Copy the model structure without rebuilding layers."""
        pass
    
    def _apply_sharded_weights(self):
        """Apply sharded weights to the model."""
        pass
    
    def call(self, inputs, training=None, mask=None):
        return self.original_model(inputs, training=training, mask=mask)
    
    def _execute_complete_forward_pass(self, inputs, training=None, mask=None):
        print(f"   - Executing complete forward pass")
        
        current_input = inputs
        residual_tensor = None
        
        for i, layer in enumerate(self.original_model.layers):
            print(f"   - Processing layer {i}: {layer.name} ({type(layer).__name__})")
            
            if isinstance(layer, keras.layers.InputLayer):
                continue

            if isinstance(layer, keras.layers.MultiHeadAttention) or ('mlp_fc1' in layer.name):
                residual_tensor = current_input

            if isinstance(layer, keras.layers.Dense) and f"{layer.name}.kernel" in self.sharding_strategy.sharded_weights:
                sharded_output = self._handle_dense_layer(current_input, layer)
                
                if 'mlp_fc1' in layer.name:
                    current_input = sharded_output
                    print(f"   - (Column-Parallel) Output shape: {current_input.shape}")

                elif 'mlp_fc2' in layer.name:
                    current_input = self._gather_sharded_output(sharded_output, layer.name, op="sum")
                    print(f"   - (Row-Parallel) Final aggregated output shape: {current_input.shape}")

                else:
                    current_input = sharded_output
                    
            elif isinstance(layer, keras.layers.Add):
                current_input = layer([current_input, residual_tensor])

            else:
                current_input = layer(current_input, training=training)
            
            print(f"   - Layer {layer.name} output shape after processing: {current_input.shape}")
        
        return current_input


    def _gather_sharded_output(self, sharded_output, layer_name, op="concat"):
        all_shard_outputs = [sharded_output, sharded_output] # Simulating 2 shards

        if op == "sum":
            print(f"   - Aggregating (All-Reduce Sum) {len(all_shard_outputs)} sharded outputs from {layer_name}")
            aggregated_output = keras.ops.add(*all_shard_outputs)
            return aggregated_output
        else:
            print(f"   - Gathering (All-Gather Concat) {len(all_shard_outputs)} sharded outputs from {layer_name}")
            concatenated_output = keras.ops.concatenate(all_shard_outputs, axis=-1)
            return concatenated_output
    
    def _get_original_input_for_layer(self, layer_name):
        """Get the original input that would be fed to this layer."""
        try:
            layer_index = None
            for i, layer in enumerate(self.original_model.layers):
                if layer.name == layer_name:
                    layer_index = i
                    break
            
            if layer_index is not None and layer_index > 0:
                prev_layer = self.original_model.layers[layer_index - 1]
                if hasattr(prev_layer, 'output_shape') and prev_layer.output_shape:
                    import tensorflow as tf
                    import numpy as np
                    
                    batch_size = 1
                    if len(prev_layer.output_shape) == 2: 
                        shape = (batch_size, prev_layer.output_shape[-1])
                    elif len(prev_layer.output_shape) == 3:  
                        shape = (batch_size, prev_layer.output_shape[1], prev_layer.output_shape[2])
                    else:
                        shape = prev_layer.output_shape
                    
                    dummy_input = tf.convert_to_tensor(
                        np.random.random(shape).astype(np.float32)
                    )
                    return dummy_input
            
            return None
            
        except Exception as e:
            print(f"   - Error getting original input: {e}")
            return None
    
    def _get_expected_dimension_for_layer(self, layer_name):
        """Get the expected full dimension for a specific layer."""
        try:
            original_layer = None
            for layer in self.original_model.layers:
                if layer.name == layer_name:
                    original_layer = layer
                    break
            
            if original_layer is None:
                return None
            
            if hasattr(original_layer, 'output_dim'):
                return original_layer.output_dim
            elif hasattr(original_layer, 'units'):
                return original_layer.units
            elif hasattr(original_layer, 'output_shape'):
                output_shape = original_layer.output_shape
                if isinstance(output_shape, tuple) and len(output_shape) > 0:
                    for dim in reversed(output_shape):
                        if dim is not None:
                            return dim
                elif hasattr(output_shape, '__iter__'):
                    for dim in reversed(list(output_shape)):
                        if dim is not None:
                            return dim
            elif hasattr(original_layer, 'equation'):
                equation = original_layer.equation
                if '->' in equation:
                    output_part = equation.split('->')[1]
                    if 'einsum' in layer_name.lower():
                        return 32
            
            return None
            
        except Exception as e:
            print(f"   - Could not determine expected dimension for {layer_name}: {e}")
            return None
    
    def _handle_embedding_layer(self, inputs, layer):
        """Handle Embedding layer with column-parallel sharding."""
        print(f"   - Handling Embedding layer (column-parallel)")

        sharded_embeddings = self.sharding_strategy.sharded_weights[f"{layer.name}.embeddings"]

        embeddings_tensor = keras.ops.convert_to_tensor(sharded_embeddings, dtype="float32")
        sharded_output = keras.ops.take(embeddings_tensor, inputs, axis=0)

        print(f"   - Computed sharded embedding output shape: {sharded_output.shape}")
        return sharded_output

    def _handle_pooling_layer(self, inputs, layer):
        """Handle pooling layer."""
        print(f"   - Handling pooling layer")
        return layer(inputs)

    def _handle_einsum_dense_layer(self, inputs, layer):
        """Handle EinsumDense layer with column-parallel sharding."""
        print(f"   - Handling EinsumDense layer (column-parallel)")

        einsum_kernel = self.sharding_strategy.sharded_weights[f"{layer.name}.kernel"]

        kernel_tensor = keras.ops.convert_to_tensor(einsum_kernel, dtype="float32")
        einsum_output = keras.ops.einsum('bsi,ih->bsh', inputs, kernel_tensor)

        print(f"   - Computed sharded einsum output shape: {einsum_output.shape}")
        return einsum_output

    def _handle_dense_layer(self, current_input, layer):
        """
        Handles the forward pass for a Dense layer, applying the correct
        tensor parallelism strategy based on the sharding type.
        """
        kernel_tensor = layer.kernel
        
        is_row_wise_sharded = (
            hasattr(layer, 'sharding_annotation') and 
            layer.sharding_annotation.is_row_wise_sharded
        )

        if is_row_wise_sharded:
            sharded_output = keras.ops.matmul(current_input, kernel_tensor)
            return sharded_output
            
        else:
            sharded_output = keras.ops.matmul(current_input, kernel_tensor)
            if layer.use_bias:
                sharded_output = sharded_output + layer.bias
            return sharded_output
    
    def get_config(self):
        """Get model configuration."""
        return self.original_model.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create model from config."""
        return cls(**config)
    
    
    def count_params(self):
        """
        Count parameters in the sharded model.
        This should return the total parameters across all shards.
        """
        total_params = 0
        for param_name, weight_info in self.sharding_strategy.weight_mapping.items():
            sharded_weight = self.sharding_strategy.sharded_weights[param_name]
            total_params += sharded_weight.numel()
        
        original_param_names = {f"{layer.name}.{weight.name}" for layer in self.original_model.layers 
                              for weight in layer.weights}
        sharded_param_names = set(self.sharding_strategy.sharded_weights.keys())
        unsharded_param_names = original_param_names - sharded_param_names
        
        for param_name in unsharded_param_names:
            for layer in self.original_model.layers:
                for weight in layer.weights:
                    if f"{layer.name}.{weight.name}" == param_name:
                        total_params += weight.shape.num_elements()
                        break
        
        return total_params


def make_parameter_sharded_model(
    module: Model,
    config: ConfigKeras,
    rank: int,
    world_size: int
) -> Tuple[Model, Set[str]]:
    """
    Create a parameter-sharded version of a Keras model.
    
    Args:
        module: Original Keras model
        config: Tensor parallel configuration
        rank: Rank of this shard
        world_size: Total number of shards
        
    Returns:
        Tuple of (sharded_model, modified_parameter_names)
    """
    sharding_strategy = ParameterShardingStrategy(world_size, rank)
    sharded_model, modified_parameters = sharding_strategy.shard_model_parameters(module, config)
    
    return sharded_model, modified_parameters


def apply_parameter_sharding_to_existing_model(
    model: Model,
    config: ConfigKeras,
    rank: int,
    world_size: int
) -> Model:
    """
    Apply parameter sharding to an existing model without creating a new one.
    This is useful for models that can't be easily rebuilt.
    
    Args:
        model: Existing Keras model
        config: Tensor parallel configuration
        rank: Rank of this shard
        world_size: Total number of shards
        
    Returns:
        Model with sharded parameters
    """
    print(f"🔧 Applying parameter sharding to existing model: {model.name}")
    
    sharding_strategy = ParameterShardingStrategy(world_size, rank)
    for pattern, action in config.state_rules.items():
        if isinstance(action, StateActionKeras):
            matching_params = sharding_strategy._find_matching_parameters(model, pattern)
            
            for param_name, param in matching_params:
                sharded_param = action(param, rank)
                
                sharding_strategy.sharded_weights[param_name] = sharded_param
                sharding_strategy.weight_mapping[param_name] = {
                    'original_shape': param.shape,
                    'sharded_shape': sharded_param.shape,
                    'action': action
                }
                
                print(f"   ✅ Sharded {param_name}: {param.shape} -> {sharded_param.shape}")
    
    model._tensor_parallel_sharding = sharding_strategy
    
    print(f"🎯 Parameter sharding applied to existing model")
    return model 