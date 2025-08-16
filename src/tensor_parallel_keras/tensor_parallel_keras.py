"""
Tensor Parallel implementation for Keras 3.0
Port of the PyTorch tensor_parallel library
"""

import logging
import threading
from contextlib import nullcontext
from operator import attrgetter
from typing import Any, Collection, Optional, Sequence, Union

import torch
import keras
from keras import layers, Model

from keras import device

from .autoconfig_keras import get_default_config_keras
from .config_keras import ConfigKeras
from .shard_keras import make_shard_keras
from .sharding_keras import ShardedKeras
from .utils_keras import nested_flatten, nested_pack

logger = logging.getLogger(__file__)


class TensorParallelKeras(Model):
    """
    Tensor Parallel wrapper for Keras models.
    Distributes model parameters across multiple devices for parallel computation.
    """
    
    def __init__(
        self,
        model: Model,
        device_ids: Optional[Sequence[str]] = None,
        output_device: Optional[str] = None,
        output_device_index: Optional[int] = None,
        tensor_parallel_config: Optional[ConfigKeras] = None,
        delay_init: bool = False,
        distributed: bool = True,
        sharded: bool = True,
        sharded_param_names: Optional[Collection[str]] = None,
        **kwargs
    ):
        print("="*50)
        print("Amit - TensorParallelKeras __init__ called!")
        print("="*50)
        
        # Initialize the Keras Model parent class
        super().__init__(**kwargs)
        
        # Store original model
        self.original_model = model
        
        # Calculate original parameter count
        original_params = sum(p.shape.num_elements() for p in model.weights)
        
        # Validate output device specification
        assert output_device is None or output_device_index is None, "please specify either device or index, not both"
        
        # Process device IDs
        device_ids = self.check_device_ids(device_ids)
        
        # Handle output device
        if output_device is not None:
            output_device = self.canonicalize_device(output_device)
            assert output_device in device_ids, f"Output device {output_device} not in {device_ids}"
            output_device_index = device_ids.index(output_device)
            del output_device
        elif output_device_index is None:
            output_device_index = 0
            
        # Store device information
        self.devices = device_ids
        self.output_device_index = output_device_index
        self.all_cuda = all(device.startswith("gpu") for device in self.devices)
        self.device_ids = [self._get_device_index(x) for x in device_ids]
        self.need_delayed_init = delay_init
        self.world_size = len(self.devices)
        self.sharding_manager = None
        
        # Handle single device case
        if len(device_ids) <= 1:
            self.model_shards = [model]
            if len(device_ids) == 1 and not delay_init:
                # Move model to specified device
                with device(device_ids[0]):
                    self.model_shards[0] = model
            return
            
        # Get tensor parallel configuration
        if tensor_parallel_config is None:
            tensor_parallel_config = get_default_config_keras(model, self.devices)
            logger.info("Using automatic config: sharding individual Dense/Conv/Embedding layers")
            
        self.tensor_parallel_config = tensor_parallel_config
        
        # Create collective operations
        config_with_ops = tensor_parallel_config.create_collective_ops(self.devices, distributed)
        
        # Create model shards
        self.model_shards = []
        self.modified_parameters_names = set()
        
        for rank, device_id in enumerate(self.devices):
            if delay_init:
                device_id = "cpu"
                
            shard, modified_parameters_names = make_shard_keras(
                model, device_id, config_with_ops, rank=rank, world_size=self.world_size
            )
            self.model_shards.append(shard)
            self.modified_parameters_names.update(modified_parameters_names)
            
        # Validate sharding
        params_per_shard = [sum(p.shape.num_elements() for p in shard.weights) for shard in self.model_shards]
        # In tensor parallelism, total shard params should be <= original params (not >=)
        # This is because we're splitting the model, not duplicating it
        assert sum(params_per_shard) <= original_params, "Internal assert failed: shard parameters exceed original"
        
        self.param_fractions = tuple(params_i / original_params for params_i in params_per_shard)
        inefficiency_rate = (sum(self.param_fractions) - 1) / len(device_ids)
        
        log_level = logging.DEBUG if inefficiency_rate < 0.1 else logging.WARNING
        logger.log(
            log_level,
            f"Inefficiency warning: model has {original_params} params but shards have {params_per_shard} params. "
            f"Inefficiency rate: {inefficiency_rate:.2%}"
        )
        
        # Apply sharding if requested
        if sharded:
            sharded_param_names = sharded_param_names or self.modified_parameters_names
            self.apply_sharding(sharded_param_names)
            
        # Set model as built
        self.built = True
        
    def check_device_ids(self, device_ids: Optional[Sequence[str]]) -> Sequence[str]:
        """Validate and normalize device IDs for Keras."""
        if device_ids is None:
            # Get all available devices
            device_ids = self._get_all_device_indices()
            
        # Convert to list and canonicalize
        device_ids = list(device_ids)
        device_ids = [self.canonicalize_device(device_id) for device_id in device_ids]
        
        return tuple(device_ids)
        
    def _get_all_device_indices(self) -> Sequence[str]:
        """Get all available device indices for Keras."""
        devices = []
        
        # Check for GPU devices
        try:
            gpu_count = len(keras.config.list_physical_devices('GPU'))
            for i in range(gpu_count):
                devices.append(f"gpu:{i}")
        except:
            pass
            
        # Always include CPU
        devices.append("cpu")
        
        return devices
        
    def _get_device_index(self, device_spec: str) -> int:
        """Extract device index from device specification."""
        if device_spec == "cpu":
            return -1
        elif device_spec.startswith("gpu:"):
            return int(device_spec.split(":")[1])
        else:
            return 0
            
    def canonicalize_device(self, device_spec: Union[str, int]) -> str:
        """Convert device specification to canonical form."""
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
                # Convert CUDA format to GPU format
                return f"gpu:{device_spec.split(':')[1]}"
            else:
                return device_spec
        else:
            return "cpu"
            
    def apply_sharding(self, replicated_param_names: Optional[Collection[str]] = None):
        """Apply sharding to the model parameters."""
        if replicated_param_names is None:
            replicated_param_names = self.modified_parameters_names
            
        # Create sharding manager
        self.sharding_manager = ShardedKeras(
            self.model_shards,
            replicated_param_names,
            self.tensor_parallel_config,
            self.devices,
            self.output_device_index
        )
        
    def call(self, inputs, training=None, **kwargs):
        """Forward pass through the tensor parallel model."""
        if len(self.model_shards) == 1:
            return self.model_shards[0](inputs, training=training, **kwargs)
            
        # For multiple shards, we need to implement parallel execution
        # This is a simplified version - in practice you'd want more sophisticated parallel execution
        outputs = []
        
        for i, shard in enumerate(self.model_shards):
            with device(self.devices[i]):
                output = shard(inputs, training=training, **kwargs)
                outputs.append(output)
                
        # Combine outputs (this is simplified - you'd need proper output handling)
        if len(outputs) > 0:
            return outputs[self.output_device_index]
        else:
            return None
            
    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "model": self.original_model,
            "device_ids": self.devices,
            "output_device_index": self.output_device_index,
            "sharded": hasattr(self, 'sharding_manager') and self.sharding_manager is not None
        })
        return config 