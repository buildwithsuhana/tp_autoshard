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
                    sharded_param_tensor = action(param, self.rank)
                    
                    # Create a ShardedWeight object which wraps the tensor in a Keras Variable
                    self.sharded_weights[param_name] = ShardedWeight(sharded_param_tensor, param_name, param.trainable)
                    
                    self.weight_mapping[param_name] = {
                        'original_shape': param.shape,
                        'sharded_shape': sharded_param_tensor.shape,
                        'action': action
                    }
                    
                    modified_parameters.add(param_name)
                    print(f"   ✅ Sharded {param_name}: {param.shape} -> {sharded_param_tensor.shape}")
        
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
                    param_name = f"{layer.name}.{weight.name.split(':')[0]}"
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
                        param_name = f"{full_name}.{weight.name.split(':')[0]}"
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
        
        self._map_sharded_weights_to_layers()
        self._build_and_cache_weights()

        if original_model.inputs:
             self.build(original_model.inputs[0].shape)

        print(f"🚀 ParameterShardedModel created successfully")

    def _map_sharded_weights_to_layers(self):
        """Creates a mapping from sharded parameter names to the layer and weight attribute name."""
        self._layer_weight_map = {}
        sharded_names = set(self.sharding_strategy.sharded_weights.keys())
        
        for layer in self.original_model.layers:
            if hasattr(layer, 'weights'):
                for weight in layer.weights:
                    param_name = f"{layer.name}.{weight.name.split(':')[0]}"
                    if param_name in sharded_names:
                        # Find the public attribute name for the weight (e.g., 'kernel', 'bias', 'embeddings')
                        weight_attr_name = None
                        for attr_name in dir(layer):
                            if not attr_name.startswith("_"):
                                try:
                                    attr_value = getattr(layer, attr_name)
                                    if attr_value is weight:
                                        weight_attr_name = attr_name
                                        break
                                except:
                                    continue
                        
                        if weight_attr_name:
                            self._layer_weight_map[param_name] = (layer, weight_attr_name)

    def _build_and_cache_weights(self):
        """
        Builds the list of trainable/non-trainable weights ONCE and caches it.
        This list includes the new sharded variables and any original, non-sharded variables.
        """
        print("   - Building and caching the definitive weights list...")
        weights_list = []
        trainable_weights_list = []
        non_trainable_weights_list = []

        sharded_param_names = set(self.sharding_strategy.sharded_weights.keys())

        # Add sharded weights
        for param_name, sharded_weight_obj in self.sharding_strategy.sharded_weights.items():
            weights_list.append(sharded_weight_obj.variable)
            if sharded_weight_obj.trainable:
                trainable_weights_list.append(sharded_weight_obj.variable)
            else:
                non_trainable_weights_list.append(sharded_weight_obj.variable)

        # Add original, non-sharded weights
        for layer in self.original_model.layers:
            if hasattr(layer, 'weights'):
                for weight in layer.weights:
                    param_name = f"{layer.name}.{weight.name.split(':')[0]}"
                    if param_name not in sharded_param_names:
                        weights_list.append(weight)
                        if weight.trainable:
                            trainable_weights_list.append(weight)
                        else:
                            non_trainable_weights_list.append(weight)
        
        self._weights = weights_list
        self._trainable_weights = trainable_weights_list
        self._non_trainable_weights = non_trainable_weights_list

    def call(self, inputs, training=None, mask=None):
        """
        Implements the 'swap-and-run' strategy for a truly general forward pass.
        """
        original_weights_backup = {}
        try:
            # 1. Backup original weights and SWAP IN sharded weights
            for param_name, sharded_weight_obj in self.sharding_strategy.sharded_weights.items():
                if param_name in self._layer_weight_map:
                    layer, weight_attr_name = self._layer_weight_map[param_name]
                    # Backup the original full weight
                    original_weights_backup[param_name] = getattr(layer, weight_attr_name)
                    # Swap in the sharded weight variable
                    setattr(layer, weight_attr_name, sharded_weight_obj.variable)

            # 2. RUN the forward pass using the original model's logic
            #    It will now automatically use the sharded weights we just swapped in.
            output = self.original_model(inputs, training=training, mask=mask)

        finally:
            # 3. RESTORE the original weights to keep the model in a clean state
            for param_name, original_weight_var in original_weights_backup.items():
                if param_name in self._layer_weight_map:
                    layer, weight_attr_name = self._layer_weight_map[param_name]
                    setattr(layer, weight_attr_name, original_weight_var)
        
        return output
    
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
        This should return the total parameters held by this specific shard.
        """
        total_params = 0
        
        # Count sharded parameters for this shard
        for sharded_weight_obj in self.sharding_strategy.sharded_weights.values():
            # FIX: Use backend-agnostic keras.ops.size
            total_params += keras.ops.size(sharded_weight_obj.variable)

        # Count non-sharded parameters
        sharded_param_names = set(self.sharding_strategy.sharded_weights.keys())
        for layer in self.original_model.layers:
            if hasattr(layer, 'weights'):
                for weight in layer.weights:
                    param_name = f"{layer.name}.{weight.name.split(':')[0]}"
                    if param_name not in sharded_param_names:
                        total_params += keras.ops.size(weight)
        
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
                sharded_param_tensor = action(param, rank)
                
                sharded_weight_obj = ShardedWeight(sharded_param_tensor, param_name, param.trainable)
                sharding_strategy.sharded_weights[param_name] = sharded_weight_obj
                sharding_strategy.weight_mapping[param_name] = {
                    'original_shape': param.shape,
                    'sharded_shape': sharded_param_tensor.shape,
                    'action': action
                }
                
                print(f"   ✅ Sharded {param_name}: {param.shape} -> {sharded_param_tensor.shape}")
    
    model._tensor_parallel_sharding = sharding_strategy
    
    print(f"🎯 Parameter sharding applied to existing model")
    return model

