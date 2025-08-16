"""
Shard creation for Keras Tensor Parallel
"""

import copy
import re
from typing import Dict, List, Set, Tuple, Any, Optional

import torch
import numpy as np
import keras
from keras import Model, layers
from keras.backend import standardize_dtype

from .config_keras import ConfigKeras
from .state_actions_keras import StateActionKeras


def safe_layer_call(layer, x):
    """Safely call a layer with proper error handling for missing arguments."""
    try:
        return layer(x)
    except TypeError as e:
        if "missing a required argument" in str(e):
            # Handle layers that need additional arguments
            if hasattr(layer, 'call'):
                import inspect
                sig = inspect.signature(layer.call)
                if 'training' in sig.parameters and 'value' in sig.parameters:
                    # Some layers need both training and value
                    return layer(x, training=False, value=x)
                elif 'training' in sig.parameters:
                    return layer(x, training=False)
                elif 'mask' in sig.parameters:
                    return layer(x, mask=None)
                elif 'value' in sig.parameters:
                    # Some layers expect a 'value' argument
                    return layer(x, value=x)
                elif 'inputs' in sig.parameters:
                    # Some layers expect 'inputs' argument
                    return layer(inputs=x)
                else:
                    print(f"Warning: Could not reconstruct layer {layer.name}, using fallback")
                    return keras.ops.convert_to_tensor(x)
        else:
            raise e


def make_shard_keras(
    module: Model,
    device: str,
    config: ConfigKeras,
    *,
    rank: int,
    world_size: int,
) -> Tuple[Model, Set[str]]:
    """
    Create a shard of a Keras model for tensor parallelism.
    
    Args:
        module: Original Keras model
        device: Target device for this shard
        config: Tensor parallel configuration
        rank: Rank of this shard
        world_size: Total number of shards
        
    Returns:
        Tuple of (sharded_model, modified_parameter_names)
    """
    # Track modified parameter names
    modified_parameters_names = set()
    
    # Process state rules to get modified parameter names
    for pattern, action in config.state_rules.items():
        if isinstance(action, StateActionKeras):
            # Find matching parameters
            matching_params = find_matching_parameters(module, pattern)
            
            for param_name, param in matching_params:
                modified_parameters_names.add(param_name)
    
    # Rebuild the model with sharded layers
    sharded_module = rebuild_model_with_sharded_layers(module, config, rank, world_size)
    
    # Process input and output rules
    # (These are handled during forward pass)
    
    return sharded_module, modified_parameters_names


def find_matching_parameters(module: Model, pattern: str) -> List[Tuple[str, torch.Tensor]]:
    """
    Find parameters in a Keras model that match a given pattern.
    
    Args:
        module: Keras model to search
        pattern: Regex pattern to match parameter names
        
    Returns:
        List of (parameter_name, parameter_tensor) tuples
    """
    matching_params = []
    
    def search_module(mod: Model, prefix: str = ""):
        for layer in mod.layers:
            name = layer.name
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this layer has parameters
            if hasattr(layer, 'weights') and layer.weights:
                for weight in layer.weights:
                    # Use the actual weight name (e.g., 'kernel', 'bias')
                    param_name = f"{full_name}.{weight.name}"
                    if re.match(pattern, param_name):
                        # Convert Keras weight to PyTorch tensor for processing
                        weight_tensor = torch.tensor(weight.numpy())
                        matching_params.append((param_name, weight_tensor))
                        
            # Recursively search submodules
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                search_module(layer, full_name)
                
    search_module(module)
    return matching_params


def apply_sharded_weights_to_model(module: Model, config: ConfigKeras, rank: int):
    """
    Apply sharded weights to the model layers.
    
    Args:
        module: Keras model to modify
        config: Tensor parallel configuration
        rank: Rank of this shard
    """
    for pattern, action in config.state_rules.items():
        if isinstance(action, StateActionKeras):
            # Find matching parameters in the original model
            matching_params = find_matching_parameters(module, pattern)
            
            for param_name, param in matching_params:
                # Apply the sharding action
                sharded_param = action(param, rank)
                
                # Apply the sharded weight to the actual layer
                apply_weight_to_layer(module, param_name, sharded_param.numpy())


def apply_weight_to_layer(module: Model, param_name: str, sharded_weight: np.ndarray):
    """
    Apply a sharded weight to a specific layer in the model.
    
    Args:
        module: Keras model
        param_name: Parameter name (e.g., "dense.kernel")
        sharded_weight: Sharded weight value
    """
    # Parse the parameter name to find the layer and weight
    parts = param_name.split('.')
    
    if len(parts) >= 2:
        layer_name = parts[0]
        weight_name = parts[1]
        
        # Find the layer by name
        target_layer = None
        for layer in module.layers:
            if layer.name == layer_name:
                target_layer = layer
                break
        
        if target_layer is not None:
            # Find the weight by name and replace it
            for weight in target_layer.weights:
                if weight.name == weight_name:
                    # Replace the weight value
                    weight.assign(sharded_weight)
                    break


def rebuild_model_with_sharded_layers(original_module: Model, config: ConfigKeras, rank: int, world_size: int) -> Model:
    """
    Rebuild the entire model with sharded layers.
    
    Args:
        original_module: Original Keras model
        config: Tensor parallel configuration
        rank: Rank of this shard
        world_size: Total number of shards
        
    Returns:
        New model with sharded layers
    """
    # Create a completely new model with sharded architecture
    # This is the proper way to handle Keras layer modifications
    
    # Get the input shape from the original model
    input_shape = original_module.inputs[0].shape
    
    # Create new input layer
    inputs = keras.Input(shape=input_shape[1:], name=f"input_shard_{rank}")
    
    # Build the model layer by layer with sharded dimensions
    x = inputs
    
    for layer in original_module.layers:
        if isinstance(layer, layers.InputLayer):
            continue  # Skip input layer
            
        elif isinstance(layer, layers.Dense):
            # Get the sharded output size for this layer
            layer_name = layer.name
            kernel_pattern = f"^{layer_name}.kernel$"
            
            if kernel_pattern in config.state_rules:
                # Get the original kernel to calculate sharded size
                original_kernel = None
                for weight in layer.weights:
                    if weight.name == "kernel":
                        original_kernel = weight
                        break
                
                if original_kernel is not None:
                    # Apply sharding to get the new dimensions
                    action = config.state_rules[kernel_pattern]
                    sharded_kernel = action(torch.tensor(original_kernel.numpy()), rank)
                    
                    # Determine the new layer dimensions based on sharding type
                    if action.sharding_type == "row":
                        # Row-wise: Split input features, keep all output features
                        new_input_size = sharded_kernel.shape[0]
                        new_output_size = sharded_kernel.shape[1]
                        
                        # Create new Dense layer with sharded input size
                        new_layer = layers.Dense(
                            units=new_output_size,
                            activation=layer.activation,
                            use_bias=layer.use_bias,
                            name=layer.name
                        )
                        # Note: We need to handle input reshaping for row-wise sharding
                        # This is more complex and requires input preprocessing
                        x = new_layer(x)
                        
                    else:  # column or auto
                        # Column-wise: Split output features, keep all input features
                        new_output_size = sharded_kernel.shape[1]
                        
                        # Create new Dense layer with sharded output size
                        new_layer = layers.Dense(
                            units=new_output_size,
                            activation=layer.activation,
                            use_bias=layer.use_bias,
                            name=layer.name
                        )
                        x = new_layer(x)
                else:
                    # Fallback to original layer
                    x = safe_layer_call(layer, x)
            else:
                # No sharding rule, use original layer
                x = layer(x)
                
        elif isinstance(layer, layers.Embedding):
            # Handle embedding layers similarly
            layer_name = layer.name
            embeddings_pattern = f"^{layer_name}.embeddings$"
            
            if embeddings_pattern in config.state_rules:
                # Get the original embeddings to calculate sharded size
                original_embeddings = None
                for weight in layer.weights:
                    if weight.name == "embeddings":
                        original_embeddings = weight
                        break
                
                if original_embeddings is not None:
                    # Apply sharding to get the new embedding dimension
                    action = config.state_rules[embeddings_pattern]
                    sharded_embeddings = action(torch.tensor(original_embeddings.numpy()), rank)
                    new_embedding_dim = sharded_embeddings.shape[1]
                    
                    # Create new Embedding layer with sharded embedding dimension
                    new_layer = layers.Embedding(
                        input_dim=layer.input_dim,
                        output_dim=new_embedding_dim,
                        name=layer.name
                    )
                    x = new_layer(x)
                else:
                    # Fallback to original layer
                    x = safe_layer_call(layer, x)
            else:
                # No sharding rule, use original layer
                x = safe_layer_call(layer, x)
                
        elif isinstance(layer, layers.LayerNormalization):
            # LayerNormalization needs to be updated to work with sharded dimensions
            # The axis parameter should work with the current tensor shape
            layer_name = layer.name
            
            # Create new LayerNormalization with the same parameters
            new_layer = layers.LayerNormalization(
                axis=layer.axis,
                epsilon=layer.epsilon,
                center=layer.center,
                scale=layer.scale,
                name=layer.name
            )
            x = new_layer(x)
                
        else:
            # For other layer types, use safe layer call
            x = safe_layer_call(layer, x)
    
    # Create the new model
    new_model = Model(inputs=inputs, outputs=x, name=f"{original_module.name}_shard_{rank}")
    
    # Build the model so we can access weights
    new_model.build(input_shape)
    

    
    return new_model


def apply_sharding_to_model(module: Model, param_name: str, sharded_param: torch.Tensor):
    """
    Apply sharding to a model by modifying layer dimensions.
    
    Args:
        module: Keras model
        param_name: Parameter name (e.g., "dense.kernel")
        sharded_param: Sharded parameter tensor
    """
    # Parse the parameter name to find the layer and weight
    parts = param_name.split('.')
    
    if len(parts) >= 2:
        layer_name = parts[0]
        weight_name = parts[1]
        
        # Find the layer by name
        target_layer = None
        for layer in module.layers:
            if layer.name == layer_name:
                target_layer = layer
                break
        
        if target_layer is not None and weight_name == "kernel":
            # For kernel weights, we need to modify the layer's output size
            if isinstance(target_layer, layers.Dense):
                # Get the new output size from the sharded kernel
                new_output_size = sharded_param.shape[1]
                
                # Create a new Dense layer with the sharded output size
                new_layer = layers.Dense(
                    units=new_output_size,
                    activation=target_layer.activation,
                    use_bias=target_layer.use_bias,
                    kernel_initializer=target_layer.kernel_initializer,
                    bias_initializer=target_layer.bias_initializer,
                    kernel_regularizer=target_layer.kernel_regularizer,
                    bias_regularizer=target_layer.bias_regularizer,
                    activity_regularizer=target_layer.activity_regularizer,
                    kernel_constraint=target_layer.kernel_constraint,
                    bias_constraint=target_layer.bias_constraint,
                    name=target_layer.name
                )
                
                # Replace the layer in the module
                layer_index = module.layers.index(target_layer)
                module.layers[layer_index] = new_layer
                
                # Set the sharded weights after building
                if hasattr(module, '_sharded_weights'):
                    module._sharded_weights[param_name] = sharded_param.numpy()
                else:
                    module._sharded_weights = {param_name: sharded_param.numpy()}
                        
        elif target_layer is not None and weight_name == "embeddings":
            # For embedding weights, we need to modify the layer's output dimension
            if isinstance(target_layer, layers.Embedding):
                # Get the new embedding dimension from the sharded embeddings
                new_embedding_dim = sharded_param.shape[1]
                
                # Create a new Embedding layer with the sharded embedding dimension
                new_layer = layers.Embedding(
                    input_dim=target_layer.input_dim,
                    output_dim=new_embedding_dim,
                    embeddings_initializer=target_layer.embeddings_initializer,
                    embeddings_regularizer=target_layer.embeddings_regularizer,
                    embeddings_constraint=target_layer.embeddings_constraint,
                    mask_zero=target_layer.mask_zero,
                    input_length=target_layer.input_length,
                    name=target_layer.name
                )
                
                # Replace the layer in the module
                layer_index = module.layers.index(target_layer)
                module.layers[layer_index] = new_layer
                
                # Set the sharded weights after building
                if hasattr(module, '_sharded_weights'):
                    module._sharded_weights[param_name] = sharded_param.numpy()
                else:
                    module._sharded_weights = {param_name: sharded_param.numpy()}


def replace_layer_with_sharded(module: Model, param_name: str, sharded_param: torch.Tensor):
    """
    Replace a layer with a sharded version that has the correct parameter shapes.
    
    Args:
        module: Keras model
        param_name: Parameter name (e.g., "dense.kernel")
        sharded_param: Sharded parameter tensor
    """
    # Parse the parameter name to find the layer and weight
    parts = param_name.split('.')
    
    if len(parts) >= 2:
        layer_name = parts[0]
        weight_name = parts[1]
        
        # Find the layer by name
        target_layer = None
        target_layer_index = -1
        for i, layer in enumerate(module.layers):
            if layer.name == layer_name:
                target_layer = layer
                target_layer_index = i
                break
        
        if target_layer is not None and weight_name == "kernel":
            # For kernel weights, we need to create a new layer with the correct output size
            if isinstance(target_layer, layers.Dense):
                # Get the new output size from the sharded kernel
                new_output_size = sharded_param.shape[1]
                
                # Create a new Dense layer with the sharded output size
                new_layer = layers.Dense(
                    units=new_output_size,
                    activation=target_layer.activation,
                    use_bias=target_layer.use_bias,
                    kernel_initializer=target_layer.kernel_initializer,
                    bias_initializer=target_layer.bias_initializer,
                    kernel_regularizer=target_layer.kernel_regularizer,
                    bias_regularizer=target_layer.bias_regularizer,
                    activity_regularizer=target_layer.activity_regularizer,
                    kernel_constraint=target_layer.kernel_constraint,
                    bias_constraint=target_layer.bias_constraint,
                    name=target_layer.name
                )
                
                # Set the sharded weights
                new_layer.build(target_layer.input_spec.shape)
                new_layer.kernel.assign(sharded_param.numpy())
                
                # Replace the layer in the module
                module.layers[target_layer_index] = new_layer
                
                # Update the model's layer references
                if hasattr(module, '_layers'):
                    module._layers[target_layer_index] = new_layer
                    
        elif target_layer is not None and weight_name == "embeddings":
            # For embedding weights, we need to create a new layer with the correct embedding dimension
            if isinstance(target_layer, layers.Embedding):
                # Get the new embedding dimension from the sharded embeddings
                new_embedding_dim = sharded_param.shape[1]
                
                # Create a new Embedding layer with the sharded embedding dimension
                new_layer = layers.Embedding(
                    input_dim=target_layer.input_dim,
                    output_dim=new_embedding_dim,
                    embeddings_initializer=target_layer.embeddings_initializer,
                    embeddings_regularizer=target_layer.embeddings_regularizer,
                    embeddings_constraint=target_layer.embeddings_constraint,
                    mask_zero=target_layer.mask_zero,
                    input_length=target_layer.input_length,
                    name=target_layer.name
                )
                
                # Set the sharded weights
                new_layer.build(target_layer.input_spec.shape)
                new_layer.embeddings.assign(sharded_param.numpy())
                
                # Replace the layer in the module
                module.layers[target_layer_index] = new_layer
                
                # Update the model's layer references
                if hasattr(module, '_layers'):
                    module._layers[target_layer_index] = new_layer


def set_parameter_value(module: Model, param_name: str, value: torch.Tensor):
    """
    Set a parameter value in a Keras model.
    
    Args:
        module: Keras model
        param_name: Parameter name (e.g., "dense.kernel")
        value: New parameter value
    """
    # Parse the parameter name to find the layer and weight
    parts = param_name.split('.')
    
    if len(parts) >= 2:
        layer_name = parts[0]
        weight_name = parts[1]
        
        # Find the layer by name
        target_layer = None
        for layer in module.layers:
            if layer.name == layer_name:
                target_layer = layer
                break
        
        if target_layer is not None:
            # Find the weight by name
            for weight in target_layer.weights:
                if weight.name == weight_name:
                    # Convert PyTorch tensor back to Keras format and assign
                    weight_value = value.numpy()
                    weight.assign(weight_value)
                    break


def make_distributed_shard_keras(module: Model, device: str, config: Optional[ConfigKeras] = None):
    """
    Create a distributed shard for a Keras model.
    
    Args:
        module: Keras model
        device: Target device
        config: Optional tensor parallel configuration
        
    Returns:
        Distributed shard
    """
    # For now, just return the module as-is
    # In a real implementation, you'd implement distributed sharding
    return module


def _maybe_wrap_submodule_keras(
    config: ConfigKeras,
    name: str,
    module: Model,
    *,
    rank: int,
    world_size: int
) -> Model:
    """
    Wrap a submodule if needed for tensor parallelism.
    
    Args:
        config: Tensor parallel configuration
        name: Module name
        module: Module to wrap
        
    Returns:
        Wrapped module
    """
    # For now, just return the module as-is
    # In a real implementation, you'd implement submodule wrapping
    return module 