"""
Automatic configuration for Keras Tensor Parallel
"""

import logging
from typing import Dict, Any, Sequence, List, Tuple, Optional
from keras import Model, layers

import torch
from .config_keras import ConfigKeras
from .state_actions_keras import SplitKeras, GatherKeras, SumKeras

logger = logging.getLogger(__name__)


def analyze_dense_layer_directly(layer: layers.Dense, module: Model, prefix: str) -> str:
    """
    Analyze a Dense layer directly to determine its MLP role.
    No name lookups - pure structural analysis.
    
    Args:
        layer: The Dense layer to analyze
        module: The parent module containing the layer
        prefix: Current layer name prefix for context
        
    Returns:
        String indicating the layer type: 'up_projection', 'down_projection', or 'generic_dense'
    """
    
    # Get layer dimensions
    input_shape = getattr(layer, 'input_shape', None)
    output_shape = getattr(layer, 'output_shape', None)
    
    if not input_shape or not output_shape:
        return 'generic_dense'
    
    input_dim = input_shape[-1] if input_shape else None
    output_dim = output_shape[-1] if output_shape else None
    
    if not input_dim or not output_dim:
        return 'generic_dense'
    
    # Check expansion/contraction patterns
    expansion_threshold = 2.0
    is_expansion = output_dim > input_dim * expansion_threshold
    is_contraction = input_dim > output_dim * expansion_threshold
    
    # Find connected Dense layers in the same module for context
    connected_dense_count = count_connected_dense_layers(module)
    
    # Analyze the pattern
    if is_expansion and connected_dense_count >= 1:
        return 'up_projection'
    elif is_contraction and connected_dense_count >= 1:
        return 'down_projection'
    else:
        return 'generic_dense'


def count_connected_dense_layers(module: Model) -> int:
    """
    Count the number of Dense layers in the module for context analysis.
    """
    dense_count = 0
    for layer in module.layers:
        if isinstance(layer, layers.Dense):
            dense_count += 1
    return dense_count


def get_default_config_keras(module: Model, device_ids: Sequence[str]) -> ConfigKeras:
    """
    Get default configuration for Keras tensor parallel.
    
    Args:
        module: Keras model to analyze
        device_ids: List of device IDs for sharding
        
    Returns:
        ConfigKeras object with sharding rules
    """
    world_size = len(device_ids)
    state_rules = {}
    output_rules = {}
    
    # Recursively process all layers
    def process_module(module: Model, prefix: str = ""):
        for layer in module.layers:
            name = layer.name
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Handle Dense layers (equivalent to PyTorch Linear)
            if isinstance(layer, layers.Dense):
                # Analyze the layer directly to determine its MLP role
                mlp_type = analyze_dense_layer_directly(layer, module, full_name)
                
                if mlp_type == 'up_projection':
                    # Column-wise sharding: split output dimension
                    state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                        world_size=world_size, 
                        dim=1,
                        sharding_type="column"
                    )
                    if layer.use_bias:
                        state_rules[f"^{full_name}.bias$"] = SplitKeras(
                            world_size=world_size, 
                            dim=0,
                            sharding_type="row"
                        )
                    
                    # Output is distributed (no communication needed)
                    output_rules[f"^{full_name}$"] = {0: "no_comm"}
                    
                    logger.info(f"Applied Column-wise sharding to MLP up-projection {full_name} (direct-analysis)")
                
                elif mlp_type == 'down_projection':
                    # Row-wise sharding: split input dimension
                    state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                        world_size=world_size, 
                        dim=0,
                        sharding_type="row"
                    )
                    if layer.use_bias:
                        state_rules[f"^{full_name}.bias$"] = SplitKeras(
                            world_size=world_size, 
                            dim=0,
                            sharding_type="row"
                        )
                    
                    # Output needs AllReduce to combine
                    output_rules[f"^{full_name}$"] = {0: "allreduce"}
                    
                    logger.info(f"Applied Row-wise sharding to MLP down-projection {full_name} (direct-analysis)")
                
                else:
                    # Generic Dense layer handling (fallback)
                    # Always use column-wise sharding for generic Dense layers (optimal approach)
                    kernel_dim = 1  # Split output features
                    bias_dim = 0    # Split bias along output dimension
                    
                    # Apply sharding rules
                    state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                        world_size=world_size, 
                        dim=kernel_dim,
                        sharding_type="column"
                    )
                    if layer.use_bias:
                        state_rules[f"^{full_name}.bias$"] = SplitKeras(
                            world_size=world_size, 
                            dim=bias_dim,
                            sharding_type="row"
                        )
                    
                    # Output needs to be gathered
                    output_rules[f"^{full_name}$"] = {0: "gather -1"}
                
            # Handle EinsumDense layers (common in transformer models like OPT)
            elif isinstance(layer, layers.EinsumDense):
                # EinsumDense has equation parameter
                # We need to determine the sharding strategy based on the equation
                equation = layer.equation
                
                # Common patterns for EinsumDense:
                # - "btd,de->bte" (input projection): split output dimension 'e'
                # - "bte,de->btd" (output projection): split input dimension 'e'
                # - "btd,de->bte" (MLP): split output dimension 'e'
                
                # Default to column-wise sharding (split output dimension)
                # This is the most common and safe approach for EinsumDense
                state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                    world_size=world_size, 
                    dim=1,  # Split along output dimension (dim=1)
                    sharding_type="column"
                )
                
                # Apply sharding to bias if it exists
                if hasattr(layer, 'bias') and layer.bias is not None:
                    state_rules[f"^{full_name}.bias$"] = SplitKeras(
                        world_size=world_size, 
                        dim=0,
                        sharding_type="row"
                    )
                
                # Output needs to be gathered
                output_rules[f"^{full_name}$"] = {0: "gather -1"}
                
                logger.info(f"Applied EinsumDense sharding to {full_name} with equation {equation}")
                
            # Handle TFDense layers (TensorFlow Dense equivalent)
            elif hasattr(layer, '__class__') and 'Dense' in layer.__class__.__name__:
                # Handle various Dense layer variants
                # This catches TFDense, Dense, and other Dense-like layers
                
                # Check if it has kernel and bias attributes
                if hasattr(layer, 'kernel') or hasattr(layer, 'weights'):
                    # Determine the weight attribute name
                    weight_attr = 'kernel' if hasattr(layer, 'kernel') else 'weights'
                    
                    # Apply sharding rules
                    state_rules[f"^{full_name}.{weight_attr}$"] = SplitKeras(
                        world_size=world_size, 
                        dim=1,
                        sharding_type="column"
                    )
                    
                    # Handle bias if it exists
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        state_rules[f"^{full_name}.bias$"] = SplitKeras(
                            world_size=world_size, 
                            dim=0,
                            sharding_type="row"
                        )
                    elif hasattr(layer, 'use_bias') and layer.use_bias:
                        # Try to find bias in weights
                        if hasattr(layer, 'weights') and len(layer.weights) > 1:
                            state_rules[f"^{full_name}.weights.1$"] = SplitKeras(
                                world_size=world_size, 
                                dim=0,
                                sharding_type="row"
                            )
                    
                    # Output needs to be gathered
                    output_rules[f"^{full_name}$"] = {0: "gather -1"}
                    
                    logger.info(f"Applied generic Dense sharding to {full_name}")
                
            # Handle Embedding layers
            elif isinstance(layer, layers.Embedding):
                # Split along embedding dimension (dim=1 for Keras)
                # Keras: embeddings shape is (input_dim, output_dim)
                # We want to split output_dim (embedding dimension), so split along dim=1
                state_rules[f"^{full_name}.embeddings$"] = SplitKeras(world_size=world_size, dim=1)
                
                # Output needs to be gathered
                output_rules[f"^{full_name}$"] = {0: "gather -1"}
                
            # Handle Attention layers with Column -> Row pattern
            elif isinstance(layer, layers.MultiHeadAttention):
                # QKV Projection: Column-sharded (split output dimension)
                # This means the weight matrices are split along the output dimension
                state_rules[f"^{full_name}.query_dense.kernel$"] = SplitKeras(world_size=world_size, dim=1)
                state_rules[f"^{full_name}.key_dense.kernel$"] = SplitKeras(world_size=world_size, dim=1)
                state_rules[f"^{full_name}.value_dense.kernel$"] = SplitKeras(world_size=world_size, dim=1)
                
                # QKV biases: Column-sharded (split output dimension)
                if hasattr(layer.query_dense, 'bias') and layer.query_dense.bias is not None:
                    state_rules[f"^{full_name}.query_dense.bias$"] = SplitKeras(world_size=world_size, dim=0)
                if hasattr(layer.key_dense, 'bias') and layer.key_dense.bias is not None:
                    state_rules[f"^{full_name}.key_dense.bias$"] = SplitKeras(world_size=world_size, dim=0)
                if hasattr(layer.value_dense, 'bias') and layer.value_dense.bias is not None:
                    state_rules[f"^{full_name}.value_dense.bias$"] = SplitKeras(world_size=world_size, dim=0)
                
                # Output Projection: Row-sharded (split input dimension)
                # This means the weight matrix is split along the input dimension
                state_rules[f"^{full_name}.output_dense.kernel$"] = SplitKeras(world_size=world_size, dim=0)
                if hasattr(layer.output_dense, 'bias') and layer.output_dense.bias is not None:
                    state_rules[f"^{full_name}.output_dense.bias$"] = SplitKeras(world_size=world_size, dim=0)
                
                # QKV outputs are distributed (no communication needed)
                # Output projection result needs AllReduce to combine
                output_rules[f"^{full_name}$"] = {0: "allreduce"}
                
                logger.info(f"Applied Column->Row pattern to {full_name}")
            
            # Handle LayerNormalization layers
            elif isinstance(layer, layers.LayerNormalization):
                # LayerNormalization needs to be aware of the sharded hidden size
                # We'll handle this by ensuring the axis parameter is correct
                pass
                
            # Handle other normalization layers
            elif isinstance(layer, (layers.BatchNormalization, layers.GroupNormalization)):
                # These layers need to be aware of the sharded dimensions
                pass
                
            # Recursively process submodules
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                process_module(layer, full_name)
                
    # Start processing from the root module
    process_module(module)
    
    return ConfigKeras(
        state_rules=state_rules,
        output_rules=output_rules
    ) 