"""
Automatic configuration for Keras Tensor Parallel
"""

import re
import logging
from typing import Dict, Sequence, Union

import torch
from keras import layers, Model

from .config_keras import ConfigKeras
from .state_actions_keras import SplitKeras, GatherKeras, SumKeras

logger = logging.getLogger(__name__)


def get_default_config_keras(module: Model, device_ids: Sequence[str], sharding_strategy: str = "auto") -> ConfigKeras:
    """
    Generate default tensor parallel configuration for Keras models.
    
    Args:
        module: Keras model to parallelize
        device_ids: List of device IDs to use
        sharding_strategy: Sharding strategy - "auto", "row", "column", or "mixed"
        
    Returns:
        ConfigKeras object with sharding rules
    """
    world_size = len(device_ids)
    state_rules = {}
    input_rules = {}
    output_rules = {}
    attr_rules = {}
    
    # Recursively process all layers
    def process_module(module: Model, prefix: str = ""):
        for layer in module.layers:
            name = layer.name
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Handle Dense layers (equivalent to PyTorch Linear)
            if isinstance(layer, layers.Dense):
                # For now, we only support column-wise sharding (output feature splitting)
                # Row-wise sharding requires complex input preprocessing that's not implemented yet
                if sharding_strategy in ["row", "mixed"]:
                    # Fall back to column-wise for unsupported strategies
                    kernel_dim = 1
                    bias_dim = 0
                    effective_strategy = "column"
                else:  # "auto" or "column"
                    # Column-wise: Split output features (dim=1)
                    kernel_dim = 1
                    bias_dim = 0
                    effective_strategy = "column"
                
                # Apply sharding rules
                state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                    world_size=world_size, 
                    dim=kernel_dim,
                    sharding_type=effective_strategy
                )
                if layer.use_bias:
                    state_rules[f"^{full_name}.bias$"] = SplitKeras(
                        world_size=world_size, 
                        dim=bias_dim,
                        sharding_type="row" if bias_dim == 0 else "column"
                    )
                
                # Output needs to be gathered
                output_rules[f"^{full_name}$"] = {0: "gather -1"}
                
            # Handle Embedding layers
            elif isinstance(layer, layers.Embedding):
                # Split along embedding dimension (dim=1 for Keras)
                # Keras: embeddings shape is (input_dim, output_dim)
                # We want to split output_dim (embedding dimension), so split along dim=1
                state_rules[f"^{full_name}.embeddings$"] = SplitKeras(world_size=world_size, dim=1)
                
                # Output needs to be gathered
                output_rules[f"^{full_name}$"] = {0: "gather -1"}
                
            # Handle Conv2D layers
            elif isinstance(layer, (layers.Conv2D, layers.Conv1D, layers.Conv3D)):
                # Split along output channels (dim=-1)
                state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size=world_size, dim=-1)
                if layer.use_bias:
                    state_rules[f"^{full_name}.bias$"] = SplitKeras(world_size=world_size, dim=-1)
                    
                # Output needs to be gathered
                output_rules[f"^{full_name}$"] = {0: "gather -1"}
                
            # Handle LSTM layers
            elif isinstance(layer, layers.LSTM):
                # Split along hidden size dimension
                state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size=world_size, dim=1)
                if layer.recurrent_activation == "sigmoid":
                    # For LSTM, we need to handle the gates properly
                    pass
                    
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
            
            # Handle MLP/Feed-Forward Network layers with Column -> Row pattern
            elif isinstance(layer, layers.Dense) and "mlp" in full_name.lower():
                # MLP layers follow the same Column -> Row pattern as attention
                # First Dense layer (up-projection): Column-sharded (split output dimension)
                if "up" in full_name.lower() or "h_to_4h" in full_name.lower() or "gate" in full_name.lower():
                    # Column-wise sharding: split output dimension
                    state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size=world_size, dim=1)
                    if layer.use_bias:
                        state_rules[f"^{full_name}.bias$"] = SplitKeras(world_size=world_size, dim=0)
                    
                    # Output is distributed (no communication needed)
                    output_rules[f"^{full_name}$"] = {0: "no_comm"}
                    
                    logger.info(f"Applied Column-wise sharding to MLP up-projection {full_name}")
                
                # Second Dense layer (down-projection): Row-sharded (split input dimension)
                elif "down" in full_name.lower() or "4h_to_h" in full_name.lower() or "proj" in full_name.lower():
                    # Row-wise sharding: split input dimension
                    state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size=world_size, dim=0)
                    if layer.use_bias:
                        state_rules[f"^{full_name}.bias$"] = SplitKeras(world_size=world_size, dim=0)
                    
                    # Output needs AllReduce to combine
                    output_rules[f"^{full_name}$"] = {0: "allreduce"}
                    
                    logger.info(f"Applied Row-wise sharding to MLP down-projection {full_name}")
                
                # Generic MLP handling (if naming doesn't match specific patterns)
                else:
                    # Assume first layer is column-sharded, second is row-sharded
                    # This is a fallback for generic MLP layers
                    if "first" in full_name.lower() or "1" in full_name:
                        state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size=world_size, dim=1)
                        output_rules[f"^{full_name}$"] = {0: "no_comm"}
                        logger.info(f"Applied Column-wise sharding to generic MLP layer {full_name}")
                    else:
                        state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size=world_size, dim=0)
                        output_rules[f"^{full_name}$"] = {0: "allreduce"}
                        logger.info(f"Applied Row-wise sharding to generic MLP layer {full_name}")
                
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
        input_rules=input_rules,
        output_rules=output_rules,
        attr_rules=attr_rules
    ) 