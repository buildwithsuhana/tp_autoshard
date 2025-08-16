"""
Automatic configuration for Keras Tensor Parallel
"""

import re
from typing import Dict, Sequence, Union

import torch
from keras import layers, Model

from .config_keras import ConfigKeras
from .state_actions_keras import SplitKeras, GatherKeras, SumKeras


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
                    
            # Handle Attention layers
            elif isinstance(layer, layers.MultiHeadAttention):
                # Split query, key, value projections
                state_rules[f"^{full_name}.query_dense.kernel$"] = SplitKeras(world_size=world_size, dim=0)
                state_rules[f"^{full_name}.key_dense.kernel$"] = SplitKeras(world_size=world_size, dim=0)
                state_rules[f"^{full_name}.value_dense.kernel$"] = SplitKeras(world_size=world_size, dim=0)
                state_rules[f"^{full_name}.output_dense.kernel$"] = SplitKeras(world_size=world_size, dim=1)
                
                # Output needs to be gathered
                output_rules[f"^{full_name}$"] = {0: "gather -1"}
                
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