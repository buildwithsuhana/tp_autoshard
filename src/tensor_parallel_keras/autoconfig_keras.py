"""
Automatic configuration for Keras Tensor Parallel
"""

import re
from typing import Dict, Sequence, Union

import torch
from keras import layers, Model

from .config_keras import ConfigKeras
from .state_actions_keras import SplitKeras, GatherKeras, SumKeras


def get_default_config_keras(module: Model, device_ids: Sequence[str]) -> ConfigKeras:
    """
    Generate default tensor parallel configuration for Keras models.
    
    Args:
        module: Keras model to parallelize
        device_ids: List of device IDs to use
        
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
                # Split weights along output dimension (dim=1)
                state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size=world_size, dim=1)
                if layer.use_bias:
                    # Bias is 1D, split along dimension 0
                    state_rules[f"^{full_name}.bias$"] = SplitKeras(world_size=world_size, dim=0)
                
                # Output needs to be gathered
                output_rules[f"^{full_name}$"] = {0: "gather -1"}
                
            # Handle Embedding layers
            elif isinstance(layer, layers.Embedding):
                # Split along embedding dimension (dim=1)
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