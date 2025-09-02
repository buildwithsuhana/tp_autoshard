import logging
from typing import Sequence
from keras import Model, layers
from .config_keras import ConfigKeras
from .state_actions_keras import SplitKeras

logger = logging.getLogger(__name__)

def analyze_dense_layer_directly(layer: layers.Dense, module: Model, prefix: str) -> str:
    if not isinstance(layer, layers.Dense):
        return 'generic_dense'
    
    output_dim = layer.units if hasattr(layer, 'units') else None
    
    if not output_dim:
        return 'generic_dense'
    
    input_dim = None
    
    prev_layer = None
    for i, l in enumerate(module.layers):
        if l.name == layer.name:
            if i > 0:
                prev_layer = module.layers[i-1]
            break
            
    if prev_layer and not isinstance(prev_layer, layers.Dense):
        prev_layer = None

    if prev_layer:
        if hasattr(prev_layer, 'units'):
            input_dim = prev_layer.units
        elif hasattr(prev_layer, 'output_shape') and prev_layer.output_shape:
            input_dim = prev_layer.output_shape[-1]
        if hasattr(module, 'input_shape') and module.input_shape:
            shape_info = module.input_shape
            if isinstance(shape_info, list):
                shape_info = shape_info[0]
            if isinstance(shape_info, (list, tuple)) and len(shape_info) > 1:
                input_dim = shape_info[-1]

    if input_dim is None:
        return 'generic_dense'
    
    expansion_threshold = 1.5
    is_expansion = output_dim > input_dim * expansion_threshold
    is_contraction = input_dim > output_dim * expansion_threshold
    
    if is_expansion:
        return 'up_projection'
    elif is_contraction:
        return 'down_projection'
    else:
        return 'generic_dense'


def get_default_config_keras(module: Model, device_ids: Sequence[str]) -> ConfigKeras:
    world_size = len(device_ids)
    state_rules = {}
    output_rules = {}
    
    def process_module(module: Model, prefix: str = ""):
        for layer in module.layers:
            name = layer.name
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(layer, layers.Dense):
                mlp_type = analyze_dense_layer_directly(layer, module, full_name)
                
                if mlp_type == 'up_projection':
                    state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                        world_size=world_size, 
                        dim=1,
                        sharding_type="column"
                    )
                    if layer.use_bias:
                        state_rules[f"^{full_name}.bias$"] = SplitKeras(
                            world_size=world_size, 
                            dim=0,
                            sharding_type="column"
                        )
                    
                    output_rules[f"^{full_name}$"] = {0: "gather"}
                    logger.info(f"Applied Column-wise sharding to MLP up-projection {full_name} (direct-analysis)")
                
                elif mlp_type == 'down_projection':
                    state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                        world_size=world_size,
                        dim=0,
                        sharding_type="row"
                    )
                    if layer.use_bias:
                        pass 

                    output_rules[f"^{full_name}$"] = {0: "allreduce"}
                    logger.info(f"Applied Row-wise sharding to MLP down-projection {full_name} (direct-analysis)")
                
                else:
                    kernel_dim = 1  
                    bias_dim = 0    
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
                    
                    output_rules[f"^{full_name}$"] = {0: "gather -1"}
                
            elif isinstance(layer, layers.EinsumDense):
                equation = layer.equation
                state_rules[f"^{full_name}.kernel$"] = SplitKeras(
                    world_size=world_size, 
                    dim=1,
                    sharding_type="column"
                )
                
                if hasattr(layer, 'bias') and layer.bias is not None:
                    state_rules[f"^{full_name}.bias$"] = SplitKeras(
                        world_size=world_size, 
                        dim=0,
                        sharding_type="row"
                    )
                
                output_rules[f"^{full_name}$"] = {0: "gather -1"}
                logger.info(f"Applied EinsumDense sharding to {full_name} with equation {equation}")
                
            elif hasattr(layer, '__class__') and 'Dense' in layer.__class__.__name__:
                if hasattr(layer, 'kernel') or hasattr(layer, 'weights'):
                    weight_attr = 'kernel' if hasattr(layer, 'kernel') else 'weights'
                    state_rules[f"^{full_name}.{weight_attr}$"] = SplitKeras(
                        world_size=world_size, 
                        dim=1,
                        sharding_type="column"
                    )
                    
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        state_rules[f"^{full_name}.bias$"] = SplitKeras(
                            world_size=world_size, 
                            dim=0,
                            sharding_type="row"
                        )
                    elif hasattr(layer, 'use_bias') and layer.use_bias:
                        if hasattr(layer, 'weights') and len(layer.weights) > 1:
                            state_rules[f"^{full_name}.weights.1$"] = SplitKeras(
                                world_size=world_size, 
                                dim=0,
                                sharding_type="row"
                            )
                    
                    output_rules[f"^{full_name}$"] = {0: "gather -1"}
                    logger.info(f"Applied generic Dense sharding to {full_name}")
                
            elif isinstance(layer, layers.Embedding):
                state_rules[f"^{full_name}.embeddings$"] = SplitKeras(world_size=world_size, dim=1)
                output_rules[f"^{full_name}$"] = {0: "no_comm"}
                logger.info(f"Applied Embedding sharding to {full_name}")
                
            elif isinstance(layer, layers.MultiHeadAttention):
                state_rules[f"^{full_name}.query_dense.kernel$"] = SplitKeras(world_size=world_size, dim=1)
                state_rules[f"^{full_name}.key_dense.kernel$"] = SplitKeras(world_size=world_size, dim=1)
                state_rules[f"^{full_name}.value_dense.kernel$"] = SplitKeras(world_size=world_size, dim=1)
                
                if hasattr(layer.query_dense, 'bias') and layer.query_dense.bias is not None:
                    state_rules[f"^{full_name}.query_dense.bias$"] = SplitKeras(world_size=world_size, dim=0)
                if hasattr(layer.key_dense, 'bias') and layer.key_dense.bias is not None:
                    state_rules[f"^{full_name}.key_dense.bias$"] = SplitKeras(world_size=world_size, dim=0)
                if hasattr(layer.value_dense, 'bias') and layer.value_dense.bias is not None:
                    state_rules[f"^{full_name}.value_dense.bias$"] = SplitKeras(world_size=world_size, dim=0)

                state_rules[f"^{full_name}.output_dense.kernel$"] = SplitKeras(world_size=world_size, dim=0)
                if hasattr(layer.output_dense, 'bias') and layer.output_dense.bias is not None:
                    state_rules[f"^{full_name}.output_dense.bias$"] = SplitKeras(world_size=world_size, dim=0)

                output_rules[f"^{full_name}$"] = {0: "allreduce"}
                logger.info(f"Applied Column->Row pattern to {full_name}")
            
            elif isinstance(layer, layers.LayerNormalization):
                pass
                
            elif isinstance(layer, (layers.BatchNormalization, layers.GroupNormalization)):
                pass
                
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                process_module(layer, full_name)
                
    process_module(module)
    
    return ConfigKeras(
        state_rules=state_rules,
        output_rules=output_rules
    ) 