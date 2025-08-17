#!/usr/bin/env python3
"""
Test MLP configuration for tensor parallelism
"""

import numpy as np
import keras
from keras.layers import Input, Dense
from src.tensor_parallel_keras.autoconfig_keras import get_default_config_keras

def create_mlp_model():
    """Create a simple MLP model with up/down projection."""
    inputs = Input(shape=(32,), name='input_tensor')
    
    # Up projection: 32 -> 128 (expansion)
    dense_up = Dense(128, activation='relu', name='dense_up')(inputs)
    
    # Down projection: 128 -> 16 (contraction)
    dense_down = Dense(16, activation='relu', name='dense_down')(dense_up)
    
    # Final output
    output = Dense(32, activation='linear', name='output_dense')(dense_down)
    
    model = keras.Model(inputs=inputs, outputs=output)
    return model

def test_mlp_config():
    """Test MLP configuration."""
    print("Testing MLP configuration...")
    
    # Create model
    model = create_mlp_model()
    print(f"Model created with {len(model.layers)} layers")
    
    # Get configuration
    config = get_default_config_keras(model, ['device:0', 'device:1'])
    
    print("\nOutput rules:")
    for pattern, rule in config.output_rules.items():
        print(f"  {pattern}: {rule}")
    
    print("\nState rules:")
    for pattern, split_rule in config.state_rules.items():
        print(f"  {pattern}: {split_rule}")
    
    # Check if MLP layers are correctly identified
    print("\nChecking MLP layer identification:")
    
    # Test the analyze_dense_layer_directly function
    from src.tensor_parallel_keras.autoconfig_keras import analyze_dense_layer_directly
    
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            mlp_type = analyze_dense_layer_directly(layer, model, layer.name)
            print(f"  {layer.name}: {mlp_type}")
            print(f"    Input shape: {getattr(layer, 'input_shape', 'N/A')}")
            print(f"    Output shape: {getattr(layer, 'output_shape', 'N/A')}")
            print(f"    Units: {getattr(layer, 'units', 'N/A')}")

if __name__ == "__main__":
    test_mlp_config() 