#!/usr/bin/env python3
"""
Test simple Dense layer tensor parallelism execution
"""

import os
import numpy as np

# 💻 Set this flag BEFORE importing jax
# This tells JAX to simulate 2 CPU devices.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

# Import JAX first to ensure device detection works
import jax
print(f"🔍 JAX Device Detection:")
print(f"   Number of JAX devices: {jax.local_device_count()}")
print(f"   Device list: {jax.devices()}")
print(f"   Device types: {[str(d) for d in jax.devices()]}")

import keras
from keras.layers import Input, Dense
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

def create_simple_model():
    """Create a simple Dense model."""
    inputs = Input(shape=(64,), name='input_tensor')
    
    # Dense layer
    output = Dense(32, activation='relu', name='dense')(inputs)
    
    model = keras.Model(inputs=inputs, outputs=output)
    return model

def test_dense_execution():
    """Test simple Dense layer tensor parallelism execution."""
    print("Testing simple Dense layer tensor parallelism execution...")
    
    # Create model
    model = create_simple_model()
    print(f"Model created with {len(model.layers)} layers")
    
    # Create tensor parallel model with JAX backend
    tp_model = TensorParallelKeras(
        model=model,
        world_size=2,
        distributed_backend='jax'  # Use JAX backend
    )
    print("Tensor parallel model created")
    
    # Create test input
    input_data = np.random.random((8, 64)).astype(np.float32)
    print(f"Input data shape: {input_data.shape}")
    
    # Run single device model
    single_output = model(input_data)
    print(f"Single device output shape: {single_output.shape}")
    
    # Run tensor parallel model
    tp_output = tp_model(input_data)
    print(f"Tensor parallel output shape: {tp_output.shape}")
    
    # Check shapes match
    shape_match = single_output.shape == tp_output.shape
    print(f"Shape match: {shape_match}")
    
    if shape_match:
        # Convert to numpy for comparison
        single_np = single_output.numpy()
        tp_np = tp_output.numpy()
        
        # Calculate differences
        abs_diff = np.abs(single_np - tp_np)
        rel_diff = abs_diff / (np.abs(single_np) + 1e-8)
        
        print(f"Max absolute difference: {np.max(abs_diff):.2e}")
        print(f"Max relative difference: {np.max(rel_diff):.2e}")
        
        # Check if within tolerance
        tolerance = 1e-5
        within_tolerance = np.max(abs_diff) < tolerance
        
        if within_tolerance:
            print("✅ MATHEMATICAL IDENTITY ACHIEVED! (within tolerance)")
        else:
            print("❌ Mathematical differences detected")
            
        # Show sample values
        print("\nSample values:")
        print(f"  Single device: {single_np[0, :5]}")
        print(f"  Tensor parallel: {tp_np[0, :5]}")
        print(f"  Differences: {abs_diff[0, :5]}")
    else:
        print("❌ Shape mismatch - execution failed")

if __name__ == "__main__":
    test_dense_execution() 