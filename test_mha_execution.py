#!/usr/bin/env python3
"""
Test Multi-Head Attention tensor parallelism execution
"""

import numpy as np
import keras
from keras.layers import Input, MultiHeadAttention, Dense
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

def create_attention_model():
    """Create a simple Multi-Head Attention model."""
    inputs = Input(shape=(10, 32), name='input_tensor')
    
    # Multi-Head Attention layer
    attn_output = MultiHeadAttention(
        num_heads=8, 
        key_dim=32, 
        name='multihead_attention'
    )(inputs, inputs)
    
    # Final output projection
    output = Dense(32, activation='linear', name='output_dense')(attn_output)
    
    model = keras.Model(inputs=inputs, outputs=output)
    return model

def test_mha_execution():
    """Test Multi-Head Attention tensor parallelism execution."""
    print("Testing Multi-Head Attention tensor parallelism execution...")
    
    # Create model
    model = create_attention_model()
    print(f"Model created with {len(model.layers)} layers")
    
    # Create tensor parallel model
    tp_model = TensorParallelKeras(
        model=model,
        world_size=2,
        distributed_backend='fallback'
    )
    print("Tensor parallel model created")
    
    # Create test input
    input_data = np.random.random((8, 10, 32)).astype(np.float32)
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
        print(f"  Single device: {single_np[0, 0, :5]}")
        print(f"  Tensor parallel: {tp_np[0, 0, :5]}")
        print(f"  Differences: {abs_diff[0, 0, :5]}")
    else:
        print("❌ Shape mismatch - execution failed")

if __name__ == "__main__":
    test_mha_execution() 