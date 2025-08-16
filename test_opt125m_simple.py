#!/usr/bin/env python3
"""
Simple test of Keras Tensor Parallel with a basic transformer-like model
"""

import numpy as np
import keras
from keras import layers, Model
from src.tensor_parallel_keras import TensorParallelKeras

def create_simple_transformer_model():
    """Create a simple transformer-like model for testing."""
    # Input layer
    inputs = keras.Input(shape=(None, 768), name="input")
    
    # First Dense layer (will be sharded)
    x = layers.Dense(768, activation='relu', name="dense1")(inputs)
    
    # Layer normalization
    x = layers.LayerNormalization(axis=-1, name="ln1")(x)
    
    # Second Dense layer (will be sharded)
    x = layers.Dense(768, activation='relu', name="dense2")(x)
    
    # Layer normalization
    x = layers.LayerNormalization(axis=-1, name="ln2")(x)
    
    # Output layer
    outputs = layers.Dense(1000, activation='softmax', name="output")(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name="simple_transformer")
    
    return model

def test_simple_transformer():
    """Test tensor parallel with a simple transformer model."""
    print("🧪 Testing Simple Transformer with Keras Tensor Parallel...")
    
    # Create model
    model = create_simple_transformer_model()
    
    # Count parameters
    total_params = sum(w.shape.num_elements() for w in model.weights)
    print(f"📊 Original model parameters: {total_params:,}")
    
    # Test single device
    print("\n🧪 Testing single device...")
    try:
        tp_single = TensorParallelKeras(
            model,
            device_ids=["cpu"],
            sharded=False
        )
        print("✅ Single device created successfully!")
        
        # Test forward pass
        test_input = np.random.random((1, 5, 768)).astype(np.float32)
        output = tp_single(test_input)
        print(f"✅ Single device forward pass successful! Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Single device test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test tensor parallel
    print("\n🧪 Testing tensor parallel (2 CPUs)...")
    try:
        tp_tp = TensorParallelKeras(
            model,
            device_ids=["cpu", "cpu"],
            sharded=True
        )
        print("✅ Tensor parallel created successfully!")
        
        # Check sharding
        print(f"📊 Number of shards: {len(tp_tp.model_shards)}")
        for i, shard in enumerate(tp_tp.model_shards):
            params = sum(w.shape.num_elements() for w in shard.weights)
            print(f"   Shard {i}: {params:,} parameters")
        
        # Test forward pass
        test_input = np.random.random((1, 5, 768)).astype(np.float32)
        output = tp_tp(test_input)
        print(f"✅ Tensor parallel forward pass successful! Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor parallel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Simple Transformer Tensor Parallel Test...")
    
    success = test_simple_transformer()
    
    if success:
        print("\n🎉 Simple transformer test completed successfully!")
    else:
        print("\n❌ Simple transformer test failed!") 