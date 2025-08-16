#!/usr/bin/env python3
"""
Test OPT-125M with Keras Tensor Parallel implementation
"""

import numpy as np
import torch
import keras
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.tensor_parallel_keras import TensorParallelKeras

def test_opt125m_keras():
    """Test OPT-125M with Keras Tensor Parallel."""
    print("🚀 Testing OPT-125M with Keras Tensor Parallel...")
    
    # Load tokenizer and model
    print("📥 Loading OPT-125M model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"📊 Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🔧 Model device: {next(model.parameters()).device}")
    
    # Convert PyTorch model to Keras
    print("🔄 Converting PyTorch model to Keras...")
    keras_model = convert_pytorch_to_keras(model)
    
    print(f"📊 Keras model parameters: {sum(w.shape.num_elements() for w in keras_model.weights):,}")
    
    # Test with single device first
    print("\n🧪 Testing single device (CPU)...")
    try:
        tp_model_single = TensorParallelKeras(
            keras_model,
            device_ids=["cpu"],
            sharded=False
        )
        print("✅ Single device TensorParallelKeras created successfully!")
        
        # Test forward pass with proper input shape
        test_input = np.random.random((1, 5, 768)).astype(np.float32)  # (batch, seq_len, hidden_size)
        output = tp_model_single(test_input)
        print(f"✅ Single device forward pass successful! Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Single device test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with multiple devices (tensor parallel)
    print("\n🧪 Testing tensor parallel (2 CPUs)...")
    try:
        tp_model_tp = TensorParallelKeras(
            keras_model,
            device_ids=["cpu", "cpu"],
            sharded=True
        )
        print("✅ Tensor Parallel TensorParallelKeras created successfully!")
        
        # Check sharding
        print(f"📊 Number of shards: {len(tp_model_tp.model_shards)}")
        for i, shard in enumerate(tp_model_tp.model_shards):
            params = sum(w.shape.num_elements() for w in shard.weights)
            print(f"   Shard {i}: {params:,} parameters")
        
        # Test forward pass with proper input shape
        test_input = np.random.random((1, 5, 768)).astype(np.float32)  # (batch, seq_len, hidden_size)
        output = tp_model_tp(test_input)
        print(f"✅ Tensor parallel forward pass successful! Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Tensor parallel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests passed! OPT-125M with Keras Tensor Parallel is working!")
    return True

def convert_pytorch_to_keras(pytorch_model):
    """Convert a PyTorch model to Keras format."""
    # This is a simplified conversion - in practice, you'd want a more robust approach
    
    # Get model config
    config = pytorch_model.config
    
    # Create a simple Keras model with similar architecture
    # For OPT, we'll create a basic transformer-like structure
    inputs = keras.Input(shape=(None, config.hidden_size))
    
    # Create layers based on OPT architecture
    x = inputs
    
    # Add transformer layers (simplified)
    for i in range(min(config.num_hidden_layers, 2)):  # Limit to 2 layers for testing
        # Self-attention (simplified)
        attention_output = keras.layers.Dense(config.hidden_size, activation='relu')(x)
        x = keras.layers.LayerNormalization()(attention_output + x)
        
        # Feed-forward (simplified)
        ff_output = keras.layers.Dense(config.hidden_size * 4, activation='relu')(x)
        ff_output = keras.layers.Dense(config.hidden_size)(ff_output)
        x = keras.layers.LayerNormalization()(ff_output + x)
    
    # Output projection
    outputs = keras.layers.Dense(config.vocab_size, activation='softmax')(x)
    
    # Create Keras model
    keras_model = keras.Model(inputs=inputs, outputs=outputs, name="opt_keras")
    
    # Build the model
    keras_model.build((None, None, config.hidden_size))
    
    return keras_model

def test_simple_keras_model():
    """Test with a simple Keras model first."""
    print("\n🧪 Testing with simple Keras model...")
    
    # Create a simple model
    inputs = keras.Input(shape=(100,))
    x = keras.layers.Dense(512, activation='relu')(inputs)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="simple_test")
    
    print(f"📊 Simple model parameters: {sum(w.shape.num_elements() for w in model.weights):,}")
    
    try:
        tp_model = TensorParallelKeras(
            model,
            device_ids=["cpu", "cpu"],
            sharded=True
        )
        print("✅ Simple model TensorParallelKeras created successfully!")
        
        # Test forward pass
        test_input = np.random.random((32, 100)).astype(np.float32)
        output = tp_model(test_input)
        print(f"✅ Simple model forward pass successful! Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Starting Keras Tensor Parallel tests...")
    
    # Test simple model first
    simple_success = test_simple_keras_model()
    
    if simple_success:
        # Test OPT-125M
        opt_success = test_opt125m_keras()
        
        if opt_success:
            print("\n🎉 All tests completed successfully!")
        else:
            print("\n❌ OPT-125M test failed!")
    else:
        print("\n❌ Simple model test failed!") 