"""
Basic tests for Keras Tensor Parallel implementation
"""

import pytest
import torch
import keras
from keras import layers, Model

# Import our Keras tensor parallel implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from tensor_parallel_keras import TensorParallelKeras


def create_simple_keras_model():
    """Create a simple Keras model for testing."""
    inputs = keras.Input(shape=(100,))
    x = layers.Dense(200, activation='relu')(inputs)
    x = layers.Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    return model


def test_keras_tensor_parallel_creation():
    """Test that TensorParallelKeras can be created."""
    model = create_simple_keras_model()
    
    # Test creation with CPU devices
    tp_model = TensorParallelKeras(
        model, 
        device_ids=["cpu", "cpu"],
        sharded=True
    )
    
    assert tp_model is not None
    assert len(tp_model.model_shards) == 2
    assert tp_model.devices == ("cpu", "cpu")


def test_keras_tensor_parallel_single_device():
    """Test TensorParallelKeras with single device."""
    model = create_simple_keras_model()
    
    tp_model = TensorParallelKeras(
        model,
        device_ids=["cpu"],
        sharded=False
    )
    
    assert len(tp_model.model_shards) == 1
    assert tp_model.model_shards[0] == model


def test_keras_tensor_parallel_device_detection():
    """Test automatic device detection."""
    model = create_simple_keras_model()
    
    tp_model = TensorParallelKeras(model)
    
    # Should detect at least CPU
    assert "cpu" in tp_model.devices
    assert len(tp_model.devices) >= 1


if __name__ == "__main__":
    # Run basic tests
    print("Testing Keras Tensor Parallel implementation...")
    
    try:
        test_keras_tensor_parallel_creation()
        print("✅ Basic creation test passed")
        
        test_keras_tensor_parallel_single_device()
        print("✅ Single device test passed")
        
        test_keras_tensor_parallel_device_detection()
        print("✅ Device detection test passed")
        
        print("\n🎉 All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 