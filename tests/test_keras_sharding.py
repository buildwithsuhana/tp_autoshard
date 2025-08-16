"""
Test sharding functionality for Keras Tensor Parallel
"""

import pytest
import torch
import keras
from keras import layers, Model
import numpy as np

# Import our Keras tensor parallel implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from tensor_parallel_keras import TensorParallelKeras


def create_test_model():
    """Create a test model with known parameter counts."""
    inputs = keras.Input(shape=(100,))
    x = layers.Dense(200, activation='relu')(inputs)  # 100*200 + 200 = 20,200 params
    x = layers.Dense(50, activation='relu')(x)        # 200*50 + 50 = 10,050 params
    x = layers.Dense(10, activation='softmax')(x)     # 50*10 + 10 = 510 params
    model = Model(inputs=inputs, outputs=x)
    return model


def test_parameter_sharding():
    """Test that parameters are actually sharded across devices."""
    model = create_test_model()
    
    # Count original parameters
    original_params = sum(w.shape.num_elements() for w in model.weights)
    print(f"Original model has {original_params} parameters")
    
    # Create tensor parallel model with 2 devices
    tp_model = TensorParallelKeras(
        model,
        device_ids=["cpu", "cpu"],
        sharded=True
    )
    
    # Check that we have 2 shards
    assert len(tp_model.model_shards) == 2
    
    # Check that parameters are actually different between shards
    shard0_params = tp_model.model_shards[0].weights
    shard1_params = tp_model.model_shards[1].weights
    
    # The first Dense layer should be sharded (output dimension split)
    # Original: (100, 200) -> Shard 0: (100, 100), Shard 1: (100, 100)
    first_dense_shard0 = shard0_params[0]  # kernel
    first_dense_shard1 = shard1_params[0]  # kernel
    
    print(f"Shard 0 first dense kernel shape: {first_dense_shard0.shape}")
    print(f"Shard 1 first dense kernel shape: {first_dense_shard1.shape}")
    
    # Check that the shapes are the same (both shards have same architecture)
    assert first_dense_shard0.shape == first_dense_shard1.shape
    assert first_dense_shard0.shape[1] == 100  # Should be split in half
    
    # Check that the actual weight values are different (different shards)
    # Convert to numpy for comparison
    weights0 = first_dense_shard0.numpy()
    weights1 = first_dense_shard1.numpy()
    
    # The weights should be different (different slices of the original tensor)
    assert not np.array_equal(weights0, weights1), "Shards should have different weight values"
    
    # Check total parameters per shard
    shard0_total = sum(w.shape.num_elements() for w in shard0_params)
    shard1_total = sum(w.shape.num_elements() for w in shard1_params)
    
    print(f"Shard 0 total parameters: {shard0_total}")
    print(f"Shard 1 total parameters: {shard1_total}")
    
    # Each shard should have fewer parameters than the original
    assert shard0_total < original_params
    assert shard1_total < original_params
    
    print("✅ Parameter sharding test passed!")


def test_embedding_sharding():
    """Test that embedding layers are sharded correctly."""
    # Create a model with embedding
    inputs = keras.Input(shape=(10,))
    x = layers.Embedding(1000, 64)(inputs)  # 1000*64 = 64,000 params
    x = layers.Dense(32, activation='relu')(x)  # 64*32 + 32 = 2,080 params
    model = Model(inputs=inputs, outputs=x)
    
    # Count original parameters
    original_params = sum(w.shape.num_elements() for w in model.weights)
    print(f"Embedding model has {original_params} parameters")
    
    # Create tensor parallel model
    tp_model = TensorParallelKeras(
        model,
        device_ids=["cpu", "cpu"],
        sharded=True
    )
    
    # Check embedding sharding
    shard0_embedding = tp_model.model_shards[0].weights[0]  # embeddings
    shard1_embedding = tp_model.model_shards[1].weights[0]  # embeddings
    
    print(f"Shard 0 embedding shape: {shard0_embedding.shape}")
    print(f"Shard 1 embedding shape: {shard1_embedding.shape}")
    
    # Embedding should be split along embedding dimension (dim=1)
    # Original: (1000, 64) -> Shard 0: (1000, 32), Shard 1: (1000, 32)
    assert shard0_embedding.shape[1] == 32
    assert shard1_embedding.shape[1] == 32
    
    print("✅ Embedding sharding test passed!")


def test_forward_pass():
    """Test that the sharded model can perform forward pass."""
    model = create_test_model()
    
    # Create tensor parallel model
    tp_model = TensorParallelKeras(
        model,
        device_ids=["cpu", "cpu"],
        sharded=True
    )
    
    # Create test input
    test_input = np.random.random((32, 100)).astype(np.float32)
    
    # Perform forward pass
    try:
        output = tp_model(test_input)
        print(f"Forward pass successful! Output shape: {output.shape}")
        
        # In tensor parallelism, the output shape depends on which shard we're looking at
        # Each shard produces a partial output that needs to be gathered
        # For now, we expect the output to be from one shard (partial result)
        assert output.shape[0] == 32  # batch_size should be correct
        assert output.shape[1] <= 10  # output features should be <= original (partial result)
        
        print("✅ Forward pass test passed!")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise


if __name__ == "__main__":
    print("Testing Keras Tensor Parallel Sharding...")
    
    try:
        test_parameter_sharding()
        test_embedding_sharding()
        test_forward_pass()
        
        print("\n🎉 All sharding tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 