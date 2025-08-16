#!/usr/bin/env python3
"""
Test MLP sharding with Column -> Row pattern
"""

import numpy as np
import keras
from keras import layers, Model
from src.tensor_parallel_keras import TensorParallelKeras

def create_mlp_model():
    """Create a simple MLP model for testing."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(64,), name='mlp_up_projection'),
        layers.Dense(256, activation='relu', name='mlp_hidden'),
        layers.Dense(64, activation='relu', name='mlp_down_projection'),
        layers.Dense(10, activation='softmax', name='mlp_output')
    ])
    return model

def test_mlp_sharding():
    """Test MLP sharding with Column -> Row pattern."""
    print("🧪 Testing MLP Sharding with Column -> Row Pattern...")
    print("=" * 60)
    
    # Create model
    model = create_mlp_model()
    
    print(f"Original model parameters: {sum(p.shape.num_elements() for p in model.weights)}")
    
    # Create tensor parallel model
    tp_model = TensorParallelKeras(
        model,
        device_ids=["cpu", "cpu"],
        sharded=True,
        sharding_strategy="auto"
    )
    
    print(f"Sharded model created with {len(tp_model.model_shards)} shards")
    
    # Test forward pass
    test_input = np.random.random((4, 64)).astype(np.float32)
    
    print(f"\n🔍 Testing forward pass...")
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    output = tp_model(test_input)
    print(f"Output shape: {output.shape}")
    
    # Check if outputs are properly gathered
    if output.shape == (4, 10):
        print("✅ Output gathering working correctly!")
    else:
        print(f"⚠️ Output shape mismatch: expected (4, 10), got {output.shape}")
    
    # Test training
    print(f"\n🔄 Testing training...")
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    tp_model.compile(optimizer=optimizer, loss=loss_fn)
    
    # Create training data
    x_train = np.random.random((32, 64)).astype(np.float32)
    y_train = np.random.randint(0, 10, (32,)).astype(np.int32)
    
    print(f"Training data: {x_train.shape} → {y_train.shape}")
    
    # Train for a few steps
    history = tp_model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=1)
    
    print("✅ MLP sharding test completed!")
    return True

if __name__ == "__main__":
    test_mlp_sharding() 