#!/usr/bin/env python3
"""
Simple test to isolate the training issue
"""

import numpy as np
import keras
from keras import layers, Model, optimizers
from src.tensor_parallel_keras import TensorParallelKeras

def test_simple_training():
    """Test simple training without complex setup."""
    print("🧪 Testing Simple Training...")
    print("=" * 50)
    
    # Create a very simple model
    inputs = keras.Input(shape=(5,), name="input")
    x = layers.Dense(10, activation='relu', name="dense1")(inputs)
    outputs = layers.Dense(3, activation='softmax', name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="simple_model")
    
    # Create tensor parallel model
    tp_model = TensorParallelKeras(
        model,
        device_ids=["cpu", "cpu"],
        sharded=True,
        sharding_strategy="column"
    )
    
    print(f"Model created with {len(tp_model.model_shards)} shards")
    
    # Compile model
    optimizer = optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    tp_model.compile(optimizer=optimizer, loss=loss_fn)
    
    print("✅ Model compiled successfully")
    
    # Create test input first
    test_input = np.random.random((2, 5)).astype(np.float32)
    
    # Test forward pass first to see actual output size
    print("\n🔍 Testing forward pass...")
    try:
        output = tp_model(test_input, training=True)
        print(f"✅ Forward pass works! Output: {output.shape}, dtype: {output.dtype}")
        
        # Get actual number of output classes from the model
        actual_output_classes = output.shape[1]
        print(f"🔍 Actual output classes: {actual_output_classes}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Create training data with correct label range
    x_train = np.random.random((8, 5)).astype(np.float32)
    y_train = np.random.randint(0, actual_output_classes, (8,)).astype(np.int64)
    
    print(f"Training data: {x_train.shape} → {y_train.shape}")
    print(f"X dtype: {x_train.dtype}, Y dtype: {y_train.dtype}")
    print(f"Label range: 0 to {actual_output_classes-1}")
    
    # Test training
    print("\n🔄 Testing training...")
    try:
        history = tp_model.fit(
            x_train, y_train,
            epochs=2,
            batch_size=4,
            verbose=1
        )
        print("✅ Training successful!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Simple Training Test...")
    print("=" * 50)
    
    success = test_simple_training()
    
    if success:
        print("\n🎉 Simple training test passed!")
    else:
        print("\n❌ Simple training test failed!") 