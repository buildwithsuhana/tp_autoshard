#!/usr/bin/env python3
"""
Test backward pass correctness for Tensor Parallelism
Compares weight updates between single-device and tensor-parallel models
"""

import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from keras import layers, Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

def create_simple_model(input_dim=128, output_dim=10):
    """Create a simple model for testing."""
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu', name="dense_1")(inputs)
    x = layers.Dense(256, activation='relu', name="dense_2")(x)
    outputs = layers.Dense(output_dim, activation='softmax', name="dense_3")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def test_backward_pass_correctness():
    """Test that backward pass produces identical weight updates."""
    print("🧪 Testing Backward Pass Correctness")
    print("=" * 60)
    
    # 1. Setup
    devices = ["cpu:0", "cpu:1"]  # Use available devices
    input_dim = 128
    output_dim = 10
    batch_size = 32
    
    print(f"🔧 Setup:")
    print(f"  - Devices: {devices}")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Output dim: {output_dim}")
    print(f"  - Batch size: {batch_size}")
    
    # Create dummy data
    np.random.seed(42)  # For reproducible results
    dummy_x = np.random.rand(batch_size, input_dim).astype("float32")
    dummy_y = np.random.randint(0, output_dim, size=(batch_size,)).astype("int32")
    dummy_y = keras.utils.to_categorical(dummy_y, output_dim)
    
    print(f"✅ Created dummy data: X shape {dummy_x.shape}, Y shape {dummy_y.shape}")
    
    # 2. Initialize and compile single-device model
    print("\n🔧 Setting up single-device model...")
    model_single = create_simple_model(input_dim, output_dim)
    optimizer_single = Adam(learning_rate=0.001)
    loss_fn = CategoricalCrossentropy()
    model_single.compile(optimizer=optimizer_single, loss=loss_fn)
    
    # Store initial weights
    initial_weights = model_single.get_weights()
    print(f"✅ Single-device model initialized with {len(initial_weights)} weight tensors")
    
    # 3. Initialize and compile Tensor Parallel model
    print("\n🔧 Setting up Tensor Parallel model...")
    model_tp_base = create_simple_model(input_dim, output_dim)
    model_tp_base.set_weights(initial_weights)  # Ensure same starting point

    optimizer_tp_base = Adam(learning_rate=0.001)
    model_tp_base.compile(optimizer=optimizer_tp_base, loss=loss_fn)
    
    model_tp = TensorParallelKeras(model_tp_base, device_ids=devices)
    optimizer_tp = Adam(learning_rate=0.001)  # This will be wrapped by TensorParallelOptimizer
    model_tp.compile(optimizer=optimizer_tp, loss=loss_fn)
    
    print(f"✅ Tensor Parallel model initialized with {len(model_tp.trainable_variables)} trainable variables")
    
    # 4. Verify initial weights match
    print("\n🔍 Verifying initial weights match...")
    weights_single_init = model_single.get_weights()
    weights_tp_init = model_tp.original_model.get_weights()
    
    for i, (w_single, w_tp) in enumerate(zip(weights_single_init, weights_tp_init)):
        if not np.allclose(w_single, w_tp, rtol=1e-6, atol=1e-6):
            print(f"❌ Initial weights at index {i} do not match!")
            print(f"   Single: {w_single.shape}, TP: {w_tp.shape}")
            return False
        else:
            print(f"   ✅ Weight {i}: {w_single.shape} - matches")
    
    print("✅ All initial weights match perfectly!")
    
    # 5. Perform one training step
    print("\n🚀 Performing one training step...")
    
    print("   Training single-device model...")
    # Keras train_on_batch returns loss, and optionally metrics. We only care about the loss here.
    loss_single = model_single.train_on_batch(dummy_x, dummy_y, return_dict=False)
    
    print("   Training Tensor Parallel model...")
    loss_tp = model_tp.train_on_batch(dummy_x, dummy_y, return_dict=False)
    
    # 6. Compare results
    print(f"\n🔍 Training Results:")
    print(f"   Single-device loss: {loss_single:.6f}")
    print(f"   Tensor Parallel loss: {loss_tp:.6f}")
    
    # The losses should be identical
    loss_diff = abs(loss_single - loss_tp)
    if loss_diff > 1e-5:
        print(f"❌ Loss difference too large: {loss_diff:.2e}")
        return False
    else:
        print(f"✅ Losses match perfectly! (difference: {loss_diff:.2e})")
    
    # 7. Compare the updated weights layer by layer
    print("\n🔍 Comparing updated weights...")
    weights_single_updated = model_single.get_weights()
    weights_tp_updated = model_tp.original_model.get_weights()
    
    all_weights_match = True
    for i, (w_single, w_tp) in enumerate(zip(weights_single_updated, weights_tp_updated)):
        try:
            np.testing.assert_allclose(
                w_single, w_tp, rtol=1e-5, atol=1e-5,
                err_msg=f"Weights at index {i} do not match after training step."
            )
            print(f"   ✅ Weight {i}: {w_single.shape} - matches perfectly")
        except AssertionError as e:
            print(f"   ❌ Weight {i}: {w_single.shape} - MISMATCH!")
            print(f"      Error: {e}")
            all_weights_match = False
    
    if all_weights_match:
        print("\n🎉 BACKWARD PASS TEST PASSED!")
        print("✅ All weights match perfectly after one training step")
        print("✅ Tensor Parallelism backward pass is mathematically correct!")
        return True
    else:
        print("\n❌ BACKWARD PASS TEST FAILED!")
        print("❌ Weight updates do not match between single-device and tensor-parallel models")
        return False

def test_multiple_training_steps():
    """Test that multiple training steps maintain correctness."""
    print("\n🧪 Testing Multiple Training Steps")
    print("=" * 60)
    
    # Setup
    devices = ["cpu:0", "cpu:1"]
    input_dim = 64
    output_dim = 8
    batch_size = 16
    num_steps = 3
    
    print(f"🔧 Setup: {num_steps} training steps")
    
    # Create data
    np.random.seed(42)
    dummy_x = np.random.rand(batch_size, input_dim).astype("float32")
    dummy_y = np.random.randint(0, output_dim, size=(batch_size,)).astype("int32")
    dummy_y = keras.utils.to_categorical(dummy_y, output_dim)
    
    # Create models
    model_single = create_simple_model(input_dim, output_dim)
    model_tp_base = create_simple_model(input_dim, output_dim)
    
    # Ensure same initial weights
    initial_weights = model_single.get_weights()
    model_tp_base.set_weights(initial_weights)
    
    # Compile models
    optimizer_single = Adam(learning_rate=0.001)
    loss_fn = CategoricalCrossentropy()
    
    model_single.compile(optimizer=optimizer_single, loss=loss_fn)

    # MODIFIED: Compile the base model before wrapping
    optimizer_tp_base = Adam(learning_rate=0.001)
    model_tp_base.compile(optimizer=optimizer_tp_base, loss=loss_fn)
    
    model_tp = TensorParallelKeras(model_tp_base, device_ids=devices)
    optimizer_tp = Adam(learning_rate=0.001)
    model_tp.compile(optimizer=optimizer_tp, loss=loss_fn)
    
    # Training loop
    print(f"🚀 Training for {num_steps} steps...")
    
    for step in range(num_steps):
        print(f"   Step {step + 1}/{num_steps}")
        
        # Train both models
        loss_single = model_single.train_on_batch(dummy_x, dummy_y, return_dict=False)
        loss_tp = model_tp.train_on_batch(dummy_x, dummy_y, return_dict=False)
        
        # Compare losses
        loss_diff = abs(loss_single - loss_tp)
        if loss_diff > 1e-5:
            print(f"      ❌ Loss mismatch at step {step + 1}: {loss_diff:.2e}")
            return False
        else:
            print(f"      ✅ Losses match: {loss_single:.6f} vs {loss_tp:.6f}")
    
    # Final weight comparison
    print("\n🔍 Final weight comparison...")
    weights_single_final = model_single.get_weights()
    weights_tp_final = model_tp.original_model.get_weights()
    
    all_match = True
    for i, (w_single, w_tp) in enumerate(zip(weights_single_final, weights_tp_final)):
        if not np.allclose(w_single, w_tp, rtol=1e-5, atol=1e-5):
            print(f"   ❌ Final weight {i} mismatch!")
            all_match = False
        else:
            print(f"   ✅ Final weight {i}: matches")
    
    if all_match:
        print("\n🎉 MULTIPLE STEPS TEST PASSED!")
        print("✅ Tensor Parallelism maintains correctness over multiple training steps!")
        return True
    else:
        print("\n❌ MULTIPLE STEPS TEST FAILED!")
        return False

if __name__ == "__main__":
    print("🧪 COMPREHENSIVE BACKWARD PASS TESTING")
    print("=" * 80)
    
    # Test 1: Single training step
    test1_passed = test_backward_pass_correctness()
    
    if test1_passed:
        # Test 2: Multiple training steps
        test2_passed = test_multiple_training_steps()
        
        print("\n" + "=" * 80)
        if test2_passed:
            print("🏆 ALL TESTS PASSED!")
            print("✅ Tensor Parallelism backward pass is mathematically correct!")
            exit(0)
        else:
            print("❌ MULTIPLE STEP TEST FAILED!")
            exit(1)
    else:
        print("\n❌ SINGLE STEP TEST FAILED!")
        print("❌ Backward pass is still not mathematically correct!")
        exit(1)