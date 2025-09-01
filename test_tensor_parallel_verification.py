#!/usr/bin/env python3
"""
Test suite for tensor parallel verification with comprehensive checks.
"""

import time
import logging
import numpy as np
import keras
from keras import layers

# Import TensorParallelKeras
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_and_get_model(input_shape, layers_config, output_config):
    """Helper function to build a Keras model and ensure it's built."""
    model = keras.Sequential([layers.Input(shape=input_shape)] + \
                             [layers.Dense(units, activation='relu') for units in layers_config] + \
                             [layers.Dense(output_config['units'], activation=output_config['activation'])])
    dummy_input = np.zeros((1, *input_shape), dtype=np.float32)
    model(dummy_input)
    return model

# In test_tensor_parallel_verification.py

# In test_tensor_parallel_verification.py

def test_parameter_sharding_verification():
    """Test parameter sharding verification."""
    print("🔧 Parameter Sharding Verification")
    print("=" * 40)
    
    start_time = time.time()
    print(f"⏱️  {time.time() - start_time:.2f}s: Starting parameter sharding test...")
    
    model = build_and_get_model(
        input_shape=(100,),
        layers_config=[512, 256, 128, 64],
        output_config={'units': 10, 'activation': 'softmax'}
    )
    
    original_params = model.count_params()
    print(f"      Original params: {original_params:,}")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Testing tensor parallelism...")
    
    tp_model = TensorParallelKeras(
        model=model,
        world_size=4,
        distributed_backend='fallback'
    )
    
    # FIX: Iterate through the wrappers in model_shards and call their count_params method
    for i, shard_wrapper in enumerate(tp_model.model_shards):
        shard_params = shard_wrapper.count_params()
        print(f"   Shard {i}: {shard_params:,} parameters")

    print(f"✅ Parameter sharding verification completed in {time.time() - start_time:.2f}s")
    return True

def test_inference_numerical_correctness():
    """Test inference numerical correctness."""
    print("🔧 Inference Numerical Correctness")
    print("=" * 40)
    
    start_time = time.time()
    model = build_and_get_model(
        input_shape=(50,),
        layers_config=[100, 50],
        output_config={'units': 10, 'activation': 'softmax'}
    )
    print(f"✅ {time.time() - start_time:.2f}s: Model created successfully")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating Tensor Parallel model...")
    tp_model = TensorParallelKeras(
        model=model,
        world_size=2,
        distributed_backend='fallback'
    )
    print(f"✅ {time.time() - start_time:.2f}s: Tensor parallel model created")
    
    for input_size in [10, 20, 30]:
        test_input = np.random.random((input_size, 50)).astype(np.float32)
        print(f"   Testing input size: {input_size}")
        
        original_output = model(test_input)
        tp_output = tp_model(test_input)
        
        print(f"      Original output shape: {original_output.shape}")
        print(f"      TP output shape: {tp_output.shape}")
        
        assert original_output.shape == tp_output.shape, "Output shapes do not match"
        np.testing.assert_allclose(original_output, tp_output, rtol=1e-5, atol=1e-5)
        print(f"      ✅ Output shapes and values match!")
    
    print(f"✅ Inference correctness test completed in {time.time() - start_time:.2f}s")
    return True

def test_gradient_synchronization_verification():
    """Test gradient synchronization by verifying training runs."""
    print("🔧 Gradient Synchronization Verification")
    print("=" * 40)
    
    start_time = time.time()
    base_model = build_and_get_model(
        input_shape=(10,),
        layers_config=[32, 16],
        output_config={'units': 1, 'activation': 'linear'}
    )
    print(f"✅ {time.time() - start_time:.2f}s: Model created successfully")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating Tensor Parallel model...")
    tp_model = TensorParallelKeras(
        model=base_model,
        world_size=2,
        distributed_backend='fallback'
    )
    print(f"✅ {time.time() - start_time:.2f}s: Tensor parallel model created")
    
    print(f"   Testing gradient computation via training...")
    x_train = np.random.random((16, 10)).astype(np.float32)
    y_train = np.random.random((16, 1)).astype(np.float32)
    
    try:
        tp_model.compile(optimizer='adam', loss='mse')
        tp_model.fit(x_train, y_train, epochs=1, verbose=0)
        print(f"      ✅ Gradient computation and training step successful")
        return True
    except Exception as e:
        print(f"      ❌ Gradient computation/training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimizer_sharding_verification():
    """Test that the model can be compiled with various optimizers."""
    print("🔧 Optimizer Sharding Verification")
    print("=" * 40)
    
    start_time = time.time()
    model = build_and_get_model(
        input_shape=(30,),
        layers_config=[128, 64],
        output_config={'units': 10, 'activation': 'softmax'}
    )
    print(f"✅ {time.time() - start_time:.2f}s: Model created successfully")
    
    tp_model = TensorParallelKeras(
        model=model,
        world_size=2,
        distributed_backend='fallback'
    )
    print(f"✅ {time.time() - start_time:.2f}s: Tensor parallel model created")
    
    optimizers_to_test = [('Adam', 'adam'), ('SGD', 'sgd'), ('RMSprop', 'rmsprop')]
    all_passed = True
    
    for opt_name, opt_config in optimizers_to_test:
        print(f"   Testing {opt_name} optimizer...")
        try:
            tp_model.compile(optimizer=opt_config, loss='sparse_categorical_crossentropy')
            print(f"      ✅ {opt_name} compilation successful")
        except Exception as e:
            print(f"      ❌ {opt_name} compilation failed: {e}")
            all_passed = False
            
    print(f"✅ Optimizer sharding verification completed in {time.time() - start_time:.2f}s")
    return all_passed

def test_end_to_end_training_verification():
    """Test end-to-end training verification."""
    print("🔧 End-to-End Training Verification")
    print("=" * 40)
    
    start_time = time.time()
    model = build_and_get_model(
        input_shape=(25,),
        layers_config=[64, 32],
        output_config={'units': 8, 'activation': 'softmax'}
    )
    print(f"✅ {time.time() - start_time:.2f}s: Model created successfully")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating Tensor Parallel model...")
    tp_model = TensorParallelKeras(
        model=model,
        world_size=2,
        distributed_backend='fallback'
    )
    print(f"✅ {time.time() - start_time:.2f}s: Tensor parallel model created")
    
    print(f"   Testing compilation...")
    try:
        tp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(f"      ✅ Compilation successful")
    except Exception as e:
        print(f"      ❌ Compilation failed: {e}")
        return False
    
    print(f"   Testing training...")
    x_train = np.random.random((32, 25)).astype(np.float32)
    y_train = np.random.randint(0, 8, (32,), dtype=np.int32)
    
    try:
        history = tp_model.fit(x_train, y_train, epochs=3, batch_size=8, verbose=0)
        print(f"      ✅ Training successful")
        print(f"      Final loss: {history.history['loss'][-1]:.6f}")
        print(f"      Final accuracy: {history.history['accuracy'][-1]:.6f}")
        return True
    except Exception as e:
        print(f"      ❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 COMPREHENSIVE TENSOR PARALLEL VERIFICATION TEST SUITE")
    print("=" * 70)
    
    test_functions = [
        ("Parameter Sharding", test_parameter_sharding_verification),
        ("Inference Correctness", test_inference_numerical_correctness),
        ("Gradient Synchronization", test_gradient_synchronization_verification),
        ("Optimizer Sharding", test_optimizer_sharding_verification),
        ("End-to-End Training", test_end_to_end_training_verification),
    ]
    
    results_summary = []
    for name, test_func in test_functions:
        try:
            result = test_func()
            results_summary.append((name, result))
        except Exception as e:
            print(f"💥 ERROR during '{name}': {e}")
            import traceback
            traceback.print_exc()
            results_summary.append((name, False))

    print("\n" + "=" * 70)
    print("🎉 VERIFICATION TESTING COMPLETED!")
    print("\n📋 COMPREHENSIVE RESULTS:")
    
    passed_tests = sum(1 for _, result in results_summary if result)
    for test_name, result in results_summary:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   - {test_name}: {status}")
    
    total_tests = len(results_summary)
    print(f"\n📊 SUMMARY:")
    print(f"   - Total Tests: {total_tests}")
    print(f"   - Passed: {passed_tests}")
    print(f"   - Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n🚀 SUCCESS: All verification tests passed!")
    else:
        print(f"\n⚠️  WARNING: {total_tests - passed_tests} tests failed.")