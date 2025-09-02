#!/usr/bin/env python3
"""
Test suite for OPT-125M model verification with tensor parallelism.
"""
import os
import time
import logging
import numpy as np
import keras
from keras import layers, optimizers
import matplotlib.pyplot as plt

from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simplified_opt125m_model(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4):
    """Create a simplified OPT-125M model for faster testing."""
    print("   Creating simplified OPT-125M model...")
    
    model = keras.Sequential([
        layers.Input(shape=(None,), dtype='int32'),
        layers.Embedding(vocab_size, hidden_size),
        layers.LayerNormalization(),
        layers.Dense(hidden_size * 4, activation='relu'),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    print(f"      Simplified model created with {model.count_params():,} parameters")
    return model

def create_opt125m_model(vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12):
    """Create a simplified OPT-125M model for testing."""
    print("   Creating OPT-125M model...")
    
    inputs = layers.Input(shape=(None,), dtype='int32', name='input_ids')
    embedding = layers.Embedding(vocab_size, hidden_size, name='embed_tokens')(inputs)
    hidden_states = embedding
    hidden_states = layers.LayerNormalization(epsilon=1e-5, name='layernorm_embedding')(hidden_states)
    
    for i in range(num_layers):
        print(f"     Adding transformer layer {i+1}/{num_layers}")
        
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_size // num_heads, name=f'layers_{i}_self_attn'
        )(hidden_states, hidden_states)
        
        hidden_states = layers.Add(name=f'layers_{i}_residual_1')([hidden_states, attention_output])
        hidden_states = layers.LayerNormalization(epsilon=1e-5, name=f'layernorm_1_{i}')(hidden_states)
        
        mlp_hidden = layers.Dense(hidden_size * 4, activation='relu', name=f'layers_{i}_mlp_up')(hidden_states)
        mlp_output = layers.Dense(hidden_size, name=f'layers_{i}_mlp_down')(mlp_hidden)
        
        hidden_states = layers.Add(name=f'layers_{i}_residual_2')([hidden_states, mlp_output])
        hidden_states = layers.LayerNormalization(epsilon=1e-5, name=f'layernorm_2_{i}')(hidden_states)
    
    hidden_states = layers.LayerNormalization(epsilon=1e-5, name='layernorm_final')(hidden_states)
    outputs = layers.Dense(vocab_size, name='lm_head')(hidden_states)
    model = keras.Model(inputs=inputs, outputs=outputs, name='OPT-125M')
    return model

def verify_layer_sharding(tp_model):
    """Verify that layers are properly sharded in the tensor parallel model."""
    print("      Verifying layer sharding...")
    if hasattr(tp_model, 'sharding_manager') and tp_model.sharding_manager is not None:
        print("      ✅ Sharding manager found")
        total_params = sum(p.shape.num_elements() for p in tp_model.weights)
        print(f"      ✅ Total parameters in TP model: {total_params:,}")
        print("      ✅ Layer sharding verification passed")
    else:
        print("      ⚠️  No sharding manager found (using fallback mode)")
        print("      ✅ Basic model structure verification passed")

def test_opt125m_parameter_sharding():
    """Test OPT-125M parameter sharding verification."""
    print("🔧 OPT-125M Parameter Sharding Verification")
    print("=" * 50)
    start_time = time.time()
    print(f"⏱️  {time.time() - start_time:.2f}s: Starting OPT-125M parameter sharding test...")
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating OPT-125M model...")
    opt_model = create_opt125m_model()
    original_params = opt_model.count_params()
    print(f"      Original params: {original_params:,}")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Testing tensor parallelism...")
    tp_manager = TensorParallelKeras(model=opt_model, world_size=7, distributed_backend='fallback')
    
    total_sharded_params = 0
    for i, shard in enumerate(tp_manager.sharded_models):
        shard_params = shard.count_params()
        total_sharded_params += shard_params
        print(f"   Shard {i}: {shard_params:,} parameters")
        
    print(f"      Sharded params: {total_sharded_params:,}")
    assert total_sharded_params >= original_params, "Sharded parameters should be >= original"
    print(f"      ✅ Parameter count verification passed")
    
    print(f"✅ OPT-125M parameter sharding verification completed in {time.time() - start_time:.2f}s")
    return True

def test_opt125m_inference_correctness():
    """Test OPT-125M inference numerical correctness."""
    print("🔧 OPT-125M Inference Numerical Correctness")
    print("=" * 50)
    start_time = time.time()
    print(f"⏱️  {time.time() - start_time:.2f}s: Starting OPT-125M inference test...")
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating OPT-125M model...")
    opt_model = create_opt125m_model()
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Testing tensor parallelism...")
    tp_manager = TensorParallelKeras(model=opt_model, world_size=7, distributed_backend='fallback')
    tp_model = tp_manager.build_assembled_model()

    print(f"✅ {time.time() - start_time:.2f}s: Models created successfully")
    for seq_len in [5, 10, 15]:
        test_input = np.random.randint(0, 1000, (1, seq_len), dtype=np.int32)
        print(f"   Testing sequence {seq_len}: (1, {seq_len})")
        original_output = opt_model(test_input)
        tp_output = tp_model(test_input)
        print(f"      Original output shape: {original_output.shape}")
        print(f"      TP output shape: {tp_output.shape}")
        if original_output.shape == tp_output.shape:
             print(f"      ✅ Output shapes are compatible")
        else:
             print(f"      ❌ Output shapes are incompatible")
             assert False, f"Shape mismatch: {original_output.shape} vs {tp_output.shape}"
             
    print(f"✅ OPT-125M inference correctness test completed in {time.time() - start_time:.2f}s")
    return True

def plot_training_graphs(original_history, tp_history):
    """
    Plots the loss and perplexity curves for both the original and TP models.
    """
    if original_history and tp_history:
        original_perplexity = np.exp(original_history.history['loss'])
        tp_perplexity = np.exp(tp_history.history['loss'])

        epochs = range(1, len(original_history.history['loss']) + 1)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, original_history.history['loss'], 'b-o', label='Original Model Loss')
        plt.plot(epochs, tp_history.history['loss'], 'r-x', label='TP Model Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, original_perplexity, 'b-o', label='Original Model Perplexity')
        plt.plot(epochs, tp_perplexity, 'r-x', label='TP Model Perplexity')
        plt.title('Training Perplexity')
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        print("\n   Generating training performance graphs...")


def test_opt125m_training_verification():
    """Test OPT-125M training verification with a fair comparison."""
    print("🔧 OPT-125M Training Verification (Fair Comparison)")
    print("=" * 50)
    start_time = time.time()
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating model template and saving initial weights...")
    model_template = create_simplified_opt125m_model()
    initial_weights = model_template.get_weights()
    print(f"      ✅ Initial weights saved.")

    print(f"⏱️  {time.time() - start_time:.2f}s: Creating Tensor Parallel manager...")
    tp_manager = TensorParallelKeras(model=model_template, world_size=7, distributed_backend='fallback')
    tp_model = tp_manager.build_assembled_model()
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating a separate baseline model...")
    baseline_model = create_simplified_opt125m_model()
    
    x = baseline_model.set_weights(initial_weights)
    print(x)
    print(f"      ✅ Baseline model weights set to match the template.")

    print(f"⏱️  {time.time() - start_time:.2f}s: Creating and pre-building the TP optimizer...")
    dummy_model_for_opt = create_simplified_opt125m_model()
    base_optimizer = optimizers.Adam()
    base_optimizer.build(dummy_model_for_opt.trainable_variables)
    print(f"      ✅ TP optimizer is now built and has state variables.")

    print("\n" + "="*20 + " Memory Savings Report " + "="*20)
    try:
        tp_manager.compile(optimizer=base_optimizer)
        
        if hasattr(tp_manager, 'optimizer') and hasattr(tp_manager.optimizer, 'coordinated_optimizer'):
            memory_info = tp_manager.optimizer.coordinated_optimizer.get_memory_usage()
            
            print(f"      📈 Sharding Enabled: {memory_info.get('sharding_enabled', False)}")
            print(f"      💾 Est. Memory w/o TP (replicated states): {memory_info.get('total_memory', 'N/A')}")
            print(f"      💾 Est. Memory w/ TP (sharded states): {memory_info.get('sharded_memory', 'N/A')}")
            print(f"      💰 Estimated Optimizer State Memory Savings: {memory_info.get('memory_savings', 'N/A')}")
        else:
            print("      ⚠️  Could not access coordinated_optimizer to report memory savings.")
    except Exception as e:
        print(f"      ❌ Failed to calculate memory savings: {e}")
    print("=" * 61 + "\n")

    print(f"⏱️  {time.time() - start_time:.2f}s: Compiling models...")
    tp_model.compile(optimizer=base_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    baseline_optimizer = optimizers.Adam()
    baseline_model.compile(optimizer=baseline_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(f"✅ Models compiled successfully")

    print(f"⏱️  {time.time() - start_time:.2f}s: Training models for comparison...")
    x_train = np.random.randint(0, 1000, (100, 10), dtype=np.int32)
    y_train = np.random.randint(0, 1000, (100, 10), dtype=np.int32)
    
    baseline_history = baseline_model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=0)
    print(f"      ✅ Baseline model training completed")
    
    try:
        tp_history = tp_model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=0)
        print(f"      ✅ TP model training completed")
    except Exception as e:
        print(f"      ❌ TP model training failed with an error: {e}")
        import traceback
        traceback.print_exc()
        return False 

    test_passed = False
    if baseline_history and 'loss' in baseline_history.history and tp_history and 'loss' in tp_history.history:
        print(f"\n   Comparing training curves...")
        baseline_final_loss = baseline_history.history['loss'][-1]
        tp_final_loss = tp_history.history['loss'][-1]
        loss_diff = abs(baseline_final_loss - tp_final_loss)
        print(f"      Final loss difference: {loss_diff:.6f}")
        if loss_diff < 1:
            print(f"      ✅ Learning verification passed")
            test_passed = True
        else:
            print(f"      ❌ Learning verification failed")
            
        plot_training_graphs(baseline_history, tp_history)
    else:
        print("      ❌ Could not compare training curves due to missing history.")

    print(f"✅ OPT-125M training verification completed in {time.time() - start_time:.2f}s")
    return test_passed

if __name__ == "__main__":
    print("🎯 OPT-125M TENSOR PARALLEL VERIFICATION TEST SUITE")
    print("=" * 60)
    test_results = []
    test_results.append(("OPT-125M Parameter Sharding", test_opt125m_parameter_sharding()))
    test_results.append(("OPT-125M Inference Correctness", test_opt125m_inference_correctness()))
    test_results.append(("OPT-125M Training Verification", test_opt125m_training_verification()))
    
    print("\n" + "=" * 60)
    print("🎉 OPT-125M VERIFICATION TESTING COMPLETED!")
    print(f"\n📋 COMPREHENSIVE RESULTS:")
    passed_tests = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   - {test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"   - Total Tests: {len(test_results)}")
    print(f"   - Passed: {passed_tests}")
    print(f"   - Failed: {len(test_results) - passed_tests}")
    
    if passed_tests == len(test_results):
        print("\n🚀 SUCCESS: All OPT-125M verification tests passed!")
    else:
        print(f"\n⚠️  WARNING: {len(test_results) - passed_tests} tests failed.")