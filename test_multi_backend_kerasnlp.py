#!/usr/bin/env python3
"""
Test suite for KerasNLP models with tensor parallelism.
This script tests the backend specified by the KERAS_BACKEND env variable.
"""
import time
import numpy as np
import keras
from keras import layers, ops
import pytest
import os
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

# This setting is only used when the JAX backend is active.
# os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=2')

def create_agnostic_model(input_dim: int = 64, num_classes: int = 2):
    """Creates a simple, backend-agnostic Keras model for compilation testing."""
    inputs = layers.Input(shape=(input_dim,), dtype="float32")
    # Use string initializers; determinism is handled by keras.utils.set_random_seed
    x = layers.Dense(32, activation="relu", kernel_initializer="glorot_uniform")(inputs)
    outputs = layers.Dense(num_classes, kernel_initializer="glorot_uniform")(x)
    model = keras.Model(inputs, outputs)
    return model

def run_kerasnlp_inference_test(model, model_tp):
    """Helper to run and validate inference on a KerasNLP model."""
    test_input = {
        'token_ids': np.random.randint(0, 1000, (2, 64), dtype=np.int32),
        'padding_mask': np.ones((2, 64), dtype=np.int32),
    }
    if 'segment_ids' in model.input:
        test_input['segment_ids'] = np.zeros((2, 64), dtype=np.int32)

    original_output = model(test_input)
    tp_output = model_tp(test_input)

    if isinstance(original_output, dict):
        original_seq_out = original_output['sequence_output']
        tp_seq_out = tp_output['sequence_output']
    else:
        original_seq_out = original_output
        tp_seq_out = tp_output

    print(f"      Original output shape: {original_seq_out.shape}")
    print(f"      TP output shape: {tp_seq_out.shape}")

    assert original_seq_out.shape == tp_seq_out.shape, "Output shapes don't match"
    print(f"      ✅ Output shapes match")

def test_kerasnlp_model(backbone_preset):
    """A generic test function for any KerasNLP backbone."""
    backend_name = keras.config.backend()
    print(f"\n🔧 Testing {backbone_preset} with {backend_name.upper()} Backend")
    print("=" * 50)
    start_time = time.time()

    try:
        import keras_nlp
        model_name_map = {
            "bert": "BertBackbone", "gpt2": "GPT2Backbone", "roberta": "RobertaBackbone",
        }
        prefix = backbone_preset.split('_')[0]
        class_name = model_name_map[prefix]
        BackboneClass = getattr(keras_nlp.models, class_name)
    except (ImportError, AttributeError, KeyError):
        pytest.skip(f"KerasNLP or {backbone_preset} backbone not available")

    print(f"⏱️  {time.time() - start_time:.2f}s: Creating {backbone_preset} model...")
    model = BackboneClass.from_preset(backbone_preset)
    print(f"✅ {time.time() - start_time:.2f}s: Model created.")

    print(f"⏱️  {time.time() - start_time:.2f}s: Warming up model to trigger build...")
    dummy_input = {key: keras.ops.zeros((2, 64), dtype="int32") for key in model.input}
    _ = model(dummy_input)
    print(f"✅ {time.time() - start_time:.2f}s: Model built via warm-up pass.")

    print(f"⏱️  {time.time() - start_time:.2f}s: Creating Tensor Parallel model...")
    tp_model = TensorParallelKeras(model=model, world_size=2)
    print(f"✅ {time.time() - start_time:.2f}s: Tensor Parallel model created.")

    print(f"⏱️  {time.time() - start_time:.2f}s: Testing inference...")
    run_kerasnlp_inference_test(model, tp_model)
    print(f"✅ {backbone_preset} with {backend_name.upper()} backend test completed in {time.time() - start_time:.2f}s")

def test_simple_model_compilation():
    """Tests if a simple sharded model can be compiled."""
    backend_name = keras.config.backend()
    print(f"\n🔧 Testing Simple Model Compilation with {backend_name.upper()} Backend")
    print("=" * 50)
    
    agnostic_model = create_agnostic_model(input_dim=32)
    dummy_x = keras.ops.zeros((2, 32), dtype="float32")
    _ = agnostic_model(dummy_x) # Warm-up
    
    tp_model = TensorParallelKeras(model=agnostic_model, world_size=2)
    tp_model.compile(optimizer='adam', loss='mse')
    print(f"      ✅ {backend_name.upper()} backend: Model compiled successfully")

def main():
    keras.utils.set_random_seed(1337)
    backend = keras.config.backend()
    print(f"🎯 RUNNING TEST SUITE FOR BACKEND: {backend.upper()} 🎯")
    
    # Run all tests for the configured backend
    test_simple_model_compilation()
    
    # We can add more tests here and they will run on the same backend
    if backend == "jax":
        test_kerasnlp_model("bert_tiny_en_uncased")
    elif backend == "torch":
        test_kerasnlp_model("gpt2_base_en")
    elif backend == "tensorflow":
        test_kerasnlp_model("roberta_base_en")

    print(f"\n🚀 SUCCESS: All tests for {backend.upper()} backend passed!")

if __name__ == "__main__":
    main()