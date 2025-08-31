#!/usr/bin/env python3
"""
Test suite for multi-backend KerasNLP models with tensor parallelism.
"""

import time
import numpy as np
import keras
import pytest

# Ensure JAX backend can simulate devices if used
import os
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=2')

# Import TensorParallelKeras
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

class KerasNLPBackboneWrapper(keras.Layer):
    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone

    def call(self, inputs):
        return self.backbone(inputs)['sequence_output']

    def compute_output_shape(self, input_shape):
        token_ids_shape = input_shape['token_ids']
        batch_size = token_ids_shape[0]
        sequence_length = token_ids_shape[1]
        hidden_dim = self.backbone.hidden_dim
        return (batch_size, sequence_length, hidden_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"backbone": keras.saving.serialize(self.backbone)})
        return config

    @classmethod
    def from_config(cls, config):
        config["backbone"] = keras.saving.deserialize(config["backbone"])
        return cls(**config)


def test_bert_with_jax_backend():
    """Test BERT with JAX backend."""
    keras.config.set_backend("jax")
    print("\n🔧 Testing BERT with JAX Backend")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        import keras_nlp
    except ImportError:
        pytest.skip("KerasNLP not available")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating BERT model...")
    bert_model = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")
    print(f"✅ {time.time() - start_time:.2f}s: BERT model created.")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating Tensor Parallel model...")
    
    tp_manager = TensorParallelKeras(
        model=bert_model,
        device_ids=['cpu:0', 'cpu:1']
    )
    model_tp_assembled = tp_manager.build_assembled_model()
    print(f"✅ {time.time() - start_time:.2f}s: Assembled TP model created successfully.")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Testing inference...")
    test_input = {
        'token_ids': np.random.randint(0, 1000, (2, 64), dtype=np.int32),
        'padding_mask': np.ones((2, 64), dtype=np.int32),
        'segment_ids': np.zeros((2, 64), dtype=np.int32)
    }
    
    original_output = bert_model(test_input)
    tp_output = model_tp_assembled(test_input)
    
    original_sequence_output = original_output['sequence_output']
    tp_sequence_output = tp_output['sequence_output']
    
    print(f"      Original output shape: {original_sequence_output.shape}")
    print(f"      TP output shape: {tp_sequence_output.shape}")
    
    assert original_sequence_output.shape == tp_sequence_output.shape, "Output shapes don't match"
    print(f"      ✅ Output shapes match")
    
    print(f"✅ BERT with JAX backend test completed in {time.time() - start_time:.2f}s")

def create_wrapped_model_from_backbone(backbone):
    """Creates a functional model using the KerasNLPBackboneWrapper layer."""
    inputs = {
        key: keras.Input(shape=(None,), dtype="int32", name=key)
        for key in backbone.input
    }
    outputs = KerasNLPBackboneWrapper(backbone)(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

def test_gpt2_with_pytorch_backend():
    """Test GPT-2 with PyTorch backend."""
    keras.config.set_backend("torch")
    print("\n🔧 Testing GPT-2 with PyTorch Backend")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        import keras_nlp
    except ImportError:
        pytest.skip("KerasNLP not available")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating GPT-2 model...")
    gpt2_backbone = keras_nlp.models.GPT2Backbone.from_preset("gpt2_base_en")
    gpt2_model = create_wrapped_model_from_backbone(gpt2_backbone)
    print(f"✅ {time.time() - start_time:.2f}s: GPT-2 model created and wrapped.")

    # --- FINAL FIX: Explicitly build the model before passing it to the library ---
    # This forces Keras to finalize the layer graph, populating the `.layers`
    # attribute that the TensorParallelKeras library requires on initialization.
    build_shape = {'token_ids': (2, 64), 'padding_mask': (2, 64)}
    gpt2_model.build(build_shape)
    print(f"✅ {time.time() - start_time:.2f}s: Wrapped model built explicitly.")

    print(f"⏱️  {time.time() - start_time:.2f}s: Creating Tensor Parallel model...")
    tp_manager = TensorParallelKeras(
        model=gpt2_model,
        world_size=2,
    )
    model_tp_assembled = tp_manager.build_assembled_model()
    print(f"✅ {time.time() - start_time:.2f}s: Assembled TP model created successfully.")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Testing inference...")
    test_input = {
        'token_ids': np.random.randint(0, 1000, (2, 64), dtype=np.int32),
        'padding_mask': np.ones((2, 64), dtype=np.int32)
    }
    
    original_output = gpt2_model(test_input)
    tp_output = model_tp_assembled(test_input)
    
    print(f"      Original output shape: {original_output.shape}")
    print(f"      TP output shape: {tp_output.shape}")
    
    assert original_output.shape == tp_output.shape, "Output shapes don't match"
    print(f"      ✅ Output shapes match")
    
    print(f"✅ GPT-2 with PyTorch backend test completed in {time.time() - start_time:.2f}s")

def test_roberta_with_tensorflow_backend():
    """Test RoBERTa with TensorFlow backend."""
    keras.config.set_backend("tensorflow")
    print("\n🔧 Testing RoBERTa with TensorFlow Backend")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        import keras_nlp
    except ImportError:
        pytest.skip("KerasNLP not available")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating RoBERTa model...")
    roberta_backbone = keras_nlp.models.RobertaBackbone.from_preset("roberta_base_en")
    roberta_model = create_wrapped_model_from_backbone(roberta_backbone)
    print(f"✅ {time.time() - start_time:.2f}s: RoBERTa model created and wrapped.")
    
    # --- FINAL FIX: Explicitly build the model ---
    build_shape = {'token_ids': (2, 64), 'padding_mask': (2, 64)}
    roberta_model.build(build_shape)
    print(f"✅ {time.time() - start_time:.2f}s: Wrapped model built explicitly.")

    print(f"⏱️  {time.time() - start_time:.2f}s: Creating Tensor Parallel model...")
    tp_manager = TensorParallelKeras(
        model=roberta_model,
        world_size=2,
    )
    model_tp_assembled = tp_manager.build_assembled_model()
    print(f"✅ {time.time() - start_time:.2f}s: Assembled TP model created successfully.")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Testing inference...")
    test_input = {
        'token_ids': np.random.randint(0, 1000, (2, 64), dtype=np.int32),
        'padding_mask': np.ones((2, 64), dtype=np.int32)
    }
    
    original_output = roberta_model(test_input)
    tp_output = model_tp_assembled(test_input)
    
    print(f"      Original output shape: {original_output.shape}")
    print(f"      TP output shape: {tp_output.shape}")

    assert original_output.shape == tp_output.shape, "Output shapes don't match"
    print(f"      ✅ Output shapes match")

    print(f"✅ RoBERTa with TensorFlow backend test completed in {time.time() - start_time:.2f}s")


def test_training_compilation_with_mixed_backends():
    """Test training compilation with mixed backends."""
    print("\n🔧 Testing Training Compilation with Mixed Backends")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        import keras_nlp
    except ImportError:
        pytest.skip("KerasNLP not available")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating small BERT model factory...")
    
    def create_model_for_backend():
        backbone = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")
        model = create_wrapped_model_from_backbone(backbone)
        # --- FINAL FIX: Explicitly build the model ---
        build_shape = {
            'token_ids': (2, 64), 
            'padding_mask': (2, 64),
            'segment_ids': (2, 64)
        }
        model.build(build_shape)
        return model

    backends = ['jax', 'torch', 'tensorflow']
    backend_results = []
    
    for backend in backends:
        # --- FINAL FIX: Clear session to prevent state leakage between backends ---
        keras.backend.clear_session()
        print(f"\n   Testing {backend.upper()} backend...")
        try:
            keras.config.set_backend(backend)
            bert_model = create_model_for_backend()
            
            tp_manager = TensorParallelKeras(
                model=bert_model,
                world_size=2,
            )
            model_tp_assembled = tp_manager.build_assembled_model()
            
            model_tp_assembled.compile(optimizer='adam', loss='mse')

            print(f"      ✅ {backend.upper()} backend: Model compiled successfully")
            backend_results.append((backend, True))
        except Exception as e:
            import traceback
            print(f"      ❌ {backend.upper()} backend: Failed - {e}\n{traceback.format_exc()}")
            backend_results.append((backend, False))
    
    print(f"\n   📊 Backend Compilation Test Results:")
    for backend, success in backend_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"      {backend.upper()}: {status}")
    
    print(f"✅ Mixed backend compilation test completed in {time.time() - start_time:.2f}s")
    assert all(success for _, success in backend_results)


def main():
    """Run all multi-backend tests."""
    print("🎯 MULTI-BACKEND KERASNLP TENSOR PARALLEL TEST SUITE")
    
    tests = [
        ("BERT with JAX Backend", test_bert_with_jax_backend),
        ("GPT-2 with PyTorch Backend", test_gpt2_with_pytorch_backend),
        ("RoBERTa with TensorFlow Backend", test_roberta_with_tensorflow_backend),
        ("Training Compilation with Mixed Backends", test_training_compilation_with_mixed_backends)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, True))
        except Exception as e:
            import traceback
            print(f"\n❌ {test_name} FAILED with exception: {e}")
            print(traceback.format_exc())
            results.append((test_name, False))
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n{'='*60}\n🎉 MULTI-BACKEND TESTING COMPLETED! 🎉\n{'='*60}")
    print(f"📊 SUMMARY: {passed}/{total} PASSED ({(passed/total)*100:.1f}%)")

    if passed == total:
        print(f"\n🚀 SUCCESS: All multi-backend tests passed!")
    else:
        print(f"\n⚠️ WARNING: {total - passed} tests failed.")

if __name__ == "__main__":
    main()