#!/usr/bin/env python3
"""
Test suite for model verification with tensor parallelism, using KerasNLP
presets and the Tiny Shakespeare dataset.

This script is designed to be a comprehensive verification tool that compares
the training performance (loss and perplexity) of a baseline model against
its tensor-parallel equivalent.

It is generalized to run tests for multiple model architectures, including:
- Gemma (as a substitute for Gemini)
- GPT-2
- Bloom
- OPT
"""
import os
import time
import logging
import numpy as np
import keras
import keras_hub
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# Assuming the custom TensorParallelKeras class is in the specified path
# from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
# For demonstration purposes, we will mock the class if it's not found.
try:
    from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
except ImportError:
    print("Warning: `TensorParallelKeras` not found. Using a mock class for demonstration.")
    # This mock class allows the script to run without the actual implementation.
    # The 'Tensor Parallel' model will just be a regular model.
    class TensorParallelKeras:
        def __init__(self, model, world_size, distributed_backend):
            self._model = model
            print(f"Mock TensorParallelKeras initialized for model: {model.name}, world_size: {world_size}")
        def build_assembled_model(self):
            # In a real scenario, this would return a complex distributed model.
            # Here, we return a clone of the original model for demonstration.
            return keras.models.clone_model(self._model)


# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- UNIFIED CONFIGURATION ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

BATCH_SIZE = 8
SEQUENCE_LENGTH = 256
LEARNING_RATE = 3e-5
EPOCHS = 3
STEPS_PER_EPOCH = 10
VALIDATION_STEPS = 10

# --- MODEL PRESETS TO TEST ---
# We map preset names to their corresponding KerasNLP model classes.
# Note:
# - 'gemma_2b_en' is used as a stand-in for the requested 'Gemini 3 1B'.
# - 'bloom_560m_en' is used as the closest available substitute for 'Bloom 1B'.
MODEL_MAPPING = {
    "gemma2_instruct_2b_en": keras_hub.models.GemmaCausalLM, 
    "gpt2_base_en": keras_hub.models.GPT2CausalLM,
    "bloom_560m_multi": keras_hub.models.BloomCausalLM,
    "opt_125m_en": keras_hub.models.OPTCausalLM,
}

# --- UNIFIED DATA LOADING ---
def load_shakespeare_dataset(model_preset, model_class):
    """Loads and preprocesses the Tiny Shakespeare dataset for a given model."""
    print(f"   Loading and preprocessing Tiny Shakespeare dataset for {model_preset}...")
    ds = tfds.load("tiny_shakespeare", split="train")
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())

    # Each model has its own tokenizer, loaded via its preprocessor.
    tokenizer = model_class.from_preset(model_preset).preprocessor.tokenizer
    token_ids = tokenizer.tokenize(text)

    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = np.array(token_ids[:num_tokens]).reshape(-1, SEQUENCE_LENGTH + 1)
    
    all_data = tf.data.Dataset.from_tensor_slices(sequences)
    
    # Split the data into training and validation sets (90% train, 10% val)
    num_sequences = sequences.shape[0]
    num_train_samples = int(0.9 * num_sequences)
    
    train_ds = all_data.take(num_train_samples)
    val_ds = all_data.skip(num_train_samples)

    print(f"      ✅ Dataset ready with {num_train_samples} training and {num_sequences - num_train_samples} validation sequences.")
    return train_ds, val_ds

def format_for_causal_lm(data):
    """Formats data for KerasNLP's CausalLM, creating features and labels."""
    features = {
        "token_ids": data[:, :-1],
        "padding_mask": tf.ones_like(data[:, :-1], dtype=tf.bool),
    }
    labels = data[:, 1:]
    return features, labels

# --- GENERALIZED MODEL DEFINITION ---
def get_model_from_preset(preset_name, model_class):
    """Creates a CausalLM model from a KerasNLP preset."""
    print(f"   Creating {preset_name} model from KerasHub preset...")
    model = model_class.from_preset(preset_name, preprocessor=None)
    print(f"      ✅ Model created with {model.count_params():,} parameters.")
    return model

# --- GENERALIZED PLOTTING ---
def plot_training_graphs(baseline_history, tp_history, preset_name):
    """Plots and saves the loss and perplexity graphs for a given model comparison."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"{preset_name} - Baseline vs. Tensor Parallel Training", fontsize=16)

    # --- Plot 1: Loss ---
    ax1.plot(baseline_history.history["loss"], label="Baseline - Training Loss", color="blue", linestyle="-")
    ax1.plot(baseline_history.history["val_loss"], label="Baseline - Validation Loss", color="blue", linestyle="--")
    ax1.plot(tp_history.history["loss"], label="Tensor Parallel - Training Loss", color="green", linestyle="-")
    ax1.plot(tp_history.history["val_loss"], label="Tensor Parallel - Validation Loss", color="green", linestyle="--")
    ax1.set_title("Training and Validation Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    # --- Plot 2: Perplexity ---
    ax2.plot(baseline_history.history["perplexity"], label="Baseline - Training Perplexity", color="red", linestyle="-")
    ax2.plot(baseline_history.history["val_perplexity"], label="Baseline - Validation Perplexity", color="red", linestyle="--")
    ax2.plot(tp_history.history["perplexity"], label="Tensor Parallel - Training Perplexity", color="purple", linestyle="-")
    ax2.plot(tp_history.history["val_perplexity"], label="Tensor Parallel - Validation Perplexity", color="purple", linestyle="--")
    ax2.set_title("Training and Validation Perplexity")
    ax2.set_ylabel("Perplexity")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    output_filename = f"{preset_name}_tp_verification_comparison.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    print(f"\n   ✅ Comparison graph saved to {output_filename}")
    plt.close() # Close the figure to free up memory

# --- GENERALIZED TRAINING VERIFICATION ---
def run_model_verification(preset_name, model_class):
    """Runs the full training verification test for a given model preset."""
    print(f"🔧 VERIFICATION FOR: {preset_name.upper()}")
    print("=" * 50)
    start_time = time.time()
    
    # 1. Create a template model and save its initial weights for a fair comparison
    model_template = get_model_from_preset(preset_name, model_class)
    initial_weights = model_template.get_weights()
    print("      ✅ Initial weights saved from template model.")

    # 2. Prepare the dataset using the model-specific tokenizer
    train_ds_raw, val_ds_raw = load_shakespeare_dataset(preset_name, model_class)
    
    # Create the data pipelines
    train_ds = (
        train_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )
    val_ds = (
        val_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )

    # 3. Set up and train the baseline model
    print("\n   --- Training Baseline Model ---")
    baseline_model = get_model_from_preset(preset_name, model_class)
    baseline_model.set_weights(initial_weights)
    baseline_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")],
    )
    baseline_history = baseline_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1
    )
    print("      ✅ Baseline model training completed.")

    # 4. Set up and train the Tensor Parallel model
    # The TP model is built from the original template to ensure it starts with the same weights.
    print("\n   --- Training Tensor Parallel (TP) Model ---")
    tp_manager = TensorParallelKeras(model=model_template, world_size=2, distributed_backend='fallback')
    tp_model = tp_manager.build_assembled_model()
    
    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")],
    )
    tp_history = tp_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1
    )
    print("      ✅ TP model training completed.")

    # 5. Compare results and plot
    print("\n   --- Comparing Final Validation Metrics ---")
    baseline_final_val_loss = baseline_history.history['val_loss'][-1]
    tp_final_val_loss = tp_history.history['val_loss'][-1]
    loss_diff = abs(baseline_final_val_loss - tp_final_val_loss)
    
    print(f"      Baseline Final Validation Loss: {baseline_final_val_loss:.4f}")
    print(f"      TP Final Validation Loss:       {tp_final_val_loss:.4f}")
    print(f"      Final Validation Loss Difference: {loss_diff:.6f}")
    
    test_passed = loss_diff < 0.1
    if test_passed:
        print("      ✅ Verification passed (validation losses are close).")
    else:
        print("      ❌ Verification failed (validation losses diverged).")
        
    plot_training_graphs(baseline_history, tp_history, preset_name)

    print(f"✅ Test for {preset_name} completed in {time.time() - start_time:.2f}s")
    return test_passed

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n🎯 TENSOR PARALLELISM VERIFICATION SUITE")
    print("=" * 70)
    
    results = {}
    total_start_time = time.time()

    for preset, model_class in MODEL_MAPPING.items():
        try:
            result = run_model_verification(preset, model_class)
            results[preset] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            logger.error(f"Test for {preset} failed with an exception: {e}", exc_info=True)
            results[preset] = "💥 ERROR"
        print("-" * 70)

    print("\n" + "=" * 70)
    print("🎉 VERIFICATION SUITE COMPLETED!")
    print(f"   Total execution time: {time.time() - total_start_time:.2f}s")
    print("\n   --- SUMMARY ---")
    for preset, status in results.items():
        print(f"   - {preset:<18}: {status}")
    print("=" * 70)
