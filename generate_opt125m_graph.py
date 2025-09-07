#!/usr/bin/env python3
"""
Test suite for OPT-125M model verification with tensor parallelism,
using a KerasNLP preset and the Tiny Shakespeare dataset.
"""
import os
import time
import logging
import numpy as np
import keras
import keras_hub  # Using keras_nlp as it's the standard
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# Assuming the custom TensorParallelKeras class is in the specified path
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- UNIFIED CONFIGURATION (from comparison script) ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

MODEL_PRESET = "opt_125m_en"
BATCH_SIZE = 8
SEQUENCE_LENGTH = 256
LEARNING_RATE = 3e-5
EPOCHS = 10
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 10


# --- UNIFIED DATA LOADING (from comparison script) ---
def load_shakespeare_dataset():
    """Loads and preprocesses the Tiny Shakespeare dataset with a 90/10 split."""
    print("   Loading and preprocessing Tiny Shakespeare dataset...")
    ds = tfds.load("tiny_shakespeare", split="train")
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())

    tokenizer = keras_hub.models.OPTCausalLM.from_preset(MODEL_PRESET).preprocessor.tokenizer
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

# --- Model Definition ---
def get_opt_model_from_preset():
    """Creates the OPT-125M model from the KerasNLP preset."""
    print("   Creating OPT-125M model from KerasHub preset...")
    model = keras_hub.models.OPTCausalLM.from_preset(MODEL_PRESET, preprocessor=None)
    print(f"      ✅ Model created with {model.count_params():,} parameters.")
    return model

# --- UPDATED PLOTTING (from comparison script) ---
def plot_training_graphs(baseline_history, tp_history):
    """Plots and saves the loss and perplexity graphs for both models."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Baseline vs. Tensor Parallel Training Comparison", fontsize=16)

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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("tp_verification_comparison.png")
    print("\n   ✅ Comparison graph saved to tp_verification_comparison.png")

# --- UPDATED TRAINING VERIFICATION ---
def test_opt125m_training_verification():
    """Test OPT-125M training verification with a fair comparison."""
    print("🔧 OPT-125M Training Verification (Fair Comparison)")
    print("=" * 50)
    start_time = time.time()
    
    # 1. Create a template model and save its initial weights for a fair comparison
    model_template = get_opt_model_from_preset()
    initial_weights = model_template.get_weights()
    print("      ✅ Initial weights saved from template model.")

    # 2. Prepare the dataset using the unified function
    train_ds_raw, val_ds_raw = load_shakespeare_dataset()
    
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
    baseline_model = get_opt_model_from_preset()
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
    test_passed = False
    print("\n   --- Comparing Final Validation Metrics ---")
    baseline_final_val_loss = baseline_history.history['val_loss'][-1]
    tp_final_val_loss = tp_history.history['val_loss'][-1]
    loss_diff = abs(baseline_final_val_loss - tp_final_val_loss)
    
    print(f"      Baseline Final Validation Loss: {baseline_final_val_loss:.4f}")
    print(f"      TP Final Validation Loss:       {tp_final_val_loss:.4f}")
    print(f"      Final Validation Loss Difference: {loss_diff:.6f}")
    
    if loss_diff < 0.1:
        print("      ✅ Verification passed (validation losses are close).")
        test_passed = True
    else:
        print("      ❌ Verification failed (validation losses diverged).")
        
    plot_training_graphs(baseline_history, tp_history)

    print(f"✅ Test completed in {time.time() - start_time:.2f}s")
    return test_passed

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n🎯 OPT-125M TENSOR PARALLEL VERIFICATION (REAL-WORLD SCENARIO)")
    print("=" * 70)
    
    try:
        result = test_opt125m_training_verification()
        status = "✅ PASS" if result else "❌ FAIL"
    except Exception as e:
        logger.error(f"Test failed with an exception: {e}", exc_info=True)
        status = "❌ FAIL"

    print("\n" + "=" * 70)
    print("🎉 VERIFICATION TESTING COMPLETED!")
    print(f"   - Training Verification Result: {status}")
    print("=" * 70)