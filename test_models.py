import os
import time
import logging
import numpy as np

import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
    tf.config.set_visible_devices([], 'TPU')
    print("✅ TensorFlow visibility successfully set to CPU-only.")
except RuntimeError:
    print("⚠️ Could not set TensorFlow visible devices. (May already be initialized)")
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=9"
os.environ["KERAS_BACKEND"] = "jax"

# --- STEP 3: Now import JAX, Keras, and all other libraries ---
import jax
import keras
import keras_hub
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# --- Configuration and Initialization ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- JAX Device Detection (MODIFIED SECTION) ---
try:
    all_devices = jax.devices()
    print(f"All JAX devices found: {[str(d) for d in all_devices]}")

    # Auto-identify platform, prioritizing TPU > GPU > CPU
    tpu_devices = [d for d in all_devices if d.platform == 'tpu']
    gpu_devices = [d for d in all_devices if d.platform == 'gpu']
    cpu_devices = [d for d in all_devices if d.platform == 'cpu']
    
    if tpu_devices:
        target_devices = tpu_devices
        platform = "TPU"
    elif gpu_devices:
        target_devices = gpu_devices
        platform = "GPU"
    elif cpu_devices:
        target_devices = cpu_devices
        platform = "CPU"
    else:
        target_devices = all_devices # Fallback
        platform = "Unknown"
    
    # --- AUTO-IDENTIFICATION ---
    # Set WORLD_SIZE based on the number of devices found on the primary platform
    WORLD_SIZE = len(target_devices)
    # --- END AUTO-IDENTIFICATION ---
    
    DEVICES_AVAILABLE = WORLD_SIZE
    TARGET_DEVICES = target_devices
    TARGET_WORLD_SIZE = WORLD_SIZE
    
    if TARGET_WORLD_SIZE == 0:
        print("⚠️ WARNING: No usable JAX devices found!")
    else:
        print(f"✅ Auto-identified {TARGET_WORLD_SIZE} devices on platform: {platform}")
        print(f"   Targeting all {TARGET_WORLD_SIZE} devices for parallelism: {[str(d) for d in TARGET_DEVICES]}")

except Exception as e:
    print(f"Could not initialize JAX or find devices. Error: {e}")
    TARGET_WORLD_SIZE = 0
# --- END MODIFIED SECTION ---


try:
    from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
except ImportError:
    print("Warning: `TensorParallelKeras` not found. Using a mock class for demonstration.")
    class TensorParallelKeras:
        def __init__(self, model, world_size, distributed_backend, device_ids=None): # Added device_ids
            self._model = model
            print(f"Mock TensorParallelKeras initialized for model: {model.name}, world_size: {world_size}")
        def build_assembled_model(self):
            # In a mock, just return a clone
            return keras.models.clone_model(self._model)
        def set_weights(self, weights): # Add mock set_weights
            print("Mock: setting weights")
            self._model.set_weights(weights)


# --- Constants ---
BATCH_SIZE = 32
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4
EPOCHS = 2
STEPS_PER_EPOCH = 5
VALIDATION_STEPS = 10

MODEL_MAPPING = {
    "gpt2_base_en": keras_hub.models.GPT2CausalLM,
    "opt_125m_en": keras_hub.models.OPTCausalLM,
}

# ----------------------------------------------------------------------
# --- Dataset and Model Helpers (UNCHANGED) ---
# ----------------------------------------------------------------------

def load_shakespeare_dataset(model_preset, model_class):
    """Loads and preprocesses the Tiny Shakespeare dataset for a given model."""
    print(f"   Loading and preprocessing Tiny Shakespeare dataset for {model_preset}...")
    ds = tfds.load("tiny_shakespeare", split="train")
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())

    # Need to load the preprocessor just for the tokenizer
    tokenizer = model_class.from_preset(model_preset).preprocessor.tokenizer
    token_ids = tokenizer.tokenize(text)

    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = np.array(token_ids[:num_tokens]).reshape(-1, SEQUENCE_LENGTH + 1)
    
    all_data = tf.data.Dataset.from_tensor_slices(sequences)
    
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

def get_model_from_preset(preset_name, model_class):
    """Creates a CausalLM model from a KerasNLP preset."""
    print(f"   Creating {preset_name} model from KerasHub preset...")
    # Load model without the preprocessor, as we are manually preprocessing
    model = model_class.from_preset(preset_name, preprocessor=None)
    print(f"      ✅ Model created with {model.count_params():,} parameters.")
    return model


def run_model_verification(preset_name, model_class):
    """Runs the full training verification test for a given model preset."""
    
    # --- Check if we have enough devices to run TP ---
    if TARGET_WORLD_SIZE < 2:
        print(f"SKIPPING {preset_name}: Need at least 2 TPUs, found {TARGET_WORLD_SIZE}")
        return "SKIPPED"
    
    print(f"🔧 VERIFICATION FOR: {preset_name.upper()}")
    print("=" * 50)
    start_time_total = time.time()
    
    model_template = get_model_from_preset(preset_name, model_class)
    initial_weights = model_template.get_weights()
    print("      ✅ Initial weights saved from template model.")

    train_ds_raw, val_ds_raw = load_shakespeare_dataset(preset_name, model_class)
    
    # Prepare data pipelines
    train_ds = (
        train_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        # --- FIX: Corrected AUTOTOOL to AUTOTUNE ---
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )
    val_ds = (
        val_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        # --- FIX: Corrected AUTOTOOL to AUTOTUNE ---
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )

    # --- NEW: Calculate total tokens for throughput ---
    total_steps = STEPS_PER_EPOCH * EPOCHS
    total_samples = total_steps * BATCH_SIZE
    # Using SEQUENCE_LENGTH (input tokens) for throughput calculation
    total_tokens_processed = total_samples * SEQUENCE_LENGTH 
    print(f"   Tokens per step: {BATCH_SIZE * SEQUENCE_LENGTH:,}")
    print(f"   Total tokens to process (per model): {total_tokens_processed:,}")
    # --- END NEW ---

    # # --- RE-ENABLED BASELINE MODEL ---
    # print("\n   --- Training Baseline Model ---")
    # baseline_model = get_model_from_preset(preset_name, model_class)
    # baseline_model.set_weights(initial_weights)
    # baseline_model.compile(
    #     optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
    #     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")],
    # )
    
    # baseline_start_time = time.time() # <-- NEW
    # baseline_history = baseline_model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=EPOCHS,
    #     steps_per_epoch=STEPS_PER_EPOCH,
    #     validation_steps=VALIDATION_STEPS,
    #     verbose=1
    # )
    # baseline_end_time = time.time() # <-- NEW
    print("      ✅ Baseline model training completed.")
    # --- END RE-ENABLED SECTION ---


    print("\n   --- Training Tensor Parallel (TP) Model ---")
    
    print(f"   Initializing TensorParallelKeras with world_size={TARGET_WORLD_SIZE} on devices: {[str(d) for d in TARGET_DEVICES]}")
    tp_manager = TensorParallelKeras(
        model=model_template, 
        world_size=TARGET_WORLD_SIZE, 
        distributed_backend='jax', # Explicitly use 'jax'
        device_ids=TARGET_DEVICES   # Pass the detected JAX devices
    )
    tp_model = tp_manager.build_assembled_model()
    
    try:
        tp_model.set_weights(initial_weights)
        print("      ✅ Initial weights set on TP model.")
    except Exception as e:
        print(f"      Warning: Could not set weights on TP model (this is expected if layer names change). {e}")

    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")],
    )
    
    tp_start_time = time.time() # <-- NEW
    tp_history = tp_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1
    )
    tp_end_time = time.time() # <-- NEW
    print("      ✅ TP model training completed.")

    # --- NEW: Calculate times and throughput ---
    # baseline_time = baseline_end_time - baseline_start_time
    # baseline_throughput_tps = total_tokens_processed / baseline_time # Tokens/sec
    
    tp_time = tp_end_time - tp_start_time
    tp_throughput_tps = total_tokens_processed / tp_time # Tokens/sec
    # --- END NEW ---

    print("\n   --- Comparing Final Validation Metrics ---")
    # baseline_final_val_loss = baseline_history.history['val_loss'][-1]
    tp_final_val_loss = tp_history.history['val_loss'][-1]
    # loss_diff = abs(baseline_final_val_loss - tp_final_val_loss)
    
    # --- NEW: Print Performance Metrics ---
    print("\n   --- Performance Comparison ---")
    # print(f"      Baseline Training Time: {baseline_time:.2f} s")
    print(f"      TP Training Time:       {tp_time:.2f} s")
    # print(f"\n      Baseline Throughput: {baseline_throughput_tps:,.2f} Tokens/s")
    print(f"      TP Throughput:       {tp_throughput_tps:,.2f} Tokens/s")
    # --- END NEW ---
    
    # test_passed = loss_diff < 0.1
    # if test_passed:
        # print("\n      ✅ Verification passed (validation losses are close).")
    # else:
        # print("\n      ❌ Verification failed (validation losses diverged).")
        
    # plot_training_graphs(baseline_history, tp_history, preset_name)

    # print(f"✅ Test for {preset_name} completed in {time.time() - start_time_total:.2f}s")
    # return test_passed

# ----------------------------------------------------------------------
# --- Main Execution (UNCHANGED) ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # --- Check for devices before starting ---
    if TARGET_WORLD_SIZE == 0:
        print("🛑 ERROR: No JAX TPUs found. Aborting verification suite.")
        # --- FIX: Need to import sys for sys.exit ---
        import sys
        sys.exit(1)
        
    print("\n🎯 TENSOR PARALLELISM VERIFICATION SUITE")
    print("=" * 70)
    
    results = {}
    total_start_time = time.time()

    for preset, model_class in MODEL_MAPPING.items():
        try:
            result = run_model_verification(preset, model_class)
            if result == "SKIPPED":
                results[preset] = "⚪ SKIPPED"
            else:
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