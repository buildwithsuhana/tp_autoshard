import os
os.environ["KERAS_BACKEND"] = "jax" 
import tensorflow as tf
import numpy as np
import keras

# You will need to make sure your project structure allows these imports
# NOTE: The bug appears to be within the `tensor_parallel_keras` library itself.
from tensor_parallel_keras import TensorParallelKeras

keras.utils.set_random_seed(42)

# --- Configuration ---
WORLD_SIZE = 2
BATCH_SIZE = 8
SEQ_LEN = 16
INPUT_DIM = 64
MLP_DIM = 256
TOLERANCE = 1e-6

def build_test_model():
    """Builds a simple two-layer MLP for testing."""
    inp = keras.Input(shape=(SEQ_LEN, INPUT_DIM))
    x = keras.layers.Dense(MLP_DIM, activation="relu", name="mlp_up")(inp)
    out = keras.layers.Dense(INPUT_DIM, name="mlp_down")(x)
    model = keras.Model(inputs=inp, outputs=out, name="OriginalMLP")
    return model

def compare_model_outputs_and_weights(
    original_model, tp_model, input_data, dummy_target
):
    """
    Compares outputs and weights after a single training step.
    """
    print("-" * 80)
    print("📊 STEP 1: COMPARING FORWARD PASS OUTPUTS 📊")
    print("-" * 80)

    output_original = original_model(input_data)
    output_tp = tp_model(input_data)
    forward_pass_diff = np.max(np.abs(np.array(output_original) - np.array(output_tp)))

    print(f"   Original model output calculated. Shape: {output_original.shape}")
    print(f"   Tensor parallel model output calculated. Shape: {output_tp.shape}")
    print(f"\n   Maximum absolute difference in forward pass: {forward_pass_diff}")

    # This part passes, which means the library correctly calculates the
    # forward pass across the shards.
    if forward_pass_diff < TOLERANCE:
        print("   ✅ PASSED: Forward pass outputs are numerically identical.")
    else:
        print("   ❌ FAILED: Forward pass outputs differ.")
        return False
    print("\n")

    # --- BACKWARD PASS TEST ---
    print("-" * 80)
    print("📊 STEP 2: COMPARING WEIGHTS AFTER ONE TRAINING STEP 📊")
    print("-" * 80)
    
    # Perform one training step on the original model
    print("   Performing one training step on original model...")
    original_model.train_on_batch(input_data, dummy_target)
    weights_original_after_step = original_model.get_weights()
    print("   Original model weights updated.")

    # Perform one training step on the tensor parallel model using its custom train_step
    print("   Performing one training step on tensor parallel model...")
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR ANALYSIS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # The traceback originates from this line. The error is:
    # `ValueError: No gradients provided for any variable.`
    #
    # This error means that when the optimizer tried to apply gradients to the
    # model's weights, the list of gradients it received was empty or contained
    # only `None` values. In TensorFlow/Keras, this happens when the `tf.GradientTape`
    # cannot find a differentiable path from the loss function back to the model's
    # trainable variables.
    #
    # A KEY CLUE from your output log is the line:
    # `✅ Sharded mlp_up.kernel: torch.Size([64, 256]) -> torch.Size([64, 128])`
    #
    # Your script explicitly sets KERAS_BACKEND="tensorflow", but the library's
    # logging is printing `torch.Size`. This strongly suggests that the library
    # is internally converting TensorFlow tensors to PyTorch tensors to perform
    # its sharding logic.
    #
    # This conversion breaks the computation graph that `tf.GradientTape` records.
    # The tape tracks operations on TensorFlow tensors, but when a tensor is
    # converted to a PyTorch tensor and back, the tape loses the connection.
    # It no longer knows how the final loss is related to the initial trainable
    # variables, and therefore `tape.gradient(loss, variables)` returns `None`
    # for all variables.
    #
    # FIX: This issue must be fixed within the `tensor_parallel_keras` library.
    # The library developers would need to ensure that any cross-backend tensor
    # conversions are done using a `tf.custom_gradient` to manually define the
    # backward pass, thereby preserving the gradient chain. Your test script
    # is correctly written and has successfully identified this bug in the library.
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    tp_model.train_on_batch(input_data, dummy_target)
    weights_tp_after_step = tp_model.original_model.get_weights()
    print("   Tensor parallel model weights updated.")

    # --- Final Comparison ---
    all_weights_match = True
    for i, (w_orig, w_tp) in enumerate(zip(weights_original_after_step, weights_tp_after_step)):
        weight_diff = np.max(np.abs(np.array(w_orig) - np.array(w_tp)))
        param_name = original_model.weights[i].name
        if weight_diff >= TOLERANCE:
            all_weights_match = False
            print(f"   ❌ MISMATCH on parameter '{param_name.split(':')[0]}': Max difference = {weight_diff}")

    if all_weights_match:
        print("\n   ✅ PASSED: All weights are numerically identical after one training step.")
    else:
        print("\n   ❌ FAILED: Weights differ between models after one training step.")
    
    return all_weights_match

def run_test():
    """Runs the full numerical stability and correctness test."""
    print("=" * 80)
    print("🚀 Starting Tensor Parallel Keras TF Numerical Stability Test 🚀")
    print("=" * 80)

    original_model = build_test_model()
    
    model_for_tp = build_test_model()
    tp_model = TensorParallelKeras(
        model_for_tp,
        world_size=WORLD_SIZE,
    )
    # Ensure both models start with the exact same weights
    tp_model.set_weights(original_model.get_weights())

    optimizer_orig = keras.optimizers.Adam(learning_rate=0.001)
    optimizer_tp = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.MeanSquaredError()
    
    original_model.compile(optimizer=optimizer_orig, loss=loss_fn)
    tp_model.compile(optimizer=optimizer_tp, loss=loss_fn)

    # ... (assuming BATCH_SIZE, SEQ_LEN, INPUT_DIM are defined)
    input_data = keras.random.normal(shape=(BATCH_SIZE, SEQ_LEN, INPUT_DIM))
    dummy_target = keras.random.normal(shape=(BATCH_SIZE, SEQ_LEN, INPUT_DIM))
    
    try:
        compare_model_outputs_and_weights(
            original_model, tp_model, input_data, dummy_target
        )
    except ValueError as e:
        print("\n" + "="*80)
        print("💥 TEST FAILED WITH EXPECTED ERROR 💥")
        print(f"   Error: {e}")
        print("   This is the expected failure due to the gradient issue explained above.")
        print("="*80)


if __name__ == "__main__":
    run_test()
