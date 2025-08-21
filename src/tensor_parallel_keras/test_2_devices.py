import os
import random
import string

# --- 1. SIMULATE MULTIPLE DEVICES ON A SINGLE CPU ---
# This must be at the very top, before other TensorFlow/Keras imports.
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('CPU')
if physical_devices:
    # Create 2 virtual devices on the CPU
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration()]
    )
    print("Created 2 virtual devices on CPU for testing.")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import keras
from keras import layers
import keras_nlp
from datasets import load_dataset

# --- 2. IMPORT TENSOR PARALLEL LIBRARY ---
# Assuming your library is named tensor_parallel_keras
import tensor_parallel_keras as tp

# --- Configuration ---
DUMMY_DATASET_PATH = "dummy_training_data.txt" 
BATCH_SIZE = 4
SEQUENCE_LENGTH = 64
EPOCHS = 10

# def create_and_save_dummy_dataset(file_path, num_sentences=500, vocab_size=500, max_len=15):
#     """Generates a small text file with random sentences."""
#     print(f"Creating a new dummy dataset at {file_path}...")
#     vocab = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 7))) for _ in range(vocab_size)]
    
#     with open(file_path, 'w') as f:
#         for _ in range(num_sentences):
#             sentence_len = random.randint(5, max_len)
#             sentence = ' '.join(random.choices(vocab, k=sentence_len))
#             f.write(sentence + '.\n')
#     print("Dummy dataset created successfully.")


def create_tiny_model(vocab_size, num_layers=2, hidden_size=128, num_heads=4, mlp_dim=256):
    """Create a very small Transformer model for fast training."""
    print("   Creating TINY model from scratch...")
    
    inputs = layers.Input(shape=(None,), dtype='int32', name='input_ids')
    
    hidden_states = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocab_size,
        sequence_length=SEQUENCE_LENGTH,
        embedding_dim=hidden_size,
        name='embed_tokens'
    )(inputs)
    
    for _ in range(num_layers):
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads
        )(hidden_states, hidden_states)
        hidden_states = layers.Add()([hidden_states, attention_output])
        hidden_states = layers.LayerNormalization(epsilon=1e-5)(hidden_states)
        
        mlp_output = layers.Dense(mlp_dim, activation='relu')(hidden_states)
        mlp_output = layers.Dense(hidden_size)(mlp_output)
        hidden_states = layers.Add()([hidden_states, mlp_output])
        hidden_states = layers.LayerNormalization(epsilon=1e-5)(hidden_states)
    
    outputs = layers.Dense(vocab_size, name='lm_head')(hidden_states)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='TinyModel')
    return model


def load_and_prepare_dataset(tokenizer, file_path):
    """
    Loads the local text dataset, creating it if it doesn't exist.
    """
    if not os.path.exists(file_path):
        create_and_save_dummy_dataset(file_path)
        
    print(f"Loading local dataset from: {file_path}...")
    dataset = load_dataset("text", data_files=file_path, split="train")

    def tokenize_function(examples):
        return {"token_ids": tokenizer.tokenize(examples["text"])}

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= SEQUENCE_LENGTH:
            total_length = (total_length // SEQUENCE_LENGTH) * SEQUENCE_LENGTH
        
        result = {
            k: [t[i : i + SEQUENCE_LENGTH] for i in range(0, total_length, SEQUENCE_LENGTH)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["token_ids"]
        return result

    grouped_dataset = tokenized_dataset.map(group_texts, batched=True)
    grouped_dataset = grouped_dataset.rename_column("token_ids", "input_ids")

    def format_for_fit(features):
        return (features["input_ids"], features["labels"])

    tf_dataset = grouped_dataset.to_tf_dataset(
        columns=["input_ids", "labels"],
        shuffle=True,
        batch_size=BATCH_SIZE
    ).map(format_for_fit)

    return tf_dataset


def main():
    """
    Main function to load, compile, and train the model.
    """
    print(f"Loading tokenizer...")
    tokenizer = keras_nlp.models.OPTTokenizer.from_preset("opt_125m_en")
    
    # Create the original, single-device model
    model = create_tiny_model(vocab_size=tokenizer.vocabulary_size())
    
    # --- 3. SHARD THE MODEL FOR TENSOR PARALLELISM ---
    print("\nSharding the model to run on 2 devices...")
    model = tp.shard(model) # This returns the new, parallelized model
    
    print("\n--- SHARDED MODEL SUMMARY ---")
    model.summary()
    
    train_dataset = load_and_prepare_dataset(tokenizer, file_path=DUMMY_DATASET_PATH)

    print("\nCompiling sharded model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    
    print(f"\n--- Starting Training on 2 (Virtual) Devices for {EPOCHS} Epochs ---")
    history = model.fit(train_dataset, epochs=EPOCHS, verbose=1)
    
    print(f"--- Finished Training ---")

    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    
    print("\n" + "="*50)
    print("         TRAINING COMPLETE: FINAL METRICS")
    print("="*50)
    print(f"  - Final Loss (Epoch {EPOCHS}): {final_loss:.4f}")
    print(f"  - Final Accuracy (Epoch {EPOCHS}): {final_accuracy:.4f}")
    print("="*50)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    main()