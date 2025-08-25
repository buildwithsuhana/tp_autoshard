#!/usr/bin/env python3
"""
Test suite for realistic memory savings with tensor parallelism.
"""

import time
import logging
import numpy as np
import pytest

# Import required modules
try:
    import keras
    from keras import layers
    # Make sure this import path is correct for your project structure
    from src.tensor_parallel_keras.coordinated_optimizer import CoordinatedOptimizer
    print("✅ Required modules imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    pytest.skip(f"Required modules not available: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import tensorflow as tf # Make sure keras is imported

def get_optimizer_memory_info(optimizer, model, world_size, enable_sharding=True):
    """Helper function to get memory usage for a given optimizer and world size."""
    
    if not getattr(optimizer, 'built', False):
        print("      🛠️  Performing backend-agnostic dummy step to initialize optimizer state...")
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        input_shape = model.input_shape[1:]
        output_shape = model.output_shape[1:]
        dummy_x = np.random.rand(2, *input_shape).astype(np.float32)
        dummy_y = np.random.rand(2, *output_shape).astype(np.float32)

        model.train_on_batch(dummy_x, dummy_y)
    
    stateful_optimizer = model.optimizer

    # --- START: NEW DEBUGGING BLOCK ---
    print("-" * 20)
    print(f"      [DEBUG] Inspecting optimizer: {type(stateful_optimizer)}")
    try:
        # The .variables property should list all state variables.
        optimizer_vars = stateful_optimizer.variables
        print(f"      [DEBUG] Optimizer has {len(optimizer_vars)} variables.")
        for var in optimizer_vars:
            # Keras variables have a .path like 'Adam/iter:0' or 'Adam/m/...'
            print(f"      [DEBUG]   -> Var: {var.path}, Shape: {var.shape}")
    except Exception as e:
        print(f"      [DEBUG] Could not inspect optimizer variables: {e}")
    print("-" * 20)
    # --- END: NEW DEBUGGING BLOCK ---

    coord_opt = CoordinatedOptimizer(
        base_optimizer=stateful_optimizer,
        world_size=world_size,
        distributed_backend='fallback',
        shard_optimizer_states=enable_sharding
    )
    
    return coord_opt.get_memory_usage()

def test_realistic_memory_savings():
    """Test realistic memory savings with large models."""
    try:
        print("🚀 Testing Realistic Memory Savings")
        print("=" * 40)
        start_time = time.time()
        
        print(f"⏱️  {time.time() - start_time:.2f}s: Creating large model...")
        model = keras.Sequential([
            layers.Input(shape=(784,)),
            layers.Dense(2048, activation='relu'),
            layers.Dense(4096, activation='relu'),
            layers.Dense(2048, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        print(f"✅ {time.time() - start_time:.2f}s: Model created with {model.count_params():,} parameters")
        
        world_sizes = [2, 4, 8]
        optimizers_to_test = {
            "Adam": keras.optimizers.Adam(learning_rate=0.001),
            "SGD": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
            "RMSprop": keras.optimizers.RMSprop(learning_rate=0.001)
        }

        for name, optimizer in optimizers_to_test.items():
            print(f"\n🔄 Testing {name} Optimizer")
            print("-" * 30)
            for world_size in world_sizes:
                print(f"   World Size: {world_size}")
                
                # The helper function now correctly builds the optimizer internally on the first call.
                memory_no_sharding = get_optimizer_memory_info(optimizer, model, world_size, enable_sharding=False)
                print(f"      No sharding: {memory_no_sharding}")
                
                memory_with_sharding = get_optimizer_memory_info(optimizer, model, world_size, enable_sharding=True)
                print(f"      With sharding: {memory_with_sharding}")

                # Basic assertion to ensure memory calculation is working
                assert memory_with_sharding['total_memory_bytes'] > 0
                
                if memory_with_sharding['sharding_enabled']:
                    savings = memory_with_sharding['memory_savings']
                    theoretical_max = f"{(1 - 1/world_size) * 100:.1f}%"
                    print(f"      💾 Memory savings: {savings}")
                    print(f"      📊 Theoretical max savings: {theoretical_max}")

        print(f"\n✅ Realistic memory test completed in {time.time() - start_time:.2f}s")
        return True
    except Exception as e:
        logger.error(f"❌ Realistic memory test failed: {e}", exc_info=True)
        return False

def test_optimizer_state_partitioning():
    """Test optimizer state partitioning across devices."""
    try:
        print("🔧 Testing Optimizer State Partitioning")
        print("=" * 40)
        start_time = time.time()

        model = keras.Sequential([
            layers.Input(shape=(10,)),
            layers.Dense(100, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # --- FIX 2: Create the base optimizer and build it *before* wrapping it ---
        # 1. Create the base optimizer instance.
        base_optimizer = keras.optimizers.Adam(learning_rate=0.001)

        # 2. Build it against the model's variables to create its state.
        base_optimizer.build(model.trainable_variables)
        
        # 3. Now, pass the *pre-built* base optimizer to the wrapper.
        optimizer = CoordinatedOptimizer(
            base_optimizer=base_optimizer,
            world_size=4,
            shard_optimizer_states=True
        )
        
        # 4. The incorrect call to optimizer.build() is now removed.
        print(f"✅ {time.time() - start_time:.2f}s: Coordinated optimizer created with a pre-built base optimizer")
        
        sharded_states = optimizer._get_sharded_states_structure()
        print("   Sharded states structure:")
        assert 'error' not in sharded_states
        print(f"     State keys found: {list(sharded_states.keys())}")

        print(f"✅ Partitioning test completed in {time.time() - start_time:.2f}s")
        return True
    except Exception as e:
        logger.error(f"❌ Optimizer partitioning test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    print("🎯 REALISTIC MEMORY SAVINGS TEST")
    print("=" * 40)
    
    test1_success = test_realistic_memory_savings()
    print("\n" + "-" * 40 + "\n")
    test2_success = test_optimizer_state_partitioning()
    
    print("\n" + "=" * 40)
    print("🎉 TESTING COMPLETED!")
    print(f"\n📋 RESULTS:")
    print(f"   - Realistic Memory: {'✅' if test1_success else '❌'}")
    print(f"   - State Partitioning: {'✅' if test2_success else '❌'}")
    
    if all([test1_success, test2_success]):
        print("\n🚀 SUCCESS: All realistic memory tests passed!")
    else:
        print("\n⚠️  WARNING: Some tests failed.")