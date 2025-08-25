#!/usr/bin/env python3
"""
Test suite for sharded optimizer states functionality.
"""

import time
import logging
import numpy as np
import pytest

# Import required modules
try:
    import keras
    from keras import layers
    from src.tensor_parallel_keras.coordinated_optimizer import CoordinatedOptimizer
    print("✅ Required modules imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    pytest.skip(f"Required modules not available: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FIX: Create a dummy model to build the optimizer against ---
def create_dummy_model():
    """Creates a small Keras model for testing purposes."""
    return keras.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(20),
        layers.Dense(5)
    ])

def test_sharded_optimizer_states():
    """Test sharded optimizer states functionality."""
    print("🚀 Testing Sharded Optimizer States")
    print("=" * 40)
    
    start_time = time.time()
    
    # --- FIX: Create a model and build the optimizer ---
    model = create_dummy_model()
    base_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    base_optimizer.build(model.trainable_variables)
    print(f"✅ {time.time() - start_time:.2f}s: Base optimizer created and built")
    
    world_sizes = [2, 4, 8]
    all_tests_passed = True
    
    for world_size in world_sizes:
        print(f"\n🔄 Testing with world_size={world_size}")
        print("-" * 30)
        
        # Test WITH optimizer state sharding
        print(f"   Testing WITH optimizer state sharding...")
        coord_opt_with_sharding = CoordinatedOptimizer(
            base_optimizer=base_optimizer,
            world_size=world_size,
            distributed_backend='fallback',
            shard_optimizer_states=True
        )
        
        memory_info = coord_opt_with_sharding.get_memory_usage()
        print(f"      Memory info: {memory_info}")
        
        if memory_info.get('sharding_enabled'):
            print(f"      ✅ Sharding enabled successfully")
            if 'memory_savings' in memory_info:
                print(f"      💾 Memory savings: {memory_info['memory_savings']}")
        else:
            all_tests_passed = False
            print(f"      ❌ Sharding not enabled")
    
    print(f"✅ Sharded optimizer test completed in {time.time() - start_time:.2f}s")
    return all_tests_passed

def test_optimizer_state_management():
    """Test optimizer state management (enable/disable sharding)."""
    print("🔧 Testing Optimizer State Management")
    print("=" * 40)
    
    start_time = time.time()

    # --- FIX: Create and build the base optimizer first ---
    model = create_dummy_model()
    base_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    base_optimizer.build(model.trainable_variables)
    print(f"✅ {time.time() - start_time:.2f}s: Base optimizer created and built")

    # Create coordinated optimizer with sharded states
    coord_opt = CoordinatedOptimizer(
        base_optimizer=base_optimizer,
        world_size=4,
        distributed_backend='fallback',
        shard_optimizer_states=True
    )
    
    # Check initial memory info
    initial_memory = coord_opt.get_memory_usage()
    print(f"   Initial memory info (sharded): {initial_memory}")
    
    # Disable optimizer state sharding
    print(f"   Disabling optimizer state sharding...")
    coord_opt.disable_optimizer_state_sharding()
    
    memory_after_disable = coord_opt.get_memory_usage()
    print(f"   Memory info after disabling: {memory_after_disable}")
    
    # Re-enable optimizer state sharding
    print(f"   Re-enabling optimizer state sharding...")
    coord_opt.enable_optimizer_state_sharding()
    
    memory_after_reenable = coord_opt.get_memory_usage()
    print(f"   Memory info after re-enabling: {memory_after_reenable}")
    
    print(f"✅ State management test completed in {time.time() - start_time:.2f}s")
    
    # Check if the enable/disable cycle worked correctly
    return (initial_memory.get('sharding_enabled') and 
            not memory_after_disable.get('sharding_enabled') and 
            memory_after_reenable.get('sharding_enabled'))

def test_tensor_parallel_optimizer():
    """Test TensorParallelOptimizer functionality."""
    # Note: This test uses CoordinatedOptimizer directly, as per your script.
    print("🚀 Testing TensorParallelOptimizer Wrapper Concept")
    print("=" * 40)
    
    start_time = time.time()

    # --- FIX: Create and build the base optimizer first ---
    model = create_dummy_model()
    base_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    base_optimizer.build(model.trainable_variables)
    print(f"✅ {time.time() - start_time:.2f}s: Base optimizer created and built")
    
    # Create the optimizer wrapper
    tp_optimizer = CoordinatedOptimizer(
        base_optimizer=base_optimizer,
        world_size=2,
        distributed_backend='fallback',
        shard_optimizer_states=True
    )
    
    print(f"✅ {time.time() - start_time:.2f}s: Optimizer wrapper created")
    
    # Print optimizer information
    print(f"   - World size: {tp_optimizer.world_size}")
    print(f"   - Base optimizer: {type(tp_optimizer.base_optimizer).__name__}")
    
    # Get memory usage
    memory_info = tp_optimizer.get_memory_usage()
    print(f"   - Memory info: {memory_info}")
    
    print(f"✅ TensorParallelOptimizer test completed in {time.time() - start_time:.2f}s")
    
    return (tp_optimizer.world_size == 2 and memory_info.get('sharding_enabled'))

if __name__ == "__main__":
    print("🎯 SHARDED OPTIMIZER STATES COMPREHENSIVE TEST")
    print("=" * 55)
    
    # Test 1: Basic sharded optimizer functionality
    test1_success = test_sharded_optimizer_states()
    print("-" * 55)
    
    # Test 2: State management methods
    test2_success = test_optimizer_state_management()
    print("-" * 55)

    # Test 3: TensorParallelOptimizer wrapper
    test3_success = test_tensor_parallel_optimizer()
    
    print("\n" + "=" * 55)
    print("🎉 TESTING COMPLETED!")
    print(f"\n📋 RESULTS:")
    print(f"   - Sharded Optimizer: {'✅' if test1_success else '❌'}")
    print(f"   - State Management: {'✅' if test2_success else '❌'}")
    print(f"   - TensorParallelOptimizer: {'✅' if test3_success else '❌'}")
    
    if all([test1_success, test2_success, test3_success]):
        print("\n🚀 SUCCESS: All sharded optimizer tests passed!")
    else:
        print("\n⚠️  WARNING: Some tests failed.")