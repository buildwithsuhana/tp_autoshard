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
    from src.tensor_parallel_keras.coordinated_optimizer import CoordinatedOptimizer
    print("✅ Required modules imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    pytest.skip(f"Required modules not available: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_large_model():
    """Create a large model to demonstrate memory savings."""
    import keras
    from keras import layers
    
    # Create a large model with many parameters
    model = keras.Sequential([
        layers.Input(shape=(1000,)),
        layers.Dense(2048, activation='relu'),
        layers.Dense(2048, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def get_optimizer_memory_info(optimizer, world_size, enable_sharding=True):
    """Helper function to get memory usage for a given optimizer and world size."""
    import keras
    from src.tensor_parallel_keras.coordinated_optimizer import CoordinatedOptimizer
    
    coord_opt = CoordinatedOptimizer(
        base_optimizer=optimizer,
        world_size=world_size,
        distributed_backend='fallback',
        shard_optimizer_states=enable_sharding
    )
    
    return coord_opt.get_memory_usage()

def test_realistic_memory_savings():
    """Test realistic memory savings with large models."""
    print("🚀 Testing Realistic Memory Savings")
    print("=" * 40)
    
    start_time = time.time()
    print(f"⏱️  {time.time() - start_time:.2f}s: Starting realistic memory test...")
    
    # Import required modules
    try:
        import keras
        from keras import layers
        print(f"✅ {time.time() - start_time:.2f}s: Modules imported successfully")
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating large model...")
    
    # Create a large model for realistic testing
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
    
    # Test different world sizes
    world_sizes = [2, 4, 8]
    
    print("\n🔄 Testing Adam Optimizer")
    print("-" * 30)
    
    for world_size in world_sizes:
        print(f"   World Size: {world_size}")
        
        # Test without sharding
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        memory_info = get_optimizer_memory_info(optimizer, world_size, enable_sharding=False)
        print(f"      No sharding: {memory_info}")
        
        # Test with sharding
        memory_info = get_optimizer_memory_info(optimizer, world_size, enable_sharding=True)
        print(f"      With sharding: {memory_info}")
        
        if memory_info['sharding_enabled']:
            savings = memory_info['memory_savings']
            theoretical_max = f"{(1 - 1/world_size) * 100:.1f}%"
            print(f"      💾 Memory savings: {savings}")
            print(f"      📊 Theoretical max savings: {theoretical_max}")
    
    print("\n🔄 Testing SGD Optimizer")
    print("-" * 30)
    
    for world_size in world_sizes:
        print(f"   World Size: {world_size}")
        
        # Test without sharding
        optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        memory_info = get_optimizer_memory_info(optimizer, world_size, enable_sharding=False)
        print(f"      No sharding: {memory_info}")
        
        # Test with sharding
        memory_info = get_optimizer_memory_info(optimizer, world_size, enable_sharding=True)
        print(f"      With sharding: {memory_info}")
        
        if memory_info['sharding_enabled']:
            savings = memory_info['memory_savings']
            theoretical_max = f"{(1 - 1/world_size) * 100:.1f}%"
            print(f"      💾 Memory savings: {savings}")
            print(f"      📊 Theoretical max savings: {theoretical_max}")
    
    print("\n🔄 Testing RMSprop Optimizer")
    print("-" * 30)
    
    for world_size in world_sizes:
        print(f"   World Size: {world_size}")
        
        # Test without sharding
        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        memory_info = get_optimizer_memory_info(optimizer, world_size, enable_sharding=False)
        print(f"      No sharding: {memory_info}")
        
        # Test with sharding
        memory_info = get_optimizer_memory_info(optimizer, world_size, enable_sharding=True)
        print(f"      With sharding: {memory_info}")
        
        if memory_info['sharding_enabled']:
            savings = memory_info['memory_savings']
            theoretical_max = f"{(1 - 1/world_size) * 100:.1f}%"
            print(f"      💾 Memory savings: {savings}")
            print(f"      📊 Theoretical max savings: {theoretical_max}")
    
    print(f"✅ Realistic memory test completed in {time.time() - start_time:.2f}s")

def test_optimizer_state_partitioning():
    """Test optimizer state partitioning across devices."""
    print("🔧 Testing Optimizer State Partitioning")
    print("=" * 40)
    
    start_time = time.time()
    print(f"⏱️  {time.time() - start_time:.2f}s: Starting partitioning test...")
    
    # Create a simple model
    model = keras.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Create coordinated optimizer with sharded states
    optimizer = CoordinatedOptimizer(
        base_optimizer=keras.optimizers.Adam(learning_rate=0.001),
        world_size=4,
        shard_optimizer_states=True
    )
    
    print(f"✅ {time.time() - start_time:.2f}s: Coordinated optimizer created")
    
    # Get sharded states structure
    sharded_states = optimizer._get_sharded_states_structure()
    print(f"   Sharded states structure:")
    
    for state_name, state_info in sharded_states.items():
        if isinstance(state_info, dict):
            print(f"     {state_name}:")
            for var_name, var_info in state_info.items():
                if isinstance(var_info, dict) and 'num_shards' in var_info:
                    print(f"       {var_name}: {var_info['num_shards']} shards")
                    for i, shape in enumerate(var_info['shard_shapes']):
                        print(f"         Shard {i}: {shape}")
                else:
                    print(f"       {var_name}: {var_info}")
        else:
            print(f"     {state_name}: {state_info}")
    
    print(f"✅ Partitioning test completed in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    print("🎯 REALISTIC MEMORY SAVINGS TEST")
    print("=" * 40)
    
    # Test 1: Realistic memory savings
    test1_success = test_realistic_memory_savings()
    
    # Test 2: State partitioning
    test2_success = test_optimizer_state_partitioning()
    
    print("\n" + "=" * 40)
    print("🎉 TESTING COMPLETED!")
    print(f"\n📋 RESULTS:")
    print(f"   - Realistic Memory: {'✅' if test1_success else '❌'}")
    print(f"   - State Partitioning: {'✅' if test2_success else '❌'}")
    
    if all([test1_success, test2_success]):
        print("\n🚀 SUCCESS: All realistic memory tests passed!")
        print("\n💡 KEY BENEFITS:")
        print("   ✅ Significant memory savings with large models")
        print("   ✅ Efficient optimizer state partitioning")
        print("   ✅ Scalable to any number of devices")
        print("   ✅ Production-ready implementation")
    else:
        print("\n⚠️  WARNING: Some tests failed.") 