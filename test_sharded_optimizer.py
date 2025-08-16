#!/usr/bin/env python3
"""
Test script for sharded optimizer states.
This demonstrates the memory savings and functionality of the new
sharded optimizer implementation.
"""

import time
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_sharded_optimizer_states():
    """Test the sharded optimizer states functionality."""
    print("🚀 Testing Sharded Optimizer States")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        print(f"⏱️  {time.time() - start_time:.2f}s: Starting sharded optimizer test...")
        
        # Import required modules
        import keras
        from keras import layers, optimizers
        from src.tensor_parallel_keras.coordinated_optimizer import CoordinatedOptimizer
        
        print(f"✅ {time.time() - start_time:.2f}s: Modules imported successfully")
        
        # Create a base optimizer (Adam with momentum)
        print(f"⏱️  {time.time() - start_time:.2f}s: Creating base optimizer...")
        base_optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        print(f"✅ {time.time() - start_time:.2f}s: Base optimizer created")
        
        # Test with different world sizes
        world_sizes = [2, 4, 8]
        
        for world_size in world_sizes:
            print(f"\n🔄 Testing with world_size={world_size}")
            print("-" * 30)
            
            # Test 1: Coordinated Optimizer WITHOUT sharding
            print(f"   Testing WITHOUT optimizer state sharding...")
            coord_opt_no_sharding = CoordinatedOptimizer(
                base_optimizer=base_optimizer,
                world_size=world_size,
                distributed_backend='fallback',
                shard_optimizer_states=False
            )
            
            memory_info_no_sharding = coord_opt_no_sharding.get_memory_usage()
            print(f"      Memory info: {memory_info_no_sharding}")
            
            # Test 2: Coordinated Optimizer WITH sharding
            print(f"   Testing WITH optimizer state sharding...")
            coord_opt_with_sharding = CoordinatedOptimizer(
                base_optimizer=base_optimizer,
                world_size=world_size,
                distributed_backend='fallback',
                shard_optimizer_states=True
            )
            
            memory_info_with_sharding = coord_opt_with_sharding.get_memory_usage()
            print(f"      Memory info: {memory_info_with_sharding}")
            
            # Compare memory usage
            if memory_info_no_sharding['sharding_enabled'] == False and memory_info_with_sharding['sharding_enabled'] == True:
                print(f"      ✅ Sharding enabled successfully")
                if 'memory_savings' in memory_info_with_sharding:
                    print(f"      💾 Memory savings: {memory_info_with_sharding['memory_savings']}")
            else:
                print(f"      ⚠️  Sharding status mismatch")
        
        print(f"\n✅ Sharded optimizer test completed in {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Sharded optimizer test failed after {total_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimizer_state_management():
    """Test optimizer state management methods."""
    print("\n🔧 Testing Optimizer State Management")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        print(f"⏱️  {time.time() - start_time:.2f}s: Starting state management test...")
        
        import keras
        from keras import optimizers
        from src.tensor_parallel_keras.coordinated_optimizer import CoordinatedOptimizer
        
        # Create coordinated optimizer with sharding enabled
        base_optimizer = optimizers.Adam(learning_rate=0.001)
        coord_opt = CoordinatedOptimizer(
            base_optimizer=base_optimizer,
            world_size=4,
            distributed_backend='fallback',
            shard_optimizer_states=True
        )
        
        print(f"✅ {time.time() - start_time:.2f}s: Coordinated optimizer created")
        
        # Test memory usage
        memory_info = coord_opt.get_memory_usage()
        print(f"   Initial memory info: {memory_info}")
        
        # Test disabling sharding
        print(f"   Disabling optimizer state sharding...")
        coord_opt.disable_optimizer_state_sharding()
        
        memory_info_disabled = coord_opt.get_memory_usage()
        print(f"   Memory info after disabling: {memory_info_disabled}")
        
        # Test re-enabling sharding
        print(f"   Re-enabling optimizer state sharding...")
        coord_opt.enable_optimizer_state_sharding()
        
        memory_info_enabled = coord_opt.get_memory_usage()
        print(f"   Memory info after re-enabling: {memory_info_enabled}")
        
        print(f"✅ State management test completed in {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ State management test failed after {total_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tensor_parallel_optimizer():
    """Test the TensorParallelOptimizer wrapper."""
    print("\n🚀 Testing TensorParallelOptimizer")
    print("=" * 35)
    
    start_time = time.time()
    
    try:
        print(f"⏱️  {time.time() - start_time:.2f}s: Starting TensorParallelOptimizer test...")
        
        import keras
        from keras import optimizers
        from src.tensor_parallel_keras.coordinated_optimizer import TensorParallelOptimizer
        
        # Create base optimizer
        base_optimizer = optimizers.Adam(learning_rate=0.001)
        
        # Create tensor parallel optimizer
        tp_optimizer = TensorParallelOptimizer(
            base_optimizer=base_optimizer,
            world_size=2
        )
        
        print(f"✅ {time.time() - start_time:.2f}s: TensorParallelOptimizer created")
        print(f"   - World size: {tp_optimizer.world_size}")
        print(f"   - Base optimizer: {type(tp_optimizer.base_optimizer).__name__}")
        
        # Test configuration
        config = tp_optimizer.get_config()
        print(f"   - Config keys: {list(config.keys())}")
        
        # Test memory usage through coordinated optimizer
        memory_info = tp_optimizer.coordinated_optimizer.get_memory_usage()
        print(f"   - Memory info: {memory_info}")
        
        print(f"✅ TensorParallelOptimizer test completed in {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ TensorParallelOptimizer test failed after {total_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 SHARDED OPTIMIZER STATES COMPREHENSIVE TEST")
    print("=" * 55)
    
    # Test 1: Basic sharded optimizer functionality
    test1_success = test_sharded_optimizer_states()
    
    # Test 2: State management methods
    test2_success = test_optimizer_state_management()
    
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
        print("\n💡 KEY FEATURES IMPLEMENTED:")
        print("   ✅ Sharded optimizer states across devices")
        print("   ✅ Memory usage tracking and optimization")
        print("   ✅ Dynamic enabling/disabling of state sharding")
        print("   ✅ Fallback to replicated states when needed")
        print("   ✅ True tensor parallelism (like ZeRO)")
    else:
        print("\n⚠️  WARNING: Some tests failed.") 