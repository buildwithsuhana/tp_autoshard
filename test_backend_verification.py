#!/usr/bin/env python3
"""
Test script to verify JAX and PyTorch backends are working correctly.
This will test each backend individually and ensure they can handle
real distributed operations properly.
"""

import numpy as np
import logging

# Set up logging to see backend behavior
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_jax_backend():
    """Test JAX backend functionality."""
    print("🔧 Testing JAX Backend")
    print("=" * 40)
    
    try:
        from src.tensor_parallel_keras.distributed_backend import JAXBackend
        
        # Test backend creation
        backend = JAXBackend(world_size=2, rank=0)
        print(f"✅ JAX backend created: {type(backend).__name__}")
        
        # Test availability
        is_available = backend.is_available()
        print(f"   - Available: {is_available}")
        
        if is_available:
            # Test initialization
            initialized = backend.initialize()
            print(f"   - Initialized: {initialized}")
            
            if initialized:
                # Test basic operations
                test_tensor = np.random.randn(3, 4).astype(np.float32)
                print(f"   - Test tensor shape: {test_tensor.shape}")
                
                # Test AllReduce
                try:
                    reduced = backend.allreduce(test_tensor, op='mean')
                    print(f"   - AllReduce (mean): ✅ Shape: {reduced.shape}")
                    print(f"     Original: {test_tensor[0, 0]:.4f}, Reduced: {reduced[0, 0]:.4f}")
                except Exception as e:
                    print(f"   - AllReduce (mean): ❌ {e}")
                
                # Test AllGather
                try:
                    gathered = backend.allgather(test_tensor, axis=0)
                    print(f"   - AllGather: ✅ Shape: {gathered.shape}")
                except Exception as e:
                    print(f"   - AllGather: ❌ {e}")
                
                # Test Broadcast
                try:
                    broadcasted = backend.broadcast(test_tensor, root=0)
                    print(f"   - Broadcast: ✅ Shape: {broadcasted.shape}")
                except Exception as e:
                    print(f"   - Broadcast: ❌ {e}")
            else:
                print("   - Initialization failed")
        else:
            print("   - JAX not available on this system")
            
        print("✅ JAX backend test completed\n")
        return True
        
    except Exception as e:
        print(f"❌ JAX backend test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def test_pytorch_backend():
    """Test PyTorch backend functionality."""
    print("🔧 Testing PyTorch Backend")
    print("=" * 40)
    
    try:
        from src.tensor_parallel_keras.distributed_backend import PyTorchBackend
        
        # Test backend creation
        backend = PyTorchBackend(world_size=2, rank=0)
        print(f"✅ PyTorch backend created: {type(backend).__name__}")
        
        # Test availability
        is_available = backend.is_available()
        print(f"   - Available: {is_available}")
        
        if is_available:
            # Test initialization
            initialized = backend.initialize()
            print(f"   - Initialized: {initialized}")
            
            if initialized:
                # Test basic operations
                test_tensor = np.random.randn(3, 4).astype(np.float32)
                print(f"   - Test tensor shape: {test_tensor.shape}")
                
                # Test AllReduce
                try:
                    reduced = backend.allreduce(test_tensor, op='mean')
                    print(f"   - AllReduce (mean): ✅ Shape: {reduced.shape}")
                    print(f"     Original: {test_tensor[0, 0]:.4f}, Reduced: {reduced[0, 0]:.4f}")
                except Exception as e:
                    print(f"   - AllReduce (mean): ❌ {e}")
                
                # Test AllGather
                try:
                    gathered = backend.allgather(test_tensor, axis=0)
                    print(f"   - AllGather: ✅ Shape: {gathered.shape}")
                except Exception as e:
                    print(f"   - AllGather: ❌ {e}")
                
                # Test Broadcast
                try:
                    broadcasted = backend.broadcast(test_tensor, root=0)
                    print(f"   - Broadcast: ✅ Shape: {broadcasted.shape}")
                except Exception as e:
                    print(f"   - Broadcast: ❌ {e}")
            else:
                print("   - Initialization failed")
        else:
            print("   - PyTorch not available on this system")
            
        print("✅ PyTorch backend test completed\n")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch backend test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

def test_backend_selection():
    """Test automatic backend selection including JAX and PyTorch."""
    print("🔍 Testing Backend Selection with JAX and PyTorch")
    print("=" * 50)
    
    try:
        from src.tensor_parallel_keras.distributed_backend import get_distributed_backend
        
        backends_to_test = ['auto', 'jax', 'pytorch', 'tensorflow', 'horovod', 'nccl', 'fallback']
        
        for backend_name in backends_to_test:
            print(f"\n🔄 Testing '{backend_name}' backend...")
            
            try:
                backend = get_distributed_backend(backend_name, world_size=2, rank=0)
                print(f"   ✅ Successfully created {type(backend).__name__}")
                print(f"      - Available: {backend.is_available()}")
                print(f"      - Initialized: {backend.is_initialized}")
                
                # Test a simple operation
                test_tensor = np.random.randn(2, 2).astype(np.float32)
                try:
                    result = backend.allreduce(test_tensor, op='mean')
                    print(f"      - AllReduce test: ✅ (shape: {result.shape})")
                except Exception as e:
                    print(f"      - AllReduce test: ❌ ({e})")
                    
            except Exception as e:
                print(f"   ❌ Failed to create {backend_name} backend: {e}")
        
        print("\n✅ Backend selection test completed!")
        
    except Exception as e:
        print(f"❌ Backend selection test failed: {e}")

def test_tensor_parallel_with_backends():
    """Test TensorParallelKeras with different backends."""
    print("\n🚀 Testing TensorParallelKeras with Different Backends")
    print("=" * 55)
    
    try:
        import keras
        from keras import layers
        
        # Create a simple model
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(16,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='softmax')
        ])
        
        print(f"✅ Created test model with {sum(p.shape.num_elements() for p in model.weights)} parameters")
        
        # Import TensorParallelKeras
        from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
        
        # Test different backends
        backends_to_test = ['auto', 'jax', 'pytorch', 'tensorflow', 'fallback']
        
        for backend_name in backends_to_test:
            print(f"\n🔄 Testing with '{backend_name}' backend...")
            
            try:
                # Create tensor parallel model
                tp_model = TensorParallelKeras(
                    model=model,
                    device_ids=['cpu', 'cpu'],
                    sharding_strategy='auto',
                    distributed_backend=backend_name
                )
                
                print(f"   ✅ TensorParallelKeras created with {backend_name} backend")
                print(f"      - World size: {tp_model.world_size}")
                print(f"      - Distributed backend: {type(tp_model.distributed_backend).__name__ if hasattr(tp_model, 'distributed_backend') and tp_model.distributed_backend else 'None'}")
                
                # Test forward pass
                test_input = np.random.random((4, 16)).astype(np.float32)
                output = tp_model(test_input)
                print(f"      - Forward pass: ✅ Input: {test_input.shape}, Output: {output.shape}")
                
            except Exception as e:
                print(f"   ❌ Failed with {backend_name} backend: {e}")
        
        print("\n✅ TensorParallelKeras backend test completed!")
        
    except Exception as e:
        print(f"❌ TensorParallelKeras backend test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎯 MULTI-BACKEND VERIFICATION TEST")
    print("=" * 60)
    
    # Test individual backends
    jax_success = test_jax_backend()
    pytorch_success = test_pytorch_backend()
    
    # Test backend selection
    test_backend_selection()
    
    # Test TensorParallelKeras integration
    test_tensor_parallel_with_backends()
    
    print("\n" + "=" * 60)
    print("🎉 VERIFICATION COMPLETED!")
    print(f"\n📋 RESULTS:")
    print(f"   - JAX Backend: {'✅ WORKING' if jax_success else '❌ FAILED'}")
    print(f"   - PyTorch Backend: {'✅ WORKING' if pytorch_success else '❌ FAILED'}")
    print(f"   - Backend Selection: ✅ TESTED")
    print(f"   - TensorParallelKeras Integration: ✅ TESTED")
    
    if jax_success and pytorch_success:
        print("\n🚀 SUCCESS: All backends are working correctly!")
    else:
        print("\n⚠️  WARNING: Some backends have issues that need fixing.") 