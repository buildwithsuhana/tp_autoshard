#!/usr/bin/env python3
"""
Quick test to identify backend issues without the full verification suite.
"""

import time
import numpy as np

def quick_jax_test():
    """Quick JAX backend test."""
    print("🔧 Quick JAX Test")
    start_time = time.time()
    
    try:
        from src.tensor_parallel_keras.distributed_backend import JAXBackend
        print(f"✅ Import time: {time.time() - start_time:.2f}s")
        
        backend = JAXBackend(world_size=2, rank=0)
        print(f"✅ Creation time: {time.time() - start_time:.2f}s")
        
        is_available = backend.is_available()
        print(f"✅ Availability check: {time.time() - start_time:.2f}s - Available: {is_available}")
        
        if is_available:
            initialized = backend.initialize()
            print(f"✅ Initialization time: {time.time() - start_time:.2f}s - Success: {initialized}")
            
            if initialized:
                test_tensor = np.random.randn(2, 2).astype(np.float32)
                result = backend.allreduce(test_tensor, op='mean')
                print(f"✅ AllReduce time: {time.time() - start_time:.2f}s - Shape: {result.shape}")
        
        print(f"✅ Total JAX test time: {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ JAX test failed at {time.time() - start_time:.2f}s: {e}")
        return False

def quick_pytorch_test():
    """Quick PyTorch backend test."""
    print("🔧 Quick PyTorch Test")
    start_time = time.time()
    
    try:
        from src.tensor_parallel_keras.distributed_backend import PyTorchBackend
        print(f"✅ Import time: {time.time() - start_time:.2f}s")
        
        backend = PyTorchBackend(world_size=2, rank=0)
        print(f"✅ Creation time: {time.time() - start_time:.2f}s")
        
        is_available = backend.is_available()
        print(f"✅ Availability check: {time.time() - start_time:.2f}s - Available: {is_available}")
        
        if is_available:
            initialized = backend.initialize()
            print(f"✅ Initialization time: {time.time() - start_time:.2f}s - Success: {initialized}")
            
            if initialized:
                test_tensor = np.random.randn(2, 2).astype(np.float32)
                result = backend.allreduce(test_tensor, op='mean')
                print(f"✅ AllReduce time: {time.time() - start_time:.2f}s - Shape: {result.shape}")
        
        print(f"✅ Total PyTorch test time: {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed at {time.time() - start_time:.2f}s: {e}")
        return False

if __name__ == "__main__":
    print("🚀 QUICK BACKEND TEST")
    print("=" * 30)
    
    jax_success = quick_jax_test()
    print()
    pytorch_success = quick_pytorch_test()
    
    print(f"\n📋 RESULTS:")
    print(f"   - JAX: {'✅' if jax_success else '❌'}")
    print(f"   - PyTorch: {'✅' if pytorch_success else '❌'}") 