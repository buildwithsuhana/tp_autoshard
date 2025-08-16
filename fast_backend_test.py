#!/usr/bin/env python3
"""
Fast backend test with detailed logging to identify performance issues.
"""

import time
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def test_backend_import():
    """Test just importing the backends."""
    print("🔧 Testing Backend Import")
    print("=" * 30)
    
    start_time = time.time()
    
    try:
        print(f"⏱️  {time.time() - start_time:.2f}s: Starting import...")
        from src.tensor_parallel_keras.distributed_backend import get_distributed_backend
        print(f"✅ {time.time() - start_time:.2f}s: Import successful")
        return True
    except Exception as e:
        print(f"❌ {time.time() - start_time:.2f}s: Import failed: {e}")
        return False

def test_jax_backend_fast():
    """Test JAX backend quickly."""
    print("\n🔧 Testing JAX Backend (Fast)")
    print("=" * 35)
    
    start_time = time.time()
    
    try:
        print(f"⏱️  {time.time() - start_time:.2f}s: Starting JAX test...")
        
        from src.tensor_parallel_keras.distributed_backend import JAXBackend
        
        print(f"⏱️  {time.time() - start_time:.2f}s: Creating JAX backend...")
        backend = JAXBackend(world_size=2, rank=0)
        
        print(f"⏱️  {time.time() - start_time:.2f}s: Checking availability...")
        is_available = backend.is_available()
        print(f"✅ {time.time() - start_time:.2f}s: Available: {is_available}")
        
        if is_available:
            print(f"⏱️  {time.time() - start_time:.2f}s: Initializing...")
            initialized = backend.initialize()
            print(f"✅ {time.time() - start_time:.2f}s: Initialized: {initialized}")
            
            if initialized:
                print(f"⏱️  {time.time() - start_time:.2f}s: Testing operation...")
                test_tensor = [1.0, 2.0, 3.0]  # Simple list to avoid numpy import
                result = backend.allreduce(test_tensor, op='mean')
                print(f"✅ {time.time() - start_time:.2f}s: Operation successful, result length: {len(result)}")
        
        total_time = time.time() - start_time
        print(f"✅ Total JAX test time: {total_time:.2f}s")
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ JAX test failed after {total_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pytorch_backend_fast():
    """Test PyTorch backend quickly."""
    print("\n🔧 Testing PyTorch Backend (Fast)")
    print("=" * 38)
    
    start_time = time.time()
    
    try:
        print(f"⏱️  {time.time() - start_time:.2f}s: Starting PyTorch test...")
        
        from src.tensor_parallel_keras.distributed_backend import PyTorchBackend
        
        print(f"⏱️  {time.time() - start_time:.2f}s: Creating PyTorch backend...")
        backend = PyTorchBackend(world_size=2, rank=0)
        
        print(f"⏱️  {time.time() - start_time:.2f}s: Checking availability...")
        is_available = backend.is_available()
        print(f"✅ {time.time() - start_time:.2f}s: Available: {is_available}")
        
        if is_available:
            print(f"⏱️  {time.time() - start_time:.2f}s: Initializing...")
            initialized = backend.initialize()
            print(f"✅ {time.time() - start_time:.2f}s: Initialized: {initialized}")
            
            if initialized:
                print(f"⏱️  {time.time() - start_time:.2f}s: Testing operation...")
                test_tensor = [1.0, 2.0, 3.0]  # Simple list to avoid numpy import
                result = backend.allreduce(test_tensor, op='mean')
                print(f"✅ {time.time() - start_time:.2f}s: Operation successful, result length: {len(result)}")
        
        total_time = time.time() - start_time
        print(f"✅ Total PyTorch test time: {total_time:.2f}s")
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ PyTorch test failed after {total_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backend_selection_fast():
    """Test backend selection quickly."""
    print("\n🔍 Testing Backend Selection (Fast)")
    print("=" * 37)
    
    start_time = time.time()
    
    try:
        print(f"⏱️  {time.time() - start_time:.2f}s: Starting backend selection test...")
        
        from src.tensor_parallel_keras.distributed_backend import get_distributed_backend
        
        # Test just one backend to avoid delays
        print(f"⏱️  {time.time() - start_time:.2f}s: Testing 'jax' backend...")
        backend = get_distributed_backend('jax', world_size=2, rank=0)
        
        total_time = time.time() - start_time
        print(f"✅ Backend selection completed in {total_time:.2f}s")
        print(f"   Selected: {type(backend).__name__}")
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Backend selection failed after {total_time:.2f}s: {e}")
        return False

if __name__ == "__main__":
    print("🚀 FAST BACKEND TEST WITH DETAILED LOGGING")
    print("=" * 50)
    
    # Test 1: Import
    import_success = test_backend_import()
    
    if import_success:
        # Test 2: JAX Backend
        jax_success = test_jax_backend_fast()
        
        # Test 3: PyTorch Backend
        pytorch_success = test_pytorch_backend_fast()
        
        # Test 4: Backend Selection
        selection_success = test_backend_selection_fast()
        
        print(f"\n📋 FINAL RESULTS:")
        print(f"   - Import: {'✅' if import_success else '❌'}")
        print(f"   - JAX: {'✅' if jax_success else '❌'}")
        print(f"   - PyTorch: {'✅' if pytorch_success else '❌'}")
        print(f"   - Selection: {'✅' if selection_success else '❌'}")
        
        if all([import_success, jax_success, pytorch_success, selection_success]):
            print("\n🎉 SUCCESS: All backends working correctly!")
        else:
            print("\n⚠️  WARNING: Some backends have issues.")
    else:
        print("\n❌ CRITICAL: Import failed, cannot test backends.") 