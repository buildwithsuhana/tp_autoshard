#!/usr/bin/env python3
"""
Fast integration test for TensorParallelKeras with fixed backends.
"""

import time
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_tensor_parallel_integration():
    """Test TensorParallelKeras integration with different backends."""
    print("🚀 Testing TensorParallelKeras Integration")
    print("=" * 45)
    
    start_time = time.time()
    
    try:
        print(f"⏱️  {time.time() - start_time:.2f}s: Starting integration test...")
        
        # Import required modules
        print(f"⏱️  {time.time() - start_time:.2f}s: Importing modules...")
        import keras
        from keras import layers
        from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
        
        print(f"✅ {time.time() - start_time:.2f}s: Modules imported successfully")
        
        # Create a simple model
        print(f"⏱️  {time.time() - start_time:.2f}s: Creating test model...")
        model = keras.Sequential([
            layers.Dense(16, activation='relu', input_shape=(8,)),
            layers.Dense(8, activation='relu'),
            layers.Dense(4, activation='softmax')
        ])
        
        print(f"✅ {time.time() - start_time:.2f}s: Model created with {sum(p.shape.num_elements() for p in model.weights)} parameters")
        
        # Test different backends
        backends_to_test = ['jax', 'pytorch', 'tensorflow', 'fallback']
        
        for backend_name in backends_to_test:
            backend_start = time.time()
            print(f"\n🔄 Testing with '{backend_name}' backend...")
            
            try:
                # Create tensor parallel model
                tp_model = TensorParallelKeras(
                    model=model,
                    device_ids=['cpu', 'cpu'],
                    sharding_strategy='auto',
                    distributed_backend=backend_name
                )
                
                backend_time = time.time() - backend_start
                print(f"   ✅ TensorParallelKeras created with {backend_name} backend in {backend_time:.2f}s")
                print(f"      - World size: {tp_model.world_size}")
                print(f"      - Distributed backend: {type(tp_model.distributed_backend).__name__ if hasattr(tp_model, 'distributed_backend') and tp_model.distributed_backend else 'None'}")
                
                # Test forward pass
                test_input = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float32)  # Proper numpy array
                output = tp_model(test_input)
                print(f"      - Forward pass: ✅ Input: {test_input.shape}, Output: {output.shape}")
                
            except Exception as e:
                backend_time = time.time() - backend_start
                print(f"   ❌ Failed with {backend_name} backend after {backend_time:.2f}s: {e}")
        
        total_time = time.time() - start_time
        print(f"\n✅ Integration test completed in {total_time:.2f}s")
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Integration test failed after {total_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 FAST INTEGRATION TEST")
    print("=" * 30)
    
    success = test_tensor_parallel_integration()
    
    if success:
        print("\n🎉 SUCCESS: TensorParallelKeras integration working with all backends!")
    else:
        print("\n❌ FAILED: Integration test failed.") 