#!/usr/bin/env python3
"""
Debug AllGather operation to understand the shape issue.
This test is updated to use the refactored AllGatherKeras class
which requires a distributed backend instance.
"""

import numpy as np
import keras

# 1. Import the necessary backend and communication classes
from src.tensor_parallel_keras.distributed_backend import DistributedBackend
from src.tensor_parallel_keras.communications_keras import AllGatherKeras

def test_allgather_debug():
    """Test AllGather operation with the required backend."""
    print("🧪 Debugging Refactored AllGather Operation")
    print("=" * 50)
    
    # --- Setup ---
    world_size = 2
    # In a real run, each process has its own rank. We'll simulate from rank 0's perspective.
    rank_to_simulate = 0 
    
    # 2. Instantiate a backend. For a simple unit test, the "numpy" backend
    #    is a perfect simulator that doesn't require real hardware.
    active_backend_name = keras.backend.backend()
    print(f"🔧 Initializing backend for the test based on active Keras backend: '{active_backend_name}'...")
    backend = DistributedBackend(active_backend_name)
    
    # --- Tensors ---
    # This list represents the tensors that exist across all devices in the distributed system.
    all_tensors = [
        keras.ops.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"), # Tensor on rank 0
        keras.ops.array([[5.0, 6.0], [7.0, 8.0]], dtype="float32")  # Tensor on rank 1
    ]
    
    print(f"Tensor on Rank 0 shape: {all_tensors[0].shape}, values:\n{keras.ops.convert_to_numpy(all_tensors[0])}")
    print(f"Tensor on Rank 1 shape: {all_tensors[1].shape}, values:\n{keras.ops.convert_to_numpy(all_tensors[1])}")
    print("-" * 50)
    
    # 3. Update the AllGatherKeras constructor to pass the required `backend` and `rank`.
    print(f"🚀 Instantiating AllGatherKeras for world_size={world_size}, simulating from rank={rank_to_simulate}...")
    allgather = AllGatherKeras(
        world_size=world_size,
        backend=backend,
        dim=-1,
        rank=rank_to_simulate
    )
    
    try:
        # The input is the list of all tensors. The op will correctly pick
        # its local tensor (`all_tensors[rank_to_simulate]`) to contribute.
        result = allgather(all_tensors)
        
        print("\n--- Results ---")
        print(f"AllGather result shape: {result.shape}")
        print(f"AllGather result values:\n{keras.ops.convert_to_numpy(result)}")
        
        # 4. Add checks for both shape and values for a more robust test.
        expected_shape = (2, 4)
        expected_result = np.array([[1., 2., 5., 6.], [3., 4., 7., 8.]])

        shape_correct = result.shape == expected_shape
        values_correct = np.allclose(keras.ops.convert_to_numpy(result), expected_result)

        if shape_correct and values_correct:
            print("\n✅ PASSED: AllGather shape and values are correct!")
        else:
            if not shape_correct:
                print(f"\n❌ FAILED: Shape is wrong! Expected {expected_shape}, got {result.shape}")
            if not values_correct:
                print(f"\n❌ FAILED: Values are wrong! Expected:\n{expected_result}\nGot:\n{keras.ops.convert_to_numpy(result)}")

    except Exception as e:
        print(f"\n❌ FAILED: AllGather raised an exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_allgather_debug()