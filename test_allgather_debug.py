#!/usr/bin/env python3
"""
Debug AllGather operation to understand the shape issue
"""

import numpy as np
import keras
from src.tensor_parallel_keras.communications_keras import AllGatherKeras

def test_allgather_debug():
    """Test AllGather operation with simple tensors."""
    print("🧪 Debugging AllGather Operation")
    print("=" * 40)
    
    tensor1 = keras.ops.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    tensor2 = keras.ops.array([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
    
    print(f"Tensor 1 shape: {tensor1.shape}, values:\n{keras.ops.convert_to_numpy(tensor1)}")
    print(f"Tensor 2 shape: {tensor2.shape}, values:\n{keras.ops.convert_to_numpy(tensor2)}")
    
    allgather = AllGatherKeras(world_size=2, dim=-1)
    
    try:
        result = allgather([tensor1, tensor2])
        print(f"AllGather result shape: {result.shape}")
        print(f"AllGather result values:\n{keras.ops.convert_to_numpy(result)}")
        
        expected_shape = (2, 4)
        if result.shape == expected_shape:
            print("✅ AllGather shape is correct!")
        else:
            print(f"❌ AllGather shape is wrong! Expected {expected_shape}, got {result.shape}")
            
    except Exception as e:
        print(f"❌ AllGather failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_allgather_debug()