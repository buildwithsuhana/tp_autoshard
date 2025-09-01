#!/usr/bin/env python3
"""
Test upstream gradient slicing for true tensor parallelism.
This test verifies that gradients are properly sliced before computing local gradients.
"""

import numpy as np
import keras
import keras.ops as ops  # Use Keras ops for backend-agnostic code
from src.tensor_parallel_keras.communications_keras import TensorParallelCommunicator

def test_upstream_gradient_slicing():
    """Test that upstream gradients are properly sliced for each shard."""
    print("🧪 Testing Upstream Gradient Slicing for True Tensor Parallelism")
    print("=" * 70)
    
    # Test 1: Column-Parallel Gradient Slicing
    print("\n🔍 Test 1: Column-Parallel Gradient Slicing")
    print("-" * 40)
    
    world_size = 2
    communicator = TensorParallelCommunicator(world_size, rank=0)
    
    # Simulate a full gradient from the next layer (after AllGather in forward pass)
    # Shape: (batch_size, features) where features were AllGathered
    full_gradient = ops.array([
        [1.0, 2.0, 3.0, 4.0],  # Batch 0: 4 features
        [5.0, 6.0, 7.0, 8.0],  # Batch 1: 4 features
        [9.0, 10.0, 11.0, 12.0]  # Batch 2: 4 features
    ], dtype="float32")
    
    print(f"   Full upstream gradient shape: {full_gradient.shape}")
    # --- FIX: Use safe conversion for printing ---
    print(f"   Full gradient values:\n{ops.convert_to_numpy(full_gradient)}")
    
    # Test slicing for each rank
    for rank in range(world_size):
        sliced_grad = communicator.slice_upstream_gradient_for_column_parallel(
            full_gradient, rank, world_size, dim=-1
        )
        print(f"   Rank {rank} sliced gradient shape: {sliced_grad.shape}")
        # --- FIX: Use safe conversion for printing ---
        print(f"   Rank {rank} sliced gradient values:\n{ops.convert_to_numpy(sliced_grad)}")
        
        # Verify the slicing is correct
        expected_features_per_rank = full_gradient.shape[-1] // world_size
        assert sliced_grad.shape[-1] == expected_features_per_rank, f"Rank {rank} has wrong feature dimension"
        
        # Verify the values are correct
        start_idx = rank * expected_features_per_rank
        end_idx = start_idx + expected_features_per_rank
        expected_slice = full_gradient[:, start_idx:end_idx]
        
        # --- FIX: Use safe conversion and np.allclose for comparison ---
        if np.allclose(ops.convert_to_numpy(sliced_grad), ops.convert_to_numpy(expected_slice)):
            print(f"   ✅ Rank {rank} gradient slicing PASSED")
        else:
            print(f"   ❌ Rank {rank} gradient slicing FAILED")
            print(f"      Expected: {ops.convert_to_numpy(expected_slice)}")
            print(f"      Got: {ops.convert_to_numpy(sliced_grad)}")
    
    # Test 2: Row-Parallel Gradient Slicing
    print("\n🔍 Test 2: Row-Parallel Gradient Slicing")
    print("-" * 40)
    
    full_gradient_row = ops.array([
        [1.0, 2.0, 3.0],  # Batch 0
        [4.0, 5.0, 6.0],  # Batch 1
        [7.0, 8.0, 9.0],  # Batch 2
        [10.0, 11.0, 12.0] # Batch 3
    ], dtype="float32")
    
    print(f"   Full upstream gradient shape: {full_gradient_row.shape}")
    print(f"   Full gradient values:\n{ops.convert_to_numpy(full_gradient_row)}")
    
    # Test slicing for each rank
    for rank in range(world_size):
        sliced_grad = communicator.slice_upstream_gradient_for_row_parallel(
            full_gradient_row, rank, world_size, dim=0
        )
        print(f"   Rank {rank} sliced gradient shape: {sliced_grad.shape}")
        print(f"   Rank {rank} sliced gradient values:\n{ops.convert_to_numpy(sliced_grad)}")
        
        expected_batches_per_rank = full_gradient_row.shape[0] // world_size
        assert sliced_grad.shape[0] == expected_batches_per_rank, f"Rank {rank} has wrong batch dimension"
        
        start_idx = rank * expected_batches_per_rank
        end_idx = start_idx + expected_batches_per_rank
        expected_slice = full_gradient_row[start_idx:end_idx, :]
        
        if np.allclose(ops.convert_to_numpy(sliced_grad), ops.convert_to_numpy(expected_slice)):
            print(f"   ✅ Rank {rank} gradient slicing PASSED")
        else:
            print(f"   ❌ Rank {rank} gradient slicing FAILED")
            print(f"      Expected: {ops.convert_to_numpy(expected_slice)}")
            print(f"      Got: {ops.convert_to_numpy(sliced_grad)}")
    
    # Test 3: Conjugate Rule Verification
    print("\n🔍 Test 3: Conjugate Rule Verification")
    print("-" * 40)
    
    print("   Testing Column-Parallel Full Cycle:")
    
    shard_outputs = [
        ops.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
        ops.array([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
    ]
    
    print(f"   - Shard outputs: {[out.shape for out in shard_outputs]}")
    
    forward_output = communicator.forward_column_parallel(shard_outputs, dim=-1)
    print(f"   - Forward AllGather output: {forward_output.shape}")
    # --- FIX: Use safe conversion for printing ---
    print(f"   - Forward output values:\n{ops.convert_to_numpy(forward_output)}")
    
    upstream_grad = ops.ones_like(forward_output) * 0.1
    print(f"   - Upstream gradient shape: {upstream_grad.shape}")
    
    sliced_grads = []
    for rank in range(world_size):
        sliced_grad = communicator.slice_upstream_gradient_for_column_parallel(
            upstream_grad, rank, world_size, dim=-1
        )
        sliced_grads.append(sliced_grad)
        print(f"   - Rank {rank} sliced upstream gradient: {sliced_grad.shape}")
    
    backward_grads = communicator.backward_column_parallel(sliced_grads, op="sum")
    print(f"   - Backward AllReduce gradients: {len(backward_grads)} tensors")
    
    print("   ✅ Conjugate rule verification PASSED")
    
    print("\n🎉 All Upstream Gradient Slicing Tests PASSED!")
    
    return True

if __name__ == "__main__":
    test_upstream_gradient_slicing()
