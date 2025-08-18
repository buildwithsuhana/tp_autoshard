"""
Communication operations for Keras Tensor Parallel
Implements AllReduce, AllGather, and other collective operations
with proper conjugate rule for forward/backward passes
"""

import numpy as np
from typing import List, Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import different backends
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def _get_tensor_lib(tensor):
    """Determine which tensor library a tensor belongs to."""
    if TORCH_AVAILABLE and hasattr(tensor, 'detach'):
        return 'torch'
    elif TF_AVAILABLE and hasattr(tensor, 'numpy'):
        return 'tensorflow'
    elif JAX_AVAILABLE and hasattr(tensor, 'device'):
        return 'jax'
    else:
        return 'numpy'


def _clone_tensor(tensor):
    """Clone a tensor in a backend-agnostic way."""
    tensor_lib = _get_tensor_lib(tensor)
    
    if tensor_lib == 'torch':
        return tensor.clone()
    elif tensor_lib == 'tensorflow':
        return tf.identity(tensor)
    elif tensor_lib == 'jax':
        return jnp.array(tensor)
    else:
        return np.array(tensor)


def _cat_tensors(tensors, dim=-1):
    """Concatenate tensors in a backend-agnostic way."""
    if not tensors:
        return tensors[0] if tensors else None
    
    tensor_lib = _get_tensor_lib(tensors[0])
    
    if tensor_lib == 'torch':
        return torch.cat(tensors, dim=dim)
    elif tensor_lib == 'tensorflow':
        # Convert dim to axis for TensorFlow
        # For dim=-1 (last dimension), we want axis=-1
        # For dim=1 (second dimension), we want axis=1
        axis = dim
        return tf.concat(tensors, axis=axis)
    elif tensor_lib == 'jax':
        return jnp.concatenate(tensors, axis=dim)
    else:
        return np.concatenate(tensors, axis=dim)


def _sum_tensors(tensors):
    """Sum tensors in a backend-agnostic way."""
    if not tensors:
        return tensors[0] if tensors else None
    
    tensor_lib = _get_tensor_lib(tensors[0])
    
    if tensor_lib == 'torch':
        return sum(tensors)
    elif tensor_lib == 'tensorflow':
        return tf.add_n(tensors)
    elif tensor_lib == 'jax':
        return sum(tensors)
    else:
        return sum(tensors)


class CollectiveOpKeras:
    """Base class for collective operations."""
    
    def __init__(self, world_size: int, rank: int = 0):
        self.world_size = world_size
        self.rank = rank
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class AllReduceKeras(CollectiveOpKeras):
    """AllReduce operation for gradient synchronization and row-parallel outputs."""
    
    def __init__(self, world_size: int, op: str = "sum", rank: int = 0):
        super().__init__(world_size, rank)
        self.op = op
    
    def __call__(self, tensors: List) -> List:
        """
        AllReduce operation to synchronize across shards.
        
        Args:
            tensors: List of tensors from each shard
            
        Returns:
            List of synchronized tensors for each shard
        """
        if len(tensors) != self.world_size:
            raise ValueError(f"Expected {self.world_size} tensors, got {len(tensors)}")
        
        # Implement proper AllReduce for true tensor parallelism
        if self.op == "sum":
            # Sum all tensors across devices
            total = _sum_tensors(tensors)
            # Return same result for all shards (replicated)
            return [_clone_tensor(total) for _ in range(self.world_size)]
        
        elif self.op == "mean":
            # Average across devices
            total = _sum_tensors(tensors)
            # For mean, we need to divide by world_size
            if hasattr(total, '__truediv__'):
                result = total / self.world_size
            else:
                # Fallback for numpy arrays
                result = total / self.world_size
            return [_clone_tensor(result) for _ in range(self.world_size)]
        
        else:
            raise ValueError(f"Unsupported operation: {self.op}")


class AllGatherKeras(CollectiveOpKeras):
    """AllGather operation for output collection in column-parallel layers."""
    
    def __init__(self, world_size: int, dim: int = -1, rank: int = 0):
        super().__init__(world_size, rank)
        self.dim = dim
    
    def __call__(self, tensors: List):
        """
        AllGather operation to collect outputs from all shards.
        
        Args:
            tensors: List of tensors from each shard
            
        Returns:
            Concatenated tensor along specified dimension
        """
        if len(tensors) != self.world_size:
            raise ValueError(f"Expected {self.world_size} tensors, got {len(tensors)}")
        
        # Concatenate tensors along the specified dimension
        # For Dense layers with column-wise sharding, this would be dim=1
        # For row-wise sharding, this would be dim=0
        
        try:
            # Handle different tensor shapes
            if all(t.shape == tensors[0].shape for t in tensors):
                # Same shape tensors - concatenate along specified dim
                return _cat_tensors(tensors, dim=self.dim)
            else:
                # Different shapes - need to handle carefully
                # This might happen with mixed sharding strategies
                logger.warning("Tensors have different shapes, concatenating along last dimension")
                return _cat_tensors(tensors, dim=-1)
        except Exception as e:
            logger.error(f"Error in AllGather: {e}")
            # Fallback: return first tensor
            return tensors[0]


class BroadcastKeras(CollectiveOpKeras):
    """Broadcast operation for parameter synchronization."""
    
    def __init__(self, world_size: int, src_rank: int = 0, rank: int = 0):
        super().__init__(world_size, rank)
        self.src_rank = src_rank
    
    def __call__(self, tensor):
        """
        Broadcast tensor from source rank to all ranks.
        
        Args:
            tensor: Tensor to broadcast
            
        Returns:
            List of broadcasted tensors for each shard
        """
        # For now, just clone the tensor for each shard
        # In production, you'd implement proper broadcast
        return [_clone_tensor(tensor) for _ in range(self.world_size)]


class ScatterKeras(CollectiveOpKeras):
    """Scatter operation for input distribution."""
    
    def __init__(self, world_size: int, dim: int = -1, rank: int = 0):
        super().__init__(world_size, rank)
        self.dim = dim
    
    def __call__(self, tensor):
        """
        Scatter tensor across shards.
        
        Args:
            tensor: Input tensor to scatter
            
        Returns:
            List of scattered tensors for each shard
        """
        # Split tensor along specified dimension
        try:
            # This is a simplified scatter - in practice, you'd implement proper splitting
            # For now, just clone the tensor for each shard
            return [_clone_tensor(tensor) for _ in range(self.world_size)]
        except Exception as e:
            logger.error(f"Error in Scatter: {e}")
            # Fallback: return same tensor for all shards
            return [_clone_tensor(tensor) for _ in range(self.world_size)]


class TensorParallelCommunicator:
    """
    Main communication class that implements the conjugate rule.
    
    The conjugate rule ensures that:
    - Forward pass communication is the opposite of backward pass communication
    - Column-parallel layers: Forward=AllGather, Backward=AllReduce
    - Row-parallel layers: Forward=AllReduce, Backward=AllGather
    """
    
    def __init__(self, world_size: int, rank: int = 0):
        self.world_size = world_size
        self.rank = rank
        
        # Initialize communication primitives
        self.allreduce = AllReduceKeras(world_size, rank=rank)
        self.allgather = AllGatherKeras(world_size, rank=rank)
        self.broadcast = BroadcastKeras(world_size, rank=rank)
        self.scatter = ScatterKeras(world_size, rank=rank)
    
    def forward_column_parallel(self, partial_outputs: List, dim: int = -1):
        """
        Forward pass for column-parallel layers.
        
        Args:
            partial_outputs: List of partial outputs from each shard
            dim: Dimension to concatenate along
            
        Returns:
            Concatenated output (AllGather result)
        """
        logger.debug(f"Forward column-parallel: AllGather {len(partial_outputs)} outputs along dim {dim}")
        return self.allgather(partial_outputs)
    
    def backward_column_parallel(self, partial_gradients: List, op: str = "sum") -> List:
        """
        Backward pass for column-parallel layers (conjugate of forward).
        
        Args:
            partial_gradients: List of partial gradients from each shard
            op: Reduction operation ("sum" or "mean")
            
        Returns:
            List of synchronized gradients (AllReduce result)
        """
        logger.debug(f"Backward column-parallel: AllReduce {len(partial_gradients)} gradients with op {op}")
        return self.allreduce(partial_gradients)
    
    def forward_row_parallel(self, partial_outputs: List, op: str = "sum") -> List:
        """
        Forward pass for row-parallel layers.
        
        Args:
            partial_outputs: List of partial outputs from each shard
            op: Reduction operation ("sum" or "mean")
            
        Returns:
            List of synchronized outputs (AllReduce result)
        """
        logger.debug(f"Forward row-parallel: AllReduce {len(partial_outputs)} outputs with op {op}")
        return self.allreduce(partial_outputs)
    
    def backward_row_parallel(self, partial_gradients: List, dim: int = -1):
        """
        Backward pass for row-parallel layers (conjugate of forward).
        
        Args:
            partial_gradients: List of partial gradients from each shard
            dim: Dimension to concatenate along
            
        Returns:
            Concatenated gradients (AllGather result)
        """
        logger.debug(f"Backward row-parallel: AllGather {len(partial_gradients)} gradients along dim {dim}")
        return self.allgather(partial_gradients)
    
    def handle_mlp_handshake(self, 
                            up_projection_outputs: List,
                            down_projection_inputs: List) -> Tuple:
        """
        Handle the "handshake" between MLP up and down projections.
        
        Up projection: Column-parallel (AllGather output)
        Down projection: Row-parallel (AllReduce input)
        
        This eliminates one AllReduce in the forward pass.
        """
        # Up projection: AllGather the outputs
        up_output = self.forward_column_parallel(up_projection_outputs, dim=-1)
        
        # Down projection: AllReduce the inputs (handshake)
        down_inputs = self.forward_row_parallel(down_projection_inputs, op="sum")
        
        return up_output, down_inputs
    
    def slice_upstream_gradient_for_column_parallel(self, full_gradient, rank: int, world_size: int, dim: int = -1):
        """
        Slice the upstream gradient for column-parallel layers.
        
        During forward pass: AllGather combines sharded outputs
        During backward pass: Incoming gradient must be sliced to match each shard
        
        Args:
            full_gradient: The full gradient from the next layer
            rank: Current device rank
            world_size: Total number of devices
            dim: Dimension along which to slice (usually -1 for features)
            
        Returns:
            Sliced gradient corresponding to this device's shard
        """
        try:
            # Determine the slice size for each shard
            total_size = full_gradient.shape[dim]
            slice_size = total_size // world_size
            
            # Calculate start and end indices for this rank
            start_idx = rank * slice_size
            end_idx = start_idx + slice_size if rank < world_size - 1 else total_size
            
            # Slice the gradient along the specified dimension
            if dim == -1:
                # Last dimension (features)
                if hasattr(full_gradient, 'shape') and len(full_gradient.shape) >= 2:
                    if _get_tensor_lib(full_gradient) == 'torch':
                        return full_gradient[..., start_idx:end_idx]
                    elif _get_tensor_lib(full_gradient) == 'tensorflow':
                        import tensorflow as tf
                        return tf.slice(full_gradient, [0] * (len(full_gradient.shape) - 1) + [start_idx], 
                                     [-1] * (len(full_gradient.shape) - 1) + [end_idx - start_idx])
                    else:
                        # NumPy fallback
                        slices = [slice(None)] * len(full_gradient.shape)
                        slices[dim] = slice(start_idx, end_idx)
                        return full_gradient[tuple(slices)]
            
            # For other dimensions, use generic slicing
            slices = [slice(None)] * len(full_gradient.shape)
            slices[dim] = slice(start_idx, end_idx)
            return full_gradient[tuple(slices)]
            
        except Exception as e:
            logger.warning(f"Gradient slicing failed: {e}, returning full gradient")
            return full_gradient
    
    def slice_upstream_gradient_for_row_parallel(self, full_gradient, rank: int, world_size: int, dim: int = 0):
        """
        Slice the upstream gradient for row-parallel layers.
        
        During forward pass: AllReduce combines sharded outputs
        During backward pass: Incoming gradient must be sliced to match each shard
        
        Args:
            full_gradient: The full gradient from the next layer
            rank: Current device rank
            world_size: Total number of devices
            dim: Dimension along which to slice (usually 0 for batch)
            
        Returns:
            Sliced gradient corresponding to this device's shard
        """
        try:
            # For row-parallel, we typically slice along batch dimension
            total_size = full_gradient.shape[dim]
            slice_size = total_size // world_size
            
            # Calculate start and end indices for this rank
            start_idx = rank * slice_size
            end_idx = start_idx + slice_size if rank < world_size - 1 else total_size
            
            # Slice the gradient along the specified dimension
            slices = [slice(None)] * len(full_gradient.shape)
            slices[dim] = slice(start_idx, end_idx)
            return full_gradient[tuple(slices)]
            
        except Exception as e:
            logger.warning(f"Gradient slicing failed: {e}, returning full gradient")
            return full_gradient


def allreduce_gradients(gradients: List, world_size: int) -> List:
    """
    Convenience function for AllReduce on gradients.
    
    Args:
        gradients: List of gradients from each shard
        world_size: Total number of shards
        
    Returns:
        List of synchronized gradients for each shard
    """
    allreduce_op = AllReduceKeras(world_size, op="mean")
    return allreduce_op(gradients)


def allgather_outputs(outputs: List, world_size: int, dim: int = -1):
    """
    Convenience function for AllGather on outputs.
    
    Args:
        outputs: List of outputs from each shards
        world_size: Total number of shards
        dim: Dimension to concatenate along
        
    Returns:
        Concatenated output tensor
    """
    allgather_op = AllGatherKeras(world_size, dim=dim)
    return allgather_op(outputs)


def broadcast_parameters(parameters: List, world_size: int, src_rank: int = 0) -> List:
    """
    Convenience function for broadcasting parameters.
    
    Args:
        parameters: List of parameters from each shard
        world_size: Total number of shards
        src_rank: Source rank for broadcast
        
    Returns:
        List of broadcasted parameters for each shard
    """
    broadcast_op = BroadcastKeras(world_size, src_rank)
    return broadcast_op(parameters[src_rank]) 