"""
Communication operations for Keras Tensor Parallel
Implements AllReduce, AllGather, and other collective operations
"""

import torch
import numpy as np
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class CollectiveOpKeras:
    """Base class for collective operations."""
    
    def __init__(self, world_size: int):
        self.world_size = world_size
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class AllReduceKeras(CollectiveOpKeras):
    """AllReduce operation for gradient synchronization."""
    
    def __init__(self, world_size: int, op: str = "sum"):
        super().__init__(world_size)
        self.op = op
    
    def __call__(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        AllReduce operation to synchronize gradients across shards.
        
        Args:
            tensors: List of tensors from each shard
            
        Returns:
            List of synchronized tensors for each shard
        """
        if len(tensors) != self.world_size:
            raise ValueError(f"Expected {self.world_size} tensors, got {len(tensors)}")
        
        # For now, implement a simple CPU-based AllReduce
        # In production, you'd use NCCL for GPU or MPI for CPU
        
        if self.op == "sum":
            # Sum all tensors
            total = sum(tensors)
            # Divide by world_size to get average
            result = total / self.world_size
            # Return same result for all shards
            return [result.clone() for _ in range(self.world_size)]
        
        elif self.op == "mean":
            # Same as sum/divide
            total = sum(tensors)
            result = total / self.world_size
            return [result.clone() for _ in range(self.world_size)]
        
        else:
            raise ValueError(f"Unsupported operation: {self.op}")


class AllGatherKeras(CollectiveOpKeras):
    """AllGather operation for output collection."""
    
    def __init__(self, world_size: int, dim: int = -1):
        super().__init__(world_size)
        self.dim = dim
    
    def __call__(self, tensors: List[torch.Tensor]) -> torch.Tensor:
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
                return torch.cat(tensors, dim=self.dim)
            else:
                # Different shapes - need to handle carefully
                # This might happen with mixed sharding strategies
                logger.warning("Tensors have different shapes, concatenating along last dimension")
                return torch.cat(tensors, dim=-1)
        except Exception as e:
            logger.error(f"Error in AllGather: {e}")
            # Fallback: return first tensor
            return tensors[0]


class BroadcastKeras(CollectiveOpKeras):
    """Broadcast operation for parameter synchronization."""
    
    def __init__(self, world_size: int, src_rank: int = 0):
        super().__init__(world_size)
        self.src_rank = src_rank
    
    def __call__(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Broadcast tensor from source rank to all ranks.
        
        Args:
            tensor: Tensor to broadcast
            
        Returns:
            List of broadcasted tensors for each shard
        """
        # For now, just clone the tensor for each shard
        # In production, you'd implement proper broadcast
        return [tensor.clone() for _ in range(self.world_size)]


class ScatterKeras(CollectiveOpKeras):
    """Scatter operation for input distribution."""
    
    def __init__(self, world_size: int, dim: int = -1):
        super().__init__(world_size)
        self.dim = dim
    
    def __call__(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Scatter tensor across shards.
        
        Args:
            tensor: Input tensor to scatter
            
        Returns:
            List of scattered tensors for each shard
        """
        # Split tensor along specified dimension
        try:
            chunks = torch.chunk(tensor, self.world_size, dim=self.dim)
            # Ensure we have exactly world_size chunks
            while len(chunks) < self.world_size:
                chunks.append(chunks[-1].clone())
            return chunks[:self.world_size]
        except Exception as e:
            logger.error(f"Error in Scatter: {e}")
            # Fallback: return same tensor for all shards
            return [tensor.clone() for _ in range(self.world_size)]


def allreduce_gradients(gradients: List[torch.Tensor], world_size: int) -> List[torch.Tensor]:
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


def allgather_outputs(outputs: List[torch.Tensor], world_size: int, dim: int = -1) -> torch.Tensor:
    """
    Convenience function for AllGather on outputs.
    
    Args:
        outputs: List of outputs from each shard
        world_size: Total number of shards
        dim: Dimension to concatenate along
        
    Returns:
        Concatenated output tensor
    """
    allgather_op = AllGatherKeras(world_size, dim=dim)
    return allgather_op(outputs)


def broadcast_parameters(parameters: List[torch.Tensor], world_size: int, src_rank: int = 0) -> List[torch.Tensor]:
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