"""
State actions for Keras Tensor Parallel
Handles parameter splitting, gathering, and reduction operations
"""

import torch
from typing import Sequence, Union
from keras import backend as K


class StateActionKeras:
    """Base class for state actions."""
    
    def __call__(self, tensor: torch.Tensor, rank: int) -> torch.Tensor:
        """Apply the action to a tensor."""
        raise NotImplementedError
        
    def undo(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Reverse the action to reconstruct the original tensor."""
        raise NotImplementedError


class SplitKeras(StateActionKeras):
    """Split a tensor along a specified dimension."""
    
    def __init__(self, world_size: int, dim: int, sharding_type: str = "auto"):
        self.world_size = world_size
        self.dim = dim
        self.sharding_type = sharding_type  # "auto", "row", "column"
        
        # Auto-determine dimension based on sharding type if dim is -1
        if dim == -1 and sharding_type != "auto":
            if sharding_type == "row":
                self.dim = 0
            elif sharding_type == "column":
                self.dim = 1
        
    def __call__(self, tensor: torch.Tensor, rank: int) -> torch.Tensor:
        """Split tensor and return the portion for this rank."""
        if self.dim == -1:
            dim = tensor.dim() - 1
        else:
            dim = self.dim
            

            
        # Calculate split sizes
        total_size = tensor.shape[dim]
        split_size = total_size // self.world_size
        remainder = total_size % self.world_size
        
        # Calculate start and end indices for this rank
        start_idx = rank * split_size + min(rank, remainder)
        end_idx = start_idx + split_size + (1 if rank < remainder else 0)
        

        
        # Split the tensor
        if dim == 0:
            return tensor[start_idx:end_idx]
        elif dim == 1:
            return tensor[:, start_idx:end_idx]
        elif dim == 2:
            return tensor[:, :, start_idx:end_idx]
        elif dim == 3:
            return tensor[:, :, :, start_idx:end_idx]
        else:
            # For higher dimensions, use advanced indexing
            slices = [slice(None)] * tensor.dim()
            slices[dim] = slice(start_idx, end_idx)
            return tensor[tuple(slices)]
            
    def undo(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Concatenate split tensors back together."""
        if self.dim == -1:
            dim = tensors[0].dim() - 1
        else:
            dim = self.dim
            
        return torch.cat(tensors, dim=dim)


class GatherKeras(StateActionKeras):
    """Gather tensors from all ranks along a specified dimension."""
    
    def __init__(self, world_size: int, dim: int):
        self.world_size = world_size
        self.dim = dim
        
    def __call__(self, tensor: torch.Tensor, rank: int) -> torch.Tensor:
        """Return the tensor as-is (gathering happens in communication layer)."""
        return tensor
        
    def undo(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Concatenate gathered tensors."""
        if self.dim == -1:
            dim = tensors[0].dim() - 1
        else:
            dim = self.dim
            
        return torch.cat(tensors, dim=dim)


class SumKeras(StateActionKeras):
    """Sum tensors from all ranks."""
    
    def __init__(self, world_size: int):
        self.world_size = world_size
        
    def __call__(self, tensor: torch.Tensor, rank: int) -> torch.Tensor:
        """Return the tensor as-is (summing happens in communication layer)."""
        return tensor
        
    def undo(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Sum the tensors."""
        return sum(tensors) 