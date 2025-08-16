"""
Cross-device operations for Keras Tensor Parallel
"""

import torch
from typing import Sequence, Optional
from keras import backend as K


def reduce_add_keras(tensors: Sequence[torch.Tensor], destination: Optional[str] = None, all_cuda: bool = None) -> torch.Tensor:
    """
    Reduce tensors by adding them together.
    
    Args:
        tensors: Sequence of tensors to reduce
        destination: Destination device (optional)
        all_cuda: Whether all tensors are on CUDA (optional)
        
    Returns:
        Reduced tensor
    """
    if not tensors:
        return torch.tensor(0.0)
        
    # Start with the first tensor
    result = tensors[0].clone()
    
    # Add all other tensors
    for tensor in tensors[1:]:
        result = result + tensor
        
    # Move to destination device if specified
    if destination is not None:
        result = result.to(destination)
        
    return result


def all_gather_keras(tensors: Sequence[torch.Tensor], all_cuda: bool = None) -> Sequence[torch.Tensor]:
    """
    Gather tensors from all devices.
    
    Args:
        tensors: Sequence of tensors to gather
        all_cuda: Whether all tensors are on CUDA (optional)
        
    Returns:
        Sequence of gathered tensors
    """
    # For now, just return the tensors as-is
    # In a real implementation, you'd implement cross-device gathering
    return tensors


def broadcast_coalesced_keras(tensors: Sequence[torch.Tensor], destination: str) -> Sequence[torch.Tensor]:
    """
    Broadcast tensors to a destination device.
    
    Args:
        tensors: Sequence of tensors to broadcast
        destination: Destination device
        
    Returns:
        Sequence of broadcasted tensors
    """
    # Move all tensors to the destination device
    return [tensor.to(destination) for tensor in tensors]


def gather_keras(tensors: Sequence[torch.Tensor], destination: str) -> torch.Tensor:
    """
    Gather tensors to a destination device.
    
    Args:
        tensors: Sequence of tensors to gather
        destination: Destination device
        
    Returns:
        Concatenated tensor on destination device
    """
    if not tensors:
        return torch.tensor([])
        
    # Move all tensors to destination and concatenate
    tensors_on_dest = [tensor.to(destination) for tensor in tensors]
    return torch.cat(tensors_on_dest, dim=0) 