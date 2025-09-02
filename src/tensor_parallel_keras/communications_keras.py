import numpy as np
from typing import List, Union, Optional, Tuple
import logging
import keras

logger = logging.getLogger(__name__)

from .distributed_backend import DistributedBackend

def _get_tensor_lib(tensor):
    if hasattr(tensor, 'detach'):
        return 'pytorch'
    elif hasattr(tensor, 'numpy'):
        return 'tensorflow'
    elif hasattr(tensor, 'device'):
        return 'jax'
    else:
        return 'numpy'


def _clone_tensor(tensor):
    current_backend_name = keras.backend.backend()
    backend = DistributedBackend(current_backend_name)

    return backend.convert_to_backend_tensor(tensor)


def _cat_tensors(tensors, dim=-1):
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]

    try:
        numpy_tensors = [keras.ops.convert_to_numpy(t) for t in tensors]
        return keras.ops.concatenate(numpy_tensors, axis=dim)
        
    except Exception as e:
        logger.error(f"Error during tensor conversion or concatenation in _cat_tensors: {e}")
        return tensors[0]


def _sum_tensors(tensors):
    if not tensors:
        return None

    total = tensors[0]
    for tensor in tensors[1:]:
        total = keras.ops.add(total, tensor)
        
    return total


class CollectiveOpKeras:
    def __init__(self, world_size: int, rank: int = 0):
        self.world_size = world_size
        self.rank = rank
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class AllReduceKeras(CollectiveOpKeras):
    
    def __init__(self, world_size: int, op: str = "sum", rank: int = 0):
        super().__init__(world_size, rank)
        self.op = op
    
    def __call__(self, tensors: List) -> List:
        if len(tensors) != self.world_size:
            raise ValueError(f"Expected {self.world_size} tensors, got {len(tensors)}")
        
        if self.op == "sum":
            total = _sum_tensors(tensors)
            return [_clone_tensor(total) for _ in range(self.world_size)]
        
        elif self.op == "mean":
            total = _sum_tensors(tensors)
            if hasattr(total, '__truediv__'):
                result = total / self.world_size
            else:
                result = total / self.world_size
            return [_clone_tensor(result) for _ in range(self.world_size)]
        
        else:
            raise ValueError(f"Unsupported operation: {self.op}")


class AllGatherKeras(CollectiveOpKeras):
    
    def __init__(self, world_size: int, dim: int = -1, rank: int = 0):
        super().__init__(world_size, rank)
        self.dim = dim
    
    def __call__(self, tensors: List):
        if len(tensors) != self.world_size:
            raise ValueError(f"Expected {self.world_size} tensors, got {len(tensors)}")
        
        try:
            if all(t.shape == tensors[0].shape for t in tensors):
                return _cat_tensors(tensors, dim=self.dim)
            else:
                logger.warning("Tensors have different shapes, concatenating along last dimension")
                return _cat_tensors(tensors, dim=-1)
        except Exception as e:
            logger.error(f"Error in AllGather: {e}")
            return tensors[0]


class BroadcastKeras(CollectiveOpKeras):
    
    def __init__(self, world_size: int, src_rank: int = 0, rank: int = 0):
        super().__init__(world_size, rank)
        self.src_rank = src_rank
    
    def __call__(self, tensor):
        return [_clone_tensor(tensor) for _ in range(self.world_size)]


class ScatterKeras(CollectiveOpKeras):
    
    def __init__(self, world_size: int, dim: int = -1, rank: int = 0):
        super().__init__(world_size, rank)
        self.dim = dim
    
    def __call__(self, tensor):
        try:
            return [_clone_tensor(tensor) for _ in range(self.world_size)]
        except Exception as e:
            logger.error(f"Error in Scatter: {e}")
            return [_clone_tensor(tensor) for _ in range(self.world_size)]


class TensorParallelCommunicator:
    
    def __init__(self, world_size: int, rank: int = 0):
        self.world_size = world_size
        self.rank = rank
        
        self.allreduce = AllReduceKeras(world_size, rank=rank)
        self.allgather = AllGatherKeras(world_size, rank=rank)
        self.broadcast = BroadcastKeras(world_size, rank=rank)
        self.scatter = ScatterKeras(world_size, rank=rank)
    
    def forward_column_parallel(self, partial_outputs: List, dim: int = -1):
        logger.debug(f"Forward column-parallel: AllGather {len(partial_outputs)} outputs along dim {dim}")
        return self.allgather(partial_outputs)
    
    def backward_column_parallel(self, partial_gradients: List, op: str = "sum") -> List:
        logger.debug(f"Backward column-parallel: AllReduce {len(partial_gradients)} gradients with op {op}")
        return self.allreduce(partial_gradients)
    
    def forward_row_parallel(self, partial_outputs: List, op: str = "sum") -> List:
        logger.debug(f"Forward row-parallel: AllReduce {len(partial_outputs)} outputs with op {op}")
        return self.allreduce(partial_outputs)
    
    def backward_row_parallel(self, partial_gradients: List, dim: int = -1):
        logger.debug(f"Backward row-parallel: AllGather {len(partial_gradients)} gradients along dim {dim}")
        return self.allgather(partial_gradients)
    
    def handle_mlp_handshake(self, 
                            up_projection_outputs: List,
                            down_projection_inputs: List) -> Tuple:
        up_output = self.forward_column_parallel(up_projection_outputs, dim=-1)
        down_inputs = self.forward_row_parallel(down_projection_inputs, op="sum")
        
        return up_output, down_inputs
    
    def slice_upstream_gradient_for_column_parallel(self, full_gradient, rank: int, world_size: int, dim: int = -1):
        try:
            total_size = full_gradient.shape[dim]
            slice_size = total_size // world_size
            
            start_idx = rank * slice_size
            end_idx = start_idx + slice_size if rank < world_size - 1 else total_size
            
            if dim == -1:
                if hasattr(full_gradient, 'shape') and len(full_gradient.shape) >= 2:
                    if _get_tensor_lib(full_gradient) == 'pytorch':
                        return full_gradient[..., start_idx:end_idx]
                    elif _get_tensor_lib(full_gradient) == 'tensorflow':
                        import tensorflow as tf
                        return tf.slice(full_gradient, [0] * (len(full_gradient.shape) - 1) + [start_idx], 
                                     [-1] * (len(full_gradient.shape) - 1) + [end_idx - start_idx])
                    else:
                        slices = [slice(None)] * len(full_gradient.shape)
                        slices[dim] = slice(start_idx, end_idx)
                        return full_gradient[tuple(slices)]
            
            slices = [slice(None)] * len(full_gradient.shape)
            slices[dim] = slice(start_idx, end_idx)
            return full_gradient[tuple(slices)]
            
        except Exception as e:
            logger.warning(f"Gradient slicing failed: {e}, returning full gradient")
            return full_gradient
    
    def slice_upstream_gradient_for_row_parallel(self, full_gradient, rank: int, world_size: int, dim: int = 0):
        try:
            total_size = full_gradient.shape[dim]
            slice_size = total_size // world_size
            
            start_idx = rank * slice_size
            end_idx = start_idx + slice_size if rank < world_size - 1 else total_size
            
            slices = [slice(None)] * len(full_gradient.shape)
            slices[dim] = slice(start_idx, end_idx)
            return full_gradient[tuple(slices)]
            
        except Exception as e:
            logger.warning(f"Gradient slicing failed: {e}, returning full gradient")
            return full_gradient


def allreduce_gradients(gradients: List, world_size: int) -> List:
    allreduce_op = AllReduceKeras(world_size, op="mean")
    return allreduce_op(gradients)


def allgather_outputs(outputs: List, world_size: int, dim: int = -1):
    allgather_op = AllGatherKeras(world_size, dim=dim)
    return allgather_op(outputs)


def broadcast_parameters(parameters: List, world_size: int, src_rank: int = 0) -> List:
    broadcast_op = BroadcastKeras(world_size, src_rank)
    return broadcast_op(parameters[src_rank]) 