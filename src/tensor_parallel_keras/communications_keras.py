#!/usr/bin/env python3
from typing import List, Tuple, Any
import keras

from .distributed_backend import DistributedBackend

def _clone_tensor(tensor):
    return keras.ops.convert_to_tensor(keras.ops.convert_to_numpy(tensor))

class CollectiveOpKeras:
    def __init__(self, world_size: int, rank: int = 0):
        self.world_size = world_size
        self.rank = rank
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class AllReduceKeras(CollectiveOpKeras):
    def __init__(self, world_size: int, backend: DistributedBackend, op: str = "sum", rank: int = 0):
        super().__init__(world_size, rank)
        self.op = op
        self.backend = backend
        self.all_reduce_fn = self.backend.get_communication_ops().get("all_reduce")
        if self.all_reduce_fn is None:
            raise NotImplementedError("AllReduce is not supported by the current backend.")

    def __call__(self, local_tensor: Any) -> Any:
        synced_tensor = self.all_reduce_fn(local_tensor, op=self.op)
        return synced_tensor


class AllGatherKeras(CollectiveOpKeras):
    def __init__(self, world_size: int, backend: DistributedBackend, dim: int = -1, rank: int = 0):
        super().__init__(world_size, rank)
        self.dim = dim
        self.backend = backend
        self.all_gather_fn = self.backend.get_communication_ops().get("all_gather")
        if self.all_gather_fn is None:
            raise NotImplementedError("AllGather is not supported by the current backend.")

    def __call__(self, local_tensor: Any) -> Any:
        full_tensor = self.all_gather_fn(local_tensor, axis=self.dim)
        return full_tensor
    
class BroadcastKeras(CollectiveOpKeras):
    def __init__(self, world_size: int, backend: DistributedBackend, src_rank: int = 0, rank: int = 0):
        super().__init__(world_size, rank)
        self.src_rank = src_rank

    def __call__(self, tensor):
        return [_clone_tensor(tensor) for _ in range(self.world_size)]

class ScatterKeras(CollectiveOpKeras):
    def __init__(self, world_size: int, backend: DistributedBackend, dim: int = -1, rank: int = 0):
        super().__init__(world_size, rank)
        self.dim = dim

    def __call__(self, tensor):
        return [_clone_tensor(tensor) for _ in range(self.world_size)]

class TensorParallelCommunicator:
    def __init__(self, world_size: int, rank: int = 0):
        self.world_size = world_size
        self.rank = rank
        self.backend = DistributedBackend(keras.backend.backend())
        
        self.allreduce = AllReduceKeras(world_size, backend=self.backend, rank=rank)
        self.allgather = AllGatherKeras(world_size, backend=self.backend, rank=rank)
        self.broadcast = BroadcastKeras(world_size, backend=self.backend, rank=rank)
        self.scatter = ScatterKeras(world_size, backend=self.backend, rank=rank)
    
    def forward_column_parallel(self, partial_outputs: List, dim: int = -1):
        self.allgather.dim = dim
        return self.allgather(partial_outputs)
    
    def backward_column_parallel(self, partial_gradients: List, op: str = "sum") -> List:
        self.allreduce.op = op
        return self.allreduce(partial_gradients)
    
    def forward_row_parallel(self, partial_outputs: List, op: str = "sum") -> List:
        self.allreduce.op = op
        return self.allreduce(partial_outputs)
    
    def backward_row_parallel(self, partial_gradients: List, dim: int = -1):
        self.allgather.dim = dim
        return self.allgather(partial_gradients)
    
    def handle_mlp_handshake(self, up_projection_outputs: List, down_projection_inputs: List) -> Tuple:
        up_output = self.forward_column_parallel(up_projection_outputs, dim=-1)
        down_inputs = self.forward_row_parallel(down_projection_inputs, op="sum")
        return up_output, down_inputs
    
    def slice_upstream_gradient_for_column_parallel(self, full_gradient, rank: int, world_size: int, dim: int = -1):
        try:
            total_size = full_gradient.shape[dim]
            slice_size = total_size // world_size
            remainder = total_size % world_size
            start_idx = rank * slice_size + min(rank, remainder)
            end_idx = start_idx + slice_size + (1 if rank < remainder else 0)
            slices = [slice(None)] * len(full_gradient.shape)
            slices[dim] = slice(start_idx, end_idx)
            return full_gradient[tuple(slices)]
        except Exception as e:
            return full_gradient

    def slice_upstream_gradient_for_row_parallel(self, full_gradient, rank: int, world_size: int, dim: int = 0):
        try:
            total_size = full_gradient.shape[dim]
            slice_size = total_size // world_size
            start_idx = rank * slice_size
            end_idx = (rank + 1) * slice_size
            if rank == world_size - 1:
                end_idx = total_size
            slices = [slice(None)] * len(full_gradient.shape)
            slices[dim] = slice(start_idx, end_idx)
            return full_gradient[tuple(slices)]
        except Exception as e:
            return full_gradient

def allreduce_gradients(gradients: List, world_size: int, backend: DistributedBackend) -> List:
    allreduce_op = AllReduceKeras(world_size, backend=backend, op="mean")
    return allreduce_op(gradients)

def allgather_outputs(outputs: List, world_size: int, backend: DistributedBackend, dim: int = -1):
    allgather_op = AllGatherKeras(world_size, backend=backend, dim=dim)
    return allgather_op(outputs)

def broadcast_parameters(parameters: List, world_size: int, backend: DistributedBackend, src_rank: int = 0) -> List:
    broadcast_op = BroadcastKeras(world_size, backend=backend, src_rank=src_rank)
    return broadcast_op(parameters[src_rank])