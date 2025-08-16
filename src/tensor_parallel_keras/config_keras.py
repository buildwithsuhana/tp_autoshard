"""
Configuration classes for Keras Tensor Parallel
"""

import dataclasses
from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch
from keras import layers

from .communications_keras import AllReduceKeras, AllGatherKeras, BroadcastKeras
from .cross_device_ops_keras import reduce_add_keras, all_gather_keras, broadcast_coalesced_keras


@dataclasses.dataclass
class ConfigKeras:
    """
    Configuration for Keras tensor parallel operations.
    """
    state_rules: Dict[str, Any]  # How to split parameters
    input_rules: Dict[str, Any]  # How to handle inputs  
    output_rules: Dict[str, Any] # How to handle outputs
    attr_rules: Dict[str, Any]   # How to handle attributes
    
    def create_collective_ops(self, devices: Sequence[str], distributed: bool = True):
        """
        Create collective operations for the configuration.
        """
        world_size = len(devices)
        all_cuda = all(device.startswith("gpu") for device in devices)
        
        # Use new communication operations
        make_allreduce = lambda ws: AllReduceKeras(ws, op="mean")
        make_allgather = lambda ws, dim: AllGatherKeras(ws, dim)
        make_broadcast = lambda ws: BroadcastKeras(ws)
            
        # Convert rules to operations
        def create_collective_ops(rules: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for pattern, actions in rules.items():
                if isinstance(actions, dict):
                    result[pattern] = {}
                    for key, action in actions.items():
                        if isinstance(action, str):
                            if action == "sum":
                                result[pattern][key] = make_allreduce(world_size)
                            elif action.startswith("gather"):
                                # Extract dimension from "gather -1" format
                                dim = -1
                                if " " in action:
                                    dim = int(action.split(" ")[1])
                                result[pattern][key] = make_allgather(world_size, dim)
                            elif action == "broadcast":
                                result[pattern][key] = make_broadcast(world_size)
                            else:
                                result[pattern][key] = action
                        else:
                            result[pattern][key] = action
                else:
                    result[pattern] = actions
            return result
            
        # Create a copy with collective operations
        return dataclasses.replace(
            self,
            input_rules=create_collective_ops(self.input_rules),
            output_rules=create_collective_ops(self.output_rules),
        ) 