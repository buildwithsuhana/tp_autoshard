import dataclasses
from typing import Any, Dict, Sequence

from .communications_keras import AllReduceKeras, AllGatherKeras, BroadcastKeras

@dataclasses.dataclass
class ConfigKeras:
    state_rules: Dict[str, Any]
    output_rules: Dict[str, Any]
    
    def create_collective_ops(self, devices: Sequence[str], distributed: bool = True):
        world_size = len(devices)
        make_allreduce = lambda ws: AllReduceKeras(ws, op="mean")
        make_allgather = lambda ws, dim: AllGatherKeras(ws, dim)
        make_broadcast = lambda ws: BroadcastKeras(ws)
            
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
            
        return dataclasses.replace(
            self,
            output_rules=create_collective_ops(self.output_rules),
        ) 