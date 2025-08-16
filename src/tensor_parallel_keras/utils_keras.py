"""
Utility functions for Keras Tensor Parallel
"""

from typing import Any, Callable, List, Sequence, Union
import torch


def nested_flatten(t: Any) -> List[Any]:
    """
    Flatten a nested structure.
    
    Args:
        t: Nested structure to flatten
        
    Returns:
        Flattened list
    """
    if isinstance(t, (list, tuple)):
        result = []
        for item in t:
            result.extend(nested_flatten(item))
        return result
    else:
        return [t]


def nested_pack(flat: List[Any], structure: Any) -> Any:
    """
    Pack a flattened list back into a nested structure.
    
    Args:
        flat: Flattened list
        structure: Original structure template
        
    Returns:
        Packed structure
    """
    if isinstance(structure, (list, tuple)):
        result = []
        for item in structure:
            if isinstance(item, (list, tuple)):
                # Recursively pack sub-structures
                sub_size = count_elements(item)
                sub_flat = flat[:sub_size]
                flat = flat[sub_size:]
                result.append(nested_pack(sub_flat, item))
            else:
                # Leaf element
                result.append(flat.pop(0))
        return type(structure)(result)
    else:
        # Leaf element
        return flat.pop(0)


def count_elements(structure: Any) -> int:
    """
    Count the number of elements in a nested structure.
    
    Args:
        structure: Nested structure
        
    Returns:
        Number of elements
    """
    if isinstance(structure, (list, tuple)):
        return sum(count_elements(item) for item in structure)
    else:
        return 1


def nested_map(fn: Callable, *args: Any) -> Any:
    """
    Apply a function to all elements in nested structures.
    
    Args:
        fn: Function to apply
        *args: Nested structures
        
    Returns:
        New nested structure with function applied
    """
    if isinstance(args[0], (list, tuple)):
        return type(args[0])(nested_map(fn, *[arg[i] for arg in args]) for i in range(len(args[0])))
    else:
        return fn(*args)


def is_namedtuple(x: Any) -> bool:
    """
    Check if an object is a namedtuple.
    
    Args:
        x: Object to check
        
    Returns:
        True if x is a namedtuple
    """
    return hasattr(x, '_fields') and hasattr(x, '_asdict')


def nested_compare(t: Any, u: Any) -> bool:
    """
    Compare two nested structures for equality.
    
    Args:
        t: First structure
        u: Second structure
        
    Returns:
        True if structures are equal
    """
    if type(t) != type(u):
        return False
        
    if isinstance(t, (list, tuple)):
        if len(t) != len(u):
            return False
        return all(nested_compare(ti, ui) for ti, ui in zip(t, u))
    else:
        return t == u 