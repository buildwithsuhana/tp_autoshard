"""
Distribution Library for Tensor Parallel Keras.
This module provides device detection and management.
"""

from typing import List, Dict, Optional, Tuple
import keras


def list_devices() -> List[str]:
    if keras.backend.backend() == 'torch':
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                return [f"cuda:{i}" for i in range(count)]
        except (ImportError, Exception) as e:
            pass

    if keras.backend.backend() == 'jax':
        try:
            import jax
            jax_devices = jax.devices()
            if jax_devices:
                formatted_devices = [f"{d.platform.lower()}:{d.id}" for d in jax_devices]
                return formatted_devices
        except (ImportError, Exception) as e:
            pass

    devices = []

    for device_type in ["TPU", "GPU", "CPU"]:
        try:
            physical_devices = keras.config.list_physical_devices(device_type)
            if physical_devices:
                devices.extend([f"{device_type.lower()}:{i}" for i in range(len(physical_devices))])
        except Exception as e:
            pass

    if not devices:
        devices.append("cpu:0")
    return devices

def get_device_info(device_id: str) -> Dict[str, any]:
    """
    Get detailed information about a specific device.
    
    Args:
        device_id: Device identifier (e.g., 'gpu:0', 'tpu:0', 'cpu:0')
        
    Returns:
        Dictionary containing device information
    """
    device_info = {
        'id': device_id,
        'type': None,
        'index': None,
        'memory': None,
        'capabilities': None
    }
    
    try:
        if device_id.startswith('gpu:'):
            device_info['type'] = 'GPU'
            device_info['index'] = int(device_id.split(':')[1])
        elif device_id.startswith('tpu:'):
            device_info['type'] = 'TPU'
            device_info['index'] = int(device_id.split(':')[1])
        elif device_id.startswith('cpu:'):
            device_info['type'] = 'CPU'
            device_info['index'] = int(device_id.split(':')[1])
            
    except Exception as e:
        pass
    return device_info

def get_best_devices(count: int = 1) -> List[str]:
    """
    Get the best available devices for tensor parallelism.
    
    Args:
        count: Number of devices needed
        
    Returns:
        List of best device identifiers
    """
    all_devices = list_devices()
    
    if count <= 0:
        return []
    
    if count > len(all_devices):
        count = len(all_devices)
    
    return all_devices[:count]

def get_device_backend(device_type: str) -> str:
    """
    Get the recommended backend for a device type.
    
    Args:
        device_type: Device type ('tpu', 'gpu', 'cpu')
        
    Returns:
        Recommended backend name
    """
    backend_mapping = {
        'tpu': 'jax',
        'gpu': 'nccl',
        'cpu': 'tensorflow'
    }
    
    return backend_mapping.get(device_type.lower(), 'auto')

def validate_device_placement(device_id: str) -> bool:
    """
    Validate if a device can be used for tensor operations.
    
    Args:
        device_id: Device identifier
        
    Returns:
        True if device is valid and available
    """
    try:
        all_devices = list_devices()
        return device_id in all_devices
    except Exception as e:
        return False

def get_device_memory_info(device_id: str) -> Optional[Dict[str, any]]:
    """
    Get memory information for a device (if available).
    
    Args:
        device_id: Device identifier
        
    Returns:
        Memory information dictionary or None if not available
    """
    try:
        if device_id.startswith('gpu:'):
            return {'type': 'GPU', 'index': int(device_id.split(':')[1]), 'memory': 'Available'}
        elif device_id.startswith('tpu:'):
            return {'type': 'TPU', 'index': int(device_id.split(':')[1]), 'memory': 'TPU Memory'}
        elif device_id.startswith('cpu:'):
            return {'type': 'CPU', 'index': int(device_id.split(':')[1]), 'memory': 'System RAM'}
    except Exception as e:
        pass
    return None 

def auto_configure_tensor_parallel(world_size: int = None, backend: str = None) -> Dict[str, any]:
    """
    Automatically configure tensor parallelism with the best available devices.
    
    Args:
        world_size: Number of devices to use (if None, uses all available)
        backend: Backend to use (if None, uses 'auto')
        
    Returns:
        Configuration dictionary with devices, backend, and other settings
    """
    try:
        all_devices = list_devices()
        
        if not all_devices:
            raise RuntimeError("No devices available for tensor parallelism")
        
        if world_size is None:
            world_size = len(all_devices)
        else:
            world_size = min(world_size, len(all_devices))
        
        selected_devices = all_devices[:world_size]
        
        recommended_backend = backend if backend else 'auto'
        
        config = {
            'devices': selected_devices,
            'world_size': world_size,
            'backend': recommended_backend
        }
        
        return config
        
    except Exception as e:
        return {
            'devices': ['cpu:0'],
            'world_size': 1,
            'backend': backend if backend else 'fallback'
        } 