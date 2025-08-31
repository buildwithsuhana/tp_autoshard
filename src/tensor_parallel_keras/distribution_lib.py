"""
Distribution Library for Tensor Parallel Keras.

This module provides device detection and management.
It's updated to correctly detect simulated devices for the JAX backend.
"""

import logging
from typing import List, Dict, Optional, Tuple
import keras

logger = logging.getLogger(__name__)

def list_devices() -> List[str]:
    """
    List all available devices.
    
    This function is updated to correctly detect simulated devices when
    using the JAX backend with flags like XLA_FLAGS. It falls back to
    checking physical devices for other backends.
    
    Returns:
        List of device identifiers in priority order (TPU > GPU > CPU)
    """
    if keras.backend.backend() == 'jax':
        try:
            import jax
            jax_devices = jax.devices()
            if jax_devices:
                logger.info(f"JAX backend detected, found {len(jax_devices)} devices via jax.devices()")
                formatted_devices = [f"{d.platform.lower()}:{d.id}" for d in jax_devices]
                return formatted_devices
        except (ImportError, Exception) as e:
            logger.warning(f"Could not use jax.devices() to detect devices: {e}")
    
    devices = []
    
    try:
        tpu_devices = keras.config.list_physical_devices('TPU')
        if tpu_devices:
            logger.info(f"Found {len(tpu_devices)} TPU devices")
            devices.extend([f"tpu:{i}" for i in range(len(tpu_devices))])
    except Exception as e:
        logger.debug(f"TPU detection failed: {e}")
    
    try:
        gpu_devices = keras.config.list_physical_devices('GPU')
        if gpu_devices:
            logger.info(f"Found {len(gpu_devices)} GPU devices")
            devices.extend([f"gpu:{i}" for i in range(len(gpu_devices))])
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")
    
    try:
        cpu_devices = keras.config.list_physical_devices('CPU')
        if cpu_devices:
            logger.info(f"Found {len(cpu_devices)} CPU devices")
            devices.extend([f"cpu:{i}" for i in range(len(cpu_devices))])
    except Exception as e:
        logger.debug(f"CPU detection failed: {e}")
    
    if not devices:
        logger.warning("No physical devices detected, using default CPU")
        devices.append("cpu:0")
    
    logger.info(f"Total available devices via physical scan: {len(devices)}")
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
        logger.warning(f"Failed to get device info for {device_id}: {e}")
    
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
        logger.warning(f"Requested {count} devices but only {len(all_devices)} available")
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
        logger.error(f"Device validation failed: {e}")
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
        logger.debug(f"Failed to get memory info for {device_id}: {e}")
    
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
        
        logger.info(f"Auto-configured tensor parallelism: {config}")
        return config
        
    except Exception as e:
        logger.error(f"Auto-configuration failed: {e}")
        return {
            'devices': ['cpu:0'],
            'world_size': 1,
            'backend': backend if backend else 'fallback'
        } 