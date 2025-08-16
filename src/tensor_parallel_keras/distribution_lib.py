"""
Distribution Library for Tensor Parallel Keras.

This module provides device detection and management using Keras APIs.
"""

import logging
from typing import List, Dict, Optional, Tuple
import keras

logger = logging.getLogger(__name__)

def list_devices() -> List[str]:
    """
    List all available devices using Keras APIs.
    
    Returns:
        List of device identifiers in priority order (TPU > GPU > CPU)
    """
    devices = []
    
    # Check for TPU devices first (highest priority)
    try:
        tpu_devices = keras.config.list_physical_devices('TPU')
        if tpu_devices:
            logger.info(f"Found {len(tpu_devices)} TPU devices")
            for i, device in enumerate(tpu_devices):
                devices.append(f"tpu:{i}")
                logger.info(f"  TPU device {i}: {device}")
    except Exception as e:
        logger.debug(f"TPU detection failed: {e}")
    
    # Check for GPU devices
    try:
        gpu_devices = keras.config.list_physical_devices('GPU')
        if gpu_devices:
            logger.info(f"Found {len(gpu_devices)} GPU devices")
            for i, device in enumerate(gpu_devices):
                devices.append(f"gpu:{i}")
                logger.info(f"  GPU device {i}: {device}")
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")
    
    # Check for CPU devices
    try:
        cpu_devices = keras.config.list_physical_devices('CPU')
        if cpu_devices:
            logger.info(f"Found {len(cpu_devices)} CPU devices")
            for i, device in enumerate(cpu_devices):
                devices.append(f"cpu:{i}")
                logger.info(f"  CPU device {i}: {device}")
    except Exception as e:
        logger.debug(f"CPU detection failed: {e}")
    
    # If no devices found, add default CPU
    if not devices:
        logger.warning("No devices detected, using default CPU")
        devices.append("cpu:0")
    
    logger.info(f"Total available devices: {len(devices)}")
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
            
            # Get GPU memory info if available
            try:
                import tensorflow as tf
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if device_info['index'] < len(gpus):
                    device_info['memory'] = 'Available'  # Could add actual memory info
            except:
                pass
                
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
    
    # Return devices in priority order (TPU > GPU > CPU)
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
        'tpu': 'jax',      # JAX is best for TPU
        'gpu': 'nccl',     # NCCL is best for GPU
        'cpu': 'tensorflow' # TensorFlow is good for CPU
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
            # Try to get GPU memory info
            try:
                import tensorflow as tf
                gpus = tf.config.experimental.list_physical_devices('GPU')
                index = int(device_id.split(':')[1])
                if index < len(gpus):
                    # Could add actual memory querying here
                    return {'type': 'GPU', 'index': index, 'memory': 'Available'}
            except:
                pass
                
        elif device_id.startswith('tpu:'):
            return {'type': 'TPU', 'index': int(device_id.split(':')[1]), 'memory': 'TPU Memory'}
            
        elif device_id.startswith('cpu:'):
            return {'type': 'CPU', 'index': int(device_id.split(':')[1]), 'memory': 'System RAM'}
            
    except Exception as e:
        logger.debug(f"Failed to get memory info for {device_id}: {e}")
    
    return None 

def auto_configure_tensor_parallel(world_size: int = None) -> Dict[str, any]:
    """
    Automatically configure tensor parallelism with the best available devices and backend.
    
    Args:
        world_size: Number of devices to use (if None, uses all available)
        
    Returns:
        Configuration dictionary with devices, backend, and other settings
    """
    try:
        from .distribution_lib import list_devices, get_device_info, get_device_backend
        
        # Get all available devices
        all_devices = list_devices()
        
        if not all_devices:
            raise RuntimeError("No devices available for tensor parallelism")
        
        # Determine world_size if not specified
        if world_size is None:
            world_size = len(all_devices)
        else:
            world_size = min(world_size, len(all_devices))
        
        # Select the best devices
        selected_devices = all_devices[:world_size]
        
        # Determine the best backend based on device types
        device_types = set()
        for device_id in selected_devices:
            device_info = get_device_info(device_id)
            device_type = device_info.get('type', '').lower()
            device_types.add(device_type)
        
        # Choose backend based on device types
        if 'tpu' in device_types:
            recommended_backend = 'jax'
        elif 'gpu' in device_types:
            recommended_backend = 'nccl'
        else:
            recommended_backend = 'tensorflow'
        
        # Create configuration
        config = {
            'devices': selected_devices,
            'world_size': world_size,
            'backend': recommended_backend,
            'device_types': list(device_types),
            'auto_configured': True
        }
        
        logger.info(f"Auto-configured tensor parallelism: {config}")
        return config
        
    except Exception as e:
        logger.error(f"Auto-configuration failed: {e}")
        # Return fallback configuration
        return {
            'devices': ['cpu:0'],
            'world_size': 1,
            'backend': 'fallback',
            'device_types': ['cpu'],
            'auto_configured': False,
            'error': str(e)
        } 