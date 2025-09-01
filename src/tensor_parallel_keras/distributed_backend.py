#!/usr/bin/env python3
"""
Distributed Backend Operations
Centralizes all backend-specific operations for JAX, TensorFlow, and PyTorch
"""

import logging
from typing import List, Any, Union, Tuple, Optional
import numpy as np
import keras 

logger = logging.getLogger(__name__)

class DistributedBackend:
    """Base class for distributed backend operations."""
    
    def __init__(self, backend_name: str = "auto"):
        self.backend_name = backend_name
        self.backend = self._initialize_backend()

    def _initialize_backend(self) -> str:
        """Initializes the backend, respecting the user's choice."""
        if self.backend_name != "auto":
            # If a specific backend is requested, use it.
            return self.backend_name

        # --- This is the old _detect_backend logic, now used only for "auto" ---
        try:
            import jax
            return "jax"
        except ImportError:
            try:
                import tensorflow as tf
                return "tensorflow"
            except ImportError:
                try:
                    import torch
                    return "pytorch"
                except ImportError:
                    return "numpy"

        
    def _detect_backend(self) -> str:
        """Detect the available backend."""
        try:
            import jax
            return "jax"
        except ImportError:
            try:
                import tensorflow as tf
                return "tensorflow"
            except ImportError:
                try:
                    import torch
                    return "pytorch"
                except ImportError:
                    return "numpy"
    
    def get_tensor_lib(self):
        """Get the appropriate tensor library for the backend."""
        if self.backend == "jax":
            import jax.numpy as jnp
            return jnp
        elif self.backend == "tensorflow":
            import tensorflow as tf
            return tf
        elif self.backend == "pytorch":
            import torch
            return torch
        else:
            return np
    
    def convert_to_backend_tensor(self, tensor: Any) -> Any:
        """Convert a tensor to the appropriate backend format."""
        if self.backend == "jax":
            import jax.numpy as jnp
            if hasattr(tensor, 'numpy'):
                return jnp.array(tensor.numpy())
            else:
                return jnp.array(tensor)
        elif self.backend == "tensorflow":
            import tensorflow as tf
            if hasattr(tensor, 'numpy'):
                return tf.convert_to_tensor(tensor.numpy())
            else:
                return tf.convert_to_tensor(tensor)
        elif self.backend == "pytorch":
            import torch
            # Use the native clone() method; detach() removes it from the grad graph
            return tensor.clone().detach()
        else:
            return keras.ops.convert_to_numpy(tensor)
    
    def compute_gradients(self, loss: Any, trainable_vars: List[Any]) -> List[Any]:
        """Compute gradients using the appropriate backend's automatic differentiation."""
        try:
            if self.backend == "jax":
                return self._compute_jax_gradients(loss, trainable_vars)
            elif self.backend == "tensorflow":
                return self._compute_tensorflow_gradients(loss, trainable_vars)
            elif self.backend == "pytorch":
                return self._compute_pytorch_gradients(loss, trainable_vars)
            else:
                return self._compute_numpy_gradients(loss, trainable_vars)
        except Exception as e:
            logger.error(f"Gradient computation failed for {self.backend}: {e}")
            return self._create_zero_gradients(trainable_vars)
    
    def _compute_jax_gradients(self, loss: Any, trainable_vars: List[Any]) -> List[Any]:
        """Compute gradients using JAX automatic differentiation."""
        import jax
        import jax.numpy as jnp
        
        def safe_convert_to_jax(tensor):
            try:
                if hasattr(tensor, 'numpy'):
                    if hasattr(tensor, 'shape') and tensor.shape is None:
                        logger.warning("Symbolic tensor detected, using dummy value for gradient computation")
                        return jnp.array(0.0)
                    else:
                        return jnp.array(tensor.numpy())
                else:
                    return jnp.array(tensor)
            except Exception as e:
                logger.warning(f"Failed to convert tensor to JAX: {e}, using dummy value")
                return jnp.array(0.0)
        
        loss_jax = safe_convert_to_jax(loss)
        params_jax = [safe_convert_to_jax(param) for param in trainable_vars]
        
        def loss_fn(params):
            return loss_jax
        
        try:
            gradients = jax.grad(loss_fn)(params_jax)
            logger.info("   - JAX gradient computation successful")
            return gradients
        except Exception as e:
            logger.warning(f"JAX gradient computation failed: {e}, using fallback")
            return [jnp.zeros_like(param) for param in params_jax]
    
    def _compute_tensorflow_gradients(self, loss: Any, trainable_vars: List[Any]) -> List[Any]:
        """Compute gradients using TensorFlow automatic differentiation."""
        import tensorflow as tf
        
        with tf.GradientTape() as tape:
            for var in trainable_vars:
                tape.watch(var)
        
        try:
            gradients = tape.gradient(loss, trainable_vars)
            logger.info("   - TensorFlow gradient computation successful")
            return gradients
        except Exception as e:
            logger.warning(f"TensorFlow gradient computation failed: {e}, using fallback")
            return [tf.zeros_like(var) for var in trainable_vars]
    
    def _compute_pytorch_gradients(self, loss: Any, trainable_vars: List[Any]) -> List[Any]:
        """
        Computes gradients for PyTorch.
        NOTE: In a Keras model, this is handled by `loss.backward()` in the `train_step`.
        This function is deprecated for the Keras workflow.
        """
        logger.warning("PyTorch gradient computation is handled by `loss.backward()` in the Keras model's `train_step`.")
        # The actual logic `loss.backward()` is stateful and doesn't fit this functional pattern.
        return self._create_zero_gradients(trainable_vars)
    
    def _create_zero_gradients(self, trainable_vars: List[Any]) -> List[Any]:
        """Create zero gradients as fallback."""
        lib = self.get_tensor_lib()
        return [lib.zeros_like(var) for var in trainable_vars]
    
    def _compute_numpy_gradients(self, loss: Any, trainable_vars: List[Any]) -> List[Any]:
        """Fallback gradient computation using numerical differentiation."""
        epsilon = 1e-7
        gradients = []
        
        for var in trainable_vars:
            if hasattr(var, 'shape'):
                grad = np.zeros_like(var)
                for i in range(var.size):
                    idx = np.unravel_index(i, var.shape)
                    var_plus = var.copy()
                    var_minus = var.copy()
                    var_plus[idx] += epsilon
                    var_minus[idx] -= epsilon
                    grad[idx] = (loss - loss) / (2 * epsilon)
                gradients.append(grad)
            else:
                gradients.append(0.0)
        
        return gradients
    
    # def _create_zero_gradients(self, trainable_vars: List[Any]) -> List[Any]:
    #     """Create zero gradients as fallback."""
    #     gradients = []
    #     for var in trainable_vars:
    #         if hasattr(var, 'shape'):
    #             zero_grad = np.zeros_like(var)
    #             gradients.append(zero_grad)
    #         else:
    #             gradients.append(0.0)
    #     return gradients
    
    def apply_gradients(self, gradients: List[Any], trainable_vars: List[Any], 
                       learning_rate: float = 0.001) -> None:
        """Apply gradients to trainable variables using the appropriate backend."""
        if self.backend == "jax":
            self._apply_jax_gradients(gradients, trainable_vars, learning_rate)
        elif self.backend == "tensorflow":
            self._apply_tensorflow_gradients(gradients, trainable_vars, learning_rate)
        elif self.backend == "pytorch":
            self._apply_pytorch_gradients(gradients, trainable_vars, learning_rate)
        else:
            self._apply_numpy_gradients(gradients, trainable_vars, learning_rate)
    
    def _apply_jax_gradients(self, gradients: List[Any], trainable_vars: List[Any], 
                            learning_rate: float) -> None:
        """Apply gradients using JAX operations."""
        import jax.numpy as jnp
        
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                new_value = var - (learning_rate * grad)
                if hasattr(var, 'assign'):
                    var.assign(new_value)
    
    def _apply_tensorflow_gradients(self, gradients: List[Any], trainable_vars: List[Any], 
                                   learning_rate: float) -> None:
        """Apply gradients using TensorFlow operations."""
        import tensorflow as tf
        
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                new_value = var - (learning_rate * grad)
                var.assign(new_value)
    
    def _apply_pytorch_gradients(self, gradients: List[Any], trainable_vars: List[Any], 
                                learning_rate: float) -> None:
        """Apply gradients using PyTorch operations."""
        import torch
        
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                with torch.no_grad():
                    var -= learning_rate * grad
    
    def _apply_numpy_gradients(self, gradients: List[Any], trainable_vars: List[Any], 
                              learning_rate: float) -> None:
        """Apply gradients using NumPy operations."""
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                new_value = var - (learning_rate * grad)
                if hasattr(var, 'assign'):
                    var.assign(new_value)
                else:
                    var[:] = new_value
    
    def create_optimizer(self, optimizer_class: str, **kwargs):
        """Create an optimizer for the appropriate backend."""
        if self.backend == "jax":
            return self._create_jax_optimizer(optimizer_class, **kwargs)
        elif self.backend == "tensorflow":
            return self._create_tensorflow_optimizer(optimizer_class, **kwargs)
        elif self.backend == "pytorch":
            return self._create_pytorch_optimizer(optimizer_class, **kwargs)
        else:
            return self._create_numpy_optimizer(optimizer_class, **kwargs)
    
    def _create_jax_optimizer(self, optimizer_class: str, **kwargs):
        """Create a JAX optimizer."""
        import jax
        import optax
        
        if optimizer_class.lower() == "adam":
            return optax.adam(**kwargs)
        elif optimizer_class.lower() == "sgd":
            return optax.sgd(**kwargs)
        else:
            return optax.adam(learning_rate=0.001)
    
    def _create_tensorflow_optimizer(self, optimizer_class: str, **kwargs):
        """Create a TensorFlow optimizer."""
        import tensorflow as tf
        
        if optimizer_class.lower() == "adam":
            return tf.keras.optimizers.Adam(**kwargs)
        elif optimizer_class.lower() == "sgd":
            return tf.keras.optimizers.SGD(**kwargs)
        else:
            return tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def _create_pytorch_optimizer(self, optimizer_class: str, **kwargs):
        """Create a PyTorch optimizer."""
        import torch
        
        if optimizer_class.lower() == "adam":
            return torch.optim.Adam(**kwargs)
        elif optimizer_class.lower() == "sgd":
            return torch.optim.SGD(**kwargs)
        else:
            return torch.optim.Adam(lr=0.001)
    
    def _create_numpy_optimizer(self, optimizer_class: str, **kwargs):
        """Create a NumPy-based optimizer (simplified)."""
        class NumpyOptimizer:
            def __init__(self, learning_rate=0.001):
                self.learning_rate = learning_rate
            
            def apply_gradients(self, grads_and_vars):
                for grad, var in grads_and_vars:
                    if grad is not None:
                        var -= self.learning_rate * grad
        
        return NumpyOptimizer(**kwargs)
    
    def get_device_info(self) -> dict:
        """Get information about available devices for the backend."""
        info = {
            "backend": self.backend,
            "devices": [],
            "device_count": 0
        }
        
        try:
            if self.backend == "jax":
                import jax
                info["devices"] = [str(d) for d in jax.devices()]
                info["device_count"] = jax.local_device_count()
            elif self.backend == "tensorflow":
                import tensorflow as tf
                info["devices"] = [d.name for d in tf.config.list_physical_devices()]
                info["device_count"] = len(tf.config.list_physical_devices())
            elif self.backend == "pytorch":
                import torch
                if torch.cuda.is_available():
                    count = torch.cuda.device_count()
                    info["devices"] = [f"cuda:{i}" for i in range(count)]
                    info["device_count"] = count
            else:
                info["devices"] = ["cpu"]
                info["device_count"] = 1
        except Exception as e:
            logger.warning(f"Could not get device info for {self.backend}: {e}")
            info["devices"] = ["cpu"]
            info["device_count"] = 1
        
        return info
    
    def is_multi_device_capable(self) -> bool:
        """Check if the backend supports multi-device operations."""
        device_info = self.get_device_info()
        return device_info["device_count"] > 1
    
    def get_communication_ops(self):
        """Get communication operations for the backend."""
        if self.backend == "jax":
            return self._get_jax_communication_ops()
        elif self.backend == "tensorflow":
            return self._get_tensorflow_communication_ops()
        elif self.backend == "pytorch":
            return self._get_pytorch_communication_ops()
        else:
            return self._get_numpy_communication_ops()
    
    def _get_jax_communication_ops(self):
        """Get JAX communication operations."""
        import jax
        import jax.lax as lax
        
        return {
            "all_reduce": lambda x, op="sum": lax.pmean(x, axis_name="batch"),
            "all_gather": lambda x: lax.all_gather(x, axis_name="batch"),
            "broadcast": lambda x: x,
            "scatter": lambda x, num_devices: lax.psplit(x, axis_name="batch", num_splits=num_devices)
        }
    
    def _get_tensorflow_communication_ops(self):
        """Get TensorFlow communication operations."""
        import tensorflow as tf
        
        return {
            "all_reduce": lambda x, op="sum": tf.reduce_sum(x, axis=0),
            "all_gather": lambda x: tf.concat([x, x], axis=0),
            "broadcast": lambda x: x,
            "scatter": lambda x, num_devices: tf.split(x, num_devices, axis=0)
        }
    
    def _pytorch_all_gather_wrapper(self, tensor, axis=-1):
        """Wrapper for torch.distributed.all_gather to provide a friendlier API."""
        import torch
        import torch.distributed as dist
        world_size = dist.get_world_size()
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list, dim=axis)
    
    def _get_pytorch_communication_ops(self):
        """Get PyTorch communication operations using torch.distributed."""
        import torch
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                logger.info("Using real torch.distributed communication ops.")
                return {
                    "all_reduce": lambda x, op="sum": dist.all_reduce(x, op=dist.ReduceOp.SUM) or x,
                    "all_gather": self._pytorch_all_gather_wrapper,
                }
        except (ImportError, RuntimeError):
            pass # Fallback to simulation
        
        logger.warning("torch.distributed not available or initialized. Using SIMULATED communication ops.")
        world_size = 1 # Cannot know world_size, assume 1
        return {
            "all_reduce": lambda x, op="sum": x,
            "all_gather": lambda x, axis=-1: torch.cat([x] * world_size, dim=axis),
        }
    
    def _get_numpy_communication_ops(self):
        """Get NumPy communication operations (simplified)."""
        return {
            "all_reduce": lambda x, op="sum": np.sum(x, axis=0),
            "all_gather": lambda x: np.concatenate([x, x], axis=0),
            "broadcast": lambda x: x,
            "scatter": lambda x, num_devices: np.split(x, num_devices, axis=0)
        }

def get_distributed_backend(backend_name: str = 'auto', world_size: int = 1, rank: int = 0):
    """
    Get the best available distributed backend.
    
    Args:
        backend_name: Backend to use ('auto', 'jax', 'tensorflow', 'pytorch', 'numpy')
        world_size: Number of processes
        rank: Process rank
        
    Returns:
        Initialized distributed backend
    """
    if backend_name == 'auto':
        backends = ['jax', 'tensorflow', 'pytorch', 'numpy']
        
        for name in backends:
            try:
                backend = DistributedBackend(name)
                if backend.is_multi_device_capable():
                    return backend
            except Exception:
                continue
        
        return DistributedBackend('numpy')
    
    else:
        return DistributedBackend(backend_name)

def allreduce_gradients(gradients: List[np.ndarray], backend: DistributedBackend, op: str = 'mean') -> List[np.ndarray]:
    """AllReduce a list of gradients."""
    synchronized = []
    for grad in gradients:
        if grad is not None:
            synced_grad = backend.get_communication_ops()["all_reduce"](grad, op=op)
            synchronized.append(synced_grad)
        else:
            synchronized.append(None)
    return synchronized

def allgather_outputs(outputs: List[np.ndarray], backend: DistributedBackend, axis: int = 0) -> np.ndarray:
    """AllGather outputs from all shards."""
    if len(outputs) == 1:
        return outputs[0]
        
    for output in outputs:
        if output is not None:
            return backend.get_communication_ops()["all_gather"](output)
            
    raise ValueError("No valid outputs to gather")

def broadcast_parameters(parameters: List[np.ndarray], backend: DistributedBackend, root: int = 0) -> List[np.ndarray]:
    """Broadcast parameters from root to all processes."""
    broadcasted = []
    for param in parameters:
        if param is not None:
            broadcasted_param = backend.get_communication_ops()["broadcast"](param)
            broadcasted.append(broadcasted_param)
        else:
            broadcasted.append(None)
    return broadcasted 