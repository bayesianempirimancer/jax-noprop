"""
Gradient utility classes for neural networks.

This module provides utility classes for computing gradients, Hessians, and function values
of neural networks without the parameter scope issues that affect Flax module wrappers.

These utility classes are NOT Flax modules - they are regular Python classes that operate
on existing neural networks and their parameters.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple


def handle_broadcasting(z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Handle broadcasting for inputs with different batch shapes.
    
    Args:
        z: Current state [batch_shape_z, z_dim] or [z_dim]
        x: Conditional input [batch_shape_x, x_dim] or [x_dim]
        t: Time values [batch_shape_t] or scalar (optional)
        
    Returns:
        Tuple of (z_broadcasted, x_broadcasted, t_broadcasted)
    """
    # Collect all input arrays (excluding None)
    arrays = [z]
    if x is not None:
        arrays.append(x)
    
    # Check if all inputs are scalars or 1D (no batching needed)
    if all(arr.ndim <= 1 for arr in arrays):
        return z, x, t
    
    # Use JAX's broadcast_shapes to determine the common broadcasted shape
    try:
        # Get batch shapes of all arrays (assumes vector inputs)
        shapes = [arr.shape[:-1] for arr in arrays]
        broadcast_batch_shape = jnp.broadcast_shapes(*shapes)
    except ValueError as e:
        raise ValueError(f"Incompatible shapes for broadcasting: {[arr.shape for arr in arrays]}") from e
    
    # Broadcast all arrays to the common shape
    z_broadcasted = jnp.broadcast_to(z, broadcast_batch_shape + z.shape[-1:])
    x_broadcasted = jnp.broadcast_to(x, broadcast_batch_shape + x.shape[-1:])
    if t is not None:
        t_broadcasted = jnp.broadcast_to(t, broadcast_batch_shape)
    else:
        t_broadcasted = t
    
    return z_broadcasted, x_broadcasted, t_broadcasted


def reduction_fn(output: jnp.ndarray, reduction_method: str) -> jnp.ndarray:
    """
    Helper function to reduce neural network output to scalar values.
    
    Args:
        output: Neural network output [batch_size, output_dim] or [output_dim]
        reduction_method: How to reduce the output ("sum", "mean", "norm", "first", "logsumexp")
        
    Returns:
        Reduced scalar output [batch_size] or scalar
    """
    if reduction_method == "sum":
        return jnp.sum(output, axis=-1)
    elif reduction_method == "mean":
        return jnp.mean(output, axis=-1)
    elif reduction_method == "norm":
        return jnp.linalg.norm(output, axis=-1)
    elif reduction_method == "first":
        return output[..., 0]
    elif reduction_method == "logsumexp":
        return jax.scipy.special.logsumexp(output, axis=-1, keepdims=False)
    else:
        raise ValueError(f"Unknown reduction_method: {reduction_method}. "
                        f"Supported methods: 'sum', 'mean', 'norm', 'first', 'logsumexp'")


class GradNetUtility:
    """
    Utility class for computing gradients of ResNet functions.
    
    This class computes gradients of any conditional ResNet function with respect to
    the input z. The ResNet output is reduced to a scalar using the specified reduction
    method before computing the gradient.
    
    Args:
        resnet: The underlying conditional ResNet module or factory function
        reduction_method: How to reduce ResNet output to scalar ("sum", "mean", "norm", "first", "logsumexp")
    """
    
    def __init__(self, resnet, reduction_method="sum"):
        self.resnet = resnet
        self.reduction_method = reduction_method
    
    def __call__(self, params, z: jnp.ndarray, x: Optional[jnp.ndarray] = None, 
                 t: Optional[jnp.ndarray] = None, training: bool = True, rngs=None) -> jnp.ndarray:
        """
        Compute gradients of the ResNet function with respect to z.
        
        Args:
            params: Flax parameters for the ResNet
            z: Current state [batch_size, z_dim] or [z_dim]
            x: Conditional input [batch_size, x_dim] or [x_dim] (optional)
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Gradient ∇_z f(z,x,t) [batch_size, z_dim]
        """
        
        # Handle broadcasting
        z_broadcasted, x_broadcasted, t_broadcasted = handle_broadcasting(z, x, t)
        
        # Check if resnet is a factory function or a module instance
        if callable(self.resnet) and not hasattr(self.resnet, 'apply'):
            # It's a factory function, call it directly
            def resnet_call(z_input, x_input, t_input):
                return self.resnet(z_input, x_input, t_input, training=training)
        else:
            # It's a Flax module, use apply
            def resnet_call(z_input, x_input, t_input):
                return self.resnet.apply(params, z_input, x_input, t_input, training=training, rngs=rngs)
        
        # Check if all inputs are scalars or 1D (no batching needed)
        if all(arr.ndim <= 1 for arr in [z_broadcasted, x_broadcasted] if x_broadcasted is not None):
            # All inputs are scalars or 1D, no batching needed
            def scalar_fn(z_input):
                resnet_output = resnet_call(z_input, x_broadcasted, t_broadcasted)
                return reduction_fn(resnet_output, self.reduction_method)
            return jax.grad(scalar_fn)(z_broadcasted)
        
        def single_grad_fn(z_single, x_single, t_single):
            def scalar_fn(z_input):
                resnet_output = resnet_call(z_input, x_single, t_single)
                return reduction_fn(resnet_output, self.reduction_method)
            
            return jax.grad(scalar_fn)(z_single)
        
        # Vmap over the batch dimension
        vmapped_grad_fn = jax.vmap(single_grad_fn, in_axes=(0, 0, 0))
        return vmapped_grad_fn(z_broadcasted, x_broadcasted, t_broadcasted)


class ValueAndGradNetUtility:
    """
    Utility class for computing both function value and gradient.
    
    This class returns both the scalar function value f(z,x,t) and its gradient ∇_z f(z,x,t).
    Useful when you need both the function value and its derivative.
    
    Args:
        resnet: The underlying conditional ResNet module
        reduction_method: How to reduce ResNet output to scalar ("sum", "mean", "norm", "first", "logsumexp")
    """
    
    def __init__(self, resnet, reduction_method="sum"):
        self.resnet = resnet
        self.reduction_method = reduction_method
    
    def __call__(self, params, z: jnp.ndarray, x: Optional[jnp.ndarray] = None, 
                 t: Optional[jnp.ndarray] = None, training: bool = True, rngs=None) -> tuple:
        """
        Compute both function value and gradient.
        
        Args:
            params: Flax parameters for the ResNet
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim] (optional)
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Tuple of (value, gradient) where:
            - value: Scalar function value f(z,x,t) [batch_size]
            - gradient: Gradient ∇_z f(z,x,t) [batch_size, z_dim]
        """
        
        def single_value_grad_fn(z_single, x_single, t_single):
            def scalar_fn(z_input):
                resnet_output = self.resnet.apply(params, z_input, x_single, t_single, training=training, rngs=rngs)
                return reduction_fn(resnet_output, self.reduction_method)
            
            return jax.value_and_grad(scalar_fn)(z_single)
        
        # Vmap over the batch dimension
        vmapped_fn = jax.vmap(single_value_grad_fn, in_axes=(0, 0, 0))
        return vmapped_fn(z, x, t)


class HessNetUtility:
    """
    Utility class for computing Hessian matrices.
    
    This class computes the Hessian matrix ∇²_z f(z,x,t) of the ResNet function.
    Useful for second-order methods and geometric analysis.
    
    Args:
        resnet: The underlying conditional ResNet module
        reduction_method: How to reduce ResNet output to scalar ("sum", "mean", "norm", "first", "logsumexp")
    """
    
    def __init__(self, resnet, reduction_method="sum"):
        self.resnet = resnet
        self.reduction_method = reduction_method
    
    def __call__(self, params, z: jnp.ndarray, x: Optional[jnp.ndarray] = None, 
                 t: Optional[jnp.ndarray] = None, training: bool = True, rngs=None) -> jnp.ndarray:
        """
        Compute Hessian matrix.
        
        Args:
            params: Flax parameters for the ResNet
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim] (optional)
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Hessian matrix ∇²_z f(z,x,t) [batch_size, z_dim, z_dim]
        """
        
        def single_hess_fn(z_single, x_single, t_single):
            def scalar_fn(z_input):
                resnet_output = self.resnet.apply(params, z_input, x_single, t_single, training=training, rngs=rngs)
                return reduction_fn(resnet_output, self.reduction_method)
            
            return jax.hessian(scalar_fn)(z_single)
        
        # Vmap over the batch dimension
        vmapped_hess_fn = jax.vmap(single_hess_fn, in_axes=(0, 0, 0))
        return vmapped_hess_fn(z, x, t)


class GradAndHessNetUtility:
    """
    Utility class for computing both gradient and Hessian.
    
    This class returns both the gradient ∇_z f(z,x,t) and Hessian ∇²_z f(z,x,t).
    Useful for second-order methods that need both first and second derivatives.
    
    Args:
        resnet: The underlying conditional ResNet module
        reduction_method: How to reduce ResNet output to scalar ("sum", "mean", "norm", "first", "logsumexp")
    """
    
    def __init__(self, resnet, reduction_method="sum"):
        self.resnet = resnet
        self.reduction_method = reduction_method
    
    def __call__(self, params, z: jnp.ndarray, x: Optional[jnp.ndarray] = None, 
                 t: Optional[jnp.ndarray] = None, training: bool = True, rngs=None) -> tuple:
        """
        Compute both gradient and Hessian.
        
        Args:
            params: Flax parameters for the ResNet
            z: Current state [batch_size, z_dim] or [z_dim]
            x: Conditional input [batch_size, x_dim] or [x_dim] (optional)
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Tuple of (gradient, hessian) where:
            - gradient: Gradient ∇_z f(z,x,t) [batch_size, z_dim]
            - hessian: Hessian ∇²_z f(z,x,t) [batch_size, z_dim, z_dim]
        """
        
        # Handle broadcasting
        z_broadcasted, x_broadcasted, t_broadcasted = handle_broadcasting(z, x, t)
        
        # Check if resnet is a factory function or a module instance
        if callable(self.resnet) and not hasattr(self.resnet, 'apply'):
            # It's a factory function, call it directly
            def resnet_call(z_input, x_input, t_input):
                return self.resnet(z_input, x_input, t_input, training=training)
        else:
            # It's a Flax module, use apply
            def resnet_call(z_input, x_input, t_input):
                return self.resnet.apply(params, z_input, x_input, t_input, training=training, rngs=rngs)
        
        # Check if all inputs are scalars or 1D (no batching needed)
        if all(arr.ndim <= 1 for arr in [z_broadcasted, x_broadcasted] if x_broadcasted is not None):
            # All inputs are scalars or 1D, no batching needed
            def scalar_fn(z_input):
                resnet_output = resnet_call(z_input, x_broadcasted, t_broadcasted)
                return reduction_fn(resnet_output, self.reduction_method)
            
            gradient = jax.grad(scalar_fn)(z_broadcasted)
            hessian = jax.jacfwd(jax.grad(scalar_fn))(z_broadcasted)
            return gradient, hessian
        
        def single_grad_hess_fn(z_single, x_single, t_single):
            def scalar_fn(z_input):
                resnet_output = resnet_call(z_input, x_single, t_single)
                return reduction_fn(resnet_output, self.reduction_method)
            
            # More efficient: compute gradient first, then Hessian from gradient using jacfwd
            gradient = jax.grad(scalar_fn)(z_single)
            hessian = jax.jacfwd(jax.grad(scalar_fn))(z_single)
            return gradient, hessian
        
        # Vmap over the batch dimension
        vmapped_fn = jax.vmap(single_grad_hess_fn, in_axes=(0, 0, 0))
        return vmapped_fn(z_broadcasted, x_broadcasted, t_broadcasted)


class ValueAndGradAndHessNetUtility:
    """
    Utility class for computing function value, gradient, and Hessian.
    
    This class returns the complete information: function value f(z,x,t), 
    gradient ∇_z f(z,x,t), and Hessian ∇²_z f(z,x,t).
    Useful for comprehensive analysis and debugging.
    
    Args:
        resnet: The underlying conditional ResNet module
        reduction_method: How to reduce ResNet output to scalar ("sum", "mean", "norm", "first", "logsumexp")
    """
    
    def __init__(self, resnet, reduction_method="sum"):
        self.resnet = resnet
        self.reduction_method = reduction_method
    
    def __call__(self, params, z: jnp.ndarray, x: Optional[jnp.ndarray] = None, 
                 t: Optional[jnp.ndarray] = None, training: bool = True, rngs=None) -> tuple:
        """
        Compute function value, gradient, and Hessian.
        
        Args:
            params: Flax parameters for the ResNet
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim] (optional)
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Tuple of (value, gradient, hessian) where:
            - value: Scalar function value f(z,x,t) [batch_size]
            - gradient: Gradient ∇_z f(z,x,t) [batch_size, z_dim]
            - hessian: Hessian ∇²_z f(z,x,t) [batch_size, z_dim, z_dim]
        """
        
        def single_value_grad_hess_fn(z_single, x_single, t_single):
            def scalar_fn(z_input):
                resnet_output = self.resnet.apply(params, z_input, x_single, t_single, training=training, rngs=rngs)
                return reduction_fn(resnet_output, self.reduction_method)
            
            # More efficient: compute value and gradient together, then Hessian from gradient using jacfwd
            value, gradient = jax.value_and_grad(scalar_fn)(z_single)
            hessian = jax.jacfwd(jax.grad(scalar_fn))(z_single)
            return value, gradient, hessian
        
        # Vmap over the batch dimension
        vmapped_fn = jax.vmap(single_value_grad_hess_fn, in_axes=(0, 0, 0))
        return vmapped_fn(z, x, t)

