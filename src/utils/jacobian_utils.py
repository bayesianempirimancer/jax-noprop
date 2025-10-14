"""
Jacobian computation utilities for neural ODEs and flow matching.

This module provides optimized functions for computing Jacobians and their traces
in the context of neural ODEs and flow matching, with a focus on memory efficiency.
"""

from typing import Any, Dict, Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0,))
def trace_jacobian(
    apply_fn: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray:
    """Simple Jacobian trace computation for vector inputs.
    
    This implementation expects z to be a batch of vectors (2D array with shape [batch_size, vector_dim]).
    It computes the Jacobian trace directly without any reshaping.
    
    Args:
        apply_fn: The model apply function
        params: Model parameters
        z: Current state [batch_size, vector_dim] (2D array of vectors)
        x: Input data [batch_size, x_dim]
        t: Current time [batch_size]
        
    Returns:
        Total trace of Jacobian [batch_size]
    """
    def flow_field_single(z_single, x_single, t_single):
        """Flow field for a single batch element."""
        return apply_fn(params, z_single[None, :], x_single[None, :], t_single[None])[0]
    
    def jacobian_trace_single(z_single, x_single, t_single):
        """Compute Jacobian trace for a single batch element."""
        jacobian = jax.jacfwd(flow_field_single, argnums=0)(z_single, x_single, t_single)
        return jnp.trace(jacobian)
    
    # Vectorize over batch dimension to compute Jacobian trace for each element
    trace = jax.vmap(jacobian_trace_single)(z, x, t)
    
    return trace


@partial(jax.jit, static_argnums=(0, 1))
def trace_jacobian_tensor(
    apply_fn: Callable,
    z_shape: Tuple[int, ...],
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray:
    """Optimized Jacobian trace computation using flattened approach.
    
    This implementation flattens the input tensor, computes the Jacobian in flattened space,
    and then computes the trace. This approach is cleaner and more readable than working
    with multi-dimensional Jacobians directly.
    
    Args:
        apply_fn: The model apply function
        z_shape: Shape of z tensor (excluding batch dimension)
        params: Model parameters
        z: Current state [batch_size, ...] (can be multi-dimensional)
        x: Input data [batch_size, x_dim]
        t: Current time [batch_size]
        
    Returns:
        Total trace of Jacobian [batch_size] - computed in flattened space
    """
    def flattened_flow_field_single(z_single_flattened, x_single, t_single):
        """Flow field that works in flattened space."""
        # Unflatten z_single_flattened to original shape
        z_single = z_single_flattened.reshape(z_shape)
        # Apply the model
        output = apply_fn(params, z_single[None, :], x_single[None, :], t_single[None])[0]
        # Flatten the output
        return output.reshape(-1)
    
    def jacobian_trace_single(z_single_flattened, x_single, t_single):
        """Compute Jacobian trace for a single batch element in flattened space."""
        jacobian = jax.jacfwd(flattened_flow_field_single, argnums=0)(z_single_flattened, x_single, t_single)
        return jnp.trace(jacobian)
    
    # Flatten z for the computation
    z_flattened = z.reshape(z.shape[0], -1)
    
    # Vectorize over batch dimension to compute Jacobian trace for each element
    trace = jax.vmap(jacobian_trace_single)(z_flattened, x, t)
    
    return trace

@partial(jax.jit, static_argnums=(0,))
def jacobian_diagonal(
    apply_fn: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray:
    """Compute only the diagonal elements of the Jacobian matrix.
    
    This is more memory efficient than computing the full Jacobian when only
    the diagonal elements are needed (e.g., for trace computation).
    
    Args:
        apply_fn: The model apply function
        params: Model parameters
        z: Current state [batch_size, target_dim]
        x: Input data [batch_size, x_dim]
        t: Current time [batch_size]
        
    Returns:
        Diagonal elements of Jacobian [batch_size, target_dim]
    """
    def flow_field_single(z_single, x_single, t_single):
        """Flow field for a single batch element."""
        return apply_fn(params, z_single[None, :], x_single[None, :], t_single[None])[0]
    
    def jacobian_diagonal_single(z_single, x_single, t_single):
        """Compute Jacobian diagonal for a single batch element."""
        # Use jax.jacfwd to get the Jacobian, then extract diagonal
        jacobian = jax.jacfwd(flow_field_single, argnums=0)(z_single, x_single, t_single)
        return jnp.diag(jacobian)
    
    # Vectorize over batch dimension
    diagonal = jax.vmap(jacobian_diagonal_single)(z, x, t)
    
    return diagonal

def divergence(
    apply_fn: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray:
    """Compute the divergence of the vector field.
    
    The divergence is the trace of the Jacobian matrix, which is useful
    for normalizing flows and log-determinant computations.
    
    Args:
        apply_fn: The model apply function
        params: Model parameters
        z: Current state [batch_size, target_dim]
        x: Input data [batch_size, x_dim]
        t: Current time [batch_size]
        
    Returns:
        Divergence [batch_size]
    """
    return trace_jacobian(apply_fn, params, z, x, t)

@partial(jax.jit, static_argnums=(0,))
def grad_potential(
    apply_fn: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray:
    """Compute the gradient of the potential function.
    
    This function computes the gradient of a potential function with respect to z.
    The potential function is typically defined as the negative log-likelihood or
    energy function in the context of flow-based models.
    
    Args:
        apply_fn: The model apply function (should return a scalar potential)
        params: Model parameters
        z: Current state [batch_size, target_dim]
        x: Input data [batch_size, x_dim]
        t: Current time [batch_size]
        
    Returns:
        Gradient of potential [batch_size, target_dim]
    """
    def potential_single(z_single, x_single, t_single):
        """Potential function for a single batch element."""
        output = apply_fn(params, z_single[None, :], x_single[None, :], t_single[None])[0]
        # If output is a vector, take the sum to make it scalar
        if output.ndim > 0:
            raise ValueError("grad_potential can only be used with scalar functions")
        return output
    
    def grad_potential_single(z_single, x_single, t_single):
        """Compute gradient of potential for a single batch element."""
        return jax.grad(potential_single, argnums=0)(z_single, x_single, t_single)
    
    # Vectorize over batch dimension
    grad = jax.vmap(grad_potential_single)(z, x, t)
    
    return grad

@partial(jax.jit, static_argnums=(0,))
def compute_log_det_jacobian(
    apply_fn: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray:
    """Compute the log determinant of the Jacobian matrix using QR factorization.
    
    This is useful for normalizing flows where the log-determinant
    is needed for the change of variables formula. Uses QR factorization
    for numerical stability.
    
    Args:
        apply_fn: The model apply function
        params: Model parameters
        z: Current state [batch_size, target_dim]
        x: Input data [batch_size, x_dim]
        t: Current time [batch_size]
        
    Returns:
        Log determinant of Jacobian [batch_size]
    """
    def flow_field_single(z_single, x_single, t_single):
        """Flow field for a single batch element."""
        return apply_fn(params, z_single[None, :], x_single[None, :], t_single[None])[0]
    
    def log_det_jacobian_single(z_single, x_single, t_single):
        """Compute log determinant of Jacobian for a single batch element."""
        jacobian = jax.jacfwd(flow_field_single, argnums=0)(z_single, x_single, t_single)
        
        # QR factorization: A = QR, det(A) = det(Q) * det(R) = ±1 * prod(diag(R))
        # Since Q is orthogonal, det(Q) = ±1, so log|det(A)| = sum(log|diag(R)|)
        Q, R = jnp.linalg.qr(jacobian)
        log_det = jnp.sum(jnp.log(jnp.abs(jnp.diag(R))))
        
        return log_det
    
    # Vectorize over batch dimension
    log_det = jax.vmap(log_det_jacobian_single)(z, x, t)
    
    return log_det
