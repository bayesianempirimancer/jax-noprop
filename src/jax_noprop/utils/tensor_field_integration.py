"""
ODE integration methods for NoProp continuous-time variants.

This module provides various numerical integration methods for solving
neural ordinary differential equations (ODEs) used in NoProp-CT and NoProp-FM.
"""

from typing import Any, Callable, Dict, Tuple, Optional
import jax
import jax.numpy as jnp
from .ode_integration import integrate_ode as integrate_ode_vec 


# =============================================================================
# MAIN INTEGRATION FUNCTION AND DEFAULTS
# =============================================================================

def integrate_tensor_ode(
    tensor_field: Callable,
    tensor_shape: Tuple[int, ...],
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    method: str = "euler",
    output_type: str = "end_point"
) -> jnp.ndarray:
    """Integrate an ODE on a tensor field using the specified method.
    
    This function integrates the ODE dz/dt = f(z, x, t) on a tensor field, using the 
    specified numerical method with scan-based implementation for better JIT compilation.

    This is a wrapper that handles reshaping of the tensor field for use with 
    standard ODE integration functions.
    
    Args:
        tensor_field: Function that computes dz/dt = f(z, x, t)
        tensor_shape: shape of z
        params: Model parameters
        z0: Initial state [batch_size, state_dim]
        x: Input data [batch_size, ...]
        time_span: Tuple of (start_time, end_time)
        num_steps: Number of integration steps
        method: Integration method ("euler", "heun", "rk4", "adaptive")
        output_type: Type of output ("end_point" or "trajectory")
        
    Returns:
        If output_type="end_point": Final state [batch_size, state_dim]
        If output_type="trajectory": Full trajectory [num_steps+1, batch_size, state_dim]
    """
    # Use scan-based JIT-compiled integration functions for better performance
    batch_shape = z0.shape[:-len(tensor_shape)]
    z_vec_dim = 1
    for dim in tensor_shape:
        z_vec_dim *= dim
    z0 = z0.reshape(batch_shape + (z_vec_dim,))
    def vector_field(params, z, x, t):
        return tensor_field(params, z.reshape(batch_shape + tensor_shape), x, t).reshape(batch_shape + (z_vec_dim,))
    
    result = integrate_ode_vec(vector_field, params, z0, x, time_span, num_steps, method, output_type)
    if output_type == 'end_point':
        result = result.reshape(batch_shape + tensor_shape)

    elif output_type == 'trajectory':
        result = result.reshape((num_steps+1,) + batch_shape + tensor_shape)
    else:
        raise ValueError(f"Unknown output_type: {output_type}. Must be 'end_point' or 'trajectory'")

    return result
