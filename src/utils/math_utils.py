"""
Mathematical utility functions for the nnef-dists project.
"""

import jax.numpy as jnp
from typing import Optional, Union, Tuple


def logsumexp(
    x: jnp.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> jnp.ndarray:
    """
    Safe log-sum-exp function that prevents overflow.
    
    Computes log(sum(exp(x))) in a numerically stable way by subtracting
    the maximum value before exponentiating.
    
    Args:
        x: Input array
        axis: Axis or axes along which to compute the log-sum-exp.
              If None, compute over all axes.
        keepdims: If True, keep the reduced dimensions with size 1.
        b: Optional array to add to x before computing log-sum-exp.
           If provided, computes log(sum(exp(x + b))).
    
    Returns:
        Log-sum-exp of x along the specified axis/axes.
    
    Examples:
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> logsumexp(x)  # log(exp(1) + exp(2) + exp(3))
        >>> logsumexp(x, axis=0)  # log-sum-exp along axis 0
        >>> logsumexp(x, axis=0, keepdims=True)  # Keep reduced dimensions
    """    
    # Find the maximum value along the specified axis
    x_max = jnp.max(x, axis=axis, keepdims=True)
    
    # Subtract the maximum to prevent overflow
    x_shifted = x - x_max
    
    # Compute log-sum-exp
    result = x_max + jnp.log(jnp.sum(jnp.exp(x_shifted), axis=axis, keepdims=keepdims))
    
    return result


def logsumexp_safe(
    x: jnp.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    b: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Alias for logsumexp for backward compatibility.
    """
    return logsumexp(x, axis=axis, keepdims=keepdims, b=b)
