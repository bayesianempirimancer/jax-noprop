from typing import Optional

import jax.numpy as jnp
import flax.linen as nn
from jax import lax


class L2Norm(nn.Module):
    """L2 normalization module that projects vectors onto unit sphere."""
    eps: float = 1e-8
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Normalize input to unit length (L2 norm = 1).
        
        Args:
            x: Input array of shape [..., features]
            
        Returns:
            Normalized array with same shape, where each vector has L2 norm = 1
        """
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + self.eps)


class RMSNormGated(nn.Module):
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray, z: Optional[jnp.ndarray] = None):
        if z is not None:
            x *= z

        y = x.astype(jnp.float32)
        norm = y * lax.rsqrt(jnp.mean(y * y, -1, keepdims=True) + self.eps)

        w = self.param('w', nn.initializers.ones, (x.shape[-1],))
        return w * norm.astype(x.dtype)


class LayerScale(nn.Module):
    """Layer scale module for scaling the output of a layer."""

    init_values: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply layer scaling to the input.

        Args:
            x (jnp.ndarray): Input array.

        Returns:
            jnp.ndarray: Scaled output.
        """
        gamma = self.param('gamma', 
                          lambda rng, shape: self.init_values * nn.initializers.ones(rng, shape), 
                          (x.shape[-1],))
        return x * gamma
