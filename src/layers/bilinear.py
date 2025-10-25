"""
Bilinear layers and blocks for neural networks.

This module provides bilinear layer implementations and residual blocks
that use bilinear transformations.
"""

import jax.numpy as jnp
import flax.linen as nn


class BilinearLayer(nn.Module):
    """Bilinear layer that computes x^T W y for input tensors x and y."""
    features: int
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Initialize bilinear weight matrix
        Wxy = self.param('W', nn.initializers.lecun_normal(), (x.shape[-1], y.shape[-1], self.features))
        Wx = self.param('Wx', nn.initializers.lecun_normal(), (x.shape[-1], self.features))
        Wy = self.param('Wy', nn.initializers.lecun_normal(), (y.shape[-1], self.features))
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        
        x_broadcast = x
        y_broadcast = y
            
        bilinear_term = jnp.einsum('...bi,ijc,...bj->...bc', x, Wxy, y)
        linear_x = x @ Wx
        linear_y = y @ Wy

        if self.dropout_rate > 0:
            bilinear_term = nn.Dropout(rate=self.dropout_rate)(bilinear_term, deterministic=not training)
            linear_x = nn.Dropout(rate=self.dropout_rate)(linear_x, deterministic=not training)
            linear_y = nn.Dropout(rate=self.dropout_rate)(linear_y, deterministic=not training)

        return bilinear_term + linear_x + linear_y + bias


class BilinearProjectionLayer(nn.Module):
    """Bilinear projection layer that computes (Wx x) * (Wy y)."""
    features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Project both inputs to the same feature space with LeCun normal initialization

        kernel_init = lambda key, shape, dtype: nn.initializers.lecun_normal()(key, shape, dtype) / jnp.sqrt(x.shape[-1])
        x_proj = nn.Dense(self.features, kernel_init=kernel_init, use_bias=False, name='x_proj')(x)
        y_proj = nn.Dense(self.features, kernel_init=nn.initializers.lecun_normal(), use_bias=False, name='y_proj')(y)
        # Element-wise multiplication (linear terms handled by bias from nn.Dense)
        x_linear = nn.Dense(self.features, kernel_init=nn.initializers.lecun_normal(), name='x_linear')(x)
        y_linear = nn.Dense(self.features, kernel_init=nn.initializers.lecun_normal(), use_bias=False, name='y_linear')(y)
        return x_proj * y_proj + x_linear + y_linear



