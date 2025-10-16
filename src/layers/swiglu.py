from typing import Callable

import jax.numpy as jnp
import flax.linen as nn


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) FFN block from Google Brain.

    This module implements the SwiGLU activation function, which is a variant of the GLU
    (Gated Linear Unit) that uses the Swish activation function.

    Args:
        in_features (int): Number of input features.
        hidden_features (int | None, optional): Number of hidden features. If None, set to in_features. Defaults to None.
        out_features (int | None, optional): Number of output features. If None, set to in_features. Defaults to None.
        bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
    """

    hidden_features: int | None = None
    out_features: int | None = None
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, in_features: int):
        """
        Forward pass of the SwiGLU module.

        Args:
            x (jnp.ndarray): Input tensor.
            in_features (int): Number of input features.

        Returns:
            jnp.ndarray: Output tensor after applying SwiGLU.
        """
        out_features = self.out_features or in_features
        hidden_features = self.hidden_features or in_features

        x12 = nn.Dense(hidden_features, use_bias=self.bias)(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = nn.silu(x1) * x2

        return nn.Dense(out_features, use_bias=self.bias)(hidden)
