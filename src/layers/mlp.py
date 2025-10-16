from typing import Callable, Tuple

import jax.numpy as jnp
import flax.linen as nn


class Mlp(nn.Module):
    """
    Simple MLP (Multi-Layer Perceptron) for Vision Transformers.

    This module implements a two-layer MLP with configurable hidden size,
    activation function, and dropout.

    Attributes:
        hidden_features (int | None): Number of hidden features. If None, set to in_features.
        out_features (int | None): Number of output features. If None, set to in_features.
        act_layer (Callable): Activation function to use.
        dropout_rate (float): Dropout rate.
        bias (bool): Whether to use bias in linear layers.

    Args:
        in_features (int): Number of input features.
    """

    hidden_features: int | None = None
    out_features: int | None = None
    act_layer: Callable = nn.gelu
    dropout_rate: float = 0.0
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, in_features: int) -> jnp.ndarray:
        """
        Forward pass of the MLP.

        Args:
            x (jnp.ndarray): Input tensor of shape (B, ..., in_features).
            in_features (int): Number of input features.

        Returns:
            jnp.ndarray: Output tensor of shape (B, ..., out_features).
        """
        out_features = self.out_features or in_features
        hidden_features = self.hidden_features or in_features

        x = nn.Dense(hidden_features, use_bias=self.bias, name='fc1')(x)
        x = self.act_layer(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=False)
        x = nn.Dense(out_features, use_bias=self.bias, name='fc2')(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=False)

        return x

class MLP(nn.Module):
    """
    Simple MLP with multiple hidden layers.

    Attributes:
        hidden_features (int | None): Number of hidden features. If None, set to out_features.
        out_features (int): Number of output features. 
        act_layer (Callable): Activation function to use.
        dropout_rate (float): Dropout rate.
        bias (bool): Whether to use bias in linear layers.

    Args:
        in_features (int): Number of input features.
    """

    out_features: int 
    hidden_features: Tuple[int, ...] | None = None
    act_layer: Callable = nn.swish
    dropout_rate: float = 0.0
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, in_features: int) -> jnp.ndarray:
        """
        Forward pass of the MLP.

        Args:
            x (jnp.ndarray): Input tensor of shape (B, ..., in_features).
            in_features (int): Number of input features.

        Returns:
            jnp.ndarray: Output tensor of shape (B, ..., out_features).
        """
        out_features = self.out_features or in_features
        hidden_features = self.hidden_features or in_features

        if hidden_features is not None:
            for feat in hidden_features:
                x = nn.Dense(feat, use_bias=self.bias)(x)
                x = self.act_layer(x)
                x = nn.Dropout(self.dropout_rate)(x, deterministic=False)
        else:
            x = nn.Dense(out_features, use_bias=self.bias)(x)
            x = self.act_layer(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=False)

        x = nn.Dense(out_features, use_bias=self.bias)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=False)

        return x