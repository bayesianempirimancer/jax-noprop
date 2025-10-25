"""
Utilities for activation functions.

This module provides utilities for converting activation function strings
to callable activation functions, which is commonly needed across different layers.
"""

from typing import Callable, Union
import flax.linen as nn
import jax.numpy as jnp

def softplusglu(x: jnp.ndarray) -> jnp.ndarray:
    """
    Softplus activation function.
    """
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([nn.softplus(x1)*nn.sigmoid(x2), nn.softplus(x2)*nn.sigmoid(x1)], axis=-1)

def switanh(x: jnp.ndarray) -> jnp.ndarray:
    """
    SwiTanh activation function.
    
    SwiTanh combines Swish activation with a gating mechanism:
    - Splits input into two equal parts
    - Applies Swish to first part: swish(x1) = x1 * sigmoid(x1)
    - Applies tanh to second part: tanh(x2)
    - Returns element-wise product: swish(x1) * tanh(x2)
    
    Args:
        x: Input tensor of shape (..., 2*dim)
        
    Returns:
        Output tensor of shape (..., dim)
        
    Note:
        Input dimension must be even (divisible by 2)
    """
    # Split input into two equal parts along the last dimension
    x1, x2 = jnp.split(x, 2, axis=-1)
    
    # Apply Swish to first part and tanh to second part
#    return jnp.concatenate([nn.swish(x1) * nn.tanh(x2), nn.swish(x2) * nn.tanh(x1)], axis=-1)
    return nn.swish(x1) * nn.tanh(x2)

def swiglu(x: jnp.ndarray) -> jnp.ndarray:
    """
    SwiGLU activation function.
    
    SwiGLU combines Swish activation with a gating mechanism:
    - Splits input into two equal parts
    - Applies Swish to first part: swish(x1) = x1 * sigmoid(x1)
    - Applies sigmoid to second part: sigmoid(x2)
    - Returns element-wise product: swish(x1) * sigmoid(x2)
    
    Args:
        x: Input tensor of shape (..., 2*dim)
        
    Returns:
        Output tensor of shape (..., dim)
        
    Note:
        Input dimension must be even (divisible by 2)
    """
    # Split input into two equal parts along the last dimension
    x1, x2 = jnp.split(x, 2, axis=-1)
    
    # Apply Swish to first part and sigmoid to second part
#    return jnp.concatenate([nn.swish(x1) * nn.sigmoid(x2), nn.swish(x2) * nn.sigmoid(x1)], axis=-1)
    return nn.swish(x1) * nn.sigmoid(x2)

def get_activation_function(activation: Union[str, Callable]) -> Callable:
    """
    Convert activation string to callable activation function.
    
    Args:
        activation: Either a string name of the activation function or a callable
        
    Returns:
        Callable activation function
        
    Raises:
        ValueError: If activation string is not recognized
    """
    if callable(activation):
        return activation
    
    activation_map = {
        "linear": lambda x: x,
        "none": lambda x: x,
        "identity": lambda x: x,
        "relu": nn.relu,
        "leaky_relu": nn.leaky_relu,
        "gelu": nn.gelu,
        "swish": nn.swish,
        "switanh": switanh,
        "swiglu": swiglu,
        "softplusglu": softplusglu,
        "sigmoid": nn.sigmoid,
        "tanh": nn.tanh,
        "softplus": nn.softplus,
        "softmax": nn.softmax,
        "log_softmax": nn.log_softmax,
        "elu": nn.elu,
        "selu": nn.selu,
    }
    
    if activation.lower() in activation_map:
        return activation_map[activation.lower()]
    else:
        raise ValueError(f"Unknown activation function: {activation}. "
                        f"Supported activations: {list(activation_map.keys())}")


def get_activation_name(activation: Union[str, Callable]) -> str:
    """
    Get the string name of an activation function.
    
    Args:
        activation: Either a string name or a callable activation function
        
    Returns:
        String name of the activation function
    """
    if isinstance(activation, str):
        return activation.lower()
    
    # Map callable functions back to their names
    callable_map = {
        nn.relu: "relu",
        nn.leaky_relu: "leaky_relu", 
        nn.gelu: "gelu",
        nn.swish: "swish",
        switanh: "switanh",
        swiglu: "swiglu",
        softplusglu: "softplusglu",
        nn.sigmoid: "sigmoid",
        nn.tanh: "tanh",
        nn.softplus: "softplus",
        nn.softmax: "softmax",
        nn.log_softmax: "log_softmax",
        nn.elu: "elu",
        nn.selu: "selu",
    }
    
    # Check for lambda functions
    if activation == (lambda x: x):
        return "linear"
    
    return callable_map.get(activation, "unknown")
