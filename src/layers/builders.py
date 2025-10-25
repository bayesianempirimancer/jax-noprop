from typing import Callable

import jax.numpy as jnp
import flax.linen as nn

from .attention import Attention, LinearAttention, CrossAttention, LinearCrossAttention
from .mamba import Mamba2Mixer, Mamba2VisionMixer, MambaVisionMixer
from .mlp import Mlp
from .norm import RMSNormGated
from .swiglu import SwiGLU

def swiglu(x: jnp.ndarray) -> jnp.ndarray:
    x1,x2 = jnp.split(x, 2, axis=-1)
    return nn.silu(x1) * x2

def get_act(act: str) -> Callable:
    """Get activation function from string name.
    
    Args:
        act: String name of activation function
        
    Returns:
        Activation function callable
        
    Raises:
        NotImplementedError: If activation function is not supported
    """
    match act.lower():
        # Standard activations
        case "swiglu":
            return swiglu
        case "relu":
            return nn.relu
        case "gelu":
            return nn.gelu
        case "silu":
            return nn.silu
        case "swish":
            return nn.swish
        case "tanh":
            return nn.tanh
        case "sigmoid":
            return nn.sigmoid
        case "softmax":
            return nn.softmax
        case "log_softmax":
            return nn.log_softmax
        case "elu":
            return nn.elu
        case "leaky_relu":
            return nn.leaky_relu
        case "selu":
            return nn.selu
        case "glu":
            return nn.glu
        # Advanced activations
        case "celu":
            return nn.celu
        case "softplus":
            return nn.softplus
        case "log_sigmoid":
            return nn.log_sigmoid
        # Linear (no activation)
        case "linear" | "none" | "identity":
            return lambda x: x
        case _:
            raise NotImplementedError(
                f"Unknown activation function: `{act}`. "
                f"Supported activations: relu, gelu, silu, swish, tanh, sigmoid, "
                f"softmax, log_softmax, elu, leaky_relu, selu, glu, "
                f"celu, softplus, log_sigmoid, linear, none, identity"
            )


def get_norm(norm: str) -> nn.Module:
    match norm:
        case "batchnorm":
            return nn.BatchNorm
        case "layernorm":
            return nn.LayerNorm
        case "rmsnormgated":
            return RMSNormGated
        case _:
            raise NotImplementedError(f"Unknown norm. Got: `{norm}`.")


def get_module(module: str) -> nn.Module:
    match module:
        case "attention":
            return Attention
        case "linearattention":
            return LinearAttention
        case "crossattention":
            return CrossAttention
        case "linearcrossattention":
            return LinearCrossAttention
        case "mamba2mixer":
            return Mamba2Mixer
        case "mamba2visionmixer":
            return Mamba2VisionMixer
        case "mambavisionmixer":
            return MambaVisionMixer
        case "mlp":
            return Mlp
        case "swiglu":
            return SwiGLU
        case _:
            raise NotImplementedError(f"Unknown module. Got: `{module}`.")
