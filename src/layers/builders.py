from typing import Callable

import flax.linen as nn

from .attention import Attention, LinearAttention
from .mamba import Mamba2Mixer, Mamba2VisionMixer, MambaVisionMixer
from .mlp import Mlp
from .norm import RMSNormGated
from .swiglu import SwiGLU


def get_act(act: str) -> Callable:
    match act:
        case "gelu":
            return nn.gelu
        case "silu":
            return nn.silu
        case "relu":
            return nn.relu
        case _:
            raise NotImplementedError(f"Unknown activation. Got: `{act}`.")


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
