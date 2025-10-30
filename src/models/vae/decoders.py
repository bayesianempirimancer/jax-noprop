"""Decoder factory functions for VAE models."""

from typing import Tuple, Dict, Any
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Any

import jax.numpy as jnp
import jax
import flax.linen as nn
from src.utils.activation_utils import get_activation_function
from src.layers.mlp import MLP


########  CONFIG CLASS   ###########

@dataclass(frozen=True)
class Config:
    """Configuration for decoder networks."""
    model_name: str = "decoder"
    config: dict = field(default_factory=lambda: {
        "model_type": "mlp", # Options: "mlp", "resnet", "identity"
        "decoder_type": "linear",  # Options: "linear", "softmax", "none"
        "input_shape": "NA",  # Will be set from main config if not specified
        "output_shape": "NA",
        "hidden_dims": (64, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.1,
    })

########  DECODER CLASSES AVAILABLE   ###########

def apply_output_transformation(output: jnp.ndarray, decoder_type: str) -> jnp.ndarray:
    """Apply output transformation based on decoder type."""
    if decoder_type == "linear":
        # Apply additional Dense layer
        return nn.Dense(output.shape[-1])(output)
    elif decoder_type == "softmax":
        # Apply softmax along the last axis for classification outputs
        return jax.nn.softmax(output, axis=-1)
    elif decoder_type == "none":
        return output  # No transformation
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")

def get_decoder_class(decoder_type: str):
    """Get decoder class by type string."""
    DECODER_CLASSES = {
        'mlp': MLPDecoder,
        'resnet': ResNetDecoder,
        'identity': IdentityDecoder,
    }
    
    if decoder_type not in DECODER_CLASSES:
        raise ValueError(f"Unknown decoder type: {decoder_type}. Available: {list(DECODER_CLASSES.keys())}")
    
    return DECODER_CLASSES[decoder_type]


########  FACTORY FUNCTION   ###########

def create_decoder(config_dict: Dict[str, Any], **kwargs) -> nn.Module:
    """Create a decoder model using the homogenized approach."""
    from src.models.vae.decoders import get_decoder_class
    
    latent_shape = kwargs.get('latent_shape')
    output_shape = kwargs.get('output_shape')

    # Convert config_dict to regular dict if needed
    if hasattr(config_dict, 'unfreeze'):
        final_config = config_dict.unfreeze()
    else:
        final_config = dict(config_dict)
    
    # Handle latent_shape
    if latent_shape is not None:
        final_config["latent_shape"] = latent_shape
    elif final_config.get("latent_shape") == "NA":
        raise ValueError("latent_shape must be provided either as parameter or in config_dict")
    else:
        latent_shape = final_config.get("latent_shape")
        
    if output_shape is not None:
        final_config["output_shape"] = output_shape
    elif final_config.get("output_shape") == "NA":
        raise ValueError("output_shape must be provided either as parameter or in config_dict")    
    else:
        output_shape = final_config.get("output_shape")
        
    model_type = final_config.get("model_type", "mlp")
    DecoderClass = get_decoder_class(model_type)
    
    # Create and return the decoder instance with homogenized approach
    return DecoderClass(
        config=final_config,
        latent_shape=latent_shape, 
        output_shape=output_shape
    )


########  DECODER MODELS  ###########

class MLPDecoder(nn.Module):
    """MLP decoder that returns single tensor of output_shape."""
    config: dict
    latent_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]

    @cached_property
    def latent_dim(self) -> int:
        total_dim = 1
        for dim in self.latent_shape:
            total_dim *= dim
        return total_dim

    @cached_property
    def output_dim(self) -> int:
        total_dim = 1
        for dim in self.output_shape:
            total_dim *= dim
        return total_dim
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Flatten input if it has structured shape
        if self.latent_shape is not None:
            # Reshape from (batch, *latent_shape) to (batch, latent_dim)
            batch_shape = x.shape[:-len(self.latent_shape)]
            x = x.reshape(batch_shape + (self.latent_dim,))
        
        # Use MLP for the main network
        mlp = MLP(
            out_features=self.output_dim,
            hidden_features=self.config["hidden_dims"],
            act_layer=get_activation_function(self.config["activation"]),
            dropout_rate=self.config["dropout_rate"] if training else 0.0,
            bias=True
        )
        
        # Apply MLP and get raw output
        output = mlp(x, x.shape[-1])
        output = output.reshape(batch_shape + self.output_shape)
        
        # Apply output transformation based on decoder type
        decoder_type = self.config["decoder_type"]
        return apply_output_transformation(output, decoder_type)


class ResNetDecoder(nn.Module):
    """ResNet decoder that returns single tensor of output_shape."""
    config: dict
    latent_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]

    @cached_property
    def latent_dim(self) -> int:
        total_dim = 1
        for dim in self.latent_shape:
            total_dim *= dim
        return total_dim

    @cached_property
    def output_dim(self) -> int:
        total_dim = 1
        for dim in self.output_shape:
            total_dim *= dim
        return total_dim

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Flatten input if it has structured shape
        if self.latent_shape is not None:
            # Reshape from (batch, *latent_shape) to (batch, latent_dim)
            batch_shape = x.shape[:-len(self.latent_shape)]
            x = x.reshape(batch_shape + (self.latent_dim,))
        
        # Use MLP for the main network (same as MLPDecoder for now)
        mlp = MLP(
            out_features=self.output_dim,
            hidden_features=self.config["hidden_dims"],
            act_layer=get_activation_function(self.config["activation"]),
            dropout_rate=self.config["dropout_rate"] if training else 0.0,
            bias=True
        )
        
        # Apply MLP and get raw output
        output = mlp(x, x.shape[-1])
        output = output.reshape(batch_shape + self.output_shape)
        
        # Apply output transformation based on decoder type
        decoder_type = self.config["decoder_type"]
        return apply_output_transformation(output, decoder_type)


class IdentityDecoder(nn.Module):
    """Deterministic identity decoder that returns single tensor."""
    config: dict
    latent_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # For identity behavior: if decoder_type is 'none', return x as-is (reshape if needed)
        decoder_type = self.config.get("decoder_type", "linear")
        if decoder_type == "none":
            # Determine desired batch shape and reshape if total elements match
            batch_ndims = x.ndim - len(self.latent_shape) if self.latent_shape is not None else x.ndim - len(self.output_shape)
            batch_ndims = max(batch_ndims, 1)
            batch_shape = x.shape[:batch_ndims]
            # If already in desired output shape, return directly
            if x.shape[-len(self.output_shape):] == self.output_shape:
                return x
            # Attempt reshape to output shape if sizes match
            total_current = 1
            for d in x.shape[batch_ndims:]:
                total_current *= d
            total_target = 1
            for d in self.output_shape:
                total_target *= d
            if total_current == total_target:
                return x.reshape(batch_shape + self.output_shape)
            # Fallback to linear projection only if shapes incompatible
        
        # Linear mapping fallback (e.g., when decoder_type != 'none' or shapes incompatible)
        output = nn.Dense(self.output_shape[0])(x)
        return apply_output_transformation(output, decoder_type)


