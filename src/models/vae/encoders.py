"""Encoder factory functions for VAE models."""

from typing import Tuple, Dict, Any
from dataclasses import dataclass, field
from functools import cached_property

import jax.numpy as jnp
import flax.linen as nn
from src.utils.activation_utils import get_activation_function
from src.layers.mlp import MLP


########  CONFIG CLASS   ###########

@dataclass(frozen=True)
class Config:
    """Configuration for encoder networks."""
    model_name: str = "encoder"
    config: dict = field(default_factory=lambda: {
        "model_type": "mlp",  # Options: "mlp", "mlp_normal", "resnet", "resnet_normal", "identity"
        "encoder_type": "normal",  # Options: "normal", "deterministic"
        "input_shape": "NA",  # Will be set from main config if not specified
        "latent_shape": "NA",
        "hidden_dims": (64, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.1,
    })

########  ENCODER CLASSES AVAILABLE   ###########

def get_encoder_class(encoder_type: str):
    """Get encoder class by type string."""
    ENCODER_CLASSES = {
        'mlp': MLPEncoder,
        'mlp_normal': MLPNormalEncoder,
        'resnet': ResNetEncoder,
        'resnet_normal': ResNetNormalEncoder,
        'identity': IdentityEncoder,
        'linear': LinearEncoder,
    }
    
    if encoder_type not in ENCODER_CLASSES:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Available: {list(ENCODER_CLASSES.keys())}")
    
    return ENCODER_CLASSES[encoder_type]


########  FACTORY FUNCTION   ###########

def create_encoder(config_dict: Dict[str, Any], **kwargs) -> nn.Module:
    """Create an encoder model using the homogenized approach."""
    from src.models.vae.encoders import get_encoder_class
    
    input_shape = kwargs.get('input_shape')
    latent_shape = kwargs.get('latent_shape')

    # Convert config_dict to regular dict if needed
    if hasattr(config_dict, 'unfreeze'):
        final_config = config_dict.unfreeze()
    else:
        final_config = dict(config_dict)
    
    # Handle input_shape and input_dim
    if input_shape is not None:
        final_config["input_shape"] = input_shape
    elif final_config.get("input_shape") == "NA":
        raise ValueError("input_shape must be provided either as parameter or in config_dict")
    
    if latent_shape is not None:
        final_config["latent_shape"] = latent_shape
    elif final_config.get("latent_shape") == "NA":
        raise ValueError("latent_shape must be provided either as parameter or in config_dict")    
    # Get the appropriate encoder class based on model_type
    model_type = final_config.get("model_type", "mlp_normal")
    EncoderClass = get_encoder_class(model_type)
    
    # Create and return the encoder instance with homogenized approach
    return EncoderClass(
        config=final_config,
        input_shape=input_shape, 
        latent_shape=latent_shape
    )


########  ENCODER MODELS  ###########

class MLPEncoder(nn.Module):
    """Deterministic MLP encoder that returns single tensor."""
    config: dict
    input_shape: Tuple[int, ...]
    latent_shape: Tuple[int, ...]

    @cached_property
    def input_dim(self) -> int:
        total_dim = 1
        for dim in self.input_shape:
            total_dim *= dim
        return total_dim

    @cached_property
    def latent_dim(self) -> int:
        total_dim = 1
        for dim in self.latent_shape:
            total_dim *= dim
        return total_dim
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Flatten input if it has structured shape
        if self.input_shape is not None:
            # Reshape from (batch, *input_shape) to (batch, input_dim)
            batch_shape = x.shape[:-len(self.input_shape)]
            x = x.reshape(batch_shape + (self.input_dim,))
        
        # Use MLP for the main network
        mlp = MLP(
            out_features=self.latent_dim,
            hidden_features=self.config["hidden_dims"],
            act_layer=get_activation_function(self.config["activation"]),
            dropout_rate=self.config["dropout_rate"] if training else 0.0,
            bias=True
        )
        
        # Apply MLP and return raw output
        output = mlp(x, x.shape[-1])
        return output.reshape(batch_shape + self.latent_shape)


class MLPNormalEncoder(nn.Module):
    """Normal MLP encoder that returns (mu, logvar) tuple."""
    config: dict
    input_shape: Tuple[int, ...]
    latent_shape: Tuple[int, ...]

    @cached_property
    def input_dim(self) -> int:
        total_dim = 1
        for dim in self.input_shape:
            total_dim *= dim
        return total_dim

    @cached_property
    def latent_dim(self) -> int:
        total_dim = 1
        for dim in self.latent_shape:
            total_dim *= dim
        return total_dim
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Flatten input if it has structured shape
        if self.input_shape is not None:
            # Reshape from (batch, *input_shape) to (batch, input_dim)
            batch_shape = x.shape[:-len(self.input_shape)]
            x = x.reshape(batch_shape + (self.input_dim,))
        
        # Use MLP for the main network
        mlp = MLP(
            out_features=self.latent_dim * 2,  # mu and logvar
            hidden_features=self.config["hidden_dims"],
            act_layer=get_activation_function(self.config["activation"]),
            dropout_rate=self.config["dropout_rate"] if training else 0.0,
            bias=True
        )
        
        # Apply MLP and split into mu and logvar
        output = mlp(x, x.shape[-1])
        output = output.reshape(batch_shape + (self.latent_dim * 2,))
        
        mu = output[..., :self.latent_dim].reshape(batch_shape + self.latent_shape)
        logvar = output[..., self.latent_dim:].reshape(batch_shape + self.latent_shape)
        
        return mu, logvar


class ResNetEncoder(nn.Module):
    """Deterministic ResNet encoder that returns single tensor."""
    config: dict
    input_shape: Tuple[int, ...]
    latent_shape: Tuple[int, ...]

    @cached_property
    def input_dim(self) -> int:
        total_dim = 1
        for dim in self.input_shape:
            total_dim *= dim
        return total_dim

    @cached_property
    def latent_dim(self) -> int:
        total_dim = 1
        for dim in self.latent_shape:
            total_dim *= dim
        return total_dim
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Flatten input if it has structured shape
        if self.input_shape is not None:
            # Reshape from (batch, *input_shape) to (batch, input_dim)
            batch_shape = x.shape[:-len(self.input_shape)]
            x = x.reshape(batch_shape + (self.input_dim,))
        
        # Use MLP for the main network (simplified ResNet for now)
        mlp = MLP(
            out_features=self.latent_dim,
            hidden_features=self.config["hidden_dims"],
            act_layer=get_activation_function(self.config["activation"]),
            dropout_rate=self.config["dropout_rate"] if training else 0.0,
            bias=True
        )
        
        # Apply MLP and return raw output
        output = mlp(x, x.shape[-1])
        return output.reshape(batch_shape + self.latent_shape)


class ResNetNormalEncoder(nn.Module):
    """Normal ResNet encoder that returns (mu, logvar) tuple."""
    config: dict
    input_shape: Tuple[int, ...]
    latent_shape: Tuple[int, ...]

    @cached_property
    def input_dim(self) -> int:
        total_dim = 1
        for dim in self.input_shape:
            total_dim *= dim
        return total_dim

    @cached_property
    def latent_dim(self) -> int:
        total_dim = 1
        for dim in self.latent_shape:
            total_dim *= dim
        return total_dim
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Flatten input if it has structured shape
        if self.input_shape is not None:
            # Reshape from (batch, *input_shape) to (batch, input_dim)
            batch_shape = x.shape[:-len(self.input_shape)]
            x = x.reshape(batch_shape + (self.input_dim,))
        
        # Use MLP for the main network (simplified ResNet for now)
        mlp = MLP(
            out_features=self.latent_dim * 2,  # mu and logvar
            hidden_features=self.config["hidden_dims"],
            act_layer=get_activation_function(self.config["activation"]),
            dropout_rate=self.config["dropout_rate"] if training else 0.0,
            bias=True
        )
        
        # Apply MLP and split into mu and logvar
        output = mlp(x, x.shape[-1])
        output = output.reshape(batch_shape + (self.latent_dim * 2,))
        
        mu = output[..., :self.latent_dim].reshape(batch_shape + self.latent_shape)
        logvar = output[..., self.latent_dim:].reshape(batch_shape + self.latent_shape)
        
        return mu, logvar


class IdentityEncoder(nn.Module):
    """Identity encoder that returns input unchanged."""
    config: dict
    input_shape: Tuple[int, ...]
    latent_shape: Tuple[int, ...]

    @cached_property
    def input_dim(self) -> int:
        total_dim = 1
        for dim in self.input_shape:
            total_dim *= dim
        return total_dim

    @cached_property
    def latent_dim(self) -> int:
        total_dim = 1
        for dim in self.latent_shape:
            total_dim *= dim
        return total_dim
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Identity function - return input unchanged
        return x


class LinearEncoder(nn.Module):
    """Linear encoder that applies a single Dense layer."""
    config: dict
    input_shape: Tuple[int, ...]
    latent_shape: Tuple[int, ...]

    @cached_property
    def input_dim(self) -> int:
        total_dim = 1
        for dim in self.input_shape:
            total_dim *= dim
        return total_dim

    @cached_property
    def latent_dim(self) -> int:
        total_dim = 1
        for dim in self.latent_shape:
            total_dim *= dim
        return total_dim
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Flatten input
        batch_shape = x.shape[:-len(self.input_shape)]
        x_flat = x.reshape(-1, self.input_dim)
        
        # Apply linear transformation
        output = nn.Dense(self.latent_dim)(x_flat)
        
        # Reshape to latent shape
        return output.reshape(batch_shape + self.latent_shape)