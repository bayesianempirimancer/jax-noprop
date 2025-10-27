"""Encoder factory functions for VAE models."""

from typing import Tuple
from dataclasses import dataclass, field
import jax.numpy as jnp
import flax.linen as nn
from src.utils.activation_utils import get_activation_function
from src.layers.mlp import MLP


@dataclass(frozen=True)
class Config:
    """Configuration for encoder networks."""
    model_name: str = "encoder"
    model_type: str = "mlp_normal"  # Options: "mlp", "mlp_normal", "resnet", "resnet_normal", "identity"
    config: dict = field(default_factory=lambda: {
        "encoder_type": "normal",  # Options: "normal", "deterministic"
        "input_dim": "NA",  # Will be set from main config if not specified
        "latent_dim": "NA",
        "hidden_dims": (64, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.1,
    })


def get_encoder_class(encoder_type: str):
    """Get encoder class by type string."""
    ENCODER_CLASSES = {
        'mlp': MLPEncoder,
        'mlp_normal': MLPNormalEncoder,
        'resnet': ResNetEncoder,
        'resnet_normal': ResNetNormalEncoder,
        'identity': IdentityEncoder,
    }
    
    if encoder_type not in ENCODER_CLASSES:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Available: {list(ENCODER_CLASSES.keys())}")
    
    return ENCODER_CLASSES[encoder_type]


def create_encoder(config_dict: dict, input_dim: int = None, latent_dim: int = None, latent_shape: tuple = None):
    """Create encoder based on config dictionary and optional parameters.
    
    Args:
        config_dict: Dictionary containing encoder configuration
        input_dim: Input dimension (optional, uses config if not provided)
        latent_dim: Latent dimension (optional, uses config if not provided)
        latent_shape: Latent shape tuple (optional, overrides latent_dim if provided)
    
    Returns:
        Instantiated encoder model
    """
    # Build final config dict with provided parameters
    # Convert FrozenDict to regular dict if needed
    if hasattr(config_dict, 'unfreeze'):
        final_config = config_dict.unfreeze()
    else:
        final_config = dict(config_dict)
    
    # Handle input_dim
    if input_dim is not None:
        final_config["input_dim"] = input_dim
    elif final_config.get("input_dim") == "NA":
        raise ValueError("input_dim must be provided either as parameter or in config_dict")
    
    # Handle latent_dim and latent_shape
    if latent_shape is not None:
        # Convert latent_shape to flattened dimension
        actual_latent_dim = 1
        for dim in latent_shape:
            actual_latent_dim *= dim
        final_config["latent_dim"] = actual_latent_dim
    elif latent_dim is not None:
        final_config["latent_dim"] = latent_dim
    elif final_config.get("latent_dim") == "NA":
        raise ValueError("latent_dim must be provided either as parameter or in config_dict")
    
    # Use model_type from config to determine the encoder class
    model_type = final_config.get("model_type", "mlp_normal")
    if model_type not in ["mlp", "mlp_normal", "resnet", "resnet_normal", "identity"]:
        raise ValueError(f"Unknown model_type: {model_type}. Available: ['mlp', 'mlp_normal', 'resnet', 'resnet_normal', 'identity']")
    
    # Get the appropriate encoder class based on model_type
    EncoderClass = get_encoder_class(model_type)
    
    # Create and return the encoder instance with final config
    return EncoderClass(
        config=final_config,
        input_dim=final_config["input_dim"],
        latent_dim=final_config["latent_dim"],
        latent_shape=latent_shape  # Pass shape for structured output
    )


class MLPEncoder(nn.Module):
    """Deterministic MLP encoder that returns single tensor."""
    config: dict
    input_dim: int
    latent_dim: int
    latent_shape: tuple = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Deterministic encoder outputs latent_dim
        mlp = MLP(
            out_features=self.latent_dim,
            hidden_features=self.config["hidden_dims"],
            act_layer=get_activation_function(self.config["activation"]),
            dropout_rate=self.config["dropout_rate"] if training else 0.0,
            bias=True
        )
        
        # Apply MLP and reshape to structured output if latent_shape is provided
        output = mlp(x, x.shape[-1])
        
        if self.latent_shape is not None:
            # Reshape from (batch, latent_dim) to (batch, *latent_shape)
            batch_shape = output.shape[:-1]
            output = output.reshape(batch_shape + self.latent_shape)
        
        return output


class MLPNormalEncoder(nn.Module):
    """Probabilistic MLP encoder that returns (mu, logvar) tuple by definition."""
    config: dict
    input_dim: int
    latent_dim: int
    latent_shape: tuple = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Probabilistic encoder outputs 2*latent_dim and splits into mu, logvar
        mlp = MLP(
            out_features=2 * self.latent_dim,
            hidden_features=self.config["hidden_dims"],
            act_layer=get_activation_function(self.config["activation"]),
            dropout_rate=self.config["dropout_rate"] if training else 0.0,
            bias=True
        )
        
        # Apply MLP and split output into mu and logvar
        output = mlp(x, x.shape[-1])
        mu, logvar = jnp.split(output, 2, axis=-1)
        
        if self.latent_shape is not None:
            # Reshape from (batch, latent_dim) to (batch, *latent_shape)
            batch_shape = mu.shape[:-1]
            mu = mu.reshape(batch_shape + self.latent_shape)
            logvar = logvar.reshape(batch_shape + self.latent_shape)
        
        return mu, logvar


class ResNetEncoder(nn.Module):
    """Deterministic ResNet encoder that returns single tensor."""
    config: dict
    input_dim: int
    latent_dim: int
    latent_shape: tuple = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Deterministic encoder outputs latent_dim
        # Using MLP for now (same as MLPEncoder)
        mlp = MLP(
            out_features=self.latent_dim,
            hidden_features=self.config["hidden_dims"],
            act_layer=get_activation_function(self.config["activation"]),
            dropout_rate=self.config["dropout_rate"] if training else 0.0,
            bias=True
        )
        
        # Apply MLP and reshape to structured output if latent_shape is provided
        output = mlp(x, x.shape[-1])
        
        if self.latent_shape is not None:
            # Reshape from (batch, latent_dim) to (batch, *latent_shape)
            batch_shape = output.shape[:-1]
            output = output.reshape(batch_shape + self.latent_shape)
        
        return output


class ResNetNormalEncoder(nn.Module):
    """Probabilistic ResNet encoder that returns (mu, logvar) tuple by definition."""
    config: dict
    input_dim: int
    latent_dim: int
    latent_shape: tuple = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Probabilistic encoder outputs 2*latent_dim and splits into mu, logvar
        # Using MLP for now (same as MLPNormalEncoder)
        mlp = MLP(
            out_features=2 * self.latent_dim,
            hidden_features=self.config["hidden_dims"],
            act_layer=get_activation_function(self.config["activation"]),
            dropout_rate=self.config["dropout_rate"] if training else 0.0,
            bias=True
        )
        
        # Apply MLP and split output into mu and logvar
        output = mlp(x, x.shape[-1])
        mu, logvar = jnp.split(output, 2, axis=-1)
        
        if self.latent_shape is not None:
            # Reshape from (batch, latent_dim) to (batch, *latent_shape)
            batch_shape = mu.shape[:-1]
            mu = mu.reshape(batch_shape + self.latent_shape)
            logvar = logvar.reshape(batch_shape + self.latent_shape)
        
        return mu, logvar


class IdentityEncoder(nn.Module):
    """Deterministic identity encoder that returns single tensor."""
    config: dict
    input_dim: int
    latent_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Identity encoder just returns the input unchanged
        return x


