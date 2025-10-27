"""Decoder factory functions for VAE models."""

from typing import Tuple
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import flax.linen as nn
from src.utils.activation_utils import get_activation_function
from src.layers.mlp import MLP


@dataclass(frozen=True)
class Config:
    """Configuration for decoder networks."""
    model_name: str = "decoder"
    model_type: str = "mlp"  # Options: "mlp", "resnet", "identity"
    config: dict = field(default_factory=lambda: {
        "decoder_type": "logits",  # Options: "logits", "normal"
        "input_dim": "NA",  # Will be set from main config if not specified
        "output_dim": "NA",
        "hidden_dims": (64, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.1,
    })

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


def create_decoder(config_dict: dict, input_dim: int = None, output_dim: int = None, input_shape: tuple = None):
    """Create decoder based on config dictionary and optional parameters.
    
    Args:
        config_dict: Dictionary containing decoder configuration
        input_dim: Input dimension (optional, uses config if not provided)
        output_dim: Output dimension (optional, uses config if not provided)
        input_shape: Input shape tuple (optional, overrides input_dim if provided)
    
    Returns:
        Instantiated decoder model
    """
    # Build final config dict with provided parameters
    # Convert FrozenDict to regular dict if needed
    if hasattr(config_dict, 'unfreeze'):
        final_config = config_dict.unfreeze()
    else:
        final_config = dict(config_dict)
    
    # Handle input_dim and input_shape
    if input_shape is not None:
        # Convert input_shape to flattened dimension
        actual_input_dim = 1
        for dim in input_shape:
            actual_input_dim *= dim
        final_config["input_dim"] = actual_input_dim
    elif input_dim is not None:
        final_config["input_dim"] = input_dim
    elif final_config.get("input_dim") == "NA":
        raise ValueError("input_dim must be provided either as parameter or in config_dict")
    
    # Handle output_dim
    if output_dim is not None:
        final_config["output_dim"] = output_dim
    elif final_config.get("output_dim") == "NA":
        raise ValueError("output_dim must be provided either as parameter or in config_dict")
    
    # Use model_type from config to determine the decoder class
    model_type = final_config.get("model_type", "mlp")
    if model_type not in ["mlp", "resnet", "identity"]:
        raise ValueError(f"Unknown model_type: {model_type}. Available: ['mlp', 'resnet', 'identity']")
    
    # Get the appropriate decoder class based on model_type
    DecoderClass = get_decoder_class(model_type)
    
    # Create and return the decoder instance with final config
    return DecoderClass(
        config=final_config,
        input_dim=final_config["input_dim"],
        output_dim=final_config["output_dim"],
        input_shape=input_shape  # Pass shape for structured input
    )


class MLPDecoder(nn.Module):
    """MLP decoder that returns single tensor of output_shape."""
    config: dict
    input_dim: int
    output_dim: int
    input_shape: tuple = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Flatten input if it has structured shape
        if self.input_shape is not None:
            # Reshape from (batch, *input_shape) to (batch, input_dim)
            batch_shape = x.shape[:-len(self.input_shape)]
            x = x.reshape(batch_shape + (self.input_dim,))
        
        # Use MLP for the main network
        mlp = MLP(
            out_features=self.output_dim,
            hidden_features=self.config["hidden_dims"],
            act_layer=get_activation_function(self.config["activation"]),
            dropout_rate=self.config["dropout_rate"] if training else 0.0,
            bias=True
        )
        
        # Apply MLP and return raw output
        # Output transformation (identity/softmax/linear) is handled at usage level
        return mlp(x, x.shape[-1])


class ResNetDecoder(nn.Module):
    """ResNet decoder that returns single tensor of output_shape."""
    config: dict
    input_dim: int
    output_dim: int
    input_shape: tuple = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Flatten input if it has structured shape
        if self.input_shape is not None:
            # Reshape from (batch, *input_shape) to (batch, input_dim)
            batch_shape = x.shape[:-len(self.input_shape)]
            x = x.reshape(batch_shape + (self.input_dim,))
        
        # Use MLP for the main network (same as MLPDecoder for now)
        mlp = MLP(
            out_features=self.output_dim,
            hidden_features=self.config["hidden_dims"],
            act_layer=get_activation_function(self.config["activation"]),
            dropout_rate=self.config["dropout_rate"] if training else 0.0,
            bias=True
        )
        
        # Apply MLP and return raw output
        # Output transformation (identity/softmax/linear) is handled at usage level
        return mlp(x, x.shape[-1])


class IdentityDecoder(nn.Module):
    """Deterministic identity decoder that returns single tensor."""
    config: dict
    input_dim: int
    output_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Identity decoder just returns input unchanged
        # Output transformation (identity/softmax/linear) is handled at usage level
        return x


