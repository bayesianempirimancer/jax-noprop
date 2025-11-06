"""Standard Variational Autoencoder (VAE) model."""

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from flax.core import FrozenDict
from functools import partial
from typing import Tuple, Optional
from dataclasses import dataclass, field

from src.configs.base_config import BaseConfig
from src.models.vae.encoders import create_encoder
from src.models.vae.decoders import create_decoder


@dataclass(frozen=True)
class VAEConfig(BaseConfig):
    """Configuration for standard VAE model using hierarchically organized frozen dict.
    
    Note: 
    - input_shape can be multi-dimensional (e.g., (28, 28) for images, (48, 2) for sequences)
    - latent_shape should be a 1D vector (e.g., (32,) for 32-dimensional latent space)
    - output_shape typically matches input_shape
    """
    # BaseConfig fields
    model_name: str = "vae_network"
    
    main: FrozenDict = field(default_factory=lambda: FrozenDict({
        "input_shape": "NA",  # Will be set based on data (can be multi-dimensional)
        "latent_shape": "NA",  # Will be set based on desired latent dimension (should be 1D vector, e.g., (32,))
        "output_shape": "NA",  # Will be set based on input_shape (usually same)
        "recon_loss_type": "mse",  # Options: "mse", "bce", "cross_entropy"
        "recon_weight": 1.0,  # Weight for reconstruction loss
        "kl_weight": 1.0,  # Weight for KL divergence loss
    }))
    
    encoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "mlp_normal",  # Options: "mlp", "mlp_normal", "resnet", "resnet_normal", "identity", "linear"
        "encoder_type": "normal",  # Options: "normal"
        "input_shape": "NA",  # Will be set from main config if not specified
        "latent_shape": "NA",  # Will be set from main config if not specified
        "hidden_dims": (64, 64, 64),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "mlp",  # Options: "mlp", "resnet", "identity", "linear"
        "decoder_type": "none",  # Options: "linear", "softmax", "none"
        "latent_shape": "NA",  # Will be set from main config if not specified
        "output_shape": "NA",  # Will be set from main config if not specified
        "hidden_dims": (64, 64, 64),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))


class VAE(nn.Module):
    """Standard Variational Autoencoder using @nn.compact methods."""
    config: VAEConfig
    
    def setup(self):
        """Initialize encoder and decoder based on config.
        
        Note: input_shape can be multi-dimensional (e.g., (28, 28) for images),
        but latent_shape should be a 1D vector (e.g., (32,)).
        """
        # Get shapes from main config
        input_shape = self.config.main["input_shape"]
        latent_shape = self.config.main["latent_shape"]
        output_shape = self.config.main["output_shape"]
        
        # Create encoder - shapes passed as kwargs will override config "NA" values
        # Encoder handles flattening multi-dimensional input to vector internally
        self.encoder = create_encoder(
            self.config.encoder,
            input_shape=input_shape,
            latent_shape=latent_shape
        )
        
        # Create decoder - shapes passed as kwargs will override config "NA" values
        # Decoder takes vector z and outputs to output_shape (can be multi-dimensional)
        self.decoder = create_decoder(
            self.config.decoder,
            latent_shape=latent_shape,
            output_shape=output_shape
        )
    
    @nn.compact
    def encode(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.encoder(x, training=training)
    
    @nn.compact
    def decode(self, z: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        return self.decoder(z, training=training)
    
    @partial(jax.jit, static_argnums=(0, 4))
    def loss(self, params: dict, x: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> Tuple[jnp.ndarray, dict]:
        """
        Compute VAE loss (ELBO).
        
        Args:
            params: Model parameters dictionary
            x: Input data [batch, *input_shape] (can be multi-dimensional)
            key: Random key for sampling
            training: Whether in training mode
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Split key for sampling
        key, sample_key = jr.split(key, 2)
        
        # Encode to get latent distribution parameters
        # Note: x can have non-trivial shape, but encoder will flatten internally
        mu, logvar = self.apply(params, x, method='encode', training=training, rngs={'dropout': key})
        
        std = jnp.exp(0.5 * logvar)
        z = mu + std * jr.normal(sample_key, mu.shape)
        
        x_recon = self.apply(params, z, method='decode', training=training, rngs={'dropout': key})
        
        recon_loss_type = self.config.main.get("recon_loss_type", "mse")
        if recon_loss_type == "mse":
            recon_loss = jnp.mean((x - x_recon) ** 2)
        elif recon_loss_type == "bce":
            # Binary cross-entropy (for binary data)
            recon_loss = jnp.mean(-x * jnp.log(x_recon + 1e-8) - (1 - x) * jnp.log(1 - x_recon + 1e-8))
        elif recon_loss_type == "cross_entropy":
            # Categorical cross-entropy (for multi-class data)
            recon_loss = jnp.mean(-jnp.sum(x * jnp.log(x_recon + 1e-8), axis=-1))
        else:
            raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")
        
        # Compute KL divergence: KL(q(z|x) || N(0, I))
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * jnp.mean(1 + logvar - jnp.square(mu) - jnp.exp(logvar))
                
        # Get loss weights
        recon_weight = self.config.main.get("recon_weight", 1.0)
        kl_weight = self.config.main.get("kl_weight", 1.0)
        
        # Compute total loss
        total_loss = recon_weight * recon_loss + kl_weight * kl_loss
        
        metrics = {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss
        }
        
        return total_loss, metrics
    
    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> jnp.ndarray:
        """
        Forward pass for initialization.
        
        Args:
            x: Input data [batch, *input_shape]
            key: Random key
            training: Whether in training mode
            
        Returns:
            Dummy output for initialization
        """
        # Initialize model components by calling them
        batch_shape = x.shape[:-len(self.config.main["input_shape"])]
        dummy_z = jnp.zeros(batch_shape + self.config.main["latent_shape"])
        
        # Call encode and decode to initialize parameters
        self.encode(x, training)
        self.decode(dummy_z, training)
        
        return jnp.zeros(batch_shape + self.config.main["output_shape"])

