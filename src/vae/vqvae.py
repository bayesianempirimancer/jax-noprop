"""Vector Quantized Variational Autoencoder (VQ-VAE) model for discrete tokenization."""

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from flax.core import FrozenDict
from functools import partial
from typing import Tuple, Optional
from dataclasses import dataclass, field

from src.configs.base_config import BaseConfig
from src.vae.encoders import create_encoder
from src.vae.decoders import create_decoder


@dataclass(frozen=True)
class VQVAEConfig(BaseConfig):
    """Configuration for Vector Quantized VAE model using hierarchically organized frozen dict.
    
    VQ-VAE learns discrete latent representations by quantizing encoder outputs
    using a codebook of embedding vectors. This enables tokenization of continuous data.
    
    Note: 
    - input_shape can be multi-dimensional (e.g., (28, 28) for images, (12, 2) for sequences)
    - codebook_size: number of discrete tokens in the codebook
    - embedding_dim: dimension of each codebook vector (should match encoder output dimension)
    """
    # BaseConfig fields
    model_name: str = "vqvae_network"
    
    main: FrozenDict = field(default_factory=lambda: FrozenDict({
        "input_shape": "NA",  # Will be set based on data (can be multi-dimensional)
        "codebook_size": 512,  # Number of discrete tokens in codebook
        "embedding_dim": "NA",  # Dimension of codebook vectors (usually matches encoder output)
        "output_shape": "NA",  # Will be set based on input_shape (usually same)
        "recon_loss_type": "mse",  # Options: "mse", "bce", "cross_entropy"
        "recon_weight": 1.0,  # Weight for reconstruction loss
        "vq_weight": 1.0,  # Weight for vector quantization loss (codebook update)
        "commitment_weight": 0.25,  # Weight for commitment loss (encoder update)
    }))
    
    encoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "mlp",  # Options: "mlp", "resnet", "identity", "linear"
        "encoder_type": "none",  # For VQ-VAE, encoder outputs deterministic features
        "input_shape": "NA",  # Will be set from main config if not specified
        "latent_shape": "NA",  # Will be set from main config (embedding_dim)
        "hidden_dims": (64, 64, 64),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "mlp",  # Options: "mlp", "resnet", "identity", "linear"
        "decoder_type": "none",  # Options: "linear", "softmax", "none"
        "latent_shape": "NA",  # Will be set from main config (embedding_dim)
        "output_shape": "NA",  # Will be set from main config if not specified
        "hidden_dims": (64, 64, 64),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))


class VectorQuantizer(nn.Module):
    """Vector quantization layer that maps continuous features to discrete codebook entries.
    
    This implements the VQ-VAE quantization mechanism:
    1. Find nearest codebook entry for each encoder output
    2. Use straight-through estimator for gradients
    3. Update codebook and encoder via loss terms
    """
    codebook_size: int  # Number of codebook entries
    embedding_dim: int  # Dimension of codebook vectors
    
    def setup(self):
        """Initialize codebook (embedding table)."""
        # Initialize codebook with better initialization to prevent collapse
        # Use uniform initialization across a reasonable range to ensure diversity
        # Standard VQ-VAE uses normal initialization, but we'll use uniform
        # with scale that matches typical encoder output range
        # Shape: [codebook_size, embedding_dim]
        # Uniform in [-1, 1] gives good initial spread
        self.embedding = self.param(
            'embedding',
            nn.initializers.uniform(scale=1.0),  # Uniform in [-1, 1] for good spread
            (self.codebook_size, self.embedding_dim)
        )
    
    def __call__(self, z_e: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Quantize encoder outputs using nearest codebook entries.
        
        Args:
            z_e: Encoder output [batch, ..., embedding_dim]
            
        Returns:
            Tuple of:
            - z_q: Quantized vectors [batch, ..., embedding_dim] (for forward pass)
            - z_q_st: Quantized vectors with straight-through [batch, ..., embedding_dim] (for gradients)
            - indices: Codebook indices [batch, ...] (discrete tokens)
        """
        # Flatten spatial dimensions if present
        original_shape = z_e.shape
        z_e_flat = z_e.reshape(-1, self.embedding_dim)  # [N, embedding_dim]
        
        # Compute distances to all codebook entries
        # dists: [N, codebook_size]
        dists = jnp.sum(
            (z_e_flat[:, None, :] - self.embedding[None, :, :]) ** 2,
            axis=2
        )
        
        # Find nearest codebook entry for each encoder output
        indices = jnp.argmin(dists, axis=1)  # [N]
        
        # Get quantized vectors
        z_q_flat = self.embedding[indices]  # [N, embedding_dim]
        
        # Reshape back to original spatial dimensions
        z_q = z_q_flat.reshape(original_shape)
        
        # Straight-through estimator: use quantized values in forward pass,
        # but allow gradients to flow through encoder outputs
        z_q_st = z_e + jax.lax.stop_gradient(z_q - z_e)
        
        # Reshape indices to match spatial dimensions (excluding embedding_dim)
        indices_shape = original_shape[:-1]
        indices = indices.reshape(indices_shape)
        
        return z_q, z_q_st, indices


class VQVAE(nn.Module):
    """Vector Quantized Variational Autoencoder using @nn.compact methods."""
    config: VQVAEConfig
    
    def setup(self):
        """Initialize encoder, vector quantizer, and decoder based on config."""
        # Get shapes from main config
        input_shape = self.config.main["input_shape"]
        embedding_dim = self.config.main["embedding_dim"]
        codebook_size = self.config.main["codebook_size"]
        output_shape = self.config.main["output_shape"]
        
        # Create encoder - outputs continuous features
        # Encoder output should match embedding_dim
        encoder_latent_shape = (embedding_dim,)
        self.encoder = create_encoder(
            self.config.encoder,
            input_shape=input_shape,
            latent_shape=encoder_latent_shape
        )
        
        # Create vector quantizer
        self.vq = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=embedding_dim
        )
        
        # Create decoder - takes quantized vectors
        decoder_latent_shape = (embedding_dim,)
        self.decoder = create_decoder(
            self.config.decoder,
            latent_shape=decoder_latent_shape,
            output_shape=output_shape
        )
    
    @nn.compact
    def encode(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Encode input to discrete tokens.
        
        Args:
            x: Input data [batch, *input_shape]
            training: Whether in training mode
            
        Returns:
            Tuple of:
            - z_e: Encoder output (continuous) [batch, ..., embedding_dim]
            - z_q: Quantized vectors [batch, ..., embedding_dim]
            - indices: Discrete token indices [batch, ...]
        """
        # Encode to continuous features
        z_e = self.encoder(x, training=training)
        
        # Quantize to discrete tokens
        z_q, z_q_st, indices = self.vq(z_e)
        
        # Return z_e (for loss computation), z_q (quantized), and indices
        # Note: We use z_q_st (straight-through) in the forward pass for decoding
        # to allow gradients to flow through the encoder
        return z_e, z_q_st, indices
    
    @nn.compact
    def decode(self, z_q: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Decode quantized vectors to output.
        
        Args:
            z_q: Quantized vectors [batch, ..., embedding_dim]
            training: Whether in training mode
            
        Returns:
            Reconstructed output [batch, *output_shape]
        """
        return self.decoder(z_q, training=training)
    
    @partial(jax.jit, static_argnums=(0, 4))
    def loss(self, params: dict, x: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> Tuple[jnp.ndarray, dict]:
        """
        Compute VQ-VAE loss.
        
        The loss consists of:
        1. Reconstruction loss: ||x - decode(quantize(encode(x)))||^2
        2. VQ loss: ||sg(z_e) - z_q||^2 (moves codebook closer to encoder outputs)
        3. Commitment loss: ||z_e - sg(z_q)||^2 (moves encoder outputs closer to codebook)
        
        Args:
            params: Model parameters dictionary
            x: Input data [batch, *input_shape] (can be multi-dimensional)
            key: Random key (for dropout)
            training: Whether in training mode
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Encode and quantize
        # encode returns (z_e, z_q_st, indices) where z_q_st is the straight-through quantized vector
        z_e, z_q_st, indices = self.apply(params, x, method='encode', training=training, rngs={'dropout': key})
        
        # Get the actual quantized vectors (without straight-through) for VQ loss
        # Compute z_q from the codebook embeddings using the indices we already have
        # Params structure: params['params']['vq']['embedding'] (Flax wraps in 'params')
        embedding = params['params']['vq']['embedding']  # [codebook_size, embedding_dim]
        z_e_flat = z_e.reshape(-1, z_e.shape[-1])  # [N, embedding_dim]
        indices_flat = indices.flatten()  # [N]
        z_q_flat = embedding[indices_flat]  # [N, embedding_dim]
        z_q = z_q_flat.reshape(z_e.shape)  # Restore original shape
        
        # Decode using straight-through quantized vectors (allows gradients to flow through encoder)
        # z_q_st = z_e + stop_gradient(z_q - z_e), so gradients flow through z_e but use z_q values
        x_recon = self.apply(params, z_q_st, method='decode', training=training, rngs={'dropout': key})
        
        # Reconstruction loss
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
        
        # VQ loss: move codebook embeddings closer to encoder outputs
        # Use stop_gradient on encoder outputs
        vq_loss = jnp.mean((jax.lax.stop_gradient(z_e) - z_q) ** 2)
        
        # Commitment loss: move encoder outputs closer to codebook embeddings
        # Use stop_gradient on quantized vectors
        commitment_loss = jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)
        
        # Get loss weights
        recon_weight = self.config.main.get("recon_weight", 1.0)
        vq_weight = self.config.main.get("vq_weight", 1.0)
        commitment_weight = self.config.main.get("commitment_weight", 0.25)
        
        # Compute total loss
        total_loss = (
            recon_weight * recon_loss +
            vq_weight * vq_loss +
            commitment_weight * commitment_loss
        )
        
        # Note: Codebook usage tracking removed from JIT-compiled loss function
        # because jnp.unique is not JIT-compatible. We'll compute it separately if needed.
        
        metrics = {
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'total_loss': total_loss,
            'indices': indices,  # Return indices for codebook usage tracking outside JIT
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
        embedding_dim = self.config.main["embedding_dim"]
        dummy_z = jnp.zeros(batch_shape + (embedding_dim,))
        
        # Call encode and decode to initialize parameters
        self.encode(x, training)
        self.decode(dummy_z, training)
        
        return jnp.zeros(batch_shape + self.config.main["output_shape"])

