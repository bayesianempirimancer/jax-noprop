"""Variational Bayesian VAE (VBVAE) model.

This model uses a GMM with Variational Bayesian EM for clustering encoder outputs.
The GMM parameters use Normal-Gamma conjugate priors for means and precision (γ = 1/σ²).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax
from flax.core import FrozenDict
from functools import partial
from typing import Tuple, Optional, Any
from dataclasses import dataclass, field

from src.configs.base_config import BaseConfig
from src.models.vae.encoders import create_encoder
from src.models.vae.decoders import create_decoder
from src.utils.math_utils import logsumexp
from jax.scipy.special import digamma


@dataclass(frozen=True)
class VBVAEConfig(BaseConfig):
    """Configuration for VBVAE model.
    
    This model uses:
    - Encoder (learned by gradient descent): x -> z_e (continuous latent)
    - GMM-VBEM (learned by VBEM): z_e -> p(cluster | z_e) -> top-k -> discrete codes
    - Decoder (learned by gradient descent): discrete codes -> x_recon
    
    The GMM parameters use Normal-Inverse-Gamma conjugate priors:
    - μ_k ~ Normal(μ₀_k, σ²_k / κ₀_k) for each cluster k
    - σ²_k,d ~ Inverse-Gamma(α₀_k,d, β₀_k,d) for each dimension d of cluster k
    - π_k ~ Dirichlet(α_mix) for mixing weights
    """
    # BaseConfig fields
    model_name: str = "vb_vae_network"
    
    main: FrozenDict = field(default_factory=lambda: FrozenDict({
        "input_shape": "NA",  # Will be set based on data (can be multi-dimensional)
        "num_clusters": 512,  # Number of GMM clusters
        "latent_dim": "NA",  # Dimension of encoder output (continuous latent z_e)
        "top_k": 1,  # Number of top clusters to use for discrete representation
        "output_shape": "NA",  # Will be set based on input_shape (usually same)
        "recon_loss_type": "mse",  # Options: "mse", "bce", "cross_entropy"
        "recon_weight": 1.0,  # Weight for reconstruction loss
        "gmm_weight": 1.0,  # Weight for GMM loss (cluster assignment)
        # Prior parameters for Normal-Gamma
        "prior_mu": 0.0,  # Prior mean for cluster means
        "prior_alpha": 2.0,  # Prior Gamma shape α (κ = 2 * α)
        "prior_beta": 2.0,  # Prior rate for Gamma (precision γ = 1/σ²)
        "prior_alpha_mix": 1.0,  # Prior for Dirichlet mixing weights
    }))
    
    encoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "mlp",  # Options: "mlp", "resnet", "identity", "linear"
        "encoder_type": "none",  # For GMM-VBEM-VAE, encoder outputs deterministic features
        "input_shape": "NA",  # Will be set from main config if not specified
        "latent_shape": "NA",  # Will be set from main config (latent_dim)
        "hidden_dims": (64, 64, 64),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "mlp",  # Options: "mlp", "resnet", "identity", "linear"
        "decoder_type": "none",  # Options: "linear", "softmax", "none"
        "latent_shape": "NA",  # Will be set from main config (latent_dim)
        "output_shape": "NA",  # Will be set from main config if not specified
        "hidden_dims": (64, 64, 64),
        "activation": "swish",
        "dropout_rate": 0.1,
    }))


class VBVAE(nn.Module):
    """Variational Bayesian VAE."""
    config: VBVAEConfig
    
    def setup(self):
        """Initialize encoder, GMM-VBEM, and decoder based on config."""
        # Get shapes from main config
        input_shape = self.config.main["input_shape"]
        latent_dim = self.config.main["latent_dim"]
        num_clusters = self.config.main["num_clusters"]
        output_shape = self.config.main["output_shape"]
        
        # Create encoder - outputs continuous features
        encoder_latent_shape = (latent_dim,)
        self.encoder = create_encoder(
            self.config.encoder,
            input_shape=input_shape,
            latent_shape=encoder_latent_shape
        )
        
        # Create GMM-VBEM component
        self.gmm_vbem = GMMVBEM(
            num_clusters=num_clusters,
            latent_dim=latent_dim,
            prior_mu=self.config.main.get("prior_mu", 0.0),
            prior_alpha=self.config.main.get("prior_alpha", 1.5),
            prior_beta=self.config.main.get("prior_beta", 0.5),
            prior_alpha_mix=self.config.main.get("prior_alpha_mix", 1.0),
        )
        
        # Create decoder - takes quantized vectors
        decoder_latent_shape = (latent_dim,)
        self.decoder = create_decoder(
            self.config.decoder,
            latent_shape=decoder_latent_shape,
            output_shape=output_shape
        )
    
    @nn.compact
    def encode(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Encode input to continuous latent representation.
        
        Args:
            x: Input data [batch, *input_shape]
            training: Whether in training mode
            
        Returns:
            z_e: Encoder output (continuous) [batch, ..., latent_dim]
        """
        return self.encoder(x, training=training)
    
    @nn.compact
    def decode(self, z_q: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Decode quantized representation to output space.
        
        Args:
            z_q: Quantized representation [batch, ..., latent_dim]
            training: Whether in training mode
            
        Returns:
            Reconstructed output [batch, *output_shape]
        """
        return self.decoder(z_q, training=training)
    
    @partial(jax.jit, static_argnums=(0, 4))
    def loss(self, params: dict, x: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> Tuple[jnp.ndarray, dict]:
        """
        Compute GMM-VBEM-VAE loss.
        
        The loss consists of:
        1. Reconstruction loss: ||x - decode(quantize(encode(x)))||^2
        2. GMM loss: negative log-likelihood of cluster assignments
        
        Args:
            params: Model parameters dictionary
            x: Input data [batch, *input_shape] (can be multi-dimensional)
            key: Random key (for dropout and sampling)
            training: Whether in training mode
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        key, sample_key = jr.split(key, 2)
        
        # Encode to continuous latent
        z_e = self.apply(params, x, method='encode', training=training, rngs={'dropout': key})
                    
        # Get quantized representation using GMM-VBEM
        # Access GMM-VBEM parameters from the params structure
        gmm_params = params['params']['gmm_vbem']
        
        # Compute log_p_tilde for VB loss
        # When using apply, Flax automatically binds the parameters, so nat_to_stats uses bound params
        z_q, logZ = self.gmm_vbem.apply({'params': gmm_params}, z_e, method='log_p_tilde')
                
        # Straight-through estimator: use z_q for forward pass, but allow gradients through z_e
        z_q_st = z_e + jax.lax.stop_gradient(z_q - z_e)
        
        # Decode using straight-through quantized vectors
        x_recon = self.apply(params, z_q_st, method='decode', training=training, rngs={'dropout': key})
        
        # Reconstruction loss
        recon_loss_type = self.config.main.get("recon_loss_type", "mse")
        if recon_loss_type == "mse":
            recon_loss = jnp.mean((x - x_recon) ** 2)
        elif recon_loss_type == "bce":
            recon_loss = jnp.mean(-x * jnp.log(x_recon + 1e-8) - (1 - x) * jnp.log(1 - x_recon + 1e-8))
        elif recon_loss_type == "cross_entropy":
            recon_loss = jnp.mean(-jnp.sum(x * jnp.log(x_recon + 1e-8), axis=-1))
        else:
            raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")
                        
        # Compute total loss
        total_loss = recon_loss
        
        metrics = {
            'gmm_loss': -logZ,
            'recon_loss': recon_loss,
            'total_loss': total_loss,
        }
        
        return total_loss, metrics
    
    def __call__(self, x: jnp.ndarray, key: jr.PRNGKey, training: bool = True) -> jnp.ndarray:
        """
        Forward pass: encode -> quantize -> decode.
        
        Args:
            x: Input data [batch, *input_shape]
            key: Random key
            training: Whether in training mode
            
        Returns:
            Reconstructed output [batch, *output_shape]
        """
        # This is for initialization - call encode and decode separately
        z_e = self.encode(x, training=training)
        z_q, _, _ = self.gmm_vbem(z_e)
        # Straight-through estimator for gradients
        z_q_st = z_e + jax.lax.stop_gradient(z_q - z_e)
        x_recon = self.decode(z_q_st, training=training)
        return x_recon
    
    @partial(jax.jit, static_argnums=(0, 5))
    def train_step(
        self,
        encoder_decoder_params: dict,
        gmm_params: dict,
        x_batch: jnp.ndarray,
        opt_state: dict,
        key: jr.PRNGKey,
        training: bool = True,
        optimizer: Optional[Any] = None
    ) -> Tuple[dict, dict, dict, jnp.ndarray, dict]:
        """
        Single training step that updates encoder/decoder via gradients, GMM params stay fixed.
        
        This method separates encoder/decoder parameters from GMM parameters, computes gradients
        only for encoder/decoder, and applies optimizer updates. GMM parameters are updated
        separately via VBEM updates (not gradient descent).
        
        Args:
            encoder_decoder_params: Encoder/decoder parameters (will be updated via gradients)
            gmm_params: GMM parameters (fixed during this step, updated separately via VBEM)
            x_batch: Batch of input data [batch_size, *input_shape]
            opt_state: Optimizer state
            key: Random key
            training: Whether in training mode
            optimizer: Optax optimizer (required for parameter updates)
            
        Returns:
            Tuple of (updated_encoder_decoder_params, gmm_params, updated_opt_state, loss, metrics)
        """
        from flax.core import freeze, unfreeze
        
        # Combine params for loss computation
        def combine_params(enc_dec_params, gmm_ps):
            enc_dec_unfrozen = unfreeze(enc_dec_params)
            enc_dec_unfrozen['params']['gmm_vbem'] = gmm_ps
            return freeze(enc_dec_unfrozen)
        
        # Compute loss and gradients
        # Only encoder/decoder params will receive gradients
        def loss_fn(params_enc_dec):
            # Combine with fixed GMM params
            full_params = combine_params(params_enc_dec, gmm_params)
            return self.loss(full_params, x_batch, key, training=training)
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(encoder_decoder_params)
        
        # Update encoder/decoder parameters via optimizer
        if optimizer is None:
            raise ValueError("Optimizer must be provided for train_step")
        
        updates, opt_state = optimizer.update(grads, opt_state, encoder_decoder_params)
        encoder_decoder_params = optax.apply_updates(encoder_decoder_params, updates)
        
        return encoder_decoder_params, gmm_params, opt_state, loss, metrics

