"""Variational Bayesian VAE (VBVAE) model.

This model uses a GMM with Variational Bayesian EM for clustering encoder outputs.
The GMM parameters use Normal-Gamma conjugate priors for means and precision (γ = 1/σ²).
"""

from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import flax.linen as nn
from flax.core import FrozenDict
from dataclasses import dataclass, field, MISSING
from typing import Tuple, Union, Optional, Dict, Any, Callable

from src.utils.math_utils import logsumexp, stable_softmax
from src.utils.kl_divergence import gamma_kl, dirichlet_kl, normal_gamma_kl
from jax.scipy.special import digamma, gammaln
from flax.core import freeze
from src.configs.base_config import BaseConfig


@dataclass(frozen=True)
class GMMVBEMConfig(BaseConfig):
    """Configuration for GMMVBEM model."""
    num_clusters: int
    latent_dim: int
    model_name: str = field(default="gmm_vbem", init=False)  # Exclude from __init__ to avoid ordering issues
    prior_mu: float = 0.0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    prior_alpha_mix: float = 0.5
    beta_mix: float = 0.1
    tie_precisions: bool = False
    
    def __post_init__(self):
        # Set model_name after initialization
        object.__setattr__(self, 'model_name', "gmm_vbem")


class GMMVBEM(nn.Module):
    """Gaussian Mixture Model with Variational Bayesian EM.
    
    Maintains posterior parameters for each cluster k:
    - μ_k ~ Normal(μₙ_k, 1/(γ_k · κₙ_k))  [posterior mean μₙ_k, precision scale κₙ_k]
    - γ_k,d = 1/σ²_k,d ~ Gamma(αₙ_k, βₙ_k,d)  [posterior shape αₙ, rate βₙ]
    - π_k ~ Dirichlet(α_mix_n)  [posterior Dirichlet parameters]
    
    Note: Using γ = 1/σ² (precision) instead of σ², with Gamma distribution.
    The posterior parameters are updated using VBEM updates.
    """
    num_clusters: int
    latent_dim: int
    prior_mu: float = 0.0  # Prior mean for cluster means [latent_dim]. Can be scalar (e.g., 0.0) or array of shape [latent_dim]. Defaults to 0.0 (converted to zeros array).
    prior_alpha: float = 1.0
    prior_beta: float = 1.0      # will be divided by num_clusters when used.
    prior_alpha_mix: float = 0.5
    beta_mix: float = 0.1  # Inverse mixing temperature. If 0.0, uses flat mixing (E_log_pi = 0). If 1.0, uses full Dirichlet posterior.
    tie_precisions: bool = False  # If True, tie all cluster precisions together (use summed alpha and beta)
    
    def setup(self):
        """Initialize GMM posterior parameters."""
        # Posterior parameters for cluster means: μₙ_k ~ Normal(μₙ_k, σ²_k / κₙ_k)
        # We store: μₙ_k (posterior mean)
        # Note: κₙ = 2 * αₙ, so we compute κₙ from αₙ
        self.mu_n = self.param(
            'mu_n',  # Posterior mean for each cluster
            nn.initializers.normal(stddev=1.0),
            (self.num_clusters, self.latent_dim)
        )

        self.alpha_n = self.param(
            'alpha_n',  # Posterior shape for Gamma (same for all latent dims)
            lambda key, shape: jnp.ones(shape) * self.prior_alpha,
            (self.num_clusters, 1)
        )
        self.beta_n = self.param(
            'beta_n',  # Posterior rate for Gamma
            lambda key, shape: jnp.ones(shape) * self.prior_beta/self.num_clusters**(2/self.latent_dim),
            (self.num_clusters, self.latent_dim)
        )
        
        # Posterior parameters for mixing weights: π ~ Dirichlet(α_mix)
        # Shape [num_clusters]
        self.alpha_mix = self.param(
            'alpha_mix',  # Posterior Dirichlet parameters
            lambda key, shape: jnp.ones(shape) * self.prior_alpha_mix,
            (self.num_clusters,)
        )


    @classmethod
    def get_initial_cluster_means(cls, num_clusters: int, latent_dim: int, x: jnp.ndarray, key: jr.PRNGKey) -> jnp.ndarray:
        """
        Initialize cluster means by randomly sampling data points.
        
        Args:
            num_clusters: Number of clusters
            latent_dim: Dimension of latent space
            x: Input data [N, latent_dim] - flattened to 2D
            key: Random key for sampling
            
        Returns:
            Initialized cluster means [num_clusters, latent_dim]
        """
        # Flatten x to 2D if needed
        x_flat = x.reshape(-1, latent_dim)  # [N, latent_dim]
        N = x_flat.shape[0]
        
        # Randomly sample num_clusters data points (without replacement)
        # If we have fewer data points than clusters, add noise
        # Use permutation for efficiency when N is large (more efficient than jr.choice with replace=False)
        
        if N >= num_clusters:
            # Shuffle indices and take first num_clusters (more efficient for large N)
            permuted_indices = jr.permutation(key, jnp.arange(N))
            idx = permuted_indices[:num_clusters]
            mu_n = x_flat[idx]  # [num_clusters, latent_dim]
        else:
            mu_n = x_flat
            M = num_clusters - N
            mu_n_opt_rand = x_flat.mean(0, keepdims=True) + x_flat.std(0, keepdims=True)*N**(-latent_dim/2) * jr.normal(key, (M, latent_dim))
            mu_n = jnp.concatenate([mu_n, mu_n_opt_rand], axis=0)
        
        return mu_n

    def extract_params(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Extract GMM parameters with gradients stopped.
        
        This method returns all GMM parameters with stop_gradient applied,
        ensuring that gradients never flow through these parameters (since they
        are updated via VBEM, not gradient descent).
        
        Returns:
            Tuple of (mu_n, alpha_n, beta_n, alpha_mix):
                - mu_n: [num_clusters, latent_dim]
                - alpha_n: [num_clusters, 1]
                - beta_n: [num_clusters, latent_dim]
                - alpha_mix: [num_clusters]
        """
        return (
            jax.lax.stop_gradient(self.mu_n),
            jax.lax.stop_gradient(self.alpha_n),
            jax.lax.stop_gradient(self.beta_n),
            jax.lax.stop_gradient(self.alpha_mix),
        )

    @nn.compact
    def fill_unused(self, x: jnp.ndarray, training: bool = False) -> dict:
        """
        Fill unused clusters by reinitializing their means to worst-fit data points
        and resetting alpha_n and beta_n to prior values.
        
        Args:
            x: Input data [batch, ..., latent_dim]
            
        Returns:
            Updated GMM parameters
        """
        # Compute log_p_tilde using current params (accessed via self.*)
        log_p_tilde = self.log_p_tilde(x, training=training)
        
        # Compute logZ for finding worst-fit points
        logZ = logsumexp(log_p_tilde, axis=-1)  # [batch, ...]
        
        # Get alpha_mix to identify unused clusters (accessed via self.*)
        alpha_mix = self.alpha_mix  # [num_clusters]
        unused_mask = alpha_mix < self.prior_alpha_mix + 1.0  # [num_clusters] boolean mask
        
        # Count unused clusters (for JIT compatibility, use sum instead of where)
        ns = jnp.sum(unused_mask.astype(jnp.int32))  # scalar
        
        # If no unused clusters, return current params
        if ns == 0:
            return {
                'mu_n': self.mu_n,
                'alpha_n': self.alpha_n,
                'beta_n': self.beta_n,
                'alpha_mix': self.alpha_mix
            }
        
        x_flat = x.reshape(-1, self.latent_dim)  # [N, latent_dim]
        logZ_flat = logZ.reshape(-1)  # [N]
        N = x_flat.shape[0]
        
        # Get unused indices - simple approach without JIT constraints
        # Sort by unused_mask (True=1, False=0) descending, then take first ns
        unused_scores = unused_mask.astype(jnp.float32)  # [num_clusters], 1.0 for unused, 0.0 for used
        sorted_indices = jnp.argsort(-unused_scores)  # Sort descending, unused clusters first
        
        # Limit number of clusters to fill to available data points
        ns_actual = jnp.minimum(ns, N)
        # Simple indexing - take first ns_actual unused indices
        unused_indices = sorted_indices[:ns_actual]  # [ns_actual]
        
        # Find ns_actual worst-fit data points (smallest logZ)
        sorted_logZ = jnp.argsort(logZ_flat)  # [N] sorted indices
        # Simple indexing - take first ns_actual worst-fit indices
        worst_fit_indices = sorted_logZ[:ns_actual]  # [ns_actual]
        worst_fit_x = x_flat[worst_fit_indices]  # [ns_actual, latent_dim]
        
        # Update params: set mu_n for unused clusters to worst-fit x values
        mu_n = self.mu_n.at[unused_indices, :].set(worst_fit_x)
        
        # Reset alpha_n and beta_n for unused clusters to prior values

        alpha_n = self.alpha_n.at[unused_indices, :].set(self.alpha_n.mean())
        beta_n = self.beta_n.at[unused_indices, :].set(self.beta_n.mean())
        alpha_mix = self.alpha_mix.at[unused_indices].set(0.1*self.alpha_mix.mean())
        beta_n = 2*beta_n
        updated_params = {
            'mu_n': mu_n,
            'alpha_n': alpha_n,
            'beta_n': beta_n,
            'alpha_mix': alpha_mix  # Return updated alpha_mix
        }
            
        return updated_params 


    @nn.compact
    def nat_to_stats(self, training: bool = False) -> dict:
        """
        Convert natural parameters to expected sufficient statistics.
        
        Uses the bound parameters from self (mu_n, alpha_n, beta_n, alpha_mix).
        
        For Normal-Gamma posterior on (μ, γ) where γ = 1/σ²:
        - μ ~ Normal(μₙ, 1/(γ·κₙ)) with natural params related to μₙ and κₙ
        - γ ~ Gamma(αₙ, βₙ) with natural params (αₙ-1, -βₙ)
        - π ~ Dirichlet(α_mix) with natural params (α_mix)
        
        Expected sufficient statistics:
        - E[μ] = μₙ
        - E[μ²] = μₙ² + 1/(γ·κₙ) where E[1/γ] = βₙ/(αₙ-1) for αₙ > 1
        - E[log(γ)] = ψ(αₙ) - log(βₙ) where ψ is digamma function
        - E[γ] = αₙ/βₙ
        - E[π] = α_mix / Σ_k α_mix_k
        - E[log(π)] = ψ(α_mix) - ψ(Σ_k α_mix_k)
                
        Returns:
            Dictionary containing expected sufficient statistics:
                - 'E_mu': [num_clusters, latent_dim] - E[μ]
                - 'E_gamma_var_mu': [num_clusters, 1] - scalar that hits an identity covariance matrix
                - 'E_log_gamma': [num_clusters, latent_dim] - E[log(γ)]
                - 'E_gamma': [num_clusters, latent_dim] - E[γ] = αₙ/βₙ
                - 'E_pi': [num_clusters] - E[π]
                - 'E_log_pi': [num_clusters] - E[log(π)]
        """
        # Extract parameters with gradients stopped (updated via VBEM, not gradient descent)
        mu_n, alpha_n, beta_n, alpha_mix = self.extract_params()
        
        # Compute κₙ from αₙ: κₙ = 2 * αₙ (keep shape [num_clusters, 1] for broadcasting)
        kappa_n = 2.0 * alpha_n  # [num_clusters, 1]
        
        E_mu = mu_n  # [num_clusters, latent_dim]
        E_gamma_var_mu = 1.0 / kappa_n  # [num_clusters, 1] is a scaler that hits and identity matrix
        
        # Expected precision: E[γ] = αₙ/βₙ
        # With prior α₀ = 1.5, posterior αₙ >= 1.5 > 1, so no clamping needed
        if self.tie_precisions:
            # Tie all cluster precisions together: use summed alpha and beta across all clusters
            alpha_sum = jnp.sum(alpha_n, axis=0, keepdims=True)  # [1, latent_dim]
            beta_sum = jnp.sum(beta_n, axis=0, keepdims=True)  # [1, latent_dim]
            E_gamma = alpha_sum / beta_sum  # [1, latent_dim], broadcast to [num_clusters, latent_dim]
            E_log_gamma = digamma(alpha_sum) - jnp.log(beta_sum)  # [1, latent_dim], broadcast to [num_clusters, latent_dim]
            # Broadcast to all clusters
            E_gamma = jnp.broadcast_to(E_gamma, (self.num_clusters, self.latent_dim))
            E_log_gamma = jnp.broadcast_to(E_log_gamma, (self.num_clusters, self.latent_dim))
        else:
            E_gamma = alpha_n / beta_n  # [num_clusters, latent_dim] (broadcasts automatically)
            E_log_gamma = digamma(alpha_n) - jnp.log(beta_n)  # [num_clusters, latent_dim] (broadcasts automatically)
        
        # Expected μ²: E[μ²] = μₙ² + Var[μ]
        # Since p(μ|γ,κ) = Normal(μ₀, 1/(κ·γ)), we have Var[μ|γ] = 1/(κ·γ)
        # In the expected log-likelihood, γ·Var[μ] simplifies to 1/κ
        # So Var[μ] = 1/κ (for the unconditional variance used in the log-likelihood)
        
        # Expected mixing weights: E[π] = α_mix / Σ_k α_mix_k
        alpha_mix_sum = jnp.sum(alpha_mix)  # scalar
        E_pi = alpha_mix / alpha_mix_sum  # [num_clusters]
        
        # Expected log mixing weights: E[log(π_k)] = β_mix * (ψ(α_mix_k) - ψ(Σ_k α_mix_k))
        # beta_mix acts as inverse temperature: 0.0 = flat mixing, 1.0 = full Dirichlet posterior
        E_log_pi_full = digamma(alpha_mix) - digamma(alpha_mix_sum)  # [num_clusters]
        E_log_pi = self.beta_mix * E_log_pi_full  # [num_clusters]

        return {
            'E_mu': E_mu,
            'E_gamma_var_mu': E_gamma_var_mu,
            'E_log_gamma': E_log_gamma,
            'E_gamma': E_gamma,
            'E_pi': E_pi,
            'E_log_pi': E_log_pi,
        }
    
    
    @nn.compact
    def log_p_tilde(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Compute unnormalized log probability of cluster assignments.
        
        Uses bound parameters from self (via nat_to_stats).
        
        log p̃(x, k) = E[log p(x | k)] + E[log π_k]
        
        Where:
        - E[log p(x | k)] is the expected Gaussian log-likelihood using E[μ_k], E[γ_k], and E[log(γ_k)]
        - E[log π_k] is the expected log mixing weight (NOT log(E[π_k]))
        
        The expectation is computed by averaging log_p(natparam * stats), so we only use:
        - E[γ] (precision) and E[log(γ)] (log precision) for the Gaussian
        - E[log(π)] for the mixing weights
        
        Args:
            x: Observations [batch, ..., latent_dim] or [N, latent_dim]
                
        Returns:
            Unnormalized expected log probabilities [batch, ..., num_clusters] or [N, num_clusters]
        """
        # Convert natural parameters to expected sufficient statistics
        # nat_to_stats uses bound parameters from self (mu_n, alpha_n, beta_n, alpha_mix)
        expectations = self.nat_to_stats(training=training)
        
        E_mu = expectations['E_mu']  # [num_clusters, latent_dim]
        E_gamma_var_mu = expectations['E_gamma_var_mu']  # [num_clusters, 1] = E[γ·Var[μ]] = 1/κ
        E_gamma = expectations['E_gamma']  # [num_clusters, latent_dim] = E[γ] (precision)
        E_log_gamma = expectations['E_log_gamma']  # [num_clusters, latent_dim] = E[log(γ)]
        E_log_pi = expectations['E_log_pi']  # [num_clusters] = E[log(π)]
        
        num_clusters = E_mu.shape[-2]
        latent_dim = E_mu.shape[-1]
        
        original_shape = x.shape
        x_flat = x.reshape(-1, latent_dim)  # [N, latent_dim]
        
        # Ensure all terms have shape [num_clusters] for consistent broadcasting
        log_probs = E_log_pi - 0.5*latent_dim*E_gamma_var_mu.squeeze(-1) + 0.5*jnp.sum(E_log_gamma, axis=-1)  # [num_clusters]

        diff = x_flat[:, None, :] - E_mu[None, :, :]  # [N, num_clusters, latent_dim]
        # Clip E_gamma to prevent extreme precision values that cause numerical overflow
        # Very high precision (low variance) can cause log_probs to become extremely negative
        E_gamma_clipped = jnp.clip(E_gamma, 0.0, 1e6)  # Cap precision at reasonable maximum
        log_probs = log_probs - 0.5 * jnp.sum(E_gamma_clipped[None, :, :] * (diff ** 2), axis=-1)  # [N, num_clusters]
        
        # Note: log_probs will be used in softmax, which handles numerical stability
        # No need to clip here - softmax will handle extreme values correctly
                
        # Reshape to original spatial dimensions
        log_probs = log_probs.reshape(original_shape[:-1] + (num_clusters,))
        
        return log_probs
    
    @nn.compact
    def quantize(self, x: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute cluster assignments and get discrete representation.
        
        During training: samples cluster assignments from the categorical distribution.
        During inference: uses the most probable cluster (argmax) deterministically.
        
        Args:
            x: Input data [batch, ..., latent_dim]
            training: Whether in training mode (if True, samples; if False, uses argmax)
            
        Returns:
            Tuple of:
            - x_q: Quantized representation (cluster mean of selected/sampled cluster) [batch, ..., latent_dim]
            - log_p_tilde: Unnormalized log probabilities [batch, ..., num_clusters]
        """
        # Compute unnormalized log probabilities
        # When called via apply, Flax automatically binds parameters, so self.mu_n etc will use the bound params
        # log_p_tilde uses bound parameters from self (via nat_to_stats)
        log_p_tilde = self.log_p_tilde(x, training=training)  # [batch, ..., num_clusters]     

        # Extract parameters with gradients stopped (updated via VBEM, not gradient descent)
        mu_n, _, _, _ = self.extract_params()
        original_shape = x.shape
        
        if training:
            # Sample cluster assignments from categorical distribution during training
            # Use self.make_rng() to get the random key (passed via rngs in apply call)
            try:
                quantize_key = self.make_rng('quantize')
                # Sample from categorical distribution using log_p_tilde as logits
                # Flatten for sampling, then reshape back
                logits = log_p_tilde.reshape(-1, self.num_clusters)  # [N, num_clusters]
                logits = logits - jnp.max(logits, axis=-1, keepdims=True)
                selected_clusters_flat = jr.categorical(quantize_key, logits=logits, axis=-1)  # [N]
                selected_clusters_flat = selected_clusters_flat.reshape(original_shape[:-1])  # [batch, ...]
            except Exception:
                # Fall back to argmax if rngs not provided (e.g., during testing without rngs)
                selected_clusters = jnp.argmax(log_p_tilde, axis=-1)  # [batch, ...]
                selected_clusters_flat = selected_clusters.flatten()  # [N]
        else:
            # Use argmax for deterministic selection (inference mode)
            selected_clusters = jnp.argmax(log_p_tilde, axis=-1)  # [batch, ...]
            selected_clusters_flat = selected_clusters.flatten()  # [N]
        
        # Get quantized representation (cluster mean of selected/sampled cluster)
        x_q_flat = mu_n[selected_clusters_flat]  # [N, latent_dim]
        x_q = x_q_flat.reshape(original_shape)  # [batch, ..., latent_dim]
        
        return x_q, log_p_tilde
    
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize all @nn.compact methods by calling them.
        
        This method is used to initialize the compact methods (quantize, log_p_tilde, nat_to_stats)
        when the module is first set up. It delegates to quantize for the forward pass.
        
        Args:
            x: Input data [batch, ..., latent_dim]
            training: Whether in training mode
            
        Returns:
            Tuple of:
            - x_q: Quantized representation (cluster mean of selected/sampled cluster) [batch, ..., latent_dim]
            - log_p_tilde: Unnormalized log probabilities [batch, ..., num_clusters]
        """
        # Delegate to quantize for the actual forward pass
        # When called via init(), this will initialize quantize and all methods it calls
        return self.quantize(x, training=training)

    @nn.compact
    def kl_prior(self, training: bool = False) -> jnp.ndarray:
        """
        Compute KL divergence between posterior and prior distributions.
        
        Computes:
        1. KL for Normal-Gamma posterior/prior for each cluster's mean and precision
        2. KL for Dirichlet posterior/prior for mixing weights
        
        Returns:
            Total KL divergence (scalar)
        """
        # Extract parameters with gradients stopped (updated via VBEM, not gradient descent)
        mu_n, alpha_n, beta_n, alpha_mix = self.extract_params()
        
        # Prior parameters
        prior_mu = self.prior_mu
        prior_alpha = self.prior_alpha
        prior_beta = self.prior_beta/self.num_clusters**(2/self.latent_dim)
        prior_alpha_mix = self.prior_alpha_mix
        
        # KL divergence for Normal-Gamma (for each cluster and dimension)
        # Use the new normal_gamma_kl function
        kappa_n = 2.0 * alpha_n  # [num_clusters, 1]
        kappa_prior = 2.0 * prior_alpha  # scalar
        
        # normal_gamma_kl expects: kappa_p, mu_p, alpha_p, beta_p, kappa_q, mu_q, alpha_q, beta_q
        # Shape: kappa_n is [num_clusters, 1], mu_n is [num_clusters, latent_dim]
        #        alpha_n is [num_clusters, 1], beta_n is [num_clusters, latent_dim]
        # We need to compute KL for each cluster and dimension
        kl_normal_gamma = normal_gamma_kl(
            kappa_p=kappa_n,  # [num_clusters, 1]
            mu_p=mu_n,  # [num_clusters, latent_dim]
            alpha_p=alpha_n,  # [num_clusters, 1]
            beta_p=beta_n,  # [num_clusters, latent_dim]
            kappa_q=kappa_prior,  # scalar
            mu_q=prior_mu,  # scalar or [latent_dim]
            alpha_q=prior_alpha,  # scalar
            beta_q=prior_beta  # scalar
        )  # Returns [num_clusters, latent_dim]
        
        # Sum over clusters and dimensions
        kl_normal_gamma = jnp.sum(kl_normal_gamma)
        
        # KL divergence for Dirichlet (mixing weights)
        # Use the new dirichlet_kl function
        kl_dirichlet = dirichlet_kl(
            alpha_p=alpha_mix,  # [num_clusters]
            alpha_q=prior_alpha_mix  # scalar
        )  # Returns [1] (keepdims=True)
        
        # Sum to get scalar
        kl_dirichlet = jnp.sum(kl_dirichlet)
        
        return kl_normal_gamma + kl_dirichlet
    
    @nn.compact
    def loss(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Compute GMM loss including negative log-likelihood and KL divergence.
        
        Loss = -E[log p(x)] + KL(q(θ) || p(θ))
        where:
        - E[log p(x)] is the expected log-likelihood (negative logZ)
        - KL(q(θ) || p(θ)) is the KL divergence between posterior and prior
        
        Args:
            x: Input data [batch, ..., latent_dim]
            
        Returns:
            Total GMM loss (scalar)
        """
        log_p_tilde = self.log_p_tilde(x, training=training)
        logZ = logsumexp(log_p_tilde, axis=-1)
        logZ = jnp.sum(logZ)
        
        # Add KL divergence between posterior and prior
        kl = self.kl_prior(training=training)
        
        return -logZ + kl

    @nn.compact
    def sample(
        self,
        key: jr.PRNGKey,
        batch_shape: Tuple[int, ...],
        training: bool = False
    ) -> jnp.ndarray:
        """
        Sample from GMM.
        
        Samples from the GMM by first sampling cluster assignments from the mixing weights,
        then sampling from the selected clusters. If top_k is provided, only considers the
        top-k clusters with highest mixing weights.
        
        Args:
            key: Random key for sampling
            batch_shape: Shape of batch dimensions (e.g., (batch_size,) or (batch_size, height, width))
            top_k: Number of top clusters to consider (if None, uses all clusters)
            training: Whether in training mode
            
        Returns:
            Samples from GMM [*batch_shape, latent_dim]
        """
        # Compute total number of samples
        # Handle both traced and concrete batch_shape
        # For traced batch_shape (tuple of traced values), we need to use jnp.prod directly
        # on the tuple elements without converting to array first
        if isinstance(batch_shape, tuple) and len(batch_shape) == 1:
            # Single element tuple - use directly
            N = batch_shape[0]
        elif isinstance(batch_shape, tuple):
            # Multiple elements - compute product element by element
            N = batch_shape[0]
            for s in batch_shape[1:]:
                N = N * s
        else:
            # Already an array or single value
            batch_shape_array = jnp.asarray(batch_shape)
            N = jnp.prod(batch_shape_array)
        
        # Get expected statistics for mixing weights and cluster parameters
        expectations = self.nat_to_stats(training=training)
        E_log_pi = expectations['E_log_pi']  # [num_clusters] - expected log mixing weights
        E_mu = expectations['E_mu']  # [num_clusters, latent_dim]
        E_gamma = expectations['E_gamma']  # [num_clusters, latent_dim] (precision)

        
        # Sample cluster assignments from categorical distribution
        key, cluster_key = jr.split(key)
        # jr.categorical accepts traced shapes, so we can use N directly
        cluster_indices = jr.categorical(
            cluster_key,
            logits=E_log_pi,
            axis=-1,
            shape=(N,)
        )  # [N] - indices into cluster_indices_all
        
        # Get means and variances for selected clusters
        variance = 1.0 / (E_gamma + 1e-8)  # [num_clusters, latent_dim]
        
        # Select means and variances for the sampled clusters
        selected_means = E_mu[cluster_indices]  # [N, latent_dim]
        selected_vars = variance[cluster_indices]  # [N, latent_dim]
        
        # Sample from Gaussian distributions
        key, noise_key = jr.split(key)
        noise = jr.normal(noise_key, (N, self.latent_dim))
        samples_flat = selected_means + noise * jnp.sqrt(selected_vars + 1e-8)
        
        # Reshape to batch_shape + latent_dim
        samples = samples_flat.reshape(batch_shape + (self.latent_dim,))
        
        return samples
    

    @nn.compact
    def sample_conditional(
        self,
        x: jnp.ndarray,
        key: jr.PRNGKey,
        top_k: Optional[int] = None,
        training: bool = False
    ) -> jnp.ndarray:

        samples, loss = self.sample_conditional_and_loss(x, key, top_k, training)

        return samples
    
    @nn.compact
    def sample_conditional_and_loss(
        self,
        x: jnp.ndarray,
        key: jr.PRNGKey,
        top_k: Optional[int] = None,
        training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Conditionally sample from GMM given input samples and return loss.
        
        For each input x, computes assignment probabilities p(k|x) and generates
        a new sample by averaging samples from each cluster weighted by p(k|x).
        Also returns the loss = -logZ + KL_prior, where logZ = logsumexp(log_p_tilde).
        
        Args:
            x: Input samples [batch, ..., latent_dim]
            key: Random key for sampling
            top_k: Number of top clusters to consider for each x (if None, uses all clusters)
            training: Whether in training mode
            
        Returns:
            Tuple of:
            - samples: Conditionally sampled points [batch, ..., latent_dim] with same shape as x
            - loss: Loss value (negative log partition function + KL prior) [scalar]
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.latent_dim)  # [N, latent_dim]
        N = x_flat.shape[0]
        
        # Get assignment probabilities for each input
        log_p_tilde = self.log_p_tilde(x_flat, training=training)  # [N, num_clusters]
        
        # Compute log partition function: logZ = logsumexp(log_p_tilde, axis=-1)
        logZ_flat = logsumexp(log_p_tilde, axis=-1, keepdims=True)
        log_p_tilde = log_p_tilde - logZ_flat 
        logZ_flat = logZ_flat.squeeze(-1) # [N]
        
        # Get expected statistics for cluster parameters
        expectations = self.nat_to_stats(training=training)
        E_mu = expectations['E_mu']  # [num_clusters, latent_dim]
        E_gamma = expectations['E_gamma']  # [num_clusters, latent_dim] (precision)
        E_var = 1.0 / (E_gamma + 1e-8)  # [num_clusters, latent_dim]
        
        # Handle top_k filtering and sampling
        if top_k is None:
            # Use all clusters
            cluster_probs = stable_softmax(log_p_tilde, axis=-1)  # [N, num_clusters]
            
            # Sample from all clusters
            key, noise_key = jr.split(key)
            noise = jr.normal(noise_key, (self.num_clusters, self.latent_dim))  # [num_clusters, latent_dim]
            all_cluster_samples = E_mu + noise * jnp.sqrt(E_var + 1e-8)  # [num_clusters, latent_dim]
            
            # Weighted sum: sum_k p(k|x) * sample_from_cluster_k
            weighted_samples = jnp.sum(
                cluster_probs[:, :, None] * all_cluster_samples[None, :, :],
                axis=1
            )  # [N, latent_dim]
        else:
            top_k_actual = min(top_k, self.num_clusters)
            # Get top-k clusters for each sample
            top_k_indices = jnp.argsort(log_p_tilde, axis=-1)[:, -top_k_actual:]  # [N, top_k]
            # Extract log probabilities for top-k clusters
            log_p_tilde_topk = jnp.take_along_axis(log_p_tilde, top_k_indices, axis=-1)  # [N, top_k]
            # Normalize to get probabilities over top-k clusters
            cluster_probs = stable_softmax(log_p_tilde_topk, axis=-1)  # [N, top_k]
            
            # Extract top-k means and variances for each input (more efficient than sampling all clusters)
            top_k_means = E_mu[top_k_indices]  # [N, top_k, latent_dim]
            top_k_vars = E_var[top_k_indices]  # [N, top_k, latent_dim]
            
            # Sample noise only for top-k clusters per input
            key, noise_key = jr.split(key)
            noise = jr.normal(noise_key, (N, top_k_actual, self.latent_dim))  # [N, top_k, latent_dim]
            
            # Compute samples for top-k clusters
            cluster_samples = top_k_means + noise * jnp.sqrt(top_k_vars + 1e-8)  # [N, top_k, latent_dim]
            
            # Weighted sum: sum_k p(k|x) * sample_from_cluster_k
            weighted_samples = jnp.sum(
                cluster_probs[:, :, None] * cluster_samples,
                axis=1
            )  # [N, latent_dim]
        
        # Reshape back to original shape
        samples = weighted_samples.reshape(original_shape)
        
        # Reshape logZ to match original shape (without last dimension)
        logZ = logZ_flat.reshape(original_shape[:-1])  # [batch, ...] (removes latent_dim)
        
        return samples, -jnp.sum(logZ) + self.kl_prior(training=training)
    
    @partial(nn.jit, static_argnames=('N_eff', 'lr', 'training'))
    @nn.compact
    def update(
        self,
        x: jnp.ndarray,
        N_eff: float = 2000.0,
        lr: float = 0.2,
        training: bool = False
    ) -> dict:
        # Access parameters via self.* (bound by @nn.compact)
        # Note: In update method, we DO want gradients for the update computation itself,
        # but we stop gradients when these params are used elsewhere (via nat_to_stats)
        mu_n = self.mu_n
        alpha_n = self.alpha_n
        beta_n = self.beta_n
        alpha_mix = self.alpha_mix

        # Compute log_p_tilde using current params
        log_p_tilde = self.log_p_tilde(x, training=training)
        
        cluster_probs = stable_softmax(log_p_tilde, axis=-1)  # [batch, ..., num_clusters]
        x_flat = x.reshape(-1, self.latent_dim)  # [N, latent_dim]

        r_nk = cluster_probs.reshape(-1, self.num_clusters)  # [M, num_clusters]
                
        N_k = jnp.sum(r_nk, axis=0)  # [num_clusters] - effective number of points in each cluster
        N_scale = N_eff / x_flat.shape[0]  # weights the contribution from the minibatch to N_eff

        alpha_mix = (1 - lr) * alpha_mix + lr * (N_scale * N_k + self.prior_alpha_mix)    

        # Compute current kappa_mu_n before updates
        kappa_mu_n = 2.0 * alpha_n * mu_n  # [num_clusters, latent_dim] - broadcasting: (3,1) * (3,2) -> (3,2)
        kappa_mu_prior = 2.0 * self.prior_alpha * self.prior_mu  # [num_clusters, latent_dim] - JAX broadcasts
        kappa_mu_like = jnp.sum(r_nk[:, :, None] * x_flat[:, None, :], axis=0)  # [num_clusters, latent_dim]

        # Compute likelihood terms flr alpha and alpha_mix
        alpha_n_like = N_scale * 0.5 * N_k[:, None]
        alpha_n = (1 - lr) * alpha_n + lr * (alpha_n_like + self.prior_alpha)        
        
        kappa_mu_n = (1 - lr) * kappa_mu_n + lr * (N_scale * kappa_mu_like + kappa_mu_prior)
        mu_n = kappa_mu_n / (2.0 * alpha_n)  # because alpha_n was already updated, add epsilon for stability
        
        diff = x_flat[:, None, :] - mu_n[None, :, :]  # [N, num_clusters, latent_dim]
        weighted_diff_sq = jnp.sum(r_nk[:, :, None] * (diff ** 2), axis=0)  # [num_clusters, latent_dim]

        prior_diff_sq =  self.prior_alpha/(alpha_n_like + self.prior_alpha)  * ((mu_n - self.prior_mu) ** 2)   # [num_clusters, latent_dim] - JAX broadcasts
        beta_n = (1 - lr) * beta_n + lr * (N_scale * 0.5 * (weighted_diff_sq + prior_diff_sq) + self.prior_beta/self.num_clusters**(2/self.latent_dim))
                        

        # # CORRECTION FACTOR UPDATES (original approach - commented out)
        # kappa_mu_n = 2.0 * alpha_n * mu_n  # [num_clusters, latent_dim] - broadcasting: (3,1) * (3,2) -> (3,2)
        # alpha_n = alpha_n + (N_k[:, None] / 2.0) - correction_factor * (alpha_n - self.prior_alpha)  # [num_clusters, 1]
        # alpha_mix = alpha_mix + N_k - correction_factor * (alpha_mix - self.prior_alpha_mix)  # [num_clusters]
        # # update kappa_mu_n
        # kappa_mu_prior = 2.0 * self.prior_alpha * jnp.broadcast_to(self.prior_mu, (self.num_clusters, self.latent_dim))  # [num_clusters, latent_dim]
        # kappa_mu_n = kappa_mu_n + weighted_sum - correction_factor * (kappa_mu_n - kappa_mu_prior)  # [num_clusters, latent_dim]
        # mu_n = 0.5 * kappa_mu_n / (alpha_n + 1e-8)  # [num_clusters, latent_dim] - add epsilon for numerical stability
        # diff = z_e_flat[:, None, :] - mu_n[None, :, :]  # [N, num_clusters, latent_dim]
        # weighted_diff_sq = jnp.sum(r_nk[:, :, None] * (diff ** 2), axis=0)  # [num_clusters, latent_dim]
        # prior_diff_sq = 2.0 * self.prior_alpha * ((mu_n - self.prior_mu) ** 2)  # [num_clusters, latent_dim]
        # beta_n = beta_n + 0.5 * weighted_diff_sq + 0.5 * prior_diff_sq - correction_factor * (beta_n - self.prior_beta/self.num_clusters)

        # Create updated dict
        updated_params = {
            'mu_n': mu_n,
            'alpha_n': alpha_n,
            'beta_n': beta_n,
            'alpha_mix': alpha_mix
        }
        
        return updated_params
    
    @classmethod
    def fit(
        cls,
        config: 'GMMVBEMConfig',
        params: dict,
        x_data: jnp.ndarray,
        apply_fn: Optional[Callable] = None,
        initialize: bool = False,
        num_epochs: int = 10,
        batch_size: int = 256,
        N_eff: Optional[float] = None,
        lr: float = 0.2,
        seed: int = 42
    ) -> dict:
        """
        Fit GMM to data using VBEM updates over multiple epochs.
        
        This is a classmethod that can be called without an instance. It uses
        either a provided apply function (e.g., from a parent module) or creates
        a temporary GMM instance internally for apply calls.
        
        Args:
            config: GMM configuration
            params: Model parameters dictionary (from model.init())
            x_data: Training data [N, latent_dim] or [batch, ..., latent_dim]
            apply_fn: Optional apply function from a parent module (e.g., planner.apply).
                     If None, will attempt to create a GMM instance (requires Flax scope).
            initialize: If True, initialize cluster means from data before fitting
            num_epochs: Number of training epochs
            batch_size: Batch size for VBEM updates
            N_eff: Effective number of data points (if None, uses x_data.shape[0])
            lr: Learning rate for VBEM updates (mixing parameter between 0 and 1)
            seed: Random seed for shuffling data and initialization
            
        Returns:
            Updated params dictionary with fitted GMM parameters
        """
        from flax.core import unfreeze, freeze
        
        # Flatten data to [N, latent_dim]
        x_flat = x_data.reshape(-1, config.latent_dim)
        N = x_flat.shape[0]
        
        if N_eff is None:
            N_eff = float(N)
        
        # Unfreeze params to allow updates
        params_unfrozen = unfreeze(params)
        gmm_params = params_unfrozen.get('params', {}).get('gmm', {})
        
        # Initialize cluster means from data if requested
        if initialize:
            if 'mu_n' not in gmm_params:
                raise ValueError("GMM params not initialized. Please call model.init() first.")
            
            # Initialize cluster means from data
            init_key = jr.PRNGKey(seed)
            mu_n = cls.get_initial_cluster_means(
                num_clusters=config.num_clusters,
                latent_dim=config.latent_dim,
                x=x_flat,
                key=init_key
            )
            gmm_params['mu_n'] = mu_n
            params_unfrozen['params']['gmm'] = gmm_params
        
        # Create batches
        num_batches = (N + batch_size - 1) // batch_size
        
        # Fit GMM for multiple epochs
        for epoch in range(num_epochs):
            # Shuffle data
            key = jr.PRNGKey(seed + epoch)
            indices = jr.permutation(key, N)
            x_shuffled = x_flat[indices]
            
            # Process in batches
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, N)
                x_batch = x_shuffled[start_idx:end_idx]
                
                # Create frozen params for apply call
                gmm_params_frozen = freeze({'params': gmm_params})
                
                # Update GMM parameters using VBEM
                # Use provided apply_fn if available, otherwise create GMM instance
                if apply_fn is not None:
                    # Use the provided apply function (e.g., from planner)
                    updated_params_dict = apply_fn(
                        freeze(params_unfrozen),
                        x_batch,
                        N_eff=N_eff,
                        lr=lr,
                        training=True,
                        method=lambda mdl, x, **kwargs: mdl.gmm.update(x, **kwargs)
                    )
                else:
                    # Create GMM instance for apply calls (requires Flax scope)
                    gmm = create_gmm_vbem(config)
                    updated_params_dict = gmm.apply(
                        gmm_params_frozen,
                        x_batch,
                        N_eff=N_eff,
                        lr=lr,
                        training=True,
                        method='update'
                    )
                
                # Update gmm_params with the returned values
                gmm_params = {
                    'mu_n': updated_params_dict['mu_n'],
                    'alpha_n': updated_params_dict['alpha_n'],
                    'beta_n': updated_params_dict['beta_n'],
                    'alpha_mix': updated_params_dict['alpha_mix']
                }
                
                # Update the nested structure
                params_unfrozen['params']['gmm'] = gmm_params
        
        # Freeze and return
        return freeze(params_unfrozen)




########  FACTORY FUNCTION   ###########

def create_gmm_vbem(
    config: Union[GMMVBEMConfig, Dict[str, Any]],
    **kwargs
) -> GMMVBEM:
    """
    Factory function to create a GMMVBEM instance from a config.
    
    Args:
        config: Either a GMMVBEMConfig instance or a dictionary with configuration values.
                If a dict, it should contain at minimum:
                - num_clusters: int
                - latent_dim: int
                And optionally:
                - prior_mu: float (default: 0.0)
                - prior_alpha: float (default: 1.0)
                - prior_beta: float (default: 1.0)
                - prior_alpha_mix: float (default: 0.5)
                - beta_mix: float (default: 0.1)
                - tie_precisions: bool (default: False)
        **kwargs: Additional keyword arguments that override config values
        
    Returns:
        GMMVBEM instance
    """
    # Handle config as dict or GMMVBEMConfig instance
    if isinstance(config, dict):
        # Extract values from dict with defaults
        num_clusters = config.get("num_clusters")
        latent_dim = config.get("latent_dim")
        
        if num_clusters is None or latent_dim is None:
            raise ValueError("config must contain 'num_clusters' and 'latent_dim'")
        
        prior_mu = config.get("prior_mu", 0.0)
        prior_alpha = config.get("prior_alpha", 1.0)
        prior_beta = config.get("prior_beta", 1.0)
        prior_alpha_mix = config.get("prior_alpha_mix", 0.5)
        beta_mix = config.get("beta_mix", 0.1)
        tie_precisions = config.get("tie_precisions", False)
    elif isinstance(config, GMMVBEMConfig):
        # Extract values from config object
        num_clusters = config.num_clusters
        latent_dim = config.latent_dim
        prior_mu = config.prior_mu
        prior_alpha = config.prior_alpha
        prior_beta = config.prior_beta
        prior_alpha_mix = config.prior_alpha_mix
        beta_mix = config.beta_mix
        tie_precisions = config.tie_precisions
    else:
        raise TypeError(f"config must be a dict or GMMVBEMConfig, got {type(config)}")
    
    # Override with kwargs if provided
    num_clusters = kwargs.get("num_clusters", num_clusters)
    latent_dim = kwargs.get("latent_dim", latent_dim)
    prior_mu = kwargs.get("prior_mu", prior_mu)
    prior_alpha = kwargs.get("prior_alpha", prior_alpha)
    prior_beta = kwargs.get("prior_beta", prior_beta)
    prior_alpha_mix = kwargs.get("prior_alpha_mix", prior_alpha_mix)
    beta_mix = kwargs.get("beta_mix", beta_mix)
    tie_precisions = kwargs.get("tie_precisions", tie_precisions)
    
    return GMMVBEM(
        num_clusters=num_clusters,
        latent_dim=latent_dim,
        prior_mu=prior_mu,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        prior_alpha_mix=prior_alpha_mix,
        beta_mix=beta_mix,
        tie_precisions=tie_precisions
    )

