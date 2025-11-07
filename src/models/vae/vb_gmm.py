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
from dataclasses import dataclass, field
from typing import Tuple, Union, Optional

from src.utils.math_utils import logsumexp, stable_softmax
from src.utils.kl_divergence import gamma_kl, dirichlet_kl, normal_gamma_kl
from jax.scipy.special import digamma, gammaln
from flax.core import freeze


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


    def initialize_cluster_means(self, params: dict, z_e: jnp.ndarray, key: jr.PRNGKey) -> dict:
        """
        Initialize cluster means by randomly sampling data points.
        
        Args:
            params: Current GMM parameters dictionary
            z_e: Encoder outputs [N, latent_dim] - flattened to 2D
            key: Random key for sampling
            
        Returns:
            Updated params dictionary with initialized cluster means
        """
        # Flatten z_e to 2D if needed
        z_e_flat = z_e.reshape(-1, self.latent_dim)  # [N, latent_dim]
        N = z_e_flat.shape[0]
        
        # Randomly sample num_clusters data points (without replacement)
        # If we have fewer data points than clusters, add noise

        
        if N >= self.num_clusters:
            idx = jr.choice(key, jnp.arange(N), (self.num_clusters,), replace=False)
            mu_n = z_e_flat[idx]  # [num_clusters, latent_dim]
        else:
            mu_n = z_e_flat
            M = self.num_clusters - N
            mu_n = jnp.concatenate([mu_n, z_e_flat.mean(0, keepdims=True) + jr.normal(key, (M, self.latent_dim))], axis=0)
      
        # Set cluster means to selected data points
        # z_e_flat[idx] has shape [num_clusters, latent_dim]
        
        # Update params
        params['mu_n'] = mu_n
        return params


    @nn.compact
    def fill_unused(self, z_e: jnp.ndarray, training: bool = True) -> dict:
        """
        Fill unused clusters by reinitializing their means to worst-fit data points
        and resetting alpha_n and beta_n to prior values.
        
        Args:
            z_e: Encoder outputs [batch, ..., latent_dim]
            
        Returns:
            Updated GMM parameters
        """
        # Compute log_p_tilde using current params (accessed via self.*)
        log_p_tilde = self.log_p_tilde(z_e, training=training)
        
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
        
        z_e_flat = z_e.reshape(-1, self.latent_dim)  # [N, latent_dim]
        logZ_flat = logZ.reshape(-1)  # [N]
        N = z_e_flat.shape[0]
        
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
        worst_fit_z_e = z_e_flat[worst_fit_indices]  # [ns_actual, latent_dim]
        
        # Update params: set mu_n for unused clusters to worst-fit z_e values
        mu_n = self.mu_n.at[unused_indices, :].set(worst_fit_z_e)
        
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
    def nat_to_stats(self, training: bool = True) -> dict:
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
        
        mu_n = self.mu_n  # [num_clusters, latent_dim]
        alpha_n = self.alpha_n  # [num_clusters, 1]
        beta_n = self.beta_n  # [num_clusters, latent_dim]
        alpha_mix = self.alpha_mix  # [num_clusters]
        
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
    def log_p_tilde(self, z_e: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Compute unnormalized log probability of cluster assignments.
        
        Uses bound parameters from self (via nat_to_stats).
        
        log p̃(z_e, k) = E[log p(z_e | k)] + E[log π_k]
        
        Where:
        - E[log p(z_e | k)] is the expected Gaussian log-likelihood using E[μ_k], E[γ_k], and E[log(γ_k)]
        - E[log π_k] is the expected log mixing weight (NOT log(E[π_k]))
        
        The expectation is computed by averaging log_p(natparam * stats), so we only use:
        - E[γ] (precision) and E[log(γ)] (log precision) for the Gaussian
        - E[log(π)] for the mixing weights
        
        Args:
            z_e: Observations [batch, ..., latent_dim] or [N, latent_dim]
                
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
        
        original_shape = z_e.shape
        z_e_flat = z_e.reshape(-1, latent_dim)  # [N, latent_dim]
        
        # Ensure all terms have shape [num_clusters] for consistent broadcasting
        log_probs = E_log_pi - 0.5*latent_dim*E_gamma_var_mu.squeeze(-1) + 0.5*jnp.sum(E_log_gamma, axis=-1)  # [num_clusters]

        diff = z_e_flat[:, None, :] - E_mu[None, :, :]  # [N, num_clusters, latent_dim]
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
    def quantize(self, z_e: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute cluster assignments and get discrete representation.
        
        During training: samples cluster assignments from the categorical distribution.
        During inference: uses the most probable cluster (argmax) deterministically.
        
        Args:
            z_e: Encoder output [batch, ..., latent_dim]
            training: Whether in training mode (if True, samples; if False, uses argmax)
            
        Returns:
            Tuple of:
            - z_q: Quantized representation (cluster mean of selected/sampled cluster) [batch, ..., latent_dim]
            - log_p_tilde: Unnormalized log probabilities [batch, ..., num_clusters]
        """
        # Compute unnormalized log probabilities
        # When called via apply, Flax automatically binds parameters, so self.mu_n etc will use the bound params
        # log_p_tilde uses bound parameters from self (via nat_to_stats)
        log_p_tilde = self.log_p_tilde(z_e, training=training)  # [batch, ..., num_clusters]     

        mu_n = self.mu_n
        original_shape = z_e.shape
        
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
        z_q_flat = mu_n[selected_clusters_flat]  # [N, latent_dim]
        z_q = z_q_flat.reshape(original_shape)  # [batch, ..., latent_dim]
        
        return z_q, log_p_tilde
    
    def __call__(self, z_e: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Initialize all @nn.compact methods by calling them.
        
        This method is used to initialize the compact methods (quantize, log_p_tilde, nat_to_stats)
        when the module is first set up. It delegates to quantize for the forward pass.
        
        Args:
            z_e: Encoder output [batch, ..., latent_dim]
            training: Whether in training mode
            
        Returns:
            Tuple of:
            - z_q: Quantized representation (cluster mean of selected/sampled cluster) [batch, ..., latent_dim]
            - log_p_tilde: Unnormalized log probabilities [batch, ..., num_clusters]
        """
        # Delegate to quantize for the actual forward pass
        # When called via init(), this will initialize quantize and all methods it calls
        return self.quantize(z_e, training=training)

    @nn.compact
    def kl_prior(self, training: bool = True) -> jnp.ndarray:
        """
        Compute KL divergence between posterior and prior distributions.
        
        Computes:
        1. KL for Normal-Gamma posterior/prior for each cluster's mean and precision
        2. KL for Dirichlet posterior/prior for mixing weights
        
        Returns:
            Total KL divergence (scalar)
        """
        # Access parameters via self.* (bound by @nn.compact)
        mu_n = self.mu_n  # [num_clusters, latent_dim]
        alpha_n = self.alpha_n  # [num_clusters, 1]
        beta_n = self.beta_n  # [num_clusters, latent_dim]
        alpha_mix = self.alpha_mix  # [num_clusters]
        
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
    def loss(self, z_e: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Compute GMM loss including negative log-likelihood and KL divergence.
        
        Loss = -E[log p(z_e)] + KL(q(θ) || p(θ))
        where:
        - E[log p(z_e)] is the expected log-likelihood (negative logZ)
        - KL(q(θ) || p(θ)) is the KL divergence between posterior and prior
        
        Args:
            z_e: Encoder outputs [batch, ..., latent_dim]
            
        Returns:
            Total GMM loss (scalar)
        """
        log_p_tilde = self.log_p_tilde(z_e, training=training)
        logZ = logsumexp(log_p_tilde, axis=-1)
        logZ = jnp.sum(logZ)
        
        # Add KL divergence between posterior and prior
        kl = self.kl_prior(training=training)
        
        return -logZ + kl

    @partial(nn.jit, static_argnames=('N_eff', 'lr', 'training'))
    @nn.compact
    def update(
        self,
        z_e: jnp.ndarray,
        N_eff: float = 2000.0,
        lr: float = 0.2,
        training: bool = True
    ) -> dict:
        # Access parameters via self.* (bound by @nn.compact)
        mu_n = self.mu_n
        alpha_n = self.alpha_n
        beta_n = self.beta_n
        alpha_mix = self.alpha_mix

        # Compute log_p_tilde using current params
        log_p_tilde = self.log_p_tilde(z_e, training=training)
        
        cluster_probs = stable_softmax(log_p_tilde, axis=-1)  # [batch, ..., num_clusters]
        z_e_flat = z_e.reshape(-1, self.latent_dim)  # [N, latent_dim]
        # Ensure cluster_probs is flattened correctly - handle any number of leading dimensions
        # Flatten all leading dimensions, keeping only the last dimension (num_clusters)
        # Compute responsibilities (soft assignments)
        # cluster_probs should already be normalized (softmax applied in calling code)
        r_nk = cluster_probs.reshape(-1, self.num_clusters)  # [M, num_clusters]
                
        N_k = jnp.sum(r_nk, axis=0)  # [num_clusters] - effective number of points in each cluster
        weighted_sum = jnp.sum(r_nk[:, :, None] * z_e_flat[:, None, :], axis=0)  # [num_clusters, latent_dim]

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

        # LEARNING RATE UPDATES
        N_scale = N_eff / z_e_flat.shape[0]  # Add epsilon to prevent division by zero
        
        # Compute current kappa_mu_n before updates
        kappa_mu_n = 2.0 * alpha_n * mu_n  # [num_clusters, latent_dim] - broadcasting: (3,1) * (3,2) -> (3,2)
        
        # Prior term for kappa_mu_n (pulls unused clusters towards prior mean)
        # When N_k is zero, we want to pull towards prior: kappa_mu_prior = 2 * prior_alpha * prior_mu
        kappa_mu_prior = 2.0 * self.prior_alpha * self.prior_mu  # [num_clusters, latent_dim] - JAX broadcasts

        # For alpha_n: when N_k is small, decay towards prior_alpha
        # Use a weighted combination: data term (when N_k > 0) and prior term (always present)

        alpha_n_like = N_scale * 0.5 * N_k[:, None]
        alpha_mix_like = N_scale * N_k

        alpha_n = (1 - lr) * alpha_n + lr * (alpha_n_like + self.prior_alpha)        
        alpha_mix = (1 - lr) * alpha_mix + lr * (alpha_mix_like + self.prior_alpha_mix)    
        
        # Update kappa_mu_n: 
        # - When weighted_sum has data (N_k > 0), use weighted_sum
        # - When weighted_sum is zero (N_k = 0), pull towards prior
        # Use a weighted combination based on whether cluster has data
        kappa_mu_n_data_term = N_scale * weighted_sum  # [num_clusters, latent_dim]
        # For unused clusters (N_k ≈ 0), add prior term to pull towards prior_mu
        # Scale the prior term by (1.0 / N_scale) to keep it reasonable when N_scale is large
        kappa_mu_n = (1 - lr) * kappa_mu_n + lr * (kappa_mu_n_data_term + kappa_mu_prior)
        mu_n = kappa_mu_n / (2.0 * alpha_n)  # because alpha_n was already updated, add epsilon for stability
        
        diff = z_e_flat[:, None, :] - mu_n[None, :, :]  # [N, num_clusters, latent_dim]
        weighted_diff_sq = jnp.sum(r_nk[:, :, None] * (diff ** 2), axis=0)  # [num_clusters, latent_dim]
        prior_diff_sq =  self.prior_alpha/(alpha_n_like + self.prior_alpha)  * ((mu_n - self.prior_mu) ** 2)   # [num_clusters, latent_dim] - JAX broadcasts
        beta_n = (1 - lr) * beta_n + lr * (N_scale * 0.5 * (weighted_diff_sq + prior_diff_sq) + self.prior_beta/self.num_clusters**(2/self.latent_dim))
                        
        # Create updated dict
        updated_params = {
            'mu_n': mu_n,
            'alpha_n': alpha_n,
            'beta_n': beta_n,
            'alpha_mix': alpha_mix
        }
        
        return updated_params        

