"""Variational Bayesian VAE (VBVAE) model.

This model uses a GMM with Variational Bayesian EM for clustering encoder outputs.
The GMM parameters use Normal-Gamma conjugate priors for means and precision (γ = 1/σ²).
"""

import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from flax.core import FrozenDict
from dataclasses import dataclass, field
from typing import Tuple

from src.utils.math_utils import logsumexp
from jax.scipy.special import digamma


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
    prior_mu: float = 0.0
    prior_alpha: float = 0.5
    prior_beta: float = 0.5      # will be divided by num_clusters when used.
    prior_alpha_mix: float = 0.5
    
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
            lambda key, shape: jnp.ones(shape) * self.prior_beta,
            (self.num_clusters, self.latent_dim)
        )
        
        # Posterior parameters for mixing weights: π ~ Dirichlet(α_mix)
        # Shape [num_clusters]
        self.alpha_mix = self.param(
            'alpha_mix',  # Posterior Dirichlet parameters
            lambda key, shape: jnp.ones(shape) * self.prior_alpha_mix,
            (self.num_clusters,)
        )


    def initialize_cluster_means(self, params: dict, z_e: jnp.ndarray, key: jr.PRNGKey) -> jnp.ndarray:
        """
        Initialize cluster means to the mean of the data points.
        
        Args:
            z_e: ALL OF THE DATA GOES IN HERE==
            
        Returns:
            mu_n: Initialized cluster means [num_clusters, latent_dim]
        """
        idx = jr.choice(key, jnp.arange(z_e.shape[0]), (self.num_clusters,), replace=False)
        mu_n = params['mu_n'].at[idx].set(z_e[idx])
        idx = idx[:self.num_clusters]
        params['mu_n'] = mu_n
        return params


    def fill_unused(
        self,
        params: dict,
        z_e: jnp.ndarray,
        cluster_probs: jnp.ndarray,
        logZ: jnp.ndarray
    ) -> Tuple[dict, jnp.ndarray]:
        """
        Fill unused clusters by reinitializing their means to worst-fit data points
        and resetting alpha_n and beta_n to prior values.
        Also updates cluster_probs to set assignment probabilities to 1.0 for data points
        used to fill the unused clusters.
        
        Args:
            params: Current GMM parameters
            z_e: Encoder outputs [batch, ..., latent_dim]
            cluster_probs: Cluster assignment probabilities [batch, ..., num_clusters]
            logZ: Log partition function [batch, ...]
            
        Returns:
            Tuple of (updated_params, updated_cluster_probs)
        """
        # Get alpha_mix to identify unused clusters
        alpha_mix = params['alpha_mix']  # [num_clusters]
        unused_mask = alpha_mix < 1.5  # [num_clusters] boolean mask
        unused_indices = jnp.where(unused_mask)[0]  # [ns] cluster indices to fill
        ns = unused_indices.shape[0]
        
        # If no unused clusters, return original params and cluster_probs
        if ns == 0:
            return params, cluster_probs
        
        # Flatten z_e, cluster_probs, and logZ for processing
        z_e_flat = z_e.reshape(-1, self.latent_dim)  # [N, latent_dim]
        cluster_probs_flat = cluster_probs.reshape(-1, self.num_clusters)  # [N, num_clusters]
        logZ_flat = logZ.reshape(-1)  # [N]
        N = z_e_flat.shape[0]
        
        # Limit number of clusters to fill to available data points
        ns = jnp.minimum(ns, N)
        unused_indices = unused_indices[:ns]  # [ns]
        
        # Find ns worst-fit data points (smallest logZ)
        worst_fit_indices = jnp.argsort(logZ_flat)[:ns]  # [ns]
        worst_fit_z_e = z_e_flat[worst_fit_indices]  # [ns, latent_dim]
        
        # Update params: set mu_n for unused clusters to worst-fit z_e values
        mu_n = params['mu_n'].at[unused_indices, :].set(worst_fit_z_e)
        
        # Reset alpha_n and beta_n to prior values for unused clusters
        alpha_n = params['alpha_n'].at[unused_indices, :].set(self.prior_alpha + 1.0)
        beta_n = params['beta_n'].at[unused_indices, :].set(self.prior_beta/self.num_clusters + 1.0)
        alpha_mix = params['alpha_mix'].at[unused_indices].set(self.prior_alpha_mix + 1.0)
        
        # Update cluster_probs: set assignment probabilities to 1.0 for worst-fit points
        # assigned to their corresponding newly filled clusters
        # Only modify cluster_probs for the worst-fit data points
        updated_cluster_probs_flat = cluster_probs_flat.copy()
        
        # For each worst-fit point i, set probability to 1.0 at cluster unused_indices[i], 0.0 elsewhere
        # Create one-hot vectors for each worst-fit point: [ns, num_clusters]
        one_hot_assignments = jnp.zeros((ns, self.num_clusters))
        one_hot_assignments = one_hot_assignments.at[jnp.arange(ns), unused_indices].set(1.0)
        
        # Update only the worst-fit points
        updated_cluster_probs_flat = updated_cluster_probs_flat.at[worst_fit_indices].set(one_hot_assignments)
        
        # Reshape cluster_probs back to original shape
        original_shape = cluster_probs.shape
        updated_cluster_probs = updated_cluster_probs_flat.reshape(original_shape)
        
        # Return updated params and cluster_probs
        updated_params = {
            'mu_n': mu_n,
            'alpha_n': alpha_n,
            'beta_n': beta_n,
            'alpha_mix': alpha_mix
        }
        return updated_params, updated_cluster_probs


    @nn.compact
    def nat_to_stats(self) -> dict:
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
        E_gamma = alpha_n / beta_n  # [num_clusters, latent_dim] (broadcasts automatically)
        
        # Expected μ²: E[μ²] = μₙ² + Var[μ]
        # Since p(μ|γ,κ) = Normal(μ₀, 1/(κ·γ)), we have Var[μ|γ] = 1/(κ·γ)
        # In the expected log-likelihood, γ·Var[μ] simplifies to 1/κ
        # So Var[μ] = 1/κ (for the unconditional variance used in the log-likelihood)
        
        # Expected log precision: E[log(γ)] = ψ(αₙ) - log(βₙ)
        E_log_gamma = digamma(alpha_n) - jnp.log(beta_n)  # [num_clusters, latent_dim] (broadcasts automatically)
        
        # Expected mixing weights: E[π] = α_mix / Σ_k α_mix_k
        alpha_mix_sum = jnp.sum(alpha_mix)  # scalar
        E_pi = alpha_mix / (alpha_mix_sum + 1e-8)  # [num_clusters]
        
        # Expected log mixing weights: E[log(π_k)] = ψ(α_mix_k) - ψ(Σ_k α_mix_k)
        E_log_pi = digamma(alpha_mix) - digamma(alpha_mix_sum)  # [num_clusters]
        E_log_pi = jnp.zeros_like(E_log_pi)
        
        return {
            'E_mu': E_mu,
            'E_gamma_var_mu': E_gamma_var_mu,
            'E_log_gamma': E_log_gamma,
            'E_gamma': E_gamma,
            'E_pi': E_pi,
            'E_log_pi': E_log_pi,
        }
    
    
    @nn.compact
    def log_p_tilde(self, z_e: jnp.ndarray) -> jnp.ndarray:
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
        expectations = self.nat_to_stats()
        
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
        log_probs = log_probs - 0.5 * jnp.sum(E_gamma[None, :, :] * (diff ** 2), axis=-1)  # [N, num_clusters]
                
        # Reshape to original spatial dimensions
        if len(original_shape) > 2:
            log_probs = log_probs.reshape(original_shape[:-1] + (num_clusters,))
        
        return log_probs
    
    def __call__(self, z_e: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute cluster assignments and get discrete representation.
        
        Uses the most probable cluster (argmax) deterministically.
        
        Args:
            z_e: Encoder output [batch, ..., latent_dim]
            
        Returns:
            Tuple of:
            - z_q: Quantized representation (cluster mean of most probable cluster) [batch, ..., latent_dim]
            - cluster_probs: Cluster probabilities [batch, ..., num_clusters]
            - logZ: Log partition function (log-sum-exp of log_p_tilde) [batch, ...]
        """
        # Compute unnormalized log probabilities
        # When called via apply, Flax automatically binds parameters, so self.mu_n etc will use the bound params
        # log_p_tilde uses bound parameters from self (via nat_to_stats)
        log_p_tilde = self.log_p_tilde(z_e)  # [batch, ..., num_clusters]
        selected_clusters = jnp.argmax(log_p_tilde, axis=-1)  # [batch, ...]

        mu_n = self.mu_n
        
        original_shape = z_e.shape
        selected_clusters = selected_clusters.flatten()  # [N]
        z_q_flat = mu_n[selected_clusters]  # [N, latent_dim]
        z_q = z_q_flat.reshape(original_shape)  # [batch, ..., latent_dim]
        
        return z_q, logsumexp(log_p_tilde, axis=-1)
    

    def loss(self, params: dict, z_e: jnp.ndarray) -> jnp.ndarray:
        return -self.apply(params, z_e)[-1]


    def update(
        self,
        params: dict,
        z_e: jnp.ndarray,
        cluster_probs: jnp.ndarray,
        logZ: jnp.ndarray,
        N_eff: float,
        lr: float = 0.2,
        use_fill_unused: bool = True
    ) -> dict:
        z_e_flat = z_e.reshape(-1, self.latent_dim)  # [N, latent_dim]
        probs_flat = cluster_probs.reshape(-1, self.num_clusters)  # [N, num_clusters]
        N = z_e_flat.shape[-2]

        # Get current posterior parameters
        mu_n = params['mu_n']  # [num_clusters, latent_dim]
        alpha_n = params['alpha_n']  # [num_clusters, latent_dim]
        beta_n = params['beta_n']  # [num_clusters, latent_dim]
        alpha_mix = params['alpha_mix']  # [num_clusters]
        
        # Compute responsibilities (soft assignments)
        r_nk = probs_flat  # [N, num_clusters]
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
        N_scale = N_eff / N
        
        # Compute current kappa_mu_n before updates
        kappa_mu_n = 2.0 * alpha_n * mu_n  # [num_clusters, latent_dim] - broadcasting: (3,1) * (3,2) -> (3,2)

        alpha_n = (1 - lr) * alpha_n + lr * N_scale * 0.5 * N_k[:, None]
        alpha_mix = (1 - lr) * alpha_mix + lr * N_scale * N_k    
        kappa_mu_n = (1 - lr) * kappa_mu_n + lr * N_scale * weighted_sum
        mu_n = kappa_mu_n / (2.0 * alpha_n)  # because alpha_n was already updated
        
        diff = z_e_flat[:, None, :] - mu_n[None, :, :]  # [N, num_clusters, latent_dim]
        weighted_diff_sq = jnp.sum(r_nk[:, :, None] * (diff ** 2), axis=0)  # [num_clusters, latent_dim]
        prior_diff_sq = 2.0 * self.prior_alpha * ((mu_n - self.prior_mu) ** 2)  # [num_clusters, latent_dim]
        beta_n = (1 - lr) * beta_n + lr * (N_scale * 0.5 * weighted_diff_sq + 0.5 * prior_diff_sq)
                        
        # Create updated dict
        updated_params = {
            'mu_n': mu_n,
            'alpha_n': alpha_n,
            'beta_n': beta_n,
            'alpha_mix': alpha_mix
        }
        
        # Fill unused clusters by reinitializing their means and resetting alpha_n/beta_n
        # Also updates cluster_probs for data points assigned to newly filled clusters
        if use_fill_unused:
            updated_params, _ = self.fill_unused(updated_params, z_e, cluster_probs, logZ)
        
        return updated_params        

