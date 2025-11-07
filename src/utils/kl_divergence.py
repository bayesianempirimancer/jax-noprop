"""
KL divergence utility functions for variational Bayesian models.

This module provides functions for computing KL divergences between
posterior and prior distributions.

Each distribution has a function:
- {distribution}_kl: Computes KL divergence between two distributions
"""

import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln


def gamma_kl(
    alpha_p: jnp.ndarray,
    beta_p: jnp.ndarray,
    alpha_q: jnp.ndarray,
    beta_q: jnp.ndarray,
    epsilon: float = 1e-8
) -> jnp.ndarray:
    """
    Compute KL divergence between two Gamma distributions.
    
    KL(Gamma(α_p, β_p) || Gamma(α_q, β_q)) =
      (α_p - α_q) * digamma(α_p) - gammaln(α_p) + gammaln(α_q)
      + α_q * log(β_p/β_q) + α_p * (β_q/β_p - 1)
    
    Args:
        alpha_p: Shape parameter of first Gamma distribution [..., 1] or [..., latent_dim]
        beta_p: Rate parameter of first Gamma distribution [..., 1] or [..., latent_dim]
        alpha_q: Shape parameter of second Gamma distribution (scalar or array)
        beta_q: Rate parameter of second Gamma distribution (scalar or array)
        epsilon: Small value for numerical stability
        
    Returns:
        KL divergence with broadcast shape of inputs
    """
    # Convert q parameters to arrays - JAX broadcasting will handle scalars
    alpha_q_arr = jnp.asarray(alpha_q)
    beta_q_arr = jnp.asarray(beta_q)
    
    # Compute KL divergence
    kl = (
        (alpha_p - alpha_q_arr) * digamma(alpha_p) -  # [..., 1] or [..., latent_dim]
        gammaln(alpha_p) +  # [..., 1] or [..., latent_dim]
        gammaln(alpha_q_arr) +  # scalar or [..., 1] or [..., latent_dim]
        alpha_q_arr * jnp.log((beta_p + epsilon) / (beta_q_arr + epsilon)) +  # [..., 1] or [..., latent_dim]
        alpha_p * (beta_q_arr / (beta_p + epsilon) - 1.0)  # [..., 1] or [..., latent_dim]
    )
    
    return kl


def dirichlet_kl(
    alpha_p: jnp.ndarray,
    alpha_q: jnp.ndarray,
    epsilon: float = 1e-8
) -> jnp.ndarray:
    """
    Compute KL divergence between two Dirichlet distributions.
    
    KL(Dirichlet(α_p) || Dirichlet(α_q)) =
      log(Γ(Σα_p)) - log(Γ(Σα_q)) - Σ[log(Γ(α_p)) - log(Γ(α_q))]
      + Σ[(α_p - α_q) * (digamma(α_p) - digamma(Σα_p))]
    
    Args:
        alpha_p: Concentration parameters of first Dirichlet distribution [..., K]
        alpha_q: Concentration parameters of second Dirichlet distribution (scalar or array)
        epsilon: Small value for numerical stability (not used, kept for consistency)
        
    Returns:
        KL divergence with broadcast shape of inputs
    """
    # Convert alpha_q to array and broadcast to match alpha_p shape if needed
    alpha_q_arr = jnp.asarray(alpha_q)
    # If alpha_q is scalar or has fewer dims, broadcast to match alpha_p
    if alpha_q_arr.ndim < alpha_p.ndim:
        alpha_q_arr = jnp.broadcast_to(alpha_q_arr, alpha_p.shape)
    
    # Sum of parameters along the last dimension (the Dirichlet dimension)
    # For shape [..., K], sum over axis=-1 to get [..., 1] or [...]
    sum_alpha_p = jnp.sum(alpha_p, axis=-1, keepdims=True)  # [..., 1]
    sum_alpha_q = jnp.sum(alpha_q_arr, axis=-1, keepdims=True)  # [..., 1]
    
    # Compute KL divergence
    kl = (
        gammaln(sum_alpha_p) -  # log(Γ(Σα_p)) [..., 1]
        gammaln(sum_alpha_q) -  # log(Γ(Σα_q)) [..., 1]
        jnp.sum(gammaln(alpha_p), axis=-1, keepdims=True) +  # -Σ[log(Γ(α_p))] [..., 1]
        jnp.sum(gammaln(alpha_q_arr), axis=-1, keepdims=True) +  # +Σ[log(Γ(α_q))] [..., 1]
        jnp.sum((alpha_p - alpha_q_arr) * (digamma(alpha_p) - digamma(sum_alpha_p)), axis=-1, keepdims=True)  # [..., 1]
    )
    
    return kl


def normal_gamma_kl(
    kappa_p: jnp.ndarray,
    mu_p: jnp.ndarray,
    alpha_p: jnp.ndarray,
    beta_p: jnp.ndarray,
    kappa_q: jnp.ndarray,
    mu_q: jnp.ndarray,
    alpha_q: jnp.ndarray,
    beta_q: jnp.ndarray,
    epsilon: float = 1e-8
) -> jnp.ndarray:
    """
    Compute KL divergence between two Normal-Gamma distributions.
    
    The Normal-Gamma distribution has:
    - Normal part: μ ~ Normal(μ₀, 1/(κ·γ)) where κ is precision scale and γ is precision
    - Gamma part: γ ~ Gamma(α, β)
    
    KL(Normal-Gamma) = KL(Normal) + KL(Gamma)
    
    KL(Normal) = 0.5 * [log(κ_q/κ_p) + κ_p/κ_q * (μ_p - μ_q)² - 1]
    where we use E[γ] = α/β for the precision
    
    KL(Gamma) = (α_p - α_q) * digamma(α_p) - gammaln(α_p) + gammaln(α_q)
                + α_q * log(β_p/β_q) + α_p * (β_q/β_p - 1)
    
    Args:
        kappa_p: Precision scale of first Normal distribution [..., 1] or broadcastable
        mu_p: Mean of first Normal distribution [..., D]
        alpha_p: Shape parameter of first Gamma distribution [..., 1] or broadcastable
        beta_p: Rate parameter of first Gamma distribution [..., D]
        kappa_q: Precision scale of second Normal distribution (scalar or array)
        mu_q: Mean of second Normal distribution (scalar or array)
        alpha_q: Shape parameter of second Gamma distribution (scalar or array)
        beta_q: Rate parameter of second Gamma distribution (scalar or array)
        epsilon: Small value for numerical stability
        
    Returns:
        KL divergence with broadcast shape of inputs
    """
    # Convert q parameters to arrays - JAX broadcasting will handle scalars
    kappa_q_arr = jnp.asarray(kappa_q)
    mu_q_arr = jnp.asarray(mu_q)
    alpha_q_arr = jnp.asarray(alpha_q)
    beta_q_arr = jnp.asarray(beta_q)
    
    # KL for Normal part
    # The conditional precision is κ·γ, but γ drops out of log and ratio terms
    # KL(Normal) = 0.5 * [log(κ_q/κ_p) + κ_p/κ_q + κ_q * E[γ_p] * (μ_p - μ_q)² - 1]
    # Note: kappa is always bounded below by the prior, so no epsilon needed
    E_gamma_p = alpha_p / beta_p  # [..., D] after broadcasting
    
    mu_diff_sq = (mu_p - mu_q_arr) ** 2  # [..., D]
    kl_normal = 0.5 * (
        jnp.log(kappa_q_arr / kappa_p) +  # [..., D] after broadcasting - γ drops out
        kappa_p / kappa_q_arr +  # [..., D] - γ drops out
        kappa_q_arr * E_gamma_p * mu_diff_sq -  # [..., D] - uses κ_q * E[γ_p]
        1.0
    )  # [..., D]
    
    # KL for Gamma part - reuse gamma_kl function
    kl_gamma = gamma_kl(alpha_p, beta_p, alpha_q, beta_q, epsilon=epsilon)
    
    # Total KL = KL(Normal) + KL(Gamma)
    kl = kl_normal + kl_gamma
    
    return kl

