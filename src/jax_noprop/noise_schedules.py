"""
Noise scheduling utilities for NoProp variants.

This module provides different noise scheduling strategies used in the NoProp paper:
- Linear schedule
- Cosine schedule  
- Sigmoid schedule
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class NoiseSchedule(ABC):
    """Abstract base class for noise schedules."""
    
    @abstractmethod
    def get_alpha_t(self, t: jnp.ndarray) -> jnp.ndarray:
        """Get alpha_t values for given timesteps t."""
        pass
    
    @abstractmethod
    def get_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
        """Get sigma_t values for given timesteps t."""
        pass
    
    def get_noise_params(self, t: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Get both alpha_t and sigma_t for given timesteps."""
        return {
            "alpha_t": self.get_alpha_t(t),
            "sigma_t": self.get_sigma_t(t),
        }


@struct.dataclass
class LinearNoiseSchedule(NoiseSchedule):
    """Linear noise schedule as used in the NoProp paper."""
    
    def get_alpha_t(self, t: jnp.ndarray) -> jnp.ndarray:
        """Linear schedule: alpha_t = 1 - t."""
        return 1.0 - t
    
    def get_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
        """Linear schedule: sigma_t = sqrt(t)."""
        return jnp.sqrt(t)


@struct.dataclass
class CosineNoiseSchedule(NoiseSchedule):
    """Cosine noise schedule for smoother transitions."""
    
    def get_alpha_t(self, t: jnp.ndarray) -> jnp.ndarray:
        """Cosine schedule: alpha_t = cos(π/2 * t)."""
        return jnp.cos(jnp.pi / 2 * t)
    
    def get_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
        """Cosine schedule: sigma_t = sin(π/2 * t)."""
        return jnp.sin(jnp.pi / 2 * t)


@struct.dataclass
class SigmoidNoiseSchedule(NoiseSchedule):
    """Sigmoid noise schedule with learnable parameters."""
    
    gamma: float = 1.0  # Controls the steepness of the sigmoid
    
    def get_alpha_t(self, t: jnp.ndarray) -> jnp.ndarray:
        """Sigmoid schedule: alpha_t = σ(-γ(t))."""
        return jax.nn.sigmoid(-self.gamma * t)
    
    def get_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
        """Sigmoid schedule: sigma_t = σ(γ(t))."""
        return jax.nn.sigmoid(self.gamma * t)


@struct.dataclass
class LearnableNoiseSchedule(NoiseSchedule):
    """Learnable noise schedule as described in Appendix B of the NoProp paper.
    
    This implements the trainable noise schedule where:
    SNR(t) = exp(-γ(t))
    γ(t) = γ₀ + (γ₁ - γ₀)(1 - γ̄(t))
    
    where γ̄(t) is a normalized neural network output.
    """
    
    gamma_0: float = 0.0  # Starting value of γ
    gamma_1: float = 1.0  # Ending value of γ
    
    def get_alpha_t(self, t: jnp.ndarray) -> jnp.ndarray:
        """Get alpha_t from the learnable schedule.
        
        For the learnable schedule, we use:
        ᾱ_t = σ(-γ(t))
        where σ is the sigmoid function.
        """
        gamma_t = self._compute_gamma_t(t)
        return jax.nn.sigmoid(-gamma_t)
    
    def get_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
        """Get sigma_t from the learnable schedule.
        
        For the learnable schedule, we use:
        σ̄_t = σ(γ(t))
        where σ is the sigmoid function.
        """
        gamma_t = self._compute_gamma_t(t)
        return jax.nn.sigmoid(gamma_t)
    
    def _compute_gamma_t(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute γ(t) for the learnable schedule.
        
        This is a simplified version. In practice, this should be computed
        using a neural network as described in the paper.
        """
        # Simplified version: linear interpolation between gamma_0 and gamma_1
        # In practice, this should use a neural network to compute γ̄(t)
        gamma_bar_t = t  # Simplified: γ̄(t) = t
        gamma_t = self.gamma_0 + (self.gamma_1 - self.gamma_0) * (1 - gamma_bar_t)
        return gamma_t
    
    def get_snr_derivative(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute the derivative of SNR with respect to time.
        
        SNR(t) = exp(-γ(t))
        dSNR/dt = -γ'(t) * exp(-γ(t))
        """
        gamma_t = self._compute_gamma_t(t)
        gamma_derivative = -(self.gamma_1 - self.gamma_0)  # Simplified derivative
        snr_derivative = -gamma_derivative * jnp.exp(-gamma_t)
        return jnp.abs(snr_derivative)  # Take absolute value for weighting


def create_noise_schedule(
    schedule_type: str = "linear", 
    **kwargs: Any
) -> NoiseSchedule:
    """Factory function to create noise schedules.
    
    Args:
        schedule_type: Type of schedule ("linear", "cosine", "sigmoid")
        **kwargs: Additional parameters for the schedule
        
    Returns:
        NoiseSchedule instance
    """
    if schedule_type == "linear":
        return LinearNoiseSchedule()
    elif schedule_type == "cosine":
        return CosineNoiseSchedule()
    elif schedule_type == "sigmoid":
        return SigmoidNoiseSchedule(**kwargs)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def add_noise(
    clean: jnp.ndarray,
    noise: jnp.ndarray, 
    alpha_t: jnp.ndarray,
    sigma_t: jnp.ndarray,
) -> jnp.ndarray:
    """Add noise to clean data using the reparameterization trick.
    
    Args:
        clean: Clean data [batch_size, ...]
        noise: Gaussian noise [batch_size, ...]
        alpha_t: Alpha values [batch_size, 1] or scalar
        sigma_t: Sigma values [batch_size, 1] or scalar
        
    Returns:
        Noisy data: alpha_t * clean + sigma_t * noise
    """
    # Ensure proper broadcasting
    if alpha_t.ndim == 0:
        alpha_t = alpha_t[None, None]
    elif alpha_t.ndim == 1:
        alpha_t = alpha_t[:, None]
        
    if sigma_t.ndim == 0:
        sigma_t = sigma_t[None, None]
    elif sigma_t.ndim == 1:
        sigma_t = sigma_t[:, None]
    
    # Reshape for broadcasting across spatial dimensions
    while alpha_t.ndim < clean.ndim:
        alpha_t = alpha_t[..., None]
    while sigma_t.ndim < clean.ndim:
        sigma_t = sigma_t[..., None]
    
    return alpha_t * clean + sigma_t * noise


def sample_noise(key: jax.random.PRNGKey, shape: tuple) -> jnp.ndarray:
    """Sample Gaussian noise with the given shape."""
    return jax.random.normal(key, shape)
