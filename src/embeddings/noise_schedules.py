"""
Noise scheduling utilities for NoProp variants.

This module provides different noise scheduling strategies used in the NoProp paper:
- Linear schedule
- Cosine schedule  
- Sigmoid schedule
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.special
import flax.linen as nn


class NoiseSchedule(nn.Module):
    """Abstract base class for noise schedules.
    
    Following the paper notation:
    - ᾱ(t) is the signal strength coefficient (INCREASING with time)

    - 1 - ᾱ(t) represents cumulative noise added by backward process

    - Backward process: z_t = sqrt(ᾱ(t)) * z_1 + sqrt(1-ᾱ(t)) * ε
      where z_1 is the target/starting point of the backward process

    - Note that we are using alpha_t to denote ᾱ(t)

    - Note that for numerical stability of the forward process alpha(t)
      miust be an increasing function bounded away from 0 and 1.

    - Note that the underlying backward OU process is given by 
      dz = δ(t)/2 * z * dt + sqrt(δ(t)) * dW(t), where δ(t) = ᾱ'(t)/ᾱ(t)
    """
    
    def get_gamma_gamma_prime_t(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get both gamma_t and gamma_prime_t values for given timesteps t.
        
        Args:
            t: Time values [batch_size]
            
        Returns:
            Tuple of (gamma_t, gamma_prime_t) where:
            - gamma_t: Gamma values [batch_size]
            - gamma_prime_t: Gamma derivative values [batch_size]
        """
        raise NotImplementedError("Subclasses must implement get_gamma_gamma_prime_t")

    # Utility methods for computing derived quantities from gamma values
    @staticmethod
    def get_alpha_from_gamma(gamma_t: jnp.ndarray) -> jnp.ndarray:
        """   
        ᾱ(t) = sigmoid(γ(t))
        """
        return jax.nn.sigmoid(gamma_t)
    
    @staticmethod
    def get_alpha_prime_from_gamma(gamma_t: jnp.ndarray, gamma_prime_t: jnp.ndarray) -> jnp.ndarray:
        """
        ᾱ'(t) = γ'(t) * sigmoid(γ(t)) * (1 - sigmoid(γ(t)))
        ᾱ'(t) = γ'(t) * ᾱ(t) * (1 - ᾱ(t))
        """
        alpha_t = NoiseSchedule.get_alpha_from_gamma(gamma_t)
        return gamma_prime_t * alpha_t * (1.0 - alpha_t)
    
    @staticmethod
    def get_sigma_from_gamma(gamma_t: jnp.ndarray) -> jnp.ndarray:
        """
        σ(t) = sqrt(1 - ᾱ(t)) = sqrt(1 - sigmoid(γ(t)))
        """
        alpha_t = NoiseSchedule.get_alpha_from_gamma(gamma_t)
        return jnp.sqrt(1.0 - alpha_t)
    
    @staticmethod
    def get_snr_from_gamma(gamma_t: jnp.ndarray) -> jnp.ndarray:
        """
        SNR(t) = ᾱ(t) / (1 - ᾱ(t)) = exp(γ(t))
        """
        return jnp.exp(gamma_t)
    
    @staticmethod
    def get_snr_prime_from_gamma(gamma_t: jnp.ndarray, gamma_prime_t: jnp.ndarray) -> jnp.ndarray:
        """        
        SNR'(t) = γ'(t) * exp(γ(t))
        """
        return gamma_prime_t * jnp.exp(gamma_t)
    
    @staticmethod
    def get_tau_inverse_from_gamma(gamma_t: jnp.ndarray, gamma_prime_t: jnp.ndarray) -> jnp.ndarray:
        """
        Time constant: 1/τ(t) = γ'(t)
        """
        return gamma_prime_t

    def get_alpha_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """
        ᾱ(t) = sigmoid(γ(t))
        """
        gamma_t, _ = self.get_gamma_gamma_prime_t(t, params)
        return self.get_alpha_from_gamma(gamma_t)
    
    def get_alpha_prime_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """
        ᾱ'(t) = γ'(t) * sigmoid(γ(t)) * (1 - sigmoid(γ(t)))
        ᾱ'(t) = γ'(t) * ᾱ(t) * (1 - ᾱ(t))
        """
        gamma_t, gamma_prime_t = self.get_gamma_gamma_prime_t(t, params)
        return self.get_alpha_prime_from_gamma(gamma_t, gamma_prime_t)

    def get_sigma_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """
        Noise associated with the backward process: z_t = sqrt(ᾱ(t)) * z_1 + σ(t) * ε
        σ(t) = sqrt(1 - ᾱ(t))
        """
        gamma_t, _ = self.get_gamma_gamma_prime_t(t, params)
        return self.get_sigma_from_gamma(gamma_t)
        
    def get_snr(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Compute the Signal-to-Noise Ratio (SNR) at given timesteps.        
        SNR(t) = ᾱ(t) / (1 - ᾱ(t))
        Since ᾱ(t) = sigmoid(γ(t)), we have:
        SNR(t) = exp(γ(t))
        """
        gamma_t, _ = self.get_gamma_gamma_prime_t(t, params)
        return self.get_snr_from_gamma(gamma_t)
    
    def get_snr_prime(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Compute the derivative of SNR with respect to time.
        SNR(t) = ᾱ(t) / (1 - ᾱ(t))
        Since ᾱ(t) = sigmoid(γ(t)), we have:
        SNR(t) = exp(γ(t))
        SNR'(t) = γ'(t) * exp(γ(t))
        """
        gamma_t, gamma_prime_t = self.get_gamma_gamma_prime_t(t, params)
        return self.get_snr_prime_from_gamma(gamma_t, gamma_prime_t)

    def get_tau_inverse(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Compute the inverse of the time constant for forward integration.
        1/τ(t) = ᾱ'(t)/ᾱ(t)/(1-ᾱ(t))
        Since ᾱ(t) = sigmoid(γ(t)) and ᾱ'(t) = γ'(t) * ᾱ(t) * (1 - ᾱ(t)), we have:
        1/τ(t) = γ'(t) * ᾱ(t) * (1 - ᾱ(t)) / ᾱ(t) / (1 - ᾱ(t)) = γ'(t)
        """
        _, gamma_prime_t = self.get_gamma_gamma_prime_t(t, params)
        return gamma_prime_t
    
    def get_noise_params(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> Dict[str, jnp.ndarray]:
        """Get noise parameters for given timesteps."""

        gamma_t, gamma_prime_t = self.get_gamma_gamma_prime_t(t, params)
        
        # Use utility functions to compute derived quantities
        alpha_t = self.get_alpha_from_gamma(gamma_t)
        alpha_prime_t = self.get_alpha_prime_from_gamma(gamma_t, gamma_prime_t)
        sigma_t = self.get_sigma_from_gamma(gamma_t)
        snr = self.get_snr_from_gamma(gamma_t)
        snr_prime = self.get_snr_prime_from_gamma(gamma_t, gamma_prime_t)

        return {
            "alpha_t": alpha_t,
            "sigma_t": sigma_t,
            "alpha_prime_t": alpha_prime_t,
            "snr": snr,
            "snr_prime": snr_prime, 
            "gamma_t": gamma_t,
            "gamma_prime_t": gamma_prime_t
        }


class LinearNoiseSchedule(NoiseSchedule):
    """Linear noise schedule as used in the NoProp paper.
    
    Following the paper notation:
    - ᾱ(t) is the signal strength coefficient (INCREASING with time)
    - 1 - ᾱ(t) represents cumulative noise added by backward process
    - Backward process: z_t = sqrt(ᾱ(t)) * z_1 + sqrt(1-ᾱ(t)) * ε
      where z_1 is the target/starting point of the backward process
    - Note that for numerical stability of the forward process alpha(t)
      miust be an increasing function bounded away from 0 and 1.
    - Note that the underlying backward OU process is given by 
      dz = δ(t)/2 * z * dt + sqrt(δ(t)) * dW(t), where δ(t) = ᾱ'(t)/ᾱ(t)
    """
    
    @nn.compact
    def __call__(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass - same as get_gamma_gamma_prime_t for compatibility."""
        return self.get_gamma_gamma_prime_t(t)
    
    @nn.compact
    def get_gamma_gamma_prime_t(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get both γ(t) and γ'(t) for the linear schedule.
        
        For linear schedule: ᾱ(t) = t, so γ(t) = logit(t)
        This ensures ᾱ(t) = sigmoid(γ(t)) = t
        γ'(t) = 1/(t*(1-t))
        
        To avoid singularities at t=0 and t=1, we use a modified linear schedule
        that stays away from the boundaries: ᾱ(t) = 0.05 + 0.9*t
        """
        # Modified linear schedule to avoid boundary singularities
        # ᾱ(t) = 0.01 + 0.98*t ranges from 0.01 to 0.99
        alpha_t = 0.01 + 0.98 * t
        alpha_prime_t = 0.98
        
        # Clip to avoid numerical issues
        alpha_t = jnp.clip(alpha_t, 0.001, 0.999)
        
        # gamma_t = logit(alpha_t) so that alpha_t = sigmoid(gamma_t)
        gamma_t = jax.scipy.special.logit(alpha_t)
        gamma_prime_t = alpha_prime_t / (alpha_t * (1.0 - alpha_t))
        
        return gamma_t, gamma_prime_t

class CosineNoiseSchedule(NoiseSchedule):
    """Cosine noise schedule for smoother transitions.
    
    - ᾱ(t) is the signal strength coefficient (INCREASING with time)
    - For cosine schedule: ᾱ(t) = sin(π/2 * t) (INCREASING function)
    - Backward process: z_t = sqrt(ᾱ(t)) * z_1 + sqrt(1-ᾱ(t)) * ε
    """
    
    @nn.compact
    def __call__(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass - same as get_gamma_gamma_prime_t for compatibility."""
        return self.get_gamma_gamma_prime_t(t)
    
    @nn.compact
    def get_gamma_gamma_prime_t(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get both γ(t) and γ'(t) for the cosine schedule.
        
        For cosine schedule: ᾱ(t) = sin(π/2 * t), so γ(t) = logit(sin(π/2 * t))
        This ensures ᾱ(t) = sigmoid(γ(t)) = sin(π/2 * t)
        γ'(t) = (π/2) * cos(π/2 * t) / (sin(π/2 * t) * (1 - sin(π/2 * t)))
        
        To avoid singularities at t=0 and t=1, we use a modified cosine schedule
        that stays away from the boundaries: ᾱ(t) = 0.5 * (1 + sin(π * (t - 0.5)))
        """
        # Modified cosine schedule to avoid boundary singularities
        # ᾱ(t) = 0.5 * (1 + sin(π * (t - 0.5))) ranges from ~0.05 to ~0.95
        alpha_t = 0.01 + 0.98 * jnp.sin(0.5*jnp.pi*t)
        alpha_prime_t = 0.98 * 0.5 * jnp.pi * jnp.cos(0.5*jnp.pi * t)
        
        # Clip to avoid numerical issues
        
        # gamma_t = jnp.log(alpha_t/(1.0 - alpha_t))
        gamma_t = jax.scipy.special.logit(alpha_t)
        gamma_prime_t = alpha_prime_t / (alpha_t * (1.0 - alpha_t))

        return gamma_t, gamma_prime_t


class SigmoidNoiseSchedule(NoiseSchedule):
    """Sigmoid noise schedule with learnable parameters.
    
    Following the paper notation:
    - ᾱ(t) is the signal strength coefficient (INCREASING with time)
    - For sigmoid schedule: ᾱ(t) = σ(γ(t - 0.5)) (INCREASING function)
    - Backward process: z_t = sqrt(ᾱ(t)) * z_1 + sqrt(1-ᾱ(t)) * ε
    """
    
    gamma_rate: float = 4.0
    offset: float = 0.5

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass - same as get_gamma_gamma_prime_t for compatibility."""
        return self.get_gamma_gamma_prime_t(t)

    @nn.compact
    def get_gamma_gamma_prime_t(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get both γ(t) and γ'(t) for the sigmoid schedule.
        
        For sigmoid schedule: ᾱ(t) = σ(γ(t - 0.5)), so γ(t) = γ(t - 0.5)
        This ensures ᾱ(t) = sigmoid(γ(t)) = σ(γ(t - 0.5))
        γ'(t) = γ
        """
        gamma_t = self.gamma_rate * (t - self.offset)
        gamma_prime_t = jnp.full_like(t, self.gamma_rate)
        
        return gamma_t, gamma_prime_t

class SimpleLearnableNoiseSchedule(NoiseSchedule):
    """Sigmoid noise schedule with learnable parameters.
    
    Following the paper notation:
    - ᾱ(t) is the signal strength coefficient (INCREASING with time)
    - For sigmoid schedule: ᾱ(t) = σ(γ(t - 0.5)) (INCREASING function)
    - Backward process: z_t = sqrt(ᾱ(t)) * z_1 + sqrt(1-ᾱ(t)) * ε
    """
    
    @nn.compact
    def __call__(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass - same as get_gamma_gamma_prime_t for compatibility."""
        return self.get_gamma_gamma_prime_t(t)

    @nn.compact
    def get_gamma_gamma_prime_t(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get both γ(t) and γ'(t) for the sigmoid schedule.
        
        For sigmoid schedule: ᾱ(t) = σ(γ(t - 0.5)), so γ(t) = γ(t - 0.5)
        This ensures ᾱ(t) = sigmoid(γ(t)) = σ(γ(t - 0.5))
        γ'(t) = γ (constant for linear schedule)
        
        But for better dynamics, let's use a quadratic schedule where γ'(t) varies with time.
        """
        gamma_rate = self.param('gamma_rate', nn.initializers.constant(4.0), ())
        gamma_offset = self.param('gamma_offset', nn.initializers.constant(0.5), ())

        gamma_t = gamma_rate * (t - gamma_offset)
        gamma_prime_t = jnp.full_like(t, gamma_rate)
        
        return gamma_t, gamma_prime_t


class PositiveDense(nn.Module):
    """Dense layer with positive weights to ensure monotonicity."""
    
    features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply dense layer with positive weights."""
        # Initialize weights normally, but apply softplus in forward pass
        kernel = self.param('kernel', nn.initializers.normal(), (x.shape[-1], self.features))
        bias = self.param('bias', 
                         lambda rng, shape: jax.random.normal(rng, shape)-0.5,
                         (self.features,))
        
        # Apply softplus to ensure weights are always positive
        positive_kernel = jax.nn.softplus(kernel-0.5)
        return jnp.dot(x, positive_kernel)/jnp.sqrt(x.shape[-1]) + bias

class SimpleMonotonicNetwork(nn.Module):

    hidden_dims: Tuple[int, ...]
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply dense layer with positive weights."""
        # Ensure input has the right shape
        if x.ndim == 0:
            # Scalar input - add batch dimension
            x = x[None, None]  # [1, 1]
        elif x.ndim == 1:
            # Batch input - add feature dimension
            x = x[:, None]  # [batch_size, 1]
        
        for hidden_dim in self.hidden_dims:
            x = PositiveDense(hidden_dim)(x)
            x = nn.relu(x)
        x = PositiveDense(1)(x)
        return x

class LearnableNoiseSchedule(NoiseSchedule):
    """Learnable noise schedule as described in Appendix B of the NoProp paper.
    
    This implements the trainable noise schedule where:
    SNR(t) = exp(γ(t))
    alpha(t) = sigmoid(γ(t))

    where γ(t) is an increasing function of time and t is between 0 and 1.
    The neural network uses positive weights and ReLU activations to ensure monotonicity.
    """
    
    hidden_dims: Tuple[int, ...] = (64, 64)  # Hidden dimensions for the neural network
    monotonic_network: nn.Module = SimpleMonotonicNetwork
    gamma_range: Tuple[float, float] = (-4.0, 4.0)


    @nn.compact
    def __call__(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get both γ(t) and γ'(t) from the learnable schedule."""
        return self._get_gamma_gamma_prime_t(t)

    @nn.compact
    def _get_gamma_gamma_prime_t(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get both γ(t) and γ'(t) from the learnable schedule.
        
        Args:
            t: Time values [batch_size]
        """
        scale_logit = self.param('scale_logit', nn.initializers.constant(0.0), ())
        gamma_min = self.param('gamma_min', nn.initializers.constant(self.gamma_range[0]), ())
        gamma_max = self.param('gamma_max', nn.initializers.constant(self.gamma_range[1]), ())

        def gamma_fn(t_input):
            # Create and call the monotonic network
            network = self.monotonic_network(hidden_dims=self.hidden_dims)
            f_t = jnp.squeeze(network(t_input))
            f_t = t_input + (1 - t_input) * t_input * nn.sigmoid(scale_logit) * nn.sigmoid(f_t)
            return gamma_min + (gamma_max-gamma_min) * f_t
                
        return jax.vmap(jax.value_and_grad(gamma_fn))(t)

    def get_gamma_gamma_prime_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.apply({"params": params}, t)


def create_noise_schedule(
    schedule_type: str, 
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





