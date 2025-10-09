"""
Noise scheduling utilities for NoProp variants.

This module provides different noise scheduling strategies used in the NoProp paper:
- Linear schedule
- Cosine schedule  
- Sigmoid schedule
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct


@struct.dataclass
class NoiseSchedule(ABC):
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
    
    @abstractmethod
    def get_gamma_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get gamma_t values for given timesteps t."""
        pass

    def get_gamma_prime_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get the derivative of gamma_t with respect to time."""
        pass

    def get_alpha_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_t values for given timesteps t."""
        return jax.nn.sigmoid(self.get_gamma_t(t, params))
    
    def get_alpha_prime_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get the derivative of ᾱ(t) with respect to time.
        
        Since ᾱ(t) = sigmoid(γ(t)), we have:
        ᾱ'(t) = γ'(t) * sigmoid(γ(t)) * (1 - sigmoid(γ(t)))
        ᾱ'(t) = γ'(t) * ᾱ(t) * (1 - ᾱ(t))
        
        This is needed for computing SNR derivatives and other time-dependent
        quantities. Note that SNR(t) = ᾱ(t) / (1 - ᾱ(t)).
        """
        gamma_prime_t = self.get_gamma_prime_t(t, params)
        alpha_t = self.get_alpha_t(t, params)
        return gamma_prime_t * alpha_t * (1.0 - alpha_t)

    def get_sigma_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get σ(t) - the noise coefficient.
        
        Following the paper: σ(t) = sqrt(1 - ᾱ(t))
        Since ᾱ(t) = sigmoid(γ(t)), we have:
        σ(t) = sqrt(1 - sigmoid(γ(t)))
        
        This ensures the backward process follows: z_t = sqrt(ᾱ(t)) * z_1 + σ(t) * ε
        """
        alpha_t = self.get_alpha_t(t, params)
        return jnp.sqrt(1.0 - alpha_t)
        
    def get_snr(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Compute the Signal-to-Noise Ratio (SNR) at given timesteps.
        
        SNR(t) = ᾱ(t) / (1 - ᾱ(t))
        Since ᾱ(t) = sigmoid(γ(t)), we have:
        SNR(t) = sigmoid(γ(t)) / (1 - sigmoid(γ(t))) = exp(γ(t))
        
        Args:
            t: Time values [batch_size]
            params: Parameters for learnable noise schedule (optional)
            
        Returns:
            SNR values [batch_size]
        """
        # Avoid division by zero
        return jnp.exp(self.get_gamma_t(t, params))
    
    def get_snr_prime(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Compute the derivative of SNR with respect to time.
        
        SNR'(t) = ᾱ'(t) / (1 - ᾱ(t))²
        Since ᾱ(t) = sigmoid(γ(t)) and ᾱ'(t) = γ'(t) * ᾱ(t) * (1 - ᾱ(t)), we have:
        SNR'(t) = γ'(t) * ᾱ(t) * (1 - ᾱ(t)) / (1 - ᾱ(t))² = γ'(t) * ᾱ(t) / (1 - ᾱ(t))
        
        Args:
            t: Time values [batch_size]
            params: Parameters for learnable noise schedule (optional)
            
        Returns:
            SNR derivative values [batch_size]
        """
        return self.get_gamma_prime_t(t, params) * jnp.exp(self.get_gamma_t(t, params))

    def get_tau_inverse(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Compute the inverse of the time constant for forward integration.
        
        Note that for numerical stability, tau inverse is probably the thing 
        to parameterize when learning a noise schedule since it is what determines
        numerical stability of the forward process.  
        
        1/τ(t) = ᾱ'(t)/ᾱ(t)/(1-ᾱ(t))
        Since ᾱ(t) = sigmoid(γ(t)) and ᾱ'(t) = γ'(t) * ᾱ(t) * (1 - ᾱ(t)), we have:
        1/τ(t) = γ'(t) * ᾱ(t) * (1 - ᾱ(t)) / ᾱ(t) / (1 - ᾱ(t)) = γ'(t)
        """
        gamma_prime_t = self.get_gamma_prime_t(t, params)
        return gamma_prime_t
    
    def get_noise_params(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> Dict[str, jnp.ndarray]:
        """Get noise parameters for given timesteps."""

        gamma_t = self.get_gamma_t(t, params)
        gamma_prime_t = self.get_gamma_prime_t(t, params)
        alpha_t = jax.nn.sigmoid(gamma_t)
        alpha_prime_t = gamma_prime_t * alpha_t * (1.0 - alpha_t)
        sigma_t = jnp.sqrt(1.0 - alpha_t)
        snr = jnp.exp(gamma_t)
        snr_prime = gamma_prime_t * jnp.exp(gamma_t)

        return {
            "alpha_t": alpha_t,
            "sigma_t": sigma_t,
            "alpha_prime_t": alpha_prime_t,
            "snr": snr,
            "snr_prime": snr_prime, 
            "gamma_t": gamma_t,
            "gamma_prime_t": gamma_prime_t
        }

    def sample_zt(
        self,
        key: jax.random.PRNGKey,
        z_target: jnp.ndarray,
        t: jnp.ndarray,
        params: Optional[Dict[str, Any]] = None
    ) -> jnp.ndarray:
        """Sample z_t from the backward process distribution.
        
        z_t ~ N(sqrt(ᾱ(t)) * z_target, (1-ᾱ(t)) * I)
        
        This is equivalent to:
        z_t = sqrt(ᾱ(t)) * z_target + sqrt(1-ᾱ(t)) * ε
        where ε ~ N(0, I)
        
        Args:
            key: Random key
            z_target: Target/clean data (z_1) [batch_size, ...]
            t: Time values [batch_size]
            params: Parameters for learnable noise schedule (optional)
            
        Returns:
            Sampled z_t [batch_size, ...]
        """
        alpha_t = self.get_alpha_t(t, params)
        
        # Compute z_t = sqrt(ᾱ(t)) * z_target + sqrt(1-ᾱ(t)) * ε
        return jnp.sqrt(alpha_t)[:, None] * z_target + jnp.sqrt(1.0 - alpha_t)[:, None] * jax.random.normal(key, z_target.shape)

@struct.dataclass
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
    
    def get_gamma_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get γ(t) - the logit of the signal strength coefficient.
        
        For linear schedule: ᾱ(t) = t, so γ(t) = logit(t)
        This ensures ᾱ(t) = sigmoid(γ(t)) = t
        """
        # Use logit with numerical stability, but handle boundary cases
        t_clipped = jnp.clip(t, 1e-8, 1.0 - 1e-8)
        return jax.scipy.special.logit(t_clipped)
    
    def get_gamma_prime_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get the derivative of γ(t) with respect to time.
        
        For linear schedule: γ(t) = logit(t), so γ'(t) = 1/(t*(1-t))
        """
        t_clipped = jnp.clip(t, 1e-7, 1.0 - 1e-7)
        return 1.0 / (t_clipped * (1.0 - t_clipped))


@struct.dataclass
class CosineNoiseSchedule(NoiseSchedule):
    """Cosine noise schedule for smoother transitions.
    
    Following the paper notation:
    - ᾱ(t) is the signal strength coefficient (INCREASING with time)
    - For cosine schedule: ᾱ(t) = sin(π/2 * t) (INCREASING function)
    - Backward process: z_t = sqrt(ᾱ(t)) * z_1 + sqrt(1-ᾱ(t)) * ε
    """
    
    def get_gamma_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get γ(t) - the logit of the signal strength coefficient.
        
        For cosine schedule: ᾱ(t) = sin(π/2 * t), so γ(t) = logit(sin(π/2 * t))
        This ensures ᾱ(t) = sigmoid(γ(t)) = sin(π/2 * t)
        """
        alpha_t = jnp.sin(jnp.pi / 2 * t)
        alpha_t_clipped = jnp.clip(alpha_t, 1e-7, 1.0 - 1e-7)
        return jax.scipy.special.logit(alpha_t_clipped)
    
    def get_gamma_prime_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get the derivative of γ(t) with respect to time.
        
        For cosine schedule: γ(t) = logit(sin(π/2 * t))
        Using chain rule: γ'(t) = (π/2) * cos(π/2 * t) / (sin(π/2 * t) * (1 - sin(π/2 * t)))
        """
        alpha_t = jnp.sin(jnp.pi / 2 * t)
        alpha_prime_t = (jnp.pi / 2) * jnp.cos(jnp.pi / 2 * t)
        alpha_t_clipped = jnp.clip(alpha_t, 1e-7, 1.0 - 1e-7)
        return alpha_prime_t / (alpha_t_clipped * (1.0 - alpha_t_clipped))


@struct.dataclass
class SigmoidNoiseSchedule(NoiseSchedule):
    """Sigmoid noise schedule with learnable parameters.
    
    Following the paper notation:
    - ᾱ(t) is the signal strength coefficient (INCREASING with time)
    - For sigmoid schedule: ᾱ(t) = σ(γ(t - 0.5)) (INCREASING function)
    - Backward process: z_t = sqrt(ᾱ(t)) * z_1 + sqrt(1-ᾱ(t)) * ε
    """
    
    gamma: float = 1.0  # Controls the steepness of the sigmoid
    
    def get_gamma_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get γ(t) - the logit of the signal strength coefficient.
        
        For sigmoid schedule: ᾱ(t) = σ(γ(t - 0.5)), so γ(t) = γ(t - 0.5)
        This ensures ᾱ(t) = sigmoid(γ(t)) = σ(γ(t - 0.5))
        """
        return self.gamma * (t - 0.5)
    
    def get_gamma_prime_t(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get the derivative of γ(t) with respect to time.
        
        For sigmoid schedule: γ(t) = γ(t - 0.5), so γ'(t) = γ
        """
        return jnp.full_like(t, self.gamma)


class PositiveDense(nn.Module):
    """Dense layer with positive weights to ensure monotonicity."""
    
    features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply dense layer with positive weights."""
        # Initialize weights normally, but apply softplus in forward pass
        kernel = self.param('kernel', 
                           lambda rng, shape: jax.random.normal(rng, shape),
                           (x.shape[-1], self.features))
        bias = self.param('bias', 
                         lambda rng, shape: jax.random.normal(rng, shape),
                         (self.features,))
        
        # Apply softplus to ensure weights are always positive
        positive_kernel = jax.nn.softplus(kernel)
        return jnp.dot(x, positive_kernel) + bias


class LearnableNoiseScheduleNetwork(nn.Module):
    """Neural network for learnable noise schedule with monotonicity guarantee.
    
    This network computes γ(t) directly using positive weights and ReLU activations
    to ensure the output is monotonically increasing with respect to time.
    The network learns gamma_min and gamma_max as parameters to ensure:
    γ(0) = gamma_min and γ(1) = gamma_max
    """
    
    hidden_dims: Tuple[int, ...] = (64, 64)
    
    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute γ(t) from time t.
        
        Args:
            t: Time values [batch_size] or [batch_size, 1]
            
        Returns:
            γ(t) values [batch_size, 1] - monotonically increasing
        """
        # Ensure t is 2D
        if t.ndim == 1:
            t = t[:, None]
        
        # Learnable boundary parameters
        gamma_min = self.param('gamma_min', lambda rng, shape: jnp.array(-5.0), ())
        gamma_max = self.param('gamma_max', lambda rng, shape: jnp.array(5.0), ())
        
        # Neural network layers with positive weights
        x = t
        for hidden_dim in self.hidden_dims:
            x = PositiveDense(hidden_dim)(x)
            x = nn.relu(x)
        
        # Final layer (no activation, just positive weights)
        x = PositiveDense(1)(x)
        
        # Enforce exact boundary conditions: γ(0) = gamma_min, γ(1) = gamma_max
        # Use a monotonic interpolation that satisfies the boundary conditions
        # γ(t) = gamma_min + (gamma_max - gamma_min) * f(t) where f(0) = 0, f(1) = 1
        # and f(t) is monotonically increasing
        
        # The network learns a monotonic function, but we need to ensure it goes from 0 to 1
        # We can do this by using the network output to modulate a linear interpolation
        # f(t) = t + (1-t) * t * network_output, which ensures f(0) = 0, f(1) = 1
                
        # Create a function that satisfies boundary conditions
        # f(t) = t + (1-t) * t * network_output
        # This ensures: f(0) = 0, f(1) = 1, and f(t) is monotonic
        
        x = jax.nn.sigmoid(x)  # Bound the network output
        f_t = t + (1 - t) * t * x
                # Apply boundary conditions
        gamma_t = gamma_min + (gamma_max - gamma_min) * f_t
        
        return gamma_t


@struct.dataclass
class LearnableNoiseSchedule(NoiseSchedule):
    """Learnable noise schedule as described in Appendix B of the NoProp paper.
    
    This implements the trainable noise schedule where:
    SNR(t) = exp(γ(t))
    alpha(t) = sigmoid(γ(t))

    where γ̄(t) is an increasing function of time and t is betwen 0 and 1.  The 
    challenging part is that we also need access to the derivative of γ(t) with 
    respect to time.  The easiest way to do this is with a neural network that
    has only positive weights and convex activation functions, e.g. ReLU.
    """
    
    hidden_dims: Tuple[int, ...] = (64, 64)  # Hidden dimensions for the neural network
    
    def __post_init__(self):
        """Initialize the neural network."""
        # Create the neural network using object.__setattr__ to bypass dataclass immutability
        object.__setattr__(self, 'gamma_network', LearnableNoiseScheduleNetwork(
            hidden_dims=self.hidden_dims
        ))
    
    def get_gamma_t(self, t: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
        """Get γ(t) from the learnable schedule.
        
        For the learnable schedule, γ(t) is computed by the neural network.
        This ensures ᾱ(t) = σ(γ(t)) is INCREASING with time.
        
        Args:
            t: Time values [batch_size]
            params: Neural network parameters (required for learnable schedule)
        """
        gamma_t = self.gamma_network.apply(params, t)  # [batch_size, 1]
        return gamma_t.squeeze(-1)  # [batch_size]
    
    def get_gamma_prime_t(self, t: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
        """Get the derivative of γ(t) with respect to time.
        
        For the learnable schedule, γ'(t) is computed using automatic differentiation.
        
        Args:
            t: Time values [batch_size]
            params: Neural network parameters (required for learnable schedule)
        """
        # Use JAX's automatic differentiation to compute γ'(t)
        def gamma_fn(t_input):
            gamma_t = self.gamma_network.apply(params, t_input)
            return gamma_t.squeeze(-1)
        
        gamma_prime_t = jax.grad(lambda t_val: jnp.sum(gamma_fn(t_val)))(t)
        return gamma_prime_t


def create_noise_schedule(
    schedule_type: str = "cosine", 
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




