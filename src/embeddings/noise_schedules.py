"""
Noise scheduling utilities for NoProp variants.

This module provides different noise scheduling strategies based on the comprehensive
review paper: "A Comprehensive Review on Noise Control of Diffusion Model" (arXiv:2502.04669).

The fundamental relationship is:
- alpha_bar(t) = sigmoid(gamma(t)), where gamma(t) is an increasing function
- alpha_bar_prime(t) = alpha_bar(t) * (1 - alpha_bar(t)) * gamma_prime(t)

Most schedules in the literature parameterize alpha_bar as decreasing (noise increases),
but in our formulation alpha_bar is increasing (signal increases).

USAGE IN NEURAL NETWORK SETTINGS:

All noise schedules are Flax Linen modules with learnable parameters. Here's how to use them:

1. As a submodule in a larger model:
   
   class MyModel(nn.Module):
       noise_schedule: NoiseSchedule
       
       @nn.compact
       def __call__(self, t):
           # Get alpha_bar and gamma_prime
           alpha_bar, gamma_prime = self.noise_schedule.get_alpha_bar_gamma_prime(t)
           # Use in your model...
           return alpha_bar, gamma_prime
   
   # Initialize model with a specific schedule
   schedule = LinearNoiseSchedule()
   model = MyModel(noise_schedule=schedule)
   params = model.init(key, t_sample)  # t_sample is sample time values

2. Standalone usage (recommended - cleaner interface):
   
   schedule = CosineNoiseSchedule()
   t = jnp.array([0.1, 0.5, 0.9])  # time values
   params = schedule.init(key, t)  # initialize parameters
   
   # Direct access methods (recommended)
   alpha_bar = schedule.alpha_bar(params, t)
   gamma_prime = schedule.gamma_prime(params, t)
   gamma = schedule.gamma(params, t)
   alpha_bar_prime = schedule.alpha_bar_prime(params, t)
   alpha_bar, gamma_prime = schedule.alpha_bar_gamma_prime(params, t)
   
   # Alternative: using apply() directly
   alpha_bar = schedule.apply({"params": params}, t, method=schedule.get_alpha_bar)
   alpha_bar, gamma_prime = schedule.apply({"params": params}, t, method=schedule.get_alpha_bar_gamma_prime)

3. Training with learnable parameters:
   
   All parameters are automatically included in the params dict and will be updated
   during training. Parameters are transformed to enforce constraints:
   - Positive parameters use softplus transformation
   - Bounded parameters use sigmoid + scaling
   - Ordering constraints (e.g., max > min) are enforced via delta parameters
   
   # Parameters are automatically learned during optimization
   loss = compute_loss(model, params, data)
   grads = jax.grad(loss)(params)  # gradients include schedule parameters
   params = optimizer.update(grads, params)

4. Using the factory function:
   
   schedule = create_noise_schedule("linear")
   schedule = create_noise_schedule("cosine")
   schedule = create_noise_schedule("sigmoid", k=10.0, t_mid=0.5)  # initial values

5. Available schedules:
   - "linear": Linear schedule with learnable bounds
   - "cosine": Cosine schedule with learnable offset
   - "sigmoid": Sigmoid schedule with learnable steepness and midpoint
   - "exponential": Exponential schedule with learnable rate and bounds
   - "cauchy": Cauchy distribution schedule with learnable location, scale, and bounds
   - "laplace": Laplace distribution schedule with learnable parameters
   - "logistic": Logistic schedule (equivalent to sigmoid)
   - "quadratic": Quadratic schedule (power of 2) with learnable bounds
   - "polynomial": Polynomial schedule with learnable power and bounds
   - "monotonic_nn" or "learnable" or "network": Neural network-based learnable schedule

All schedules implement:
- get_alpha_bar(t, params=None): Returns alpha_bar(t)
- get_alpha_bar_gamma_prime(t, params=None): Returns (alpha_bar(t), gamma_prime(t))
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


class NoiseSchedule(nn.Module):
    """Abstract base class for noise schedules.
    
    The fundamental relationship is:
    - alpha_bar(t) = sigmoid(gamma(t)), where gamma(t) is an increasing function
    - alpha_bar_prime(t) = alpha_bar(t) * (1 - alpha_bar(t)) * gamma_prime(t)
    
    Subclasses must implement two methods:
    - get_alpha_bar(t): returns alpha_bar(t)
    - get_alpha_bar_gamma_prime(t): returns (alpha_bar(t), gamma_prime(t))
    
    Args:
        learnable: Whether schedule parameters should be learnable. If False, 
                  stop_gradient is applied to outputs to freeze parameters.
    """
    learnable: bool = True  # Whether parameters are learnable
    
    def _get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Internal method to get alpha_bar(t) - subclasses must implement this.
        
        Args:
            t: Time values [batch_size]
            params: Optional parameters for learnable schedules
            
        Returns:
            alpha_bar(t) values [batch_size]
        """
        raise NotImplementedError("Subclasses must implement _get_alpha_bar")
    
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Internal method to get both alpha_bar(t) and gamma_prime(t) - subclasses must implement this.
        
        Args:
            t: Time values [batch_size]
            params: Optional parameters for learnable schedules
            
        Returns:
            Tuple of (alpha_bar(t), gamma_prime(t)) where:
            - alpha_bar(t): alpha_bar values [batch_size]
            - gamma_prime(t): gamma derivative values [batch_size]
        """
        raise NotImplementedError("Subclasses must implement _get_alpha_bar_gamma_prime")
    
    # Base class methods - subclasses should override these and call _apply_stop_gradient
    def _apply_stop_gradient(self, alpha_bar_t: jnp.ndarray, gamma_prime_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Helper method to apply stop_gradient if learnable=False."""
        if not self.learnable:
            alpha_bar_t = jax.lax.stop_gradient(alpha_bar_t)
            gamma_prime_t = jax.lax.stop_gradient(gamma_prime_t)
        return alpha_bar_t, gamma_prime_t
    
    def _apply_stop_gradient_alpha_bar(self, alpha_bar_t: jnp.ndarray) -> jnp.ndarray:
        """Helper method to apply stop_gradient to alpha_bar if learnable=False."""
        if not self.learnable:
            alpha_bar_t = jax.lax.stop_gradient(alpha_bar_t)
        return alpha_bar_t
    
    @nn.compact
    def get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) = sigmoid(gamma(t)).
        
        Applies stop_gradient if learnable=False.
        
        Args:
            t: Time values [batch_size]
            params: Optional parameters for learnable schedules
            
        Returns:
            alpha_bar(t) values [batch_size]
        """
        # Call the internal method (which is @nn.compact in subclasses)
        alpha_bar_t = self._get_alpha_bar(t, params)
        # Apply stop_gradient if learnable=False
        if not self.learnable:
            alpha_bar_t = jax.lax.stop_gradient(alpha_bar_t)
        return alpha_bar_t
    
    @nn.compact
    def get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get both alpha_bar(t) and gamma_prime(t).
        
        Applies stop_gradient if learnable=False.
        
        Args:
            t: Time values [batch_size]
            params: Optional parameters for learnable schedules
            
        Returns:
            Tuple of (alpha_bar(t), gamma_prime(t)) where:
            - alpha_bar(t): alpha_bar values [batch_size]
            - gamma_prime(t): gamma derivative values [batch_size]
        """
        # Call the internal method (which is @nn.compact in subclasses)
        alpha_bar_t, gamma_prime_t = self._get_alpha_bar_gamma_prime(t, params)
        # Apply stop_gradient if learnable=False
        if not self.learnable:
            alpha_bar_t = jax.lax.stop_gradient(alpha_bar_t)
            gamma_prime_t = jax.lax.stop_gradient(gamma_prime_t)
        return alpha_bar_t, gamma_prime_t
    
    # Helper methods for convenient access without apply()
    def alpha_bar(self, variables: Dict[str, Any], t: jnp.ndarray) -> jnp.ndarray:
        """Convenience method to get alpha_bar(t) from variables.
        
        Usage: alpha_bar = schedule.alpha_bar(variables, t)
        
        Args:
            variables: Variables dict from model initialization (e.g., {"params": {...}})
            t: Time values [batch_size]
            
        Returns:
            alpha_bar(t) values [batch_size]
        """
        return self.apply(variables, t, method=self.get_alpha_bar)
    
    def gamma_prime(self, variables: Dict[str, Any], t: jnp.ndarray) -> jnp.ndarray:
        """Convenience method to get gamma_prime(t) from variables.
        
        Usage: gamma_prime = schedule.gamma_prime(variables, t)
        
        Args:
            variables: Variables dict from model initialization (e.g., {"params": {...}})
            t: Time values [batch_size]
            
        Returns:
            gamma_prime(t) values [batch_size]
        """
        _, gamma_prime_t = self.apply(variables, t, method=self.get_alpha_bar_gamma_prime)
        return gamma_prime_t
    
    def alpha_bar_gamma_prime(
        self, variables: Dict[str, Any], t: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Convenience method to get both alpha_bar(t) and gamma_prime(t) from variables.
        
        Usage: alpha_bar, gamma_prime = schedule.alpha_bar_gamma_prime(variables, t)
        
        Args:
            variables: Variables dict from model initialization (e.g., {"params": {...}})
            t: Time values [batch_size]
            
        Returns:
            Tuple of (alpha_bar(t), gamma_prime(t))
        """
        return self.apply(variables, t, method=self.get_alpha_bar_gamma_prime)
    
    def gamma(self, variables: Dict[str, Any], t: jnp.ndarray) -> jnp.ndarray:
        """Convenience method to get gamma(t) from variables.
        
        Usage: gamma = schedule.gamma(variables, t)
        
        Computes gamma(t) = logit(alpha_bar(t))
        
        Args:
            variables: Variables dict from model initialization (e.g., {"params": {...}})
            t: Time values [batch_size]
            
        Returns:
            gamma(t) values [batch_size]
        """
        alpha_bar_t = self.alpha_bar(variables, t)
        return jax.scipy.special.logit(alpha_bar_t)
    
    def alpha_bar_prime(self, variables: Dict[str, Any], t: jnp.ndarray) -> jnp.ndarray:
        """Convenience method to get alpha_bar_prime(t) from variables.
        
        Usage: alpha_bar_prime = schedule.alpha_bar_prime(variables, t)
        
        Computes alpha_bar_prime(t) = alpha_bar(t) * (1 - alpha_bar(t)) * gamma_prime(t)
        
        Args:
            variables: Variables dict from model initialization (e.g., {"params": {...}})
            t: Time values [batch_size]
            
        Returns:
            alpha_bar_prime(t) values [batch_size]
        """
        alpha_bar_t, gamma_prime_t = self.alpha_bar_gamma_prime(variables, t)
        return alpha_bar_t * (1.0 - alpha_bar_t) * gamma_prime_t


class LinearNoiseSchedule(NoiseSchedule):
    """Linear noise schedule with learnable parameters.
    
    From paper Section III-A1: Linear Schedule
    Typically parameterized as beta(t) linear, which gives alpha_bar as decreasing.
    In our formulation: alpha_bar(t) = alpha_bar_min + t * (alpha_bar_max - alpha_bar_min)
    
    All parameters are learnable:
    - alpha_bar_min: bounded to [0.001, 0.999]
    - alpha_bar_max: alpha_bar_min + delta_fraction * (0.999 - alpha_bar_min) to ensure max > min and max <= 0.999
    
    Args:
        alpha_bar_min: Initial value for alpha_bar_min (default: 0.01, which corresponds to logit -4.6)
        alpha_bar_max: Initial value for alpha_bar_max (default: 0.99, computed from delta_fraction)
    """
    
    alpha_bar_min: float = 0.01  # Initial alpha_bar_min value
    alpha_bar_max: float = 0.99  # Initial alpha_bar_max value
    
    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "alpha_bar_min": 0.01,
            "alpha_bar_max": 0.99,
        }
    
    @nn.compact
    def _get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for linear schedule."""
        if params is not None:
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            delta_fraction_logit = params['delta_fraction_logit']
        else:
            # Compute initial logit values from initial alpha_bar values
            # alpha_bar_min = 0.001 + 0.998 * sigmoid(logit) -> logit = logit((alpha_bar_min - 0.001) / 0.998)
            alpha_bar_min_logit_val = jax.scipy.special.logit((self.alpha_bar_min - 0.001) / 0.998)
            # delta_fraction = sigmoid(delta_fraction_logit) where delta_fraction = (alpha_bar_max - alpha_bar_min) / (0.999 - alpha_bar_min)
            # Ensure initial alpha_bar_max is valid
            alpha_bar_min_clamped = jnp.clip(self.alpha_bar_min, 0.001, 0.998)
            alpha_bar_max_clamped = jnp.clip(self.alpha_bar_max, alpha_bar_min_clamped, 0.999)
            delta_max_init = 0.999 - alpha_bar_min_clamped
            delta_actual_init = alpha_bar_max_clamped - alpha_bar_min_clamped
            # Use jnp.where for JIT compatibility
            delta_fraction_init = jnp.where(delta_max_init > 0, delta_actual_init / delta_max_init, 0.0)
            delta_fraction_logit_val = jax.scipy.special.logit(jnp.clip(delta_fraction_init, 0.001, 0.999))
            
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            delta_fraction_logit = self.param('delta_fraction_logit',
                                              nn.initializers.constant(delta_fraction_logit_val), ())
        
        # Transform to bounded values - optimized: cache intermediate values
        alpha_bar_min = 0.001 + 0.998 * jax.nn.sigmoid(alpha_bar_min_logit)  # [0.001, 0.999]
        delta_fraction = jax.nn.sigmoid(delta_fraction_logit)  # [0, 1]
        delta_max = 0.999 - alpha_bar_min  # Maximum possible delta
        delta_alpha = delta_fraction * delta_max  # Actual delta in [0, delta_max]
        alpha_bar_max = alpha_bar_min + delta_alpha  # Guaranteed to be in [alpha_bar_min, 0.999]
        
        # Linear interpolation - no clipping needed as bounds are guaranteed
        alpha_bar_t = alpha_bar_min + t * delta_alpha
        alpha_bar_t = jnp.clip(alpha_bar_t, 0.001, 0.999)
        # Apply stop_gradient if learnable=False (handled in base class get_alpha_bar)
        return alpha_bar_t
    
    @nn.compact
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) and gamma_prime(t) for linear schedule."""
        if params is not None:
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            delta_fraction_logit = params['delta_fraction_logit']
        else:
            # Compute initial logit values from initial alpha_bar values
            alpha_bar_min_logit_val = jax.scipy.special.logit((self.alpha_bar_min - 0.001) / 0.998)
            # delta_fraction = sigmoid(delta_fraction_logit) where delta_fraction = (alpha_bar_max - alpha_bar_min) / (0.999 - alpha_bar_min)
            alpha_bar_min_clamped = jnp.clip(self.alpha_bar_min, 0.001, 0.998)
            alpha_bar_max_clamped = jnp.clip(self.alpha_bar_max, alpha_bar_min_clamped, 0.999)
            delta_max_init = 0.999 - alpha_bar_min_clamped
            delta_actual_init = alpha_bar_max_clamped - alpha_bar_min_clamped
            # Use jnp.where for JIT compatibility
            delta_fraction_init = jnp.where(delta_max_init > 0, delta_actual_init / delta_max_init, 0.0)
            delta_fraction_logit_val = jax.scipy.special.logit(jnp.clip(delta_fraction_init, 0.001, 0.999))
            
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            delta_fraction_logit = self.param('delta_fraction_logit',
                                              nn.initializers.constant(delta_fraction_logit_val), ())
        
        # Transform to bounded values - optimized: cache intermediate values
        alpha_bar_min = 0.001 + 0.998 * jax.nn.sigmoid(alpha_bar_min_logit)
        delta_fraction = jax.nn.sigmoid(delta_fraction_logit)
        delta_max = 0.999 - alpha_bar_min
        delta_alpha = delta_fraction * delta_max
        alpha_bar_max = alpha_bar_min + delta_alpha  # Guaranteed to be in [alpha_bar_min, 0.999]
        
        # Linear interpolation
        alpha_bar_t = alpha_bar_min + t * delta_alpha
        alpha_bar_t = jnp.clip(alpha_bar_t, 0.001, 0.999)
        
        # For linear: alpha_bar_prime is constant (delta_alpha)
        # Compute gamma_prime efficiently: cache alpha_bar_t * (1 - alpha_bar_t)
        alpha_bar_t_one_minus = alpha_bar_t * (1.0 - alpha_bar_t)
        gamma_prime_t = delta_alpha / alpha_bar_t_one_minus
        
        # Apply stop_gradient if learnable=False
        return self._apply_stop_gradient(alpha_bar_t, gamma_prime_t)


class CosineNoiseSchedule(NoiseSchedule):
    """Cosine noise schedule with learnable parameters.
    
    From paper Section III-A3: Cosine Schedule
    We use the increasing version: alpha_bar(t) = sin^2((t + s) / (1 + s) * pi/2)
    which ranges from 0 to 1 as t goes from 0 to 1.
    
    All parameters are learnable:
    - s: positive offset (default: 0.008, enforced via softplus)
    
    Args:
        s: Initial value for s (default: 0.008)
    """
    
    s: float = 0.008  # Initial s value
    
    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "s": 0.008,
        }
    
    @nn.compact
    def get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for cosine schedule."""
        if params is not None:
            s_logit = params['s_logit']
        else:
            # Compute initial logit from initial s: s = softplus(logit) -> logit = log(exp(s) - 1)
            # Use stable softplus inverse
            s_logit_val = jnp.log1p(jnp.expm1(self.s)) if self.s > 0 else -10.0
            s_logit = self.param('s_logit', nn.initializers.constant(s_logit_val), ())
        
        s = jax.nn.softplus(s_logit)
        one_plus_s = 1.0 + s  # Cache denominator
        cos_arg = (t + s) / one_plus_s * (0.5 * jnp.pi)  # Cache pi/2
        sin_arg = jnp.sin(cos_arg)
        alpha_bar_t = sin_arg * sin_arg  # More efficient than sin^2
        return jnp.clip(alpha_bar_t, 0.001, 0.999)
    
    @nn.compact
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) and gamma_prime(t) for cosine schedule."""
        if params is not None:
            s_logit = params['s_logit']
        else:
            s_logit_val = jnp.log1p(jnp.expm1(self.s)) if self.s > 0 else -10.0
            s_logit = self.param('s_logit', nn.initializers.constant(s_logit_val), ())
        
        s = jax.nn.softplus(s_logit)
        one_plus_s = 1.0 + s  # Cache denominator
        pi_half = 0.5 * jnp.pi  # Cache pi/2
        cos_arg = (t + s) / one_plus_s * pi_half
        sin_arg = jnp.sin(cos_arg)
        cos_arg_val = jnp.cos(cos_arg)
        
        alpha_bar_t = sin_arg * sin_arg  # More efficient than sin^2
        alpha_bar_t = jnp.clip(alpha_bar_t, 0.001, 0.999)
        
        # Derivative: d/dx sin^2(x) = 2*sin(x)*cos(x)
        # Cache pi_half / one_plus_s
        pi_half_over_one_plus_s = pi_half / one_plus_s
        alpha_bar_prime_t = 2.0 * sin_arg * cos_arg_val * pi_half_over_one_plus_s
        
        # Compute gamma_prime efficiently
        alpha_bar_t_one_minus = alpha_bar_t * (1.0 - alpha_bar_t)
        gamma_prime_t = alpha_bar_prime_t / alpha_bar_t_one_minus
        
        return alpha_bar_t, gamma_prime_t


class SigmoidNoiseSchedule(NoiseSchedule):
    """Sigmoid noise schedule with learnable parameters.
    
    From paper Section III-A4: Sigmoid Schedule
    Typically parameterized with k (steepness) and t_mid (midpoint).
    In our formulation: alpha_bar(t) = sigmoid(k * (t - t_mid))
    
    All parameters are learnable:
    - k: positive steepness parameter (enforced via softplus)
    - t_mid: unbounded midpoint parameter
    
    Args:
        k: Initial value for k (default: 10.0)
        t_mid: Initial value for t_mid (default: 0.5)
    """
    
    k: float = 10.0  # Initial k value
    t_mid: float = 0.5  # Initial t_mid value
    
    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "k": 10.0,
            "t_mid": 0.5,
        }
    
    @nn.compact
    def get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for sigmoid schedule."""
        if params is not None:
            k_logit = params['k_logit']
            t_mid = params['t_mid']
        else:
            # Compute initial logit from initial k: k = softplus(logit) -> logit = log(exp(k) - 1)
            # Use stable softplus inverse
            k_logit_val = jnp.log1p(jnp.expm1(self.k)) if self.k > 0 else -10.0
            k_logit = self.param('k_logit', nn.initializers.constant(k_logit_val), ())
            t_mid = self.param('t_mid', nn.initializers.constant(self.t_mid), ())
        
        k = jax.nn.softplus(k_logit)
        gamma_t = k * (t - t_mid)
        alpha_bar_t = jax.nn.sigmoid(gamma_t)
        return alpha_bar_t
    
    @nn.compact
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) and gamma_prime(t) for sigmoid schedule."""
        if params is not None:
            k_logit = params['k_logit']
            t_mid = params['t_mid']
        else:
            k_logit_val = jnp.log(jnp.exp(self.k) - 1.0) if self.k > 0 else -10.0
            k_logit = self.param('k_logit', nn.initializers.constant(k_logit_val), ())
            t_mid = self.param('t_mid', nn.initializers.constant(self.t_mid), ())
        
        k = jax.nn.softplus(k_logit)
        gamma_t = k * (t - t_mid)
        gamma_prime_t = jnp.full_like(t, k)
        
        alpha_bar_t = jax.nn.sigmoid(gamma_t)
        
        return alpha_bar_t, gamma_prime_t


class ExponentialNoiseSchedule(NoiseSchedule):
    """Exponential noise schedule with learnable parameters.
    
    From paper Section III-A5: Exponential Schedule
    In our increasing formulation: alpha_bar(t) = alpha_bar_min + (alpha_bar_max - alpha_bar_min) * (1 - exp(-beta * t))
    
    All parameters are learnable:
    - beta: positive exponential decay rate (enforced via softplus)
    - alpha_bar_min: bounded to [0.001, 0.999]
    - alpha_bar_max: alpha_bar_min + delta_fraction * (0.999 - alpha_bar_min) to ensure max > min and max <= 0.999
    
    Args:
        beta: Initial value for beta (default: 2.0)
        alpha_bar_min: Initial value for alpha_bar_min (default: 0.01)
        alpha_bar_max: Initial value for alpha_bar_max (default: 0.99)
    """
    
    beta: float = 2.0  # Initial beta value
    alpha_bar_min: float = 0.01  # Initial alpha_bar_min value
    alpha_bar_max: float = 0.99  # Initial alpha_bar_max value
    
    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "beta": 2.0,
            "alpha_bar_min": 0.01,
            "alpha_bar_max": 0.99,
        }
    
    @nn.compact
    def get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for exponential schedule."""
        if params is not None:
            beta_logit = params['beta_logit']
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            delta_fraction_logit = params['delta_fraction_logit']
        else:
            # Compute initial logit values from initial values
            # Use stable softplus inverse
            beta_logit_val = jnp.log1p(jnp.expm1(self.beta)) if self.beta > 0 else -10.0
            alpha_bar_min_logit_val = jax.scipy.special.logit((self.alpha_bar_min - 0.001) / 0.998)
            # delta_fraction = sigmoid(delta_fraction_logit) where delta_fraction = (alpha_bar_max - alpha_bar_min) / (0.999 - alpha_bar_min)
            alpha_bar_min_clamped = jnp.clip(self.alpha_bar_min, 0.001, 0.998)
            alpha_bar_max_clamped = jnp.clip(self.alpha_bar_max, alpha_bar_min_clamped, 0.999)
            delta_max_init = 0.999 - alpha_bar_min_clamped
            delta_actual_init = alpha_bar_max_clamped - alpha_bar_min_clamped
            # Use jnp.where for JIT compatibility
            delta_fraction_init = jnp.where(delta_max_init > 0, delta_actual_init / delta_max_init, 0.0)
            delta_fraction_logit_val = jax.scipy.special.logit(jnp.clip(delta_fraction_init, 0.001, 0.999))
            
            beta_logit = self.param('beta_logit', nn.initializers.constant(beta_logit_val), ())
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            delta_fraction_logit = self.param('delta_fraction_logit',
                                             nn.initializers.constant(delta_fraction_logit_val), ())
        
        beta = jax.nn.softplus(beta_logit)
        alpha_bar_min = 0.001 + 0.998 * jax.nn.sigmoid(alpha_bar_min_logit)
        delta_fraction = jax.nn.sigmoid(delta_fraction_logit)
        delta_max = 0.999 - alpha_bar_min
        delta_alpha = delta_fraction * delta_max
        alpha_bar_max = alpha_bar_min + delta_alpha  # Guaranteed to be in [alpha_bar_min, 0.999]
        
        # Optimized: cache exp(-beta * t) and (1 - exp_val)
        beta_t_neg = -beta * t  # Cache -beta * t
        exp_val = jnp.exp(beta_t_neg)
        one_minus_exp = 1.0 - exp_val  # Cache (1 - exp)
        alpha_bar_t = alpha_bar_min + delta_alpha * one_minus_exp
        return jnp.clip(alpha_bar_t, 0.001, 0.999)
    
    @nn.compact
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) and gamma_prime(t) for exponential schedule."""
        if params is not None:
            beta_logit = params['beta_logit']
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            delta_fraction_logit = params['delta_fraction_logit']
        else:
            # Use stable softplus inverse
            beta_logit_val = jnp.log1p(jnp.expm1(self.beta)) if self.beta > 0 else -10.0
            alpha_bar_min_logit_val = jax.scipy.special.logit((self.alpha_bar_min - 0.001) / 0.998)
            # delta_fraction = sigmoid(delta_fraction_logit) where delta_fraction = (alpha_bar_max - alpha_bar_min) / (0.999 - alpha_bar_min)
            alpha_bar_min_clamped = jnp.clip(self.alpha_bar_min, 0.001, 0.998)
            alpha_bar_max_clamped = jnp.clip(self.alpha_bar_max, alpha_bar_min_clamped, 0.999)
            delta_max_init = 0.999 - alpha_bar_min_clamped
            delta_actual_init = alpha_bar_max_clamped - alpha_bar_min_clamped
            # Use jnp.where for JIT compatibility
            delta_fraction_init = jnp.where(delta_max_init > 0, delta_actual_init / delta_max_init, 0.0)
            delta_fraction_logit_val = jax.scipy.special.logit(jnp.clip(delta_fraction_init, 0.001, 0.999))
            
            beta_logit = self.param('beta_logit', nn.initializers.constant(beta_logit_val), ())
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            delta_fraction_logit = self.param('delta_fraction_logit',
                                             nn.initializers.constant(delta_fraction_logit_val), ())
        
        beta = jax.nn.softplus(beta_logit)
        alpha_bar_min = 0.001 + 0.998 * jax.nn.sigmoid(alpha_bar_min_logit)
        delta_fraction = jax.nn.sigmoid(delta_fraction_logit)
        delta_max = 0.999 - alpha_bar_min
        delta_alpha = delta_fraction * delta_max
        alpha_bar_max = alpha_bar_min + delta_alpha  # Guaranteed to be in [alpha_bar_min, 0.999]
        
        # Optimized: cache exp(-beta * t)
        beta_t_neg = -beta * t  # Cache -beta * t
        exp_val = jnp.exp(beta_t_neg)
        one_minus_exp = 1.0 - exp_val  # Cache (1 - exp)
        alpha_bar_t = alpha_bar_min + delta_alpha * one_minus_exp
        alpha_bar_t = jnp.clip(alpha_bar_t, 0.001, 0.999)
        
        # Derivative: d/dt [1 - exp(-beta*t)] = beta * exp(-beta*t)
        alpha_bar_prime_t = delta_alpha * beta * exp_val
        
        # Compute gamma_prime efficiently
        alpha_bar_t_one_minus = alpha_bar_t * (1.0 - alpha_bar_t)
        gamma_prime_t = alpha_bar_prime_t / alpha_bar_t_one_minus
        
        return alpha_bar_t, gamma_prime_t


class CauchyNoiseSchedule(NoiseSchedule):
    """Cauchy distribution-based noise schedule with learnable parameters.
    
    From paper Section III-A6: Cauchy Distribution
    Uses Cauchy cumulative distribution function.
    For increasing schedule: alpha_bar(t) = CDF((t - loc) / scale)
    
    All parameters are learnable:
    - loc: unbounded location parameter
    - scale: positive scale parameter (enforced via softplus)
    - alpha_bar_min: bounded to [0.001, 0.999]
    - alpha_bar_max: alpha_bar_min + delta_fraction * (0.999 - alpha_bar_min) to ensure max > min and max <= 0.999
    
    Args:
        loc: Initial value for loc (default: 0.5)
        scale: Initial value for scale (default: 0.1)
        alpha_bar_min: Initial value for alpha_bar_min (default: 0.01)
        alpha_bar_max: Initial value for alpha_bar_max (default: 0.99)
    """
    
    loc: float = 0.5  # Initial loc value
    scale: float = 0.1  # Initial scale value
    alpha_bar_min: float = 0.01  # Initial alpha_bar_min value
    alpha_bar_max: float = 0.99  # Initial alpha_bar_max value
    
    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "loc": 0.5,
            "scale": 0.1,
            "alpha_bar_min": 0.01,
            "alpha_bar_max": 0.99,
        }
    
    @nn.compact
    def get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for Cauchy schedule."""
        if params is not None:
            loc = params['loc']
            scale_logit = params['scale_logit']
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            delta_fraction_logit = params['delta_fraction_logit']
        else:
            # Compute initial logit from initial scale: scale = softplus(logit) -> logit = log(exp(scale) - 1)
            # Use stable softplus inverse
            scale_logit_val = jnp.log1p(jnp.expm1(self.scale)) if self.scale > 0 else -10.0
            alpha_bar_min_logit_val = jax.scipy.special.logit((self.alpha_bar_min - 0.001) / 0.998)
            # delta_fraction = sigmoid(delta_fraction_logit) where delta_fraction = (alpha_bar_max - alpha_bar_min) / (0.999 - alpha_bar_min)
            alpha_bar_min_clamped = jnp.clip(self.alpha_bar_min, 0.001, 0.998)
            alpha_bar_max_clamped = jnp.clip(self.alpha_bar_max, alpha_bar_min_clamped, 0.999)
            delta_max_init = 0.999 - alpha_bar_min_clamped
            delta_actual_init = alpha_bar_max_clamped - alpha_bar_min_clamped
            # Use jnp.where for JIT compatibility
            delta_fraction_init = jnp.where(delta_max_init > 0, delta_actual_init / delta_max_init, 0.0)
            delta_fraction_logit_val = jax.scipy.special.logit(jnp.clip(delta_fraction_init, 0.001, 0.999))
            
            loc = self.param('loc', nn.initializers.constant(self.loc), ())
            scale_logit = self.param('scale_logit', nn.initializers.constant(scale_logit_val), ())
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            delta_fraction_logit = self.param('delta_fraction_logit',
                                             nn.initializers.constant(delta_fraction_logit_val), ())
        
        scale = jax.nn.softplus(scale_logit)
        alpha_bar_min = 0.001 + 0.998 * jax.nn.sigmoid(alpha_bar_min_logit)
        delta_fraction = jax.nn.sigmoid(delta_fraction_logit)
        delta_max = 0.999 - alpha_bar_min
        delta_alpha = delta_fraction * delta_max
        alpha_bar_max = alpha_bar_min + delta_alpha  # Guaranteed to be in [alpha_bar_min, 0.999]
        
        # Optimized: cache 1/scale and 1/pi
        scale_inv = 1.0 / scale  # Cache 1/scale
        pi_inv = 1.0 / jnp.pi  # Cache 1/pi
        normalized = (t - loc) * scale_inv  # More efficient than (t - loc) / scale
        cdf_val = 0.5 + pi_inv * jnp.arctan(normalized)  # Cache CDF value
        alpha_bar_t = alpha_bar_min + delta_alpha * cdf_val
        return jnp.clip(alpha_bar_t, 0.001, 0.999)
    
    @nn.compact
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) and gamma_prime(t) for Cauchy schedule."""
        if params is not None:
            loc = params['loc']
            scale_logit = params['scale_logit']
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            delta_fraction_logit = params['delta_fraction_logit']
        else:
            # Use stable softplus inverse
            scale_logit_val = jnp.log1p(jnp.expm1(self.scale)) if self.scale > 0 else -10.0
            alpha_bar_min_logit_val = jax.scipy.special.logit((self.alpha_bar_min - 0.001) / 0.998)
            # delta_fraction = sigmoid(delta_fraction_logit) where delta_fraction = (alpha_bar_max - alpha_bar_min) / (0.999 - alpha_bar_min)
            alpha_bar_min_clamped = jnp.clip(self.alpha_bar_min, 0.001, 0.998)
            alpha_bar_max_clamped = jnp.clip(self.alpha_bar_max, alpha_bar_min_clamped, 0.999)
            delta_max_init = 0.999 - alpha_bar_min_clamped
            delta_actual_init = alpha_bar_max_clamped - alpha_bar_min_clamped
            # Use jnp.where for JIT compatibility
            delta_fraction_init = jnp.where(delta_max_init > 0, delta_actual_init / delta_max_init, 0.0)
            delta_fraction_logit_val = jax.scipy.special.logit(jnp.clip(delta_fraction_init, 0.001, 0.999))
            
            loc = self.param('loc', nn.initializers.constant(self.loc), ())
            scale_logit = self.param('scale_logit', nn.initializers.constant(scale_logit_val), ())
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            delta_fraction_logit = self.param('delta_fraction_logit',
                                             nn.initializers.constant(delta_fraction_logit_val), ())
        
        scale = jax.nn.softplus(scale_logit)
        alpha_bar_min = 0.001 + 0.998 * jax.nn.sigmoid(alpha_bar_min_logit)
        delta_fraction = jax.nn.sigmoid(delta_fraction_logit)
        delta_max = 0.999 - alpha_bar_min
        delta_alpha = delta_fraction * delta_max
        alpha_bar_max = alpha_bar_min + delta_alpha  # Guaranteed to be in [alpha_bar_min, 0.999]
        
        # Optimized: cache 1/scale and 1/pi
        scale_inv = 1.0 / scale  # Cache 1/scale
        pi_inv = 1.0 / jnp.pi  # Cache 1/pi
        normalized = (t - loc) * scale_inv  # More efficient than (t - loc) / scale
        normalized_sq = normalized * normalized  # Cache normalized^2
        cdf_val = 0.5 + pi_inv * jnp.arctan(normalized)  # Cache CDF value
        alpha_bar_t = alpha_bar_min + delta_alpha * cdf_val
        alpha_bar_t = jnp.clip(alpha_bar_t, 0.001, 0.999)
        
        # Cauchy PDF: 1 / (pi * scale * (1 + normalized^2))
        # Optimized: cache 1 + normalized^2
        one_plus_normalized_sq = 1.0 + normalized_sq  # Cache (1 + normalized^2)
        pdf_val = pi_inv * scale_inv / one_plus_normalized_sq  # Cache PDF value
        alpha_bar_prime_t = delta_alpha * pdf_val
        
        # Compute gamma_prime efficiently
        alpha_bar_t_one_minus = alpha_bar_t * (1.0 - alpha_bar_t)
        gamma_prime_t = alpha_bar_prime_t / alpha_bar_t_one_minus
        
        return alpha_bar_t, gamma_prime_t


class LaplaceNoiseSchedule(NoiseSchedule):
    """Laplace distribution-based noise schedule with learnable parameters.
    
    From paper Section III-A7: Laplace Distribution
    Uses Laplace cumulative distribution function.
    For increasing schedule: alpha_bar(t) = CDF((t - loc) / scale)
    
    All parameters are learnable:
    - loc: unbounded location parameter
    - scale: positive scale parameter (enforced via softplus)
    - alpha_bar_min: bounded to [0.001, 0.999]
    - alpha_bar_max: alpha_bar_min + delta_fraction * (0.999 - alpha_bar_min) to ensure max > min and max <= 0.999
    
    Args:
        loc: Initial value for loc (default: 0.5)
        scale: Initial value for scale (default: 0.1)
        alpha_bar_min: Initial value for alpha_bar_min (default: 0.01)
        alpha_bar_max: Initial value for alpha_bar_max (default: 0.99)
    """
    
    loc: float = 0.5  # Initial loc value
    scale: float = 0.1  # Initial scale value
    alpha_bar_min: float = 0.01  # Initial alpha_bar_min value
    alpha_bar_max: float = 0.99  # Initial alpha_bar_max value
    
    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "loc": 0.5,
            "scale": 0.1,
            "alpha_bar_min": 0.01,
            "alpha_bar_max": 0.99,
        }
    
    @nn.compact
    def get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for Laplace schedule."""
        if params is not None:
            loc = params['loc']
            scale_logit = params['scale_logit']
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            delta_fraction_logit = params['delta_fraction_logit']
        else:
            # Use stable softplus inverse
            scale_logit_val = jnp.log1p(jnp.expm1(self.scale)) if self.scale > 0 else -10.0
            alpha_bar_min_logit_val = jax.scipy.special.logit((self.alpha_bar_min - 0.001) / 0.998)
            # delta_fraction = sigmoid(delta_fraction_logit) where delta_fraction = (alpha_bar_max - alpha_bar_min) / (0.999 - alpha_bar_min)
            alpha_bar_min_clamped = jnp.clip(self.alpha_bar_min, 0.001, 0.998)
            alpha_bar_max_clamped = jnp.clip(self.alpha_bar_max, alpha_bar_min_clamped, 0.999)
            delta_max_init = 0.999 - alpha_bar_min_clamped
            delta_actual_init = alpha_bar_max_clamped - alpha_bar_min_clamped
            # Use jnp.where for JIT compatibility
            delta_fraction_init = jnp.where(delta_max_init > 0, delta_actual_init / delta_max_init, 0.0)
            delta_fraction_logit_val = jax.scipy.special.logit(jnp.clip(delta_fraction_init, 0.001, 0.999))
            
            loc = self.param('loc', nn.initializers.constant(self.loc), ())
            scale_logit = self.param('scale_logit', nn.initializers.constant(scale_logit_val), ())
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            delta_fraction_logit = self.param('delta_fraction_logit',
                                             nn.initializers.constant(delta_fraction_logit_val), ())
        
        scale = jax.nn.softplus(scale_logit)
        alpha_bar_min = 0.001 + 0.998 * jax.nn.sigmoid(alpha_bar_min_logit)
        delta_fraction = jax.nn.sigmoid(delta_fraction_logit)
        delta_max = 0.999 - alpha_bar_min
        delta_alpha = delta_fraction * delta_max
        alpha_bar_max = alpha_bar_min + delta_alpha  # Guaranteed to be in [alpha_bar_min, 0.999]
        
        # Optimized: cache 1/scale and compute normalized efficiently
        scale_inv = 1.0 / scale  # Cache 1/scale
        normalized = (t - loc) * scale_inv  # More efficient than (t - loc) / scale
        sign = jnp.sign(normalized)
        abs_val = jnp.abs(normalized)
        exp_neg_abs = jnp.exp(-abs_val)  # Cache exp(-abs_val)
        cdf_val = 0.5 + 0.5 * sign * (1.0 - exp_neg_abs)  # Cache CDF value
        alpha_bar_t = alpha_bar_min + delta_alpha * cdf_val
        return jnp.clip(alpha_bar_t, 0.001, 0.999)
    
    @nn.compact
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) and gamma_prime(t) for Laplace schedule."""
        if params is not None:
            loc = params['loc']
            scale_logit = params['scale_logit']
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            delta_fraction_logit = params['delta_fraction_logit']
        else:
            # Use stable softplus inverse
            scale_logit_val = jnp.log1p(jnp.expm1(self.scale)) if self.scale > 0 else -10.0
            alpha_bar_min_logit_val = jax.scipy.special.logit((self.alpha_bar_min - 0.001) / 0.998)
            # delta_fraction = sigmoid(delta_fraction_logit) where delta_fraction = (alpha_bar_max - alpha_bar_min) / (0.999 - alpha_bar_min)
            alpha_bar_min_clamped = jnp.clip(self.alpha_bar_min, 0.001, 0.998)
            alpha_bar_max_clamped = jnp.clip(self.alpha_bar_max, alpha_bar_min_clamped, 0.999)
            delta_max_init = 0.999 - alpha_bar_min_clamped
            delta_actual_init = alpha_bar_max_clamped - alpha_bar_min_clamped
            # Use jnp.where for JIT compatibility
            delta_fraction_init = jnp.where(delta_max_init > 0, delta_actual_init / delta_max_init, 0.0)
            delta_fraction_logit_val = jax.scipy.special.logit(jnp.clip(delta_fraction_init, 0.001, 0.999))
            
            loc = self.param('loc', nn.initializers.constant(self.loc), ())
            scale_logit = self.param('scale_logit', nn.initializers.constant(scale_logit_val), ())
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            delta_fraction_logit = self.param('delta_fraction_logit',
                                             nn.initializers.constant(delta_fraction_logit_val), ())
        
        scale = jax.nn.softplus(scale_logit)
        alpha_bar_min = 0.001 + 0.998 * jax.nn.sigmoid(alpha_bar_min_logit)
        delta_fraction = jax.nn.sigmoid(delta_fraction_logit)
        delta_max = 0.999 - alpha_bar_min
        delta_alpha = delta_fraction * delta_max
        alpha_bar_max = alpha_bar_min + delta_alpha  # Guaranteed to be in [alpha_bar_min, 0.999]
        
        # Optimized: cache 1/scale and compute normalized efficiently
        scale_inv = 1.0 / scale  # Cache 1/scale
        normalized = (t - loc) * scale_inv  # More efficient than (t - loc) / scale
        sign = jnp.sign(normalized)
        abs_val = jnp.abs(normalized)
        exp_neg_abs = jnp.exp(-abs_val)  # Cache exp(-abs_val)
        cdf_val = 0.5 + 0.5 * sign * (1.0 - exp_neg_abs)  # Cache CDF value
        alpha_bar_t = alpha_bar_min + delta_alpha * cdf_val
        alpha_bar_t = jnp.clip(alpha_bar_t, 0.001, 0.999)
        
        # Laplace PDF: (1/(2*scale)) * exp(-|(t - loc)/scale|)
        pdf_val = 0.5 * scale_inv * exp_neg_abs  # Cache PDF value
        alpha_bar_prime_t = delta_alpha * pdf_val
        
        # Compute gamma_prime efficiently
        alpha_bar_t_one_minus = alpha_bar_t * (1.0 - alpha_bar_t)
        gamma_prime_t = alpha_bar_prime_t / alpha_bar_t_one_minus
        
        return alpha_bar_t, gamma_prime_t


class QuadraticNoiseSchedule(NoiseSchedule):
    """Quadratic noise schedule with learnable parameters.
    
    Uses a quadratic parameterization: alpha_bar(t) = alpha_bar_min + (alpha_bar_max - alpha_bar_min) * t^2
    
    All parameters are learnable:
    - alpha_bar_min: bounded to [0.001, 0.999]
    - alpha_bar_max: alpha_bar_min + delta_fraction * (0.999 - alpha_bar_min) to ensure max > min and max <= 0.999
    
    Args:
        alpha_bar_min: Initial value for alpha_bar_min (default: 0.01)
        alpha_bar_max: Initial value for alpha_bar_max (default: 0.99)
    """
    
    alpha_bar_min: float = 0.01  # Initial alpha_bar_min value
    alpha_bar_max: float = 0.99  # Initial alpha_bar_max value
    
    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "alpha_bar_min": 0.01,
            "alpha_bar_max": 0.99,
        }
    
    @nn.compact
    def get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for quadratic schedule."""
        if params is not None:
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            delta_fraction_logit = params['delta_fraction_logit']
        else:
            # Compute initial logit values from initial alpha_bar values
            alpha_bar_min_logit_val = jax.scipy.special.logit((self.alpha_bar_min - 0.001) / 0.998)
            # delta_fraction = sigmoid(delta_fraction_logit) where delta_fraction = (alpha_bar_max - alpha_bar_min) / (0.999 - alpha_bar_min)
            alpha_bar_min_clamped = jnp.clip(self.alpha_bar_min, 0.001, 0.998)
            alpha_bar_max_clamped = jnp.clip(self.alpha_bar_max, alpha_bar_min_clamped, 0.999)
            delta_max_init = 0.999 - alpha_bar_min_clamped
            delta_actual_init = alpha_bar_max_clamped - alpha_bar_min_clamped
            # Use jnp.where for JIT compatibility
            delta_fraction_init = jnp.where(delta_max_init > 0, delta_actual_init / delta_max_init, 0.0)
            delta_fraction_logit_val = jax.scipy.special.logit(jnp.clip(delta_fraction_init, 0.001, 0.999))
            
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            delta_fraction_logit = self.param('delta_fraction_logit',
                                             nn.initializers.constant(delta_fraction_logit_val), ())
        
        # Transform to bounded values
        alpha_bar_min = 0.001 + 0.998 * jax.nn.sigmoid(alpha_bar_min_logit)  # [0.001, 0.999]
        delta_fraction = jax.nn.sigmoid(delta_fraction_logit)
        delta_max = 0.999 - alpha_bar_min
        delta_alpha = delta_fraction * delta_max
        alpha_bar_max = alpha_bar_min + delta_alpha  # Guaranteed to be in [alpha_bar_min, 0.999]
        
        # Quadratic: alpha_bar(t) = alpha_bar_min + (alpha_bar_max - alpha_bar_min) * t^2
        alpha_bar_t = alpha_bar_min + (alpha_bar_max - alpha_bar_min) * t ** 2
        alpha_bar_t = jnp.clip(alpha_bar_t, 0.001, 0.999)
        return alpha_bar_t
    
    @nn.compact
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) and gamma_prime(t) for quadratic schedule."""
        if params is not None:
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            delta_fraction_logit = params['delta_fraction_logit']
        else:
            # Compute initial logit values from initial alpha_bar values
            alpha_bar_min_logit_val = jax.scipy.special.logit((self.alpha_bar_min - 0.001) / 0.998)
            # delta_fraction = sigmoid(delta_fraction_logit) where delta_fraction = (alpha_bar_max - alpha_bar_min) / (0.999 - alpha_bar_min)
            alpha_bar_min_clamped = jnp.clip(self.alpha_bar_min, 0.001, 0.998)
            alpha_bar_max_clamped = jnp.clip(self.alpha_bar_max, alpha_bar_min_clamped, 0.999)
            delta_max_init = 0.999 - alpha_bar_min_clamped
            delta_actual_init = alpha_bar_max_clamped - alpha_bar_min_clamped
            # Use jnp.where for JIT compatibility
            delta_fraction_init = jnp.where(delta_max_init > 0, delta_actual_init / delta_max_init, 0.0)
            delta_fraction_logit_val = jax.scipy.special.logit(jnp.clip(delta_fraction_init, 0.001, 0.999))
            
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            delta_fraction_logit = self.param('delta_fraction_logit',
                                             nn.initializers.constant(delta_fraction_logit_val), ())
        
        # Transform to bounded values - optimized: cache intermediate values
        alpha_bar_min = 0.001 + 0.998 * jax.nn.sigmoid(alpha_bar_min_logit)
        delta_fraction = jax.nn.sigmoid(delta_fraction_logit)
        delta_max = 0.999 - alpha_bar_min
        delta_alpha = delta_fraction * delta_max
        alpha_bar_max = alpha_bar_min + delta_alpha  # Guaranteed to be in [alpha_bar_min, 0.999]
        
        # Quadratic: alpha_bar(t) = alpha_bar_min + delta_alpha * t^2
        # Use t * t instead of t ** 2 for efficiency
        t_squared = t * t
        alpha_bar_t = alpha_bar_min + delta_alpha * t_squared
        alpha_bar_t = jnp.clip(alpha_bar_t, 0.001, 0.999)
        
        # Derivative: d/dt [t^2] = 2*t
        alpha_bar_prime_t = delta_alpha * 2.0 * t
        
        # Compute gamma_prime efficiently
        alpha_bar_t_one_minus = alpha_bar_t * (1.0 - alpha_bar_t)
        gamma_prime_t = alpha_bar_prime_t / alpha_bar_t_one_minus
        
        return alpha_bar_t, gamma_prime_t


class PolynomialNoiseSchedule(NoiseSchedule):
    """Polynomial noise schedule with learnable parameters.
    
    Uses a polynomial parameterization: alpha_bar(t) = alpha_bar_min + (alpha_bar_max - alpha_bar_min) * t^power
    
    All parameters are learnable:
    - power: positive polynomial power (enforced via softplus), typically >= 1.0
    - alpha_bar_min: bounded to [0.001, 0.999]
    - alpha_bar_max: alpha_bar_min + delta_fraction * (0.999 - alpha_bar_min) to ensure max > min and max <= 0.999
    
    Args:
        power: Initial value for polynomial power (default: 2.0)
        alpha_bar_min: Initial value for alpha_bar_min (default: 0.01)
        alpha_bar_max: Initial value for alpha_bar_max (default: 0.99)
    """
    
    power: float = 2.0  # Initial power value
    alpha_bar_min: float = 0.01  # Initial alpha_bar_min value
    alpha_bar_max: float = 0.99  # Initial alpha_bar_max value
    
    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "power": 2.0,
            "alpha_bar_min": 0.01,
            "alpha_bar_max": 0.99,
        }

    @nn.compact
    def get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for polynomial schedule."""
        if params is not None:
            power_logit = params['power_logit']
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            delta_fraction_logit = params['delta_fraction_logit']
        else:
            # Compute initial logit values from initial values
            # power = 1.0 + softplus(logit) to ensure power >= 1.0
            # So logit = log(exp(power - 1.0) - 1) if power > 1.0 else -10.0
            # Use stable softplus inverse
            power_logit_val = jnp.log1p(jnp.expm1(self.power - 1.0)) if self.power > 1.0 else -10.0
            alpha_bar_min_logit_val = jax.scipy.special.logit((self.alpha_bar_min - 0.001) / 0.998)
            # delta_fraction = sigmoid(delta_fraction_logit) where delta_fraction = (alpha_bar_max - alpha_bar_min) / (0.999 - alpha_bar_min)
            alpha_bar_min_clamped = jnp.clip(self.alpha_bar_min, 0.001, 0.998)
            alpha_bar_max_clamped = jnp.clip(self.alpha_bar_max, alpha_bar_min_clamped, 0.999)
            delta_max_init = 0.999 - alpha_bar_min_clamped
            delta_actual_init = alpha_bar_max_clamped - alpha_bar_min_clamped
            # Use jnp.where for JIT compatibility
            delta_fraction_init = jnp.where(delta_max_init > 0, delta_actual_init / delta_max_init, 0.0)
            delta_fraction_logit_val = jax.scipy.special.logit(jnp.clip(delta_fraction_init, 0.001, 0.999))
            
            power_logit = self.param('power_logit', nn.initializers.constant(power_logit_val), ())
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            delta_fraction_logit = self.param('delta_fraction_logit',
                                             nn.initializers.constant(delta_fraction_logit_val), ())
        
        # Transform to bounded values - optimized: cache intermediate values
        power = 1.0 + jax.nn.softplus(power_logit)  # Ensure power >= 1.0 for monotonicity
        alpha_bar_min = 0.001 + 0.998 * jax.nn.sigmoid(alpha_bar_min_logit)
        delta_fraction = jax.nn.sigmoid(delta_fraction_logit)
        delta_max = 0.999 - alpha_bar_min
        delta_alpha = delta_fraction * delta_max
        alpha_bar_max = alpha_bar_min + delta_alpha  # Guaranteed to be in [alpha_bar_min, 0.999]
        
        # Polynomial: alpha_bar(t) = alpha_bar_min + delta_alpha * t^power
        t_power = t ** power
        alpha_bar_t = alpha_bar_min + delta_alpha * t_power
        return jnp.clip(alpha_bar_t, 0.001, 0.999)

    @nn.compact
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) and gamma_prime(t) for polynomial schedule."""
        if params is not None:
            power_logit = params['power_logit']
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            delta_fraction_logit = params['delta_fraction_logit']
        else:
            power_logit_val = jnp.log(jnp.exp(self.power - 1.0) - 1.0) if self.power > 1.0 else -10.0
            alpha_bar_min_logit_val = jax.scipy.special.logit((self.alpha_bar_min - 0.001) / 0.998)
            # delta_fraction = sigmoid(delta_fraction_logit) where delta_fraction = (alpha_bar_max - alpha_bar_min) / (0.999 - alpha_bar_min)
            alpha_bar_min_clamped = jnp.clip(self.alpha_bar_min, 0.001, 0.998)
            alpha_bar_max_clamped = jnp.clip(self.alpha_bar_max, alpha_bar_min_clamped, 0.999)
            delta_max_init = 0.999 - alpha_bar_min_clamped
            delta_actual_init = alpha_bar_max_clamped - alpha_bar_min_clamped
            # Use jnp.where for JIT compatibility
            delta_fraction_init = jnp.where(delta_max_init > 0, delta_actual_init / delta_max_init, 0.0)
            delta_fraction_logit_val = jax.scipy.special.logit(jnp.clip(delta_fraction_init, 0.001, 0.999))
            
            power_logit = self.param('power_logit', nn.initializers.constant(power_logit_val), ())
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            delta_fraction_logit = self.param('delta_fraction_logit',
                                             nn.initializers.constant(delta_fraction_logit_val), ())
        
        # Transform to bounded values - optimized: cache intermediate values
        power = 1.0 + jax.nn.softplus(power_logit)  # Ensure power >= 1.0 for monotonicity
        alpha_bar_min = 0.001 + 0.998 * jax.nn.sigmoid(alpha_bar_min_logit)
        delta_fraction = jax.nn.sigmoid(delta_fraction_logit)
        delta_max = 0.999 - alpha_bar_min
        delta_alpha = delta_fraction * delta_max
        alpha_bar_max = alpha_bar_min + delta_alpha  # Guaranteed to be in [alpha_bar_min, 0.999]
        
        # Polynomial: alpha_bar(t) = alpha_bar_min + delta_alpha * t^power
        t_power = t ** power
        alpha_bar_t = alpha_bar_min + delta_alpha * t_power
        alpha_bar_t = jnp.clip(alpha_bar_t, 0.001, 0.999)
        
        # Derivative: d/dt [t^power] = power * t^(power-1)
        power_minus_one = power - 1.0
        t_power_minus_one = t ** power_minus_one
        alpha_bar_prime_t = delta_alpha * power * t_power_minus_one
        
        # Compute gamma_prime efficiently
        alpha_bar_t_one_minus = alpha_bar_t * (1.0 - alpha_bar_t)
        gamma_prime_t = alpha_bar_prime_t / alpha_bar_t_one_minus
        
        return alpha_bar_t, gamma_prime_t


class LogisticNoiseSchedule(NoiseSchedule):
    """Logistic noise schedule with learnable parameters.
    
    From paper Section III-A8: Logistic Schedule
    Similar to sigmoid but with specific parameterization.
    alpha_bar(t) = 1 / (1 + exp(-k * (t - t_mid)))
    
    Note: This is mathematically equivalent to sigmoid(k * (t - t_mid))
    
    All parameters are learnable:
    - k: positive steepness parameter (enforced via softplus)
    - t_mid: unbounded midpoint parameter
    
    Args:
        k: Initial value for k (default: 10.0)
        t_mid: Initial value for t_mid (default: 0.5)
    """
    
    k: float = 10.0  # Initial k value
    t_mid: float = 0.5  # Initial t_mid value
    
    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "k": 10.0,
            "t_mid": 0.5,
        }

    @nn.compact
    def get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for logistic schedule."""
        if params is not None:
            k_logit = params['k_logit']
            t_mid = params['t_mid']
        else:
            k_logit_val = jnp.log(jnp.exp(self.k) - 1.0) if self.k > 0 else -10.0
            k_logit = self.param('k_logit', nn.initializers.constant(k_logit_val), ())
            t_mid = self.param('t_mid', nn.initializers.constant(self.t_mid), ())
        
        k = jax.nn.softplus(k_logit)
        gamma_t = k * (t - t_mid)
        alpha_bar_t = jax.nn.sigmoid(gamma_t)
        return alpha_bar_t

    @nn.compact
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) and gamma_prime(t) for logistic schedule."""
        if params is not None:
            k_logit = params['k_logit']
            t_mid = params['t_mid']
        else:
            k_logit_val = jnp.log(jnp.exp(self.k) - 1.0) if self.k > 0 else -10.0
            k_logit = self.param('k_logit', nn.initializers.constant(k_logit_val), ())
            t_mid = self.param('t_mid', nn.initializers.constant(self.t_mid), ())
        
        k = jax.nn.softplus(k_logit)
        gamma_t = k * (t - t_mid)
        gamma_prime_t = jnp.full_like(t, k)
        
        alpha_bar_t = jax.nn.sigmoid(gamma_t)
        
        return alpha_bar_t, gamma_prime_t


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
    """Monotonic neural network with positive weights and ReLU activations."""

    hidden_dims: Tuple[int, ...]
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply monotonic network."""
        # Ensure input has the right shape
        x = jnp.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            # Scalar input - add batch dimension
            x = x[None, None]  # [1, 1]
            scalar_input = True
        elif x.ndim == 1:
            # Batch input - add feature dimension
            x = x[:, None]  # [batch_size, 1]
        
        for hidden_dim in self.hidden_dims:
            x = PositiveDense(hidden_dim)(x)
            x = nn.relu(x)
        x = PositiveDense(1)(x)

        if scalar_input: 
            x = x.squeeze(-1)
        return x.squeeze(-1)


class NoiseScheduleNetwork(NoiseSchedule):
    """Neural network-based noise schedule.
    
    From paper Section III-A9: Monotonic Neural Network
    Uses a learnable neural network with monotonic constraints to parameterize gamma.
    
    Args:
        hidden_dims: Hidden dimensions for the neural network (default: (64, 64))
        gamma_range: Range for gamma values (default: (-4.0, 4.0))
    """
    
    hidden_dims: Tuple[int, ...] = (64, 64)
    monotonic_network: nn.Module = SimpleMonotonicNetwork
    gamma_range: Tuple[float, float] = (-4.0, 4.0)

    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Note: hidden_dims is not included here as it's a structural parameter
        that should be specified at the top level of the config.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "gamma_range": (-4.0, 4.0),
        }

    @nn.compact
    def _get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for learnable schedule."""
        # For NoiseScheduleNetwork, we get alpha_bar from _get_alpha_bar_gamma_prime
        alpha_bar_t, _ = self._get_alpha_bar_gamma_prime(t, params)
        return alpha_bar_t

    @nn.compact
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) and gamma_prime(t) for learnable schedule."""
        scale_logit = self.param('scale_logit', nn.initializers.constant(0.0), ())
        gamma_min = self.param('gamma_min', nn.initializers.constant(self.gamma_range[0]), ())
        gamma_max = self.param('gamma_max', nn.initializers.constant(self.gamma_range[1]), ())

        # Hoist the network so parameters are shared across vectorization
        network = self.monotonic_network(hidden_dims=self.hidden_dims)

        def gamma_fn_scalar(t_input):
            # Ensure scalar/1D input compatibility; network handles shaping internally
            f_t = network(t_input)
            f_t = t_input + (1 - t_input) * t_input * nn.sigmoid(scale_logit) * nn.sigmoid(f_t)
            return gamma_min + (gamma_max - gamma_min) * f_t
        
        t = jnp.asarray(t)
        t_flat = t.reshape(-1)
        vals, grads = jax.vmap(jax.value_and_grad(gamma_fn_scalar))(t_flat)
        gamma_t = vals.reshape(t.shape)
        gamma_prime_t = grads.reshape(t.shape)
        
        # Compute alpha_bar from gamma
        alpha_bar_t = jax.nn.sigmoid(gamma_t)
        
        return alpha_bar_t, gamma_prime_t


# Alias for backward compatibility
LearnableNoiseSchedule = NoiseScheduleNetwork


def create_noise_schedule(
    schedule_type: str, 
    **kwargs: Any
) -> NoiseSchedule:
    """Factory function to create noise schedules.
    
    Args:
        schedule_type: Type of schedule. Options:
            - "linear": Linear schedule
            - "cosine": Cosine schedule
            - "sigmoid": Sigmoid schedule
            - "exponential": Exponential schedule
            - "cauchy": Cauchy distribution schedule
            - "laplace": Laplace distribution schedule
            - "logistic": Logistic schedule
            - "monotonic_nn" or "learnable": Monotonic neural network schedule
        **kwargs: Additional parameters for the schedule
        
    Returns:
        NoiseSchedule instance
    """
    schedule_type = schedule_type.lower()
    
    if schedule_type == "linear":
        return LinearNoiseSchedule(**kwargs)
    elif schedule_type == "cosine":
        return CosineNoiseSchedule(**kwargs)
    elif schedule_type == "sigmoid":
        return SigmoidNoiseSchedule(**kwargs)
    elif schedule_type == "exponential":
        return ExponentialNoiseSchedule(**kwargs)
    elif schedule_type == "cauchy":
        return CauchyNoiseSchedule(**kwargs)
    elif schedule_type == "laplace":
        return LaplaceNoiseSchedule(**kwargs)
    elif schedule_type == "logistic":
        return LogisticNoiseSchedule(**kwargs)
    elif schedule_type == "quadratic":
        return QuadraticNoiseSchedule(**kwargs)
    elif schedule_type == "polynomial":
        return PolynomialNoiseSchedule(**kwargs)
    elif schedule_type in ["monotonic_nn", "learnable", "monotonic_neural_network", "network"]:
        return NoiseScheduleNetwork(**kwargs)
    else:
        raise ValueError(
            f"Unknown schedule type: {schedule_type}. "
            f"Options: linear, cosine, sigmoid, exponential, "
            f"cauchy, laplace, logistic, quadratic, polynomial, monotonic_nn/learnable/network"
        )
