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

import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from tensorflow_probability.substrates.jax import math as tfpmath

clip = tfpmath.clip_by_value_preserve_gradient

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
        gamma_prime_max: Maximum value for clipping gamma_prime_t. Default is 100.0.
    """
    learnable: bool = True  # Whether parameters are learnable
    gamma_prime_max: float = 100.0  # Maximum value for clipping gamma_prime_t
    
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
    - alpha_bar_min: bounded to [0, 1] via sigmoid
    - alpha_bar_max: bounded to [alpha_bar_min, 1] via sigmoid
    
    Args:
        alpha_bar_min: Initial and bound values for alpha_bar_min (default: 0.02)
        alpha_bar_max: Initial and bound values for alpha_bar_max (default: 0.98)
    """
    
    alpha_bar_min: float = 0.01  # Initial alpha_bar_min value (safe: max gamma_prime ~18.95)
    alpha_bar_max: float = 0.99  # Initial alpha_bar_max value (safe: max gamma_prime ~18.95)
    
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
    def _get_alpha_bar_gamma_prime(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) for linear schedule."""
        if params is not None:
            alpha_bar_min_logit = params['alpha_bar_min_logit']
            alpha_bar_max_logit = params['alpha_bar_max_logit']
        else:
            alpha_bar_min_logit_val = jax.scipy.special.logit(self.alpha_bar_min)
            alpha_bar_max_logit_val = jax.scipy.special.logit(self.alpha_bar_max)
            alpha_bar_min_logit = self.param('alpha_bar_min_logit', 
                                            nn.initializers.constant(alpha_bar_min_logit_val), ())
            alpha_bar_max_logit = self.param('alpha_bar_max_logit',
                                             nn.initializers.constant(alpha_bar_max_logit_val), ())        
        alpha_bar_min = jax.nn.sigmoid(alpha_bar_min_logit)
        alpha_bar_min = jnp.clip(alpha_bar_min, self.alpha_bar_min, self.alpha_bar_max) 
        alpha_bar_max = jax.nn.sigmoid(alpha_bar_max_logit)
        alpha_bar_max = jnp.clip(alpha_bar_max, alpha_bar_min, self.alpha_bar_max)

        delta_alpha = alpha_bar_max - alpha_bar_min
        alpha_bar_t = alpha_bar_min + t * delta_alpha
        gamma_prime_t = delta_alpha / (alpha_bar_t * (1.0 - alpha_bar_t))
        gamma_prime_t = jnp.clip(gamma_prime_t, 0.0, self.gamma_prime_max)

        return alpha_bar_t, gamma_prime_t
    
    def _get_alpha_bar(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> jnp.ndarray:
        return self._get_alpha_bar_gamma_prime(t, params)[0]
        
class CosineNoiseSchedule(NoiseSchedule):
    """Cosine noise schedule with learnable parameters.
    
    From paper Section III-A3: Cosine Schedule
    We use the increasing version: alpha_bar(t) = sin^2((t + s) / (1 + s) * pi/2)
    which ranges from 0 to 1 as t goes from 0 to 1.
    
    All parameters are learnable:
    - s: positive offset (default: 0.008, enforced via softplus)
    
    Args:
        alpha_bar_min: Initial and bound values for alpha_bar_min (default: 0.02)
        alpha_bar_max: Initial and bound values for alpha_bar_max (default: 0.98)
    """
    alpha_bar_min: float = 0.01  # Initial alpha_bar_min value (safe: max gamma_prime ~3.20 with alpha_bar 0.05-0.95)
    alpha_bar_max: float = 0.99  # Initial alpha_bar_max value (safe: max gamma_prime ~3.20 with alpha_bar 0.05-0.95)
    
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
        """Get alpha_bar(t) for cosine schedule."""
        return self._get_alpha_bar_gamma_prime(t, params)[0]
    
    @nn.compact
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if params is not None:
            s_min_val = params['s_min']
            s_max_val = params['s_max']
        else:
            s_min = jnp.asin(jnp.sqrt(self.alpha_bar_min))
            s_max = jnp.asin(jnp.sqrt(self.alpha_bar_max))

            s_min_val = self.param('s_min', nn.initializers.constant(s_min), ())
            s_max_val = self.param('s_max', nn.initializers.constant(s_max), ())

            s_min_val = jnp.clip(s_min_val, s_min, jnp.inf)
            s_max_val = jnp.clip(s_max_val, s_min_val, s_max)

        s = s_min_val + (s_max_val-s_min_val) * t        
        sin_s_squared = jnp.sin(s)**2
        alpha_bar_t = self.alpha_bar_min + (self.alpha_bar_max - self.alpha_bar_min) * sin_s_squared
        alpha_bar_prime_t = 2.0 * jnp.sin(s) * jnp.cos(s) * (s_max_val - s_min_val) * (self.alpha_bar_max - self.alpha_bar_min)

        gamma_prime_t = alpha_bar_prime_t / (alpha_bar_t * (1.0 - alpha_bar_t))
        gamma_prime_t = jnp.clip(gamma_prime_t, 0.0, self.gamma_prime_max)
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
    
    alpha_bar_min: float = 0.01  # Initial alpha_bar_min value (safe: max gamma_prime ~18.95)
    alpha_bar_max: float = 0.99  # Initial alpha_bar_max value (safe: max gamma_prime ~18.95)
    
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
    def _get_alpha_bar_gamma_prime(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) for sigmoid schedule."""
        if params is not None:
            alpha_bar_min = jnp.clip(params['alpha_bar_min'], self.alpha_bar_min, self.alpha_bar_max)
            alpha_bar_max = jnp.clip(params['alpha_bar_max'], alpha_bar_min,self.alpha_bar_max) 
        else:
            alpha_bar_min = self.param('alpha_bar_min', nn.initializers.constant(self.alpha_bar_min), ())
            alpha_bar_max = self.param('alpha_bar_max', nn.initializers.constant(self.alpha_bar_max), ())
            alpha_bar_min = jnp.clip(alpha_bar_min, self.alpha_bar_min, self.alpha_bar_max)
            alpha_bar_max = jnp.clip(alpha_bar_max, alpha_bar_min, self.alpha_bar_max) 

        k_times_t_mid = - jax.scipy.special.logit(alpha_bar_min)
        k = jax.scipy.special.logit(alpha_bar_max) + k_times_t_mid

        gamma_t = k * t - k_times_t_mid
        gamma_prime_t = jnp.full_like(t, k)
        gamma_prime_t = jnp.clip(gamma_prime_t, 0.0, self.gamma_prime_max)

        alpha_bar_t = jax.nn.sigmoid(gamma_t)
        
        return alpha_bar_t, gamma_prime_t

    @nn.compact
    def _get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for sigmoid schedule."""
        return self._get_alpha_bar_gamma_prime(t, params)[0]
    

class ExponentialNoiseSchedule(NoiseSchedule):
    """Exponential noise schedule with learnable parameters.
    
    From paper Section III-A5: Exponential Schedule
    In our increasing formulation: alpha_bar(t) = alpha_bar_min + (alpha_bar_max - alpha_bar_min) * (1 - exp(-beta * t))
    
    All parameters are learnable:
    - beta: positive exponential decay rate (enforced via softplus)
    - alpha_bar_min: bounded to [0.001, 0.999]
    - alpha_bar_max: alpha_bar_min + delta_fraction * (0.999 - alpha_bar_min) to ensure max > min and max <= 0.999
    
    Args:
        beta: Initial value for beta (default: 0.5)
        alpha_bar_min: Initial value for alpha_bar_min (default: 0.05)
        alpha_bar_max: Initial value for alpha_bar_max (default: 0.95)
    """
    
    beta: float = 0.5  # Initial beta value (safe: max gamma_prime ~18.46 with alpha_bar 0.05-0.95)
    alpha_bar_min: float = 0.01  # Initial alpha_bar_min value (safe: max gamma_prime ~18.46)
    alpha_bar_max: float = 0.99  # Initial alpha_bar_max value (safe: max gamma_prime ~18.46)
    
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
    def _get_alpha_bar_gamma_prime(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) for exponential schedule."""
        if params is not None:
            alpha_bar_min = jnp.clip(params['alpha_bar_min'], self.alpha_bar_min, self.alpha_bar_max)
            alpha_bar_max = jnp.clip(params['alpha_bar_max'], alpha_bar_min,self.alpha_bar_max) 
        else:
            alpha_bar_min = self.param('alpha_bar_min', nn.initializers.constant(self.alpha_bar_min), ())
            alpha_bar_max = self.param('alpha_bar_max', nn.initializers.constant(self.alpha_bar_max), ())
            alpha_bar_min = jnp.clip(alpha_bar_min, self.alpha_bar_min, self.alpha_bar_max)
            alpha_bar_max = jnp.clip(alpha_bar_max, alpha_bar_min, self.alpha_bar_max) 

        b = jnp.log(alpha_bar_min)
        a = jnp.log(alpha_bar_max) - b

        alpha_bar_t = jnp.exp(a * t + b)
        gamma_prime_t = a/(1-alpha_bar_t)
        gamma_prime_t = jnp.clip(gamma_prime_t, 0.0, self.gamma_prime_max)

        return alpha_bar_t, gamma_prime_t
    
    @nn.compact
    def _get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for exponential schedule."""        
        return self._get_alpha_bar_gamma_prime(t, params)[0]


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
        scale: Initial value for scale (default: 0.3)
        alpha_bar_min: Initial value for alpha_bar_min (default: 0.05)
        alpha_bar_max: Initial value for alpha_bar_max (default: 0.95)
    """
    
    log_scale: float = -1.2  # Initial log_scale value (exp(-1.2) ≈ 0.3, safe: max gamma_prime ~1.34 with alpha_bar 0.05-0.95)
    alpha_bar_min: float = 0.01  # Initial alpha_bar_min value
    alpha_bar_max: float = 0.99  # Initial alpha_bar_max value
    
    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "log_scale": -1.0,
            "alpha_bar_min": 0.01,
            "alpha_bar_max": 0.99,
        }
    
    @nn.compact
    def _get_alpha_bar_gamma_prime(
        self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) and gamma_prime(t) for Cauchy schedule."""
        if params is not None:
            alpha_bar_min = jnp.clip(params['alpha_bar_min'], self.alpha_bar_min, self.alpha_bar_max)
            alpha_bar_max = jnp.clip(params['alpha_bar_max'], alpha_bar_min, self.alpha_bar_max) 
            log_scale = params['log_scale']
        else:
            alpha_bar_min = self.param('alpha_bar_min', nn.initializers.constant(self.alpha_bar_min), ())
            alpha_bar_max = self.param('alpha_bar_max', nn.initializers.constant(self.alpha_bar_max), ())
            log_scale = self.param('log_scale', nn.initializers.constant(self.log_scale), ())
            alpha_bar_min = jnp.clip(alpha_bar_min, self.alpha_bar_min, self.alpha_bar_max)
            alpha_bar_max = jnp.clip(alpha_bar_max, alpha_bar_min, self.alpha_bar_max) 

        scale = jnp.exp(log_scale)
        gamma_max = jax.scipy.special.logit(alpha_bar_max)
        gamma_min = jax.scipy.special.logit(alpha_bar_min)
        loc = 0.5*(gamma_max + gamma_min)
        rad_eps = 0.5*jnp.pi - jnp.atan(0.5*(gamma_max - gamma_min)/scale)
        rad_eps = jnp.clip(rad_eps, 1e-6, jnp.pi/2 - 1e-6)

        rad_arg = (0.5*jnp.pi - rad_eps) * (2.0*t-1.0)
        gamma_t = loc + scale*jnp.tan(rad_arg)
        gamma_prime_t = 2.0*(jnp.pi/2 - rad_eps)*scale / (jnp.cos(rad_arg)**2)
        gamma_prime_t = jnp.clip(gamma_prime_t, 0.0, self.gamma_prime_max)

        alpha_bar_t = jax.nn.sigmoid(gamma_t)

        return alpha_bar_t, gamma_prime_t

    def _get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for Cauchy schedule."""
        return self._get_alpha_bar_gamma_prime(t, params)[0]
    
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
        scale: Initial value for scale (default: 0.3)
        alpha_bar_min: Initial value for alpha_bar_min (default: 0.05)
        alpha_bar_max: Initial value for alpha_bar_max (default: 0.95)
    """
    
    loc: float = 0.5  # Initial loc value
    log_scale: float = -1.0  # Initial scale value (safe: max gamma_prime ~2.11 with alpha_bar 0.05-0.95)
    
    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "loc": 0.5,
            "log_scale": -1.0,
            "eps": 0.001,
        }
    
    @nn.compact
    def _get_alpha_bar_gamma_prime(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) for Laplace schedule."""
        eps = 0.001  # Small epsilon constant, not a learnable parameter
        if params is not None:
            loc = params['loc']
            log_scale = params['log_scale']
        else:
            loc = self.param('loc', nn.initializers.constant(self.loc), ())
            log_scale = self.param('log_scale', nn.initializers.constant(self.log_scale), ())

        scale = jnp.exp(log_scale)
        gamma_t = loc + scale*jnp.sign(0.5-t)*jnp.log(1.0 - jnp.abs(1.0-2.0*t) + eps)

        gamma_prime_t = scale/(1.0 - jnp.abs(1.0-2.0*t) + eps)
        gamma_prime_t = jnp.clip(gamma_prime_t, 0.0, self.gamma_prime_max)

        alpha_bar_t = jax.nn.sigmoid(gamma_t)

        return alpha_bar_t, gamma_prime_t

    def _get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for Laplace schedule."""
        return self._get_alpha_bar_gamma_prime(t, params)[0]


class QuadraticNoiseSchedule(NoiseSchedule):
    """Quadratic noise schedule with learnable parameters.
    
    Uses a quadratic parameterization: alpha_bar(t) = alpha_bar_min + (alpha_bar_max - alpha_bar_min) * t^2
    
    All parameters are learnable:
    - alpha_bar_min: bounded to [0.001, 0.999]
    - alpha_bar_max: alpha_bar_min + delta_fraction * (0.999 - alpha_bar_min) to ensure max > min and max <= 0.999
    
    Args:
        alpha_bar_min: Initial value for alpha_bar_min (default: 0.05)
        alpha_bar_max: Initial value for alpha_bar_max (default: 0.95)
    """
    
    alpha_bar_min: float = 0.01  # Initial alpha_bar_min value (safe: max gamma_prime ~7.50)
    alpha_bar_max: float = 0.99  # Initial alpha_bar_max value (safe: max gamma_prime ~7.50)
    beta: float = 0.5  # Initial scale value (safe: max gamma_prime ~7.50)
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
    def _get_alpha_bar_gamma_prime(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) and gamma_prime(t) for quadratic schedule."""
        if params is not None:
            alpha_bar_min = jnp.clip(params['alpha_bar_min'], self.alpha_bar_min, self.alpha_bar_max)
            alpha_bar_max = jnp.clip(params['alpha_bar_max'], alpha_bar_min, self.alpha_bar_max) 
            beta = params['beta']
        else:
            alpha_bar_min = self.param('alpha_bar_min', nn.initializers.constant(self.alpha_bar_min), ())
            alpha_bar_max = self.param('alpha_bar_max', nn.initializers.constant(self.alpha_bar_max), ())
            beta = self.param('beta', nn.initializers.constant(self.beta), ())
            alpha_bar_min = jnp.clip(alpha_bar_min, self.alpha_bar_min, self.alpha_bar_max)
            alpha_bar_max = jnp.clip(alpha_bar_max, alpha_bar_min, self.alpha_bar_max) 

        delta_alpha = alpha_bar_max - alpha_bar_min

        beta = jnp.clip(beta, -delta_alpha, delta_alpha)

        alpha_bar_t = alpha_bar_min + delta_alpha*t + beta*t*(t-1.0)
        alpha_bar_prime_t = delta_alpha + beta*(2.0*t-1.0)

        gamma_prime_t = alpha_bar_prime_t / (alpha_bar_t * (1.0 - alpha_bar_t))
        gamma_prime_t = jnp.clip(gamma_prime_t, 0.0, self.gamma_prime_max)
        return alpha_bar_t, gamma_prime_t

    @nn.compact
    def _get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar'(t) for quadratic schedule."""
        return self._get_alpha_bar_gamma_prime(t, params)[0]


class PolynomialNoiseSchedule(NoiseSchedule):
    """Polynomial noise schedule with learnable parameters.
    
    Uses a polynomial parameterization: alpha_bar(t) = alpha_bar_min + (alpha_bar_max - alpha_bar_min) * t^power
    
    All parameters are learnable:
    - power: positive polynomial power (enforced via softplus), typically >= 1.0
    - alpha_bar_min: bounded to [0.001, 0.999]
    - alpha_bar_max: alpha_bar_min + delta_fraction * (0.999 - alpha_bar_min) to ensure max > min and max <= 0.999
    
    Args:
        power: Initial value for polynomial power (default: 1.0)
        alpha_bar_min: Initial value for alpha_bar_min (default: 0.05)
        alpha_bar_max: Initial value for alpha_bar_max (default: 0.95)
    """
    
    log_power: float = 0.0  # Initial log_power value (exp(0.0) = 1.0, safe: max gamma_prime ~18.95 with alpha_bar 0.05-0.95)
    alpha_bar_min: float = 0.05  # Initial alpha_bar_min value (safe: max gamma_prime ~18.95)
    alpha_bar_max: float = 0.95  # Initial alpha_bar_max value (safe: max gamma_prime ~18.95)
    
    @staticmethod
    def default_params() -> Dict[str, Any]:
        """Return default parameter dictionary for this schedule.
        
        Returns:
            Dictionary with default initial parameter values
        """
        return {
            "log_power": 0.0,
            "alpha_bar_min": 0.01,
            "alpha_bar_max": 0.99,
        }

    @nn.compact
    def _get_alpha_bar_gamma_prime(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get alpha_bar(t) for polynomial schedule."""
        if params is not None:
            log_power = params['log_power']
            alpha_bar_min = params['alpha_bar_min']
            alpha_bar_max = params['alpha_bar_max']
            power = jnp.exp(log_power)
            alpha_bar_min = jnp.clip(alpha_bar_min, self.alpha_bar_min, self.alpha_bar_max)
            alpha_bar_max = jnp.clip(alpha_bar_max, alpha_bar_min, self.alpha_bar_max)
        else:
            log_power = self.param('log_power', nn.initializers.constant(self.log_power), ())
            alpha_bar_min = self.param('alpha_bar_min', nn.initializers.constant(self.alpha_bar_min), ())
            alpha_bar_max = self.param('alpha_bar_max', nn.initializers.constant(self.alpha_bar_max), ())
            power = jnp.exp(log_power)
            alpha_bar_min = jnp.clip(alpha_bar_min, self.alpha_bar_min, self.alpha_bar_max)
            alpha_bar_max = jnp.clip(alpha_bar_max, alpha_bar_min, self.alpha_bar_max)

        alpha_bar_t = alpha_bar_min + (alpha_bar_max - alpha_bar_min)*t**power
        alpha_bar_prime_t = (alpha_bar_max - alpha_bar_min)*power*t**(power-1)
        gamma_prime_t = alpha_bar_prime_t / (alpha_bar_t * (1.0 - alpha_bar_t))
        gamma_prime_t = jnp.clip(gamma_prime_t, 0.0, self.gamma_prime_max)

        return alpha_bar_t, gamma_prime_t

    def _get_alpha_bar(self, t: jnp.ndarray, params: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
        """Get alpha_bar(t) for polynomial schedule."""
        return self._get_alpha_bar_gamma_prime(t, params)[0]


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
            "gamma_range": (-5.0, 5.0),
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
            g0 = network(jnp.zeros_like(t_input))
            g1 = network(jnp.ones_like(t_input))
            gt = clip((f_t - g0) / (g1 - g0 + 1e-8), 0.0, 1.0)
            gamma_t = gamma_min + (gamma_max - gamma_min) * ( 1 - gt)
            return gamma_t
        
        t = jnp.asarray(t)
        t_flat = t.reshape(-1)
        vals, grads = jax.vmap(jax.value_and_grad(gamma_fn_scalar))(t_flat)
        gamma_t = vals.reshape(t.shape) 
        gamma_prime_t = grads.reshape(t.shape)
        gamma_prime_t = clip(gamma_prime_t, - self.gamma_prime_max, 0.0)
        
        # Compute alpha_bar from gamma
        alpha_bar_t = jax.nn.sigmoid(- gamma_t)
        
        return alpha_bar_t, - gamma_prime_t


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
            - "monotonic_nn" or "learnable": Monotonic neural network schedule
        **kwargs: Additional parameters for the schedule
        
    Returns:
        NoiseSchedule instance
    """
    schedule_type = schedule_type.lower()

    if schedule_type == 'constant':
        return ConstantNoiseSchedule(**kwargs)
    elif schedule_type == "linear":
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
            f"cauchy, laplace, quadratic, polynomial, monotonic_nn/learnable/network"
        )
