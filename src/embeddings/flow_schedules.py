"""
Noise scheduling utilities for Generalized Flow Models

This module provides a base class for building flexible noise schedules for flow 
models. Unlike the previous version, schedules are not limited to variance preserving
diffusion models. Here we consider general affine flows that are characterized by two 
functions on the unit interval:

    α(t) - a monotonically increasing function with α(0) = 0 and α(1) = 1
    σ(t) - a monotonically decreasing function with σ(0) = 1 and σ(1) = 0
    
For affine flow models, the functions α(t) and σ(t) (and their derivatives) determine the 
relationship between various quantities in the flow model. Specifically, the flow itself is 
defined by 

    X_t = α(t) · X_1 + σ(t) · X_0

where we have adopted the convention that X_1 = x_target, and X_0 is noise so that X_t has mean 
α(t) and variance σ²(t). For example, a standard variance preserving diffusion model has

    α(t) = √ᾱ(t)  and  σ²(t) = 1 - ᾱ(t)

The generalized flow model rests on the relationship between the flow, u(t), and the initial target
states is determined by the noise schedule. See https://arxiv.org/pdf/2412.06264 wherein it is 
shown that since the expected velocity field is given by

    u(t) = α̇ · E[X_1|X_t]) + σ̇ · E[X_0|X_t]

we can conclude that 

    u(t) = (α̇ / α) · X_t - (α̇ /α - σ̇ / σ ) · σ · X_0

    u(t) = (σ̇ / σ) · X_t + (α̇ /α - σ̇ / σ ) · α · X_1

This module provides a base class that implements these functional relationships for a user-specified
set of noise schedule functions (α(t), σ(t)).  The model has been implemented using flax.linen modules
so that the functions alpha(t) and sigma(t) may be instantiated with learnable parameters.  

NOTE:  In practice, we can't ever let alpha or sigma actually reach 0.  1 is ok, but 0 is not.  We
       choose to deal with this issue by settign a minimum value for each quantity.  Also, not that 
       for flows we have α(1) = 1 and σ(0) = 1 by definition.   

"""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from dataclasses import dataclass, field, MISSING
from src.configs.base_config import BaseConfig

class Config(BaseConfig):
    alpha_min: float = 1e-3
    sigma_min: float = 1e-3
    alpha_fun = 'linear'
    sigma_fun = 'linear'


@dataclass(frozen=True)
class FlowScheduleConfig(BaseConfig):
    """Configuration for FlowSchedule models."""
    # BaseConfig fields (model_name comes from BaseConfig with default)
    model_name: str = "flow_schedule"
    
    # Required fields (must have defaults to follow BaseConfig pattern)
    schedule_type: str = "linear"  # Type of schedule: "linear", "cosine", "sigmoid", "exponential", "cauchy", "laplace", "polynomial", "network"
    ndims: int = 1  # Number of dimensions in the data shape
    
    # Optional fields
    learnable: bool = False
    
    # Common parameters for most schedules
    alpha_min: float = 0.05
    alpha_max: float = 0.95
    sigma_min: float = 0.05
    sigma_max: float = 0.95
    
    # Schedule-specific parameters (optional, with defaults)
    k: float = 10.0  # For sigmoid schedule (steepness)
    beta: float = 2.0  # For exponential schedule (rate)
    loc: float = 0.5  # For cauchy/laplace schedules (location)
    log_scale: float = -1.0  # For cauchy/laplace schedules (log scale)
    log_power: float = 0.69  # For polynomial schedule (log power, default ~2.0)
    hidden_dims: Tuple[int, ...] = field(default_factory=lambda: (64, 64))  # For network schedule


class FlowSchedule(nn.Module):
    """Base class for flow schedules.
    
    Args:
        alpha_min: Minimum value for alpha
        sigma_min: Minimum value for sigma
        alpha_fun: Function for alpha
        sigma_fun: Function for sigma
    """

    ndims: int  # specifies the number of dimensions of x which has shape batch_shape + x_shape
                            # ndims = len(x_shape)
    learnable: bool = False  # whether the alpha and sigma functions are learnable

    def __call__(self, x_0, x_target, t: jnp.ndarray) -> jnp.ndarray:
        '''Dummy call used to initialize parameters.  Use individual nn.compacct methods.'''
        x_t = self.x_t(x_0, x_target, t)
        return  self.flow_from_target(x_t, x_target, t)

    @nn.compact
    def alpha(self, t): 
        pass

    @nn.compact
    def sigma(self, t):
        pass

    @nn.compact
    def log_alpha_prime(self, t):
        shape = t.shape
        t = t.reshape(-1)
        def log_alpha_single(t):
            return jnp.log(self.alpha(t))
        grads = jax.vmap(jax.grad(log_alpha_single))(t)
        return grads.reshape(shape)

    @nn.compact
    def log_sigma_prime(self, t):
        shape = t.shape
        t = t.reshape(-1)
        def log_sigma_single(t):
            return jnp.log(self.sigma(t))
        grads = jax.vmap(jax.grad(log_sigma_single))(t)
        return grads.reshape(shape)


    @nn.compact
    def snr(self, t):
        return (self.alpha(t)/self.sigma(t))**2

    @nn.compact
    def log_snr_prime(self, t):
        return 2.0*(self.log_alpha_prime(t) - self.log_sigma_prime(t))

    @nn.compact
    def tau_inverse(self, t):
        return 1.0/(self.log_alpha_prime(t) - self.log_sigma_prime(t))

    @nn.compact
    def sigma_prime(self, t):
        return self.log_sigma_prime(t)

    @nn.compact
    def x_t(self, x_0, x_target, t: jnp.ndarray) -> jnp.ndarray:
        a = jnp.expand_dims(self.alpha(t), axis=tuple(range(-self.ndims, 0)))
        b = jnp.expand_dims(self.sigma(t), axis=tuple(range(-self.ndims, 0)))
        x_t = a * x_target + b * x_0
        return x_t

    @nn.compact 
    def flow_from_target(self, x_t, x_target, t: jnp.ndarray) -> jnp.ndarray:
        '''
        Args:
            x_t: state of the flow at time t
            x_target: target state of the flow
            t: time step
        Returns:
            x_0: initial state of the flow
        '''

        a = self.log_sigma_prime(t)
        b = (self.log_alpha_prime(t) - self.log_sigma_prime(t))*self.alpha(t)

        u_t = jnp.expand_dims(a, tuple(range(-self.ndims,0)))*x_t  
        u_t += jnp.expand_dims(b, tuple(range(-self.ndims,0)))*x_target

        return u_t

    @nn.compact
    def target_from_flow(self, x_t, u_t, t: jnp.ndarray) -> jnp.ndarray:
        '''
        Args:
            x_t: state of the flow at time t
            u_t: expected velocity field of the flow at time t
            t: time step
        Returns:
            x_target: target state of the flow
        '''
        
        a = self.log_sigma_prime(t)
        b = (self.log_alpha_prime(t) - self.log_sigma_prime(t))*self.alpha(t)

        target = u_t - jnp.expand_dims(a, tuple(range(-self.ndims,0)))*x_t
        target = target/jnp.expand_dims(b, tuple(range(-self.ndims,0)))

        return target

    @nn.compact 
    def flow_from_noise(self, x_t, x_0, t: jnp.ndarray) -> jnp.ndarray:
        '''
        Args:
            x_t: state of the flow at time t
            x_0: initial 'noise' state of the flow
            t: time step
        Returns:
            x_0: initial state of the flow
        '''
        a = self.log_alpha_prime(t)
        b = (self.log_sigma_prime(t) - self.log_alpha_prime(t))*self.sigma(t)

        u_t = jnp.expand_dims(a, tuple(range(-self.ndims,0)))*x_t  
        u_t += jnp.expand_dims(b, tuple(range(-self.ndims,0)))*x_0

        return u_t

    @nn.compact
    def target_from_noise(self, x_t, x_0, t: jnp.ndarray) -> jnp.ndarray:
        '''
        Args:
            x_t: state of the flow at time t
            x_0: initial 'noise' state of the flow
            t: time step
        Returns:
            x_target: target state of the flow
        '''
        a = self.alpha(t)
        b = -self.sigma(t)/self.alpha(t)

        x_target = jnp.expand_dims(a, tuple(range(-self.ndims,0))) * x_t
        x_target += jnp.expand_dims(b, tuple(range(-self.ndims,0))) * x_0

        return x_target

    @nn.compact
    def score_from_target(self, x_t, x_target, t: jnp.ndarray) -> jnp.ndarray:
        '''
        Args:
            x_t: state of the flow at time t
            x_target: target state of the flow
            t: time step
        Returns:
            score: score of the flow at time t
        '''

        alpha_t = jnp.expand_dims(self.alpha(t), axis=tuple(range(-self.ndims, 0)))
        sigma_t = jnp.expand_dims(self.sigma(t), axis=tuple(range(-self.ndims, 0)))
        return (x_target*alpha_t - x_t) / sigma_t**2


class LinearFlowSchedule(FlowSchedule):
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    sigma_min: float = 0.0
    sigma_max: float = 1.0

    @nn.compact
    def alpha(self, t): 
        if self.learnable:
            # Create learnable parameters in logit space for better optimization
            # Initialize from current alpha_min and alpha_max values
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.alpha_min, 1e-6, 1.0 - 1e-6))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.alpha_max, 1e-6, 1.0 - 1e-6))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            
            # Transform from logit space to [0, 1] and ensure ordering
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)            
        else:
            # Use fixed values
            alpha_min = self.alpha_min
            alpha_max = self.alpha_max
        
        alpha_min = jnp.clip(alpha_min, 1e-6, 1.0 - 1e-6)
        alpha_max = jnp.clip(alpha_max, alpha_min, 1.0 - 1e-6)
        return alpha_min + (alpha_max - alpha_min) * t

    @nn.compact
    def sigma(self, t):
        if self.learnable:
            # Create learnable parameters in logit space for better optimization
            # Initialize from current sigma_min and sigma_max values
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.sigma_min, 1e-6, 1.0 - 1e-6))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.sigma_max, 1e-6, 1.0 - 1e-6))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            # Transform from logit space to [0, 1] and ensure ordering
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
            
        else:
            # Use fixed values
            sigma_min = self.sigma_min
            sigma_max = self.sigma_max
        
        sigma_min = jnp.clip(sigma_min, 1e-6, 1.0 - 1e-6)
        sigma_max = jnp.clip(sigma_max, sigma_min, 1.0 - 1e-6)
        return sigma_max + (sigma_min - sigma_max) * t


class CosineFlowSchedule(FlowSchedule):
    """Cosine flow schedule with learnable parameters.
    
    Uses cosine interpolation: alpha(t) = sin^2(πt/2), sigma(t) = cos^2(πt/2)
    With learnable bounds for alpha and sigma.
    """
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    sigma_min: float = 0.0
    sigma_max: float = 1.0

    @nn.compact
    def alpha(self, t):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.alpha_min, 1e-6, 1.0 - 1e-6))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.alpha_max, 1e-6, 1.0 - 1e-6))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
        else:
            alpha_min = self.alpha_min
            alpha_max = self.alpha_max
        
        alpha_min = jnp.clip(alpha_min, 1e-6, 1.0 - 1e-6)
        alpha_max = jnp.clip(alpha_max, alpha_min, 1.0 - 1e-6)
        
        # Cosine interpolation: sin^2(πt/2) maps [0,1] to [0,1]
        cos_val = jnp.cos(jnp.pi * t / 2.0)
        sin_squared = 1.0 - cos_val ** 2
        return alpha_min + (alpha_max - alpha_min) * sin_squared

    @nn.compact
    def sigma(self, t):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.sigma_min, 1e-6, 1.0 - 1e-6))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.sigma_max, 1e-6, 1.0 - 1e-6))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
        else:
            sigma_min = self.sigma_min
            sigma_max = self.sigma_max
        
        sigma_min = jnp.clip(sigma_min, 1e-6, 1.0 - 1e-6)
        sigma_max = jnp.clip(sigma_max, sigma_min, 1.0 - 1e-6)
        
        # Cosine interpolation: cos^2(πt/2) maps [0,1] to [1,0]
        # We want sigma to go from sigma_max (at t=0) to sigma_min (at t=1)
        # So we use sin^2 which goes from 0 to 1
        cos_val = jnp.cos(jnp.pi * t / 2.0)
        sin_squared = 1.0 - cos_val ** 2
        return sigma_max + (sigma_min - sigma_max) * sin_squared


class SigmoidFlowSchedule(FlowSchedule):
    """Sigmoid flow schedule with learnable parameters.
    
    Uses sigmoid interpolation with learnable steepness and bounds.
    """
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    sigma_min: float = 0.0
    sigma_max: float = 1.0
    k: float = 10.0  # Steepness parameter

    @nn.compact
    def alpha(self, t):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.alpha_min, 1e-6, 1.0 - 1e-6))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.alpha_max, 1e-6, 1.0 - 1e-6))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            k_log = self.param('k_log', nn.initializers.constant(jnp.log(self.k)), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            k = jnp.exp(k_log)  # Ensure positive
        else:
            alpha_min = self.alpha_min
            alpha_max = self.alpha_max
            k = self.k
        
        alpha_min = jnp.clip(alpha_min, 1e-6, 1.0 - 1e-6)
        alpha_max = jnp.clip(alpha_max, alpha_min, 1.0 - 1e-6)
        
        # Sigmoid interpolation: sigmoid(k * (t - 0.5) + 0.5) maps [0,1] to [0,1]
        sigmoid_val = jax.nn.sigmoid(k * (t - 0.5))
        return alpha_min + (alpha_max - alpha_min) * sigmoid_val

    @nn.compact
    def sigma(self, t):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.sigma_min, 1e-6, 1.0 - 1e-6))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.sigma_max, 1e-6, 1.0 - 1e-6))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
        else:
            sigma_min = self.sigma_min
            sigma_max = self.sigma_max
        
        sigma_min = jnp.clip(sigma_min, 1e-6, 1.0 - 1e-6)
        sigma_max = jnp.clip(sigma_max, sigma_min, 1.0 - 1e-6)
        
        # Reverse sigmoid for decreasing function
        if self.learnable:
            k_log = self.param('k_log', nn.initializers.constant(jnp.log(self.k)), ())
            k = jnp.exp(k_log)
        else:
            k = self.k
        sigmoid_val = jax.nn.sigmoid(k * (t - 0.5))
        return sigma_max + (sigma_min - sigma_max) * sigmoid_val


class ExponentialFlowSchedule(FlowSchedule):
    """Exponential flow schedule with learnable parameters.
    
    Uses exponential interpolation with learnable rate and bounds.
    """
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    sigma_min: float = 0.0
    sigma_max: float = 1.0
    beta: float = 2.0  # Exponential rate parameter

    @nn.compact
    def alpha(self, t):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.alpha_min, 1e-6, 1.0 - 1e-6))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.alpha_max, 1e-6, 1.0 - 1e-6))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            
            # Compute beta from actual parameter values (alpha_min_logit and alpha_max_logit)
            # Transform logits to actual values to compute beta
            alpha_min_param = jax.nn.sigmoid(alpha_min_logit)
            alpha_max_param = jax.nn.sigmoid(alpha_max_logit)
            # Ensure ordering and compute beta: alpha_max = alpha_min * exp(beta), so beta = log(alpha_max / alpha_min)
            alpha_min_param = jnp.clip(alpha_min_param, 1e-6, 1.0 - 1e-6)
            alpha_max_param = jnp.clip(alpha_max_param, alpha_min_param, 1.0 - 1e-6)
            beta_init = jnp.log(alpha_max_param / alpha_min_param + 1e-8)
            beta_log = self.param('beta_log', nn.initializers.constant(beta_init), ())
            
            alpha_min = alpha_min_param
            alpha_max = alpha_max_param
            beta = beta_log  # beta is already in log space, use directly
        else:
            alpha_min = self.alpha_min
            alpha_max = self.alpha_max
            beta = self.beta
        
        alpha_min = jnp.clip(alpha_min, 1e-6, 1.0 - 1e-6)
        alpha_max = jnp.clip(alpha_max, alpha_min, 1.0 - 1e-6)
        
        # Pure exponential growth: alpha(t) = alpha_min * exp(beta * t)
        # For fixed beta, compute it from boundaries to ensure alpha(1) = alpha_max
        # For learnable beta, use the learnable value (boundaries may not be exactly met)
        if not self.learnable:
            # Compute beta from boundaries: alpha_max = alpha_min * exp(beta)
            # So: beta = log(alpha_max / alpha_min)
            beta = jnp.log(alpha_max / alpha_min + 1e-8)
        
        # Pure exponential growth
        alpha_val = alpha_min * jnp.exp(beta * t)
        # Clip to reasonable bounds but allow growth beyond initial offset values
        return jnp.clip(alpha_val, 1e-6, 1.0 - 1e-6)

    @nn.compact
    def sigma(self, t):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.sigma_min, 1e-6, 1.0 - 1e-6))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.sigma_max, 1e-6, 1.0 - 1e-6))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            # Compute beta from actual parameter values (sigma_min_logit and sigma_max_logit)
            # Transform logits to actual values to compute beta
            sigma_min_param = jax.nn.sigmoid(sigma_min_logit)
            sigma_max_param = jax.nn.sigmoid(sigma_max_logit)
            # Ensure ordering and compute beta: sigma_min = sigma_max * exp(-beta), so beta = -log(sigma_min / sigma_max)
            sigma_min_param = jnp.clip(sigma_min_param, 1e-6, 1.0 - 1e-6)
            sigma_max_param = jnp.clip(sigma_max_param, sigma_min_param, 1.0 - 1e-6)
            beta_init = -jnp.log(sigma_min_param / sigma_max_param + 1e-8)
            beta_log = self.param('beta_log', nn.initializers.constant(beta_init), ())
            
            sigma_min = sigma_min_param
            sigma_max = sigma_max_param
            beta = beta_log  # beta is already in log space, use directly
        else:
            sigma_min = self.sigma_min
            sigma_max = self.sigma_max
        
        sigma_min = jnp.clip(sigma_min, 1e-6, 1.0 - 1e-6)
        sigma_max = jnp.clip(sigma_max, sigma_min, 1.0 - 1e-6)
        
        # Pure exponential decay: sigma(t) = sigma_max * exp(-beta * t)
        # For fixed beta, compute it from boundaries to ensure sigma(1) = sigma_min
        if not self.learnable:
            # Compute beta from boundaries: sigma_min = sigma_max * exp(-beta)
            # So: beta = -log(sigma_min / sigma_max)
            beta = -jnp.log(sigma_min / sigma_max + 1e-8)
        else:
            # Compute beta from boundaries: sigma_min = sigma_max * exp(-beta)
            # So: beta = -log(sigma_min / sigma_max)
            beta = -jnp.log(sigma_min / sigma_max + 1e-8)
        
        # Pure exponential decay
        sigma_val = sigma_max * jnp.exp(-beta * t)
        # Clip to reasonable bounds but allow decay beyond initial offset values
        return jnp.clip(sigma_val, 1e-6, 1.0 - 1e-6)


class CauchyFlowSchedule(FlowSchedule):
    """Cauchy distribution-based flow schedule with learnable parameters.
    
    Uses Cauchy CDF for interpolation with learnable location, scale, and bounds.
    """
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    sigma_min: float = 0.0
    sigma_max: float = 1.0
    loc: float = 0.5  # Location parameter
    log_scale: float = -1.0  # Log scale parameter (scale = exp(log_scale))

    @nn.compact
    def alpha(self, t):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.alpha_min, 1e-6, 1.0 - 1e-6))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.alpha_max, 1e-6, 1.0 - 1e-6))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            loc = self.param('loc', nn.initializers.constant(self.loc), ())
            log_scale = self.param('log_scale', nn.initializers.constant(self.log_scale), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            scale = jnp.exp(log_scale)  # Ensure positive
        else:
            alpha_min = self.alpha_min
            alpha_max = self.alpha_max
            loc = self.loc
            scale = jnp.exp(self.log_scale)
        
        alpha_min = jnp.clip(alpha_min, 1e-6, 1.0 - 1e-6)
        alpha_max = jnp.clip(alpha_max, alpha_min, 1.0 - 1e-6)
        
        # Cauchy CDF: maps t in [0,1] to [0,1] using arctan
        # Normalize to ensure proper boundary conditions: CDF(0) -> 0, CDF(1) -> 1
        z = (t - loc) / scale
        z_0 = (0.0 - loc) / scale
        z_1 = (1.0 - loc) / scale
        
        cauchy_cdf_t = 0.5 + (1.0 / jnp.pi) * jnp.arctan(z)
        cauchy_cdf_0 = 0.5 + (1.0 / jnp.pi) * jnp.arctan(z_0)
        cauchy_cdf_1 = 0.5 + (1.0 / jnp.pi) * jnp.arctan(z_1)
        
        # Normalize so that at t=0 we get 0 and at t=1 we get 1
        normalized = (cauchy_cdf_t - cauchy_cdf_0) / (cauchy_cdf_1 - cauchy_cdf_0 + 1e-8)
        normalized = jnp.clip(normalized, 0.0, 1.0)
        return alpha_min + (alpha_max - alpha_min) * normalized

    @nn.compact
    def sigma(self, t):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.sigma_min, 1e-6, 1.0 - 1e-6))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.sigma_max, 1e-6, 1.0 - 1e-6))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
        else:
            sigma_min = self.sigma_min
            sigma_max = self.sigma_max
        
        sigma_min = jnp.clip(sigma_min, 1e-6, 1.0 - 1e-6)
        sigma_max = jnp.clip(sigma_max, sigma_min, 1.0 - 1e-6)
        
        # Reverse Cauchy CDF for decreasing function
        # Normalize to ensure proper boundary conditions: at t=0 -> sigma_max, at t=1 -> sigma_min
        if self.learnable:
            loc = self.param('loc', nn.initializers.constant(self.loc), ())
            log_scale = self.param('log_scale', nn.initializers.constant(self.log_scale), ())
        else:
            loc = self.loc
            log_scale = self.log_scale
        scale = jnp.exp(log_scale)
        
        z = (t - loc) / scale
        z_0 = (0.0 - loc) / scale
        z_1 = (1.0 - loc) / scale
        
        cauchy_cdf_t = 0.5 + (1.0 / jnp.pi) * jnp.arctan(z)
        cauchy_cdf_0 = 0.5 + (1.0 / jnp.pi) * jnp.arctan(z_0)
        cauchy_cdf_1 = 0.5 + (1.0 / jnp.pi) * jnp.arctan(z_1)
        
        # Normalize so that at t=0 we get 0 and at t=1 we get 1
        normalized = (cauchy_cdf_t - cauchy_cdf_0) / (cauchy_cdf_1 - cauchy_cdf_0 + 1e-8)
        normalized = jnp.clip(normalized, 0.0, 1.0)
        # For sigma (decreasing), we want: sigma(0) = sigma_max, sigma(1) = sigma_min
        return sigma_max + (sigma_min - sigma_max) * normalized


class LaplaceFlowSchedule(FlowSchedule):
    """Laplace distribution-based flow schedule with learnable parameters.
    
    Uses Laplace CDF for interpolation with learnable location, scale, and bounds.
    """
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    sigma_min: float = 0.0
    sigma_max: float = 1.0
    loc: float = 0.5  # Location parameter
    log_scale: float = -1.0  # Log scale parameter

    @nn.compact
    def alpha(self, t):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.alpha_min, 1e-6, 1.0 - 1e-6))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.alpha_max, 1e-6, 1.0 - 1e-6))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            loc = self.param('loc', nn.initializers.constant(self.loc), ())
            log_scale = self.param('log_scale', nn.initializers.constant(self.log_scale), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            scale = jnp.exp(log_scale)  # Ensure positive
        else:
            alpha_min = self.alpha_min
            alpha_max = self.alpha_max
            loc = self.loc
            scale = jnp.exp(self.log_scale)
        
        alpha_min = jnp.clip(alpha_min, 1e-6, 1.0 - 1e-6)
        alpha_max = jnp.clip(alpha_max, alpha_min, 1.0 - 1e-6)
        
        # Laplace CDF: 0.5 * (1 + sign(t - loc) * (1 - exp(-|t - loc| / scale)))
        # Normalize to ensure proper boundary conditions: CDF(0) -> 0, CDF(1) -> 1
        z = (t - loc) / scale
        z_0 = (0.0 - loc) / scale
        z_1 = (1.0 - loc) / scale
        
        laplace_cdf_t = 0.5 * (1.0 + jnp.sign(z) * (1.0 - jnp.exp(-jnp.abs(z))))
        laplace_cdf_0 = 0.5 * (1.0 + jnp.sign(z_0) * (1.0 - jnp.exp(-jnp.abs(z_0))))
        laplace_cdf_1 = 0.5 * (1.0 + jnp.sign(z_1) * (1.0 - jnp.exp(-jnp.abs(z_1))))
        
        # Normalize so that at t=0 we get 0 and at t=1 we get 1
        normalized = (laplace_cdf_t - laplace_cdf_0) / (laplace_cdf_1 - laplace_cdf_0 + 1e-8)
        normalized = jnp.clip(normalized, 0.0, 1.0)
        return alpha_min + (alpha_max - alpha_min) * normalized

    @nn.compact
    def sigma(self, t):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.sigma_min, 1e-6, 1.0 - 1e-6))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.sigma_max, 1e-6, 1.0 - 1e-6))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
        else:
            sigma_min = self.sigma_min
            sigma_max = self.sigma_max
        
        sigma_min = jnp.clip(sigma_min, 1e-6, 1.0 - 1e-6)
        sigma_max = jnp.clip(sigma_max, sigma_min, 1.0 - 1e-6)
        
        # Reverse Laplace CDF for decreasing function
        # Normalize to ensure proper boundary conditions: at t=0 -> sigma_max, at t=1 -> sigma_min
        if self.learnable:
            loc = self.param('loc', nn.initializers.constant(self.loc), ())
            log_scale = self.param('log_scale', nn.initializers.constant(self.log_scale), ())
        else:
            loc = self.loc
            log_scale = self.log_scale
        scale = jnp.exp(log_scale)
        
        z = (t - loc) / scale
        z_0 = (0.0 - loc) / scale
        z_1 = (1.0 - loc) / scale
        
        laplace_cdf_t = 0.5 * (1.0 + jnp.sign(z) * (1.0 - jnp.exp(-jnp.abs(z))))
        laplace_cdf_0 = 0.5 * (1.0 + jnp.sign(z_0) * (1.0 - jnp.exp(-jnp.abs(z_0))))
        laplace_cdf_1 = 0.5 * (1.0 + jnp.sign(z_1) * (1.0 - jnp.exp(-jnp.abs(z_1))))
        
        # Normalize so that at t=0 we get 0 and at t=1 we get 1
        normalized = (laplace_cdf_t - laplace_cdf_0) / (laplace_cdf_1 - laplace_cdf_0 + 1e-8)
        normalized = jnp.clip(normalized, 0.0, 1.0)
        # For sigma (decreasing), we want: sigma(0) = sigma_max, sigma(1) = sigma_min
        return sigma_max + (sigma_min - sigma_max) * normalized


class PolynomialFlowSchedule(FlowSchedule):
    """Polynomial flow schedule with learnable parameters.
    
    Uses polynomial interpolation: t^power with learnable power and bounds.
    """
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    sigma_min: float = 0.0
    sigma_max: float = 1.0
    log_power: float = 2  # Log power parameter (power = exp(log_power), typically >= 1.0)

    @nn.compact
    def alpha(self, t):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.alpha_min, 1e-6, 1.0 - 1e-6))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.alpha_max, 1e-6, 1.0 - 1e-6))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            log_power = self.param('log_power', nn.initializers.constant(self.log_power), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            power = jnp.exp(log_power)  # Ensure positive, typically >= 1.0
            power = jnp.clip(power, 0.1, 10.0)  # Reasonable bounds
        else:
            alpha_min = self.alpha_min
            alpha_max = self.alpha_max
            power = jnp.exp(self.log_power)
        
        alpha_min = jnp.clip(alpha_min, 1e-6, 1.0 - 1e-6)
        alpha_max = jnp.clip(alpha_max, alpha_min, 1.0 - 1e-6)
        
        # Polynomial interpolation: t^power maps [0,1] to [0,1]
        poly_val = t ** power
        return alpha_min + (alpha_max - alpha_min) * poly_val

    @nn.compact
    def sigma(self, t):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(0.05 + 0.95*self.sigma_min, 1e-6, 1.0 - 1e-6))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(0.95*self.sigma_max, 1e-6, 1.0 - 1e-6))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
        else:
            sigma_min = self.sigma_min
            sigma_max = self.sigma_max
        
        sigma_min = jnp.clip(sigma_min, 1e-6, 1.0 - 1e-6)
        sigma_max = jnp.clip(sigma_max, sigma_min, 1.0 - 1e-6)
        
        # Polynomial for decreasing function: use t^power which goes from 0 to 1
        # We want sigma to go from sigma_max (at t=0) to sigma_min (at t=1)
        log_power = self.log_power if not self.learnable else self.param('log_power', nn.initializers.constant(self.log_power), ())
        power = jnp.exp(log_power)
        power = jnp.clip(power, 0.1, 10.0)
        poly_val = t ** power  # Goes from 0 to 1 as t goes from 0 to 1
        return sigma_max + (sigma_min - sigma_max) * poly_val




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


class FlowScheduleNetwork(FlowSchedule):
    """Neural network-based flow schedule.
    
    Uses two learnable monotonic neural networks:
    - One for alpha (increasing from ~0 to ~1)
    - One for 1-sigma (increasing, so sigma decreases from ~1 to ~0)
    
    The networks are normalized to ensure proper boundary conditions.
    
    Args:
        hidden_dims: Hidden dimensions for the neural networks (default: (64, 64))
        monotonic_network: Network class to use (default: SimpleMonotonicNetwork)
        alpha_min: Minimum value for alpha (default: 0.0)
        alpha_max: Maximum value for alpha (default: 1.0)
        sigma_min: Minimum value for sigma (default: 0.0)
        sigma_max: Maximum value for sigma (default: 1.0)
    """
    
    hidden_dims: Tuple[int, ...] = (64, 64)
    monotonic_network: nn.Module = SimpleMonotonicNetwork
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    sigma_min: float = 0.0
    sigma_max: float = 1.0

    @nn.compact
    def alpha(self, t):
        """Compute alpha(t) using a monotonic neural network."""
        # Create the monotonic network for alpha
        alpha_network = self.monotonic_network(hidden_dims=self.hidden_dims)
        
        def alpha_fn_scalar(t_input):
            """Scalar function for alpha that ensures boundary conditions."""
            # Network output (monotonic increasing)
            f_t = alpha_network(t_input)
            # Get values at boundaries for normalization
            f_0 = alpha_network(jnp.zeros_like(t_input))
            f_1 = alpha_network(jnp.ones_like(t_input))
            # Normalize to [0, 1] range
            normalized = (f_t - f_0) / (f_1 - f_0 + 1e-8)
            normalized = jnp.clip(normalized, 0.0, 1.0)
            # Scale to [alpha_min, alpha_max]
            alpha_val = self.alpha_min + (self.alpha_max - self.alpha_min) * normalized
            return alpha_val
        
        # Vectorize over t
        t = jnp.asarray(t)
        t_flat = t.reshape(-1)
        alpha_vals = jax.vmap(alpha_fn_scalar)(t_flat)
        alpha_vals = alpha_vals.reshape(t.shape)
        
        # Ensure values are in valid range
        alpha_vals = jnp.clip(alpha_vals, 1e-6, 1.0 - 1e-6)
        return alpha_vals

    @nn.compact
    def sigma(self, t):
        """Compute sigma(t) using a monotonic neural network for 1-sigma."""
        # Create the monotonic network for 1-sigma (so sigma is decreasing)
        one_minus_sigma_network = self.monotonic_network(hidden_dims=self.hidden_dims)
        
        def sigma_fn_scalar(t_input):
            """Scalar function for sigma via 1-sigma network."""
            # Network output for 1-sigma (monotonic increasing)
            f_t = one_minus_sigma_network(t_input)
            # Get values at boundaries for normalization
            f_0 = one_minus_sigma_network(jnp.zeros_like(t_input))
            f_1 = one_minus_sigma_network(jnp.ones_like(t_input))
            # Normalize to [0, 1] range
            normalized = (f_t - f_0) / (f_1 - f_0 + 1e-8)
            normalized = jnp.clip(normalized, 0.0, 1.0)
            # The normalized value represents how much 1-sigma has increased
            # At t=0: normalized=0, we want sigma=sigma_max
            # At t=1: normalized=1, we want sigma=sigma_min
            # So: sigma = sigma_max + (sigma_min - sigma_max) * normalized
            sigma_val = self.sigma_max + (self.sigma_min - self.sigma_max) * normalized
            return sigma_val
        
        # Vectorize over t
        t = jnp.asarray(t)
        t_flat = t.reshape(-1)
        sigma_vals = jax.vmap(sigma_fn_scalar)(t_flat)
        sigma_vals = sigma_vals.reshape(t.shape)
        
        # Ensure values are in valid range
        sigma_vals = jnp.clip(sigma_vals, 1e-6, 1.0 - 1e-6)
        return sigma_vals

# Alias for backward compatibility
LearnableFlowSchedule = FlowScheduleNetwork


########  FACTORY FUNCTION   ###########

def create_flow_schedule(
    config: Union[FlowScheduleConfig, Dict[str, Any]],
    **kwargs
) -> FlowSchedule:
    """
    Factory function to create a FlowSchedule instance from a config.
    
    Args:
        config: Either a FlowScheduleConfig instance or a dictionary with configuration values.
                If a dict, it should contain at minimum:
                - schedule_type: str (one of: "linear", "cosine", "sigmoid", "exponential", 
                                     "cauchy", "laplace", "polynomial", "network")
                - ndims: int
                And optionally:
                - learnable: bool (default: False)
                - alpha_min, alpha_max, sigma_min, sigma_max: float (defaults: 0.05, 0.95, 0.05, 0.95)
                - Schedule-specific parameters (k, beta, loc, log_scale, log_power, hidden_dims)
        **kwargs: Additional keyword arguments that override config values
        
    Returns:
        FlowSchedule instance
    """
    # Handle config as dict or FlowScheduleConfig instance
    if isinstance(config, dict):
        schedule_type = config.get("schedule_type")
        ndims = config.get("ndims")
        
        if schedule_type is None or ndims is None:
            raise ValueError("config must contain 'schedule_type' and 'ndims'")
        
        learnable = config.get("learnable", False)
        alpha_min = config.get("alpha_min", 0.05)
        alpha_max = config.get("alpha_max", 0.95)
        sigma_min = config.get("sigma_min", 0.05)
        sigma_max = config.get("sigma_max", 0.95)
        k = config.get("k", 10.0)
        beta = config.get("beta", 2.0)
        loc = config.get("loc", 0.5)
        log_scale = config.get("log_scale", -1.0)
        log_power = config.get("log_power", 0.69)
        hidden_dims = config.get("hidden_dims", (64, 64))
    elif isinstance(config, FlowScheduleConfig):
        schedule_type = config.schedule_type
        ndims = config.ndims
        learnable = config.learnable
        alpha_min = config.alpha_min
        alpha_max = config.alpha_max
        sigma_min = config.sigma_min
        sigma_max = config.sigma_max
        k = config.k
        beta = config.beta
        loc = config.loc
        log_scale = config.log_scale
        log_power = config.log_power
        hidden_dims = config.hidden_dims
    else:
        raise TypeError(f"config must be a dict or FlowScheduleConfig, got {type(config)}")
    
    # Override with kwargs if provided
    schedule_type = kwargs.get("schedule_type", schedule_type)
    ndims = kwargs.get("ndims", ndims)
    learnable = kwargs.get("learnable", learnable)
    alpha_min = kwargs.get("alpha_min", alpha_min)
    alpha_max = kwargs.get("alpha_max", alpha_max)
    sigma_min = kwargs.get("sigma_min", sigma_min)
    sigma_max = kwargs.get("sigma_max", sigma_max)
    k = kwargs.get("k", k)
    beta = kwargs.get("beta", beta)
    loc = kwargs.get("loc", loc)
    log_scale = kwargs.get("log_scale", log_scale)
    log_power = kwargs.get("log_power", log_power)
    hidden_dims = kwargs.get("hidden_dims", hidden_dims)
    
    # Create the appropriate schedule based on schedule_type
    schedule_type = schedule_type.lower()
    
    if schedule_type == "linear":
        return LinearFlowSchedule(
            ndims=ndims,
            learnable=learnable,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
    elif schedule_type == "cosine":
        return CosineFlowSchedule(
            ndims=ndims,
            learnable=learnable,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )
    elif schedule_type == "sigmoid":
        return SigmoidFlowSchedule(
            ndims=ndims,
            learnable=learnable,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            k=k
        )
    elif schedule_type == "exponential":
        return ExponentialFlowSchedule(
            ndims=ndims,
            learnable=learnable,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            beta=beta
        )
    elif schedule_type == "cauchy":
        return CauchyFlowSchedule(
            ndims=ndims,
            learnable=learnable,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            loc=loc,
            log_scale=log_scale
        )
    elif schedule_type == "laplace":
        return LaplaceFlowSchedule(
            ndims=ndims,
            learnable=learnable,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            loc=loc,
            log_scale=log_scale
        )
    elif schedule_type == "polynomial":
        return PolynomialFlowSchedule(
            ndims=ndims,
            learnable=learnable,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            log_power=log_power
        )
    elif schedule_type in ["network", "neural", "learnable"]:
        return FlowScheduleNetwork(
            ndims=ndims,
            learnable=learnable,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            hidden_dims=hidden_dims
        )
    else:
        raise ValueError(
            f"Unknown schedule_type: {schedule_type}. "
            f"Options: linear, cosine, sigmoid, exponential, cauchy, laplace, polynomial, network/neural/learnable"
        )

