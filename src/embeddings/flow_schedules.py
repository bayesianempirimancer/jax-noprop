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

or for Optimal Transport flows we have

    α(t) = t  and  σ(t) = 1 - t

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

@dataclass(frozen=True)
class FlowScheduleConfig(BaseConfig):
    """Configuration for FlowSchedule models."""
    # BaseConfig fields (model_name comes from BaseConfig with default)
    model_name: str = "flow_schedule"
    
    # Required fields (must have defaults to follow BaseConfig pattern)
    schedule_type: str = "linear"  # Type of schedule: "linear", "cosine", "sigmoid", "exponential", "cauchy", "laplace", "polynomial", "network"
    latent_shape: Tuple[int] = ()  # Number of dimensions in the data shape
    
    # Optional fields
    learnable: bool = False
    
    # Common parameters for most schedules
    alpha_min: float = 0.01
    alpha_max: float = 1.0
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    
    # Schedule-specific parameters (optional, with defaults)
    k: float = 10.0  # For sigmoid schedule (steepness)
    beta: float = 2.0  # For exponential schedule (rate)
    softplus_beta: float = 50.0  # For softplus schedule (smoothness)
    loc: float = 0.5  # For cauchy/laplace schedules (location)
    log_scale: float = -1.0  # For cauchy/laplace schedules (log scale)
    log_power: float = 0.69  # For polynomial schedule (log power, default ~0.69)
    hidden_dims: Tuple[int, ...] = field(default_factory=lambda: (64, 64))  # For network schedule
    eps: float = 1e-6  # Epsilon for numerical stability


class FlowSchedule(nn.Module):
    """Base class for flow schedules.
    
    Args:
        alpha_min: Minimum value for alpha
        sigma_min: Minimum value for sigma
        alpha_fun: Function for alpha
        sigma_fun: Function for sigma
    """

    config: FlowScheduleConfig

    def __call__(self, x_0, x_target, t: jnp.ndarray) -> jnp.ndarray:
        '''Dummy call used to initialize parameters.  Use individual nn.compacct methods.'''
        x_t = self.x_t(x_0, x_target, t)
        return  self.flow_from_target(x_t, x_target, t)

    @property 
    def learnable(self):
        return self.config.learnable

    @property 
    def ndims(self):
        return len(self.config.latent_shape)

    @nn.compact
    def alpha(self, t): 
        raise NotImplementedError("alpha is not implemented")

    @nn.compact
    def sigma(self, t):
        raise NotImplementedError("sigma is not implemented")

    @nn.compact
    def log_alpha_prime(self, t):
        shape = t.shape
        t = t.reshape(-1)
        def log_alpha_single(t):
            return jnp.log(self.alpha(t) + self.config.eps)
        grads = jax.vmap(jax.grad(log_alpha_single))(t)
        return grads.reshape(shape)

    @nn.compact
    def log_sigma_prime(self, t):
        shape = t.shape
        t = t.reshape(-1)
        def log_sigma_single(t):
            return jnp.log(self.sigma(t) + self.config.eps)
        grads = jax.vmap(jax.grad(log_sigma_single))(t)
        return grads.reshape(shape)

    @nn.compact
    def alpha_prime(self, t):
        return self.log_alpha_prime(t)*self.alpha(t)

    @nn.compact
    def sigma_prime(self, t):
        return self.log_sigma_prime(t)*self.sigma(t)

    @nn.compact
    def snr(self, t):
        return (self.alpha(t)**2) / (self.sigma(t)**2 + self.config.eps)

    @nn.compact
    def log_snr_prime(self, t):
        return 2.0*(self.log_alpha_prime(t) - self.log_sigma_prime(t))

    @nn.compact
    def snr_prime(self, t):
        return self.log_snr_prime(t)*self.snr(t)

    @nn.compact
    def tau_inverse(self, t):
        return self.log_alpha_prime(t) - self.log_sigma_prime(t)

    @nn.compact
    def tau(self, t):
        return 1.0/(self.tau_inverse(t) + self.config.eps)

    @nn.compact
    def target_snr_weight(self, t):
        return self.tau_inverse(t)*self.snr(t)

    @nn.compact
    def noise_snr_weight(self, t):
        return self.tau_inverse(t)  # target_snr_weight(t)/self.snr(t)

    @nn.compact
    def flow_snr_weight(self, t):
        return 1.0/(self.tau_inverse(t)*(self.sigma(t)**2 + self.config.eps) + self.config.eps)

    @nn.compact
    def sample_x_0(self, x_target: jnp.ndarray, key: jr.PRNGKey) -> jnp.ndarray:
        """Sample initial latent state from unit normal distribution.  This is meant to 
        be replaced by more sophisticated sampling methods later."""
        return jr.normal(key, x_target.shape)

    @nn.compact
    def x_t(self, x_0, x_target, t: jnp.ndarray) -> jnp.ndarray:
        a = jnp.expand_dims(self.alpha(t), axis=tuple(range(-self.ndims, 0)))
        b = jnp.expand_dims(self.sigma(t), axis=tuple(range(-self.ndims, 0)))
        x_t = a * x_target + b * x_0
        return x_t

    @nn.compact
    def flow_from_endpoints(self, x_0, x_target, t: jnp.ndarray) -> jnp.ndarray:
        a = jnp.expand_dims(self.alpha_prime(t), axis=tuple(range(-self.ndims, 0)))
        b = jnp.expand_dims(self.sigma_prime(t), axis=tuple(range(-self.ndims, 0)))
        return a*x_target + b*x_0

    @nn.compact 
    def flow_from_target(self, x_t, x_target, t: jnp.ndarray) -> jnp.ndarray:

        a = self.log_sigma_prime(t)
        b = (self.log_alpha_prime(t) - self.log_sigma_prime(t))*self.alpha(t)

        u_t = jnp.expand_dims(a, tuple(range(-self.ndims,0)))*x_t  
        u_t += jnp.expand_dims(b, tuple(range(-self.ndims,0)))*x_target

        return u_t

    @nn.compact
    def target_from_flow(self, x_t, u_t, t: jnp.ndarray) -> jnp.ndarray:
        
        a = self.log_sigma_prime(t)
        b = (self.log_alpha_prime(t) - self.log_sigma_prime(t))*self.alpha(t)

        target = u_t - jnp.expand_dims(a, tuple(range(-self.ndims,0)))*x_t
        target = target/(jnp.expand_dims(b, tuple(range(-self.ndims,0))) + self.config.eps)

        return target

    @nn.compact 
    def flow_from_noise(self, x_t, x_0, t: jnp.ndarray) -> jnp.ndarray:

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
        b = self.sigma(t)

        x_target = x_t - jnp.expand_dims(b, tuple(range(-self.ndims,0))) * x_0
        x_target = x_target/(jnp.expand_dims(a, tuple(range(-self.ndims,0))) + self.config.eps)

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
        return (x_target*alpha_t - x_t) / (sigma_t**2 + self.config.eps)


class LinearFlowSchedule(FlowSchedule):
    @nn.compact
    def get_alpha_params(self):
        if self.learnable:
            # Create learnable parameters in logit space for better optimization
            # Initialize from current alpha_min and alpha_max values
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_min, self.config.eps, 1.0 - self.config.eps))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_max, self.config.eps, 1.0 - self.config.eps))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            
            # Transform from logit space to [0, 1] and ensure ordering
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            alpha_max = jnp.maximum(alpha_max, alpha_min + self.config.eps)            
        else:
            # Use fixed values
            alpha_min = self.config.alpha_min
            alpha_max = self.config.alpha_max
        
        return alpha_min, alpha_max

    @nn.compact
    def alpha(self, t): 
        alpha_min, alpha_max = self.get_alpha_params()
        return alpha_min + (alpha_max - alpha_min) * t

    @nn.compact 
    def alpha_prime(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        return (alpha_max - alpha_min)

    @nn.compact
    def log_alpha_prime(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        return (alpha_max - alpha_min)/(alpha_min + (alpha_max - alpha_min) * t + self.config.eps)

    @nn.compact
    def get_sigma_params(self):
        if self.learnable:
            # Create learnable parameters in logit space for better optimization
            # Initialize from current sigma_min and sigma_max values
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_min, self.config.eps, 1.0 - self.config.eps))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_max, self.config.eps, 1.0 - self.config.eps))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            # Transform from logit space to [0, 1] and ensure ordering
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
            sigma_max = jnp.maximum(sigma_max, sigma_min + self.config.eps)
            
        else:
            # Use fixed values
            sigma_min = self.config.sigma_min
            sigma_max = self.config.sigma_max
        
        return sigma_min, sigma_max

    @nn.compact
    def sigma(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        return sigma_max + (sigma_min - sigma_max) * t

    @nn.compact
    def sigma_prime(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        return sigma_min - sigma_max

    @nn.compact
    def log_sigma_prime(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        return (sigma_min - sigma_max)/(sigma_max + (sigma_min - sigma_max) * t + self.config.eps)


class CosineFlowSchedule(FlowSchedule):
    """Cosine flow schedule with learnable parameters.
    
    Uses cosine interpolation: alpha(t) = sin^2(πt/2), sigma(t) = cos^2(πt/2)
    With learnable bounds for alpha and sigma.
    """

    @nn.compact
    def get_alpha_params(self):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_min, 1e-6, 1.0 - 1e-6))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_max, 1e-6, 1.0 - 1e-6))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            alpha_max = jnp.maximum(alpha_max, alpha_min + self.config.eps)
        else:
            alpha_min = self.config.alpha_min
            alpha_max = self.config.alpha_max
        
        return alpha_min, alpha_max

    @nn.compact
    def alpha(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        # Cosine interpolation: sin^2(πt/2) maps [0,1] to [0,1]
        sin_squared = jnp.sin(jnp.pi * t / 2.0) ** 2
        return alpha_min + (alpha_max - alpha_min) * sin_squared

    @nn.compact
    def alpha_prime(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        # alpha' = (alpha_max - alpha_min) * pi/2 * sin(pi*t)
        # Since alpha(t) = alpha_min + (alpha_max - alpha_min) * sin^2(pi*t/2)
        # d/dt sin^2(pi*t/2) = 2*sin(pi*t/2)*cos(pi*t/2) * pi/2 = sin(pi*t) * pi/2
        return (alpha_max - alpha_min) * (jnp.pi / 2.0) * jnp.sin(jnp.pi * t)

    @nn.compact
    def log_alpha_prime(self, t):
        # Override to use explicit alpha_prime
        return self.alpha_prime(t) / (self.alpha(t) + self.config.eps)

    @nn.compact
    def get_sigma_params(self):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_min, 1e-6, 1.0 - 1e-6))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_max, 1e-6, 1.0 - 1e-6))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
            sigma_max = jnp.maximum(sigma_max, sigma_min + self.config.eps)
        else:
            sigma_min = self.config.sigma_min
            sigma_max = self.config.sigma_max
        
        return sigma_min, sigma_max

    @nn.compact
    def sigma(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        # Cosine interpolation: cos^2(πt/2) maps [0,1] to [1,0]
        # We want sigma to go from sigma_max (at t=0) to sigma_min (at t=1)
        # So we use sin^2 which goes from 0 to 1
        sin_squared = jnp.sin(jnp.pi * t / 2.0) ** 2
        return sigma_max + (sigma_min - sigma_max) * sin_squared

    @nn.compact
    def sigma_prime(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        # sigma' = (sigma_min - sigma_max) * pi/2 * sin(pi*t)
        # Since sigma(t) = sigma_max + (sigma_min - sigma_max) * sin^2(pi*t/2)
        return (sigma_min - sigma_max) * (jnp.pi / 2.0) * jnp.sin(jnp.pi * t)

    @nn.compact
    def log_sigma_prime(self, t):
        # Override to use explicit sigma_prime
        return self.sigma_prime(t) / (self.sigma(t) + self.config.eps)


class SigmoidFlowSchedule(FlowSchedule):
    """Sigmoid flow schedule with learnable parameters.
    
    Uses sigmoid interpolation with learnable steepness and bounds.
    """

    @nn.compact
    def get_alpha_params(self):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_min, 1e-6, 1.0 - 1e-6))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_max, 1e-6, 1.0 - 1e-6))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            alpha_max = jnp.maximum(alpha_max, alpha_min + self.config.eps)
        else:
            alpha_min = self.config.alpha_min
            alpha_max = self.config.alpha_max
        
        return alpha_min, alpha_max

    @nn.compact
    def get_k_param(self):
        if self.learnable:
            k_log = self.param('k_log', nn.initializers.constant(jnp.log(self.config.k)), ())
            k = jnp.exp(k_log)  # Ensure positive
        else:
            k = self.config.k
        return k

    @nn.compact
    def alpha(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        k = self.get_k_param()
        # Sigmoid interpolation: sigmoid(k * (t - 0.5)) maps [0,1] to [0,1]
        sigmoid_val = jax.nn.sigmoid(k * (t - 0.5))
        sigmoid_1 = jax.nn.sigmoid(k * (0.5))
        sigmoid_0 = jax.nn.sigmoid(k * (-0.5))
        sigmoid_val = (sigmoid_val - sigmoid_0) / (sigmoid_1 - sigmoid_0)
        return alpha_min + (alpha_max - alpha_min) * sigmoid_val

    @nn.compact
    def alpha_prime(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        k = self.get_k_param()
        # sigmoid' = k * sigmoid(x) * (1 - sigmoid(x))
        # x = k * (t - 0.5)
        sigmoid_val = jax.nn.sigmoid(k * (t - 0.5))
        sigmoid_prime = k * sigmoid_val * (1.0 - sigmoid_val)
        
        # Normalization constants
        sigmoid_1 = jax.nn.sigmoid(k * 0.5)
        sigmoid_0 = jax.nn.sigmoid(k * -0.5)
        
        return (alpha_max - alpha_min) * sigmoid_prime / (sigmoid_1 - sigmoid_0)

    @nn.compact
    def log_alpha_prime(self, t):
        return self.alpha_prime(t) / (self.alpha(t) + self.config.eps)

    @nn.compact
    def get_sigma_params(self):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_min, 1e-6, 1.0 - 1e-6))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_max, 1e-6, 1.0 - 1e-6))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
            sigma_max = jnp.maximum(sigma_max, sigma_min + self.config.eps)
        else:
            sigma_min = self.config.sigma_min
            sigma_max = self.config.sigma_max
        
        return sigma_min, sigma_max

    @nn.compact
    def sigma(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        k = self.get_k_param()
        # Reverse sigmoid for decreasing function
        sigmoid_val = jax.nn.sigmoid(k * (t - 0.5))
        sigmoid_1 = jax.nn.sigmoid(k * (0.5))
        sigmoid_0 = jax.nn.sigmoid(k * (- 0.5))
        sigmoid_val = (sigmoid_val - sigmoid_0) / (sigmoid_1 - sigmoid_0)
        return sigma_max + (sigma_min - sigma_max) * sigmoid_val

    @nn.compact
    def sigma_prime(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        k = self.get_k_param()
        # sigmoid' = k * sigmoid(x) * (1 - sigmoid(x))
        sigmoid_val = jax.nn.sigmoid(k * (t - 0.5))
        sigmoid_prime = k * sigmoid_val * (1.0 - sigmoid_val)
        
        # Normalization constants
        sigmoid_1 = jax.nn.sigmoid(k * 0.5)
        sigmoid_0 = jax.nn.sigmoid(k * -0.5)
        
        return (sigma_min - sigma_max) * sigmoid_prime / (sigmoid_1 - sigmoid_0)

    @nn.compact
    def log_sigma_prime(self, t):
        return self.sigma_prime(t) / (self.sigma(t) + self.config.eps)


class SoftplusFlowSchedule(FlowSchedule):
    """Softplus flow schedule with learnable parameters.
    
    Uses softplus interpolation: f(t) = (1/beta) * log(1 + exp(beta*t))
    Normalized such that f(1) = 1.
    """

    @nn.compact
    def get_alpha_params(self):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_min, self.config.eps, 1.0 - self.config.eps))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_max, self.config.eps, 1.0 - self.config.eps))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            alpha_max = jnp.maximum(alpha_max, alpha_min + self.config.eps)
        else:
            alpha_min = self.config.alpha_min
            alpha_max = self.config.alpha_max
        
        return alpha_min, alpha_max

    @nn.compact
    def get_beta_param(self):
        if self.learnable:
            # Initialize with log of softplus_beta (default 50.0) or config.softplus_beta if specified
            
            init_val = self.config.softplus_beta if self.config.softplus_beta != 50.0 else 50.0
            
            beta_log = self.param('beta_log', nn.initializers.constant(jnp.log(init_val)), ())
            beta = jnp.exp(beta_log)
        else:
            beta = self.config.softplus_beta
        return beta

    @nn.compact
    def alpha(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        beta = self.get_beta_param()
        
        # Softplus: f(t) = (1/beta) * log(1 + exp(beta*t))
        # Normalized: f_norm(t) = f(t) / f(1)
        # f(t) / f(1) = log(1 + exp(beta*t)) / log(1 + exp(beta))
        
        log_1_plus_exp_beta_t = jnp.logaddexp(0.0, beta * t)
        log_1_plus_exp_beta = jnp.logaddexp(0.0, beta)
        
        softplus_norm = log_1_plus_exp_beta_t / log_1_plus_exp_beta
        
        # Use alpha_min as an offset, scaling the range (alpha_max - alpha_min)
        # alpha(t) = alpha_min + (alpha_max - alpha_min) * softplus_norm(t)
        # Note: softplus_norm(0) > 0, so alpha(0) > alpha_min.
        # This preserves the "well behaved log" property (natural softplus floor)
        # while allowing alpha_min to shift the whole curve up.
        
        alpha_val = alpha_min + (alpha_max - alpha_min) * softplus_norm
        return jnp.clip(alpha_val, 0.0, 1.0)

    @nn.compact
    def alpha_prime(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        beta = self.get_beta_param()
        
        # f(t) = log(1 + exp(beta*t)) / C
        # f'(t) = (1/C) * (beta * exp(beta*t)) / (1 + exp(beta*t))
        #       = (1/C) * beta * sigmoid(beta*t)
        # alpha'(t) = (alpha_max - alpha_min) * f'(t)
        
        log_1_plus_exp_beta = jnp.logaddexp(0.0, beta)
        
        # sigmoid(x) = 1 / (1 + exp(-x))
        sigmoid_beta_t = jax.nn.sigmoid(beta * t)
        
        return (alpha_max - alpha_min) * (beta * sigmoid_beta_t) / log_1_plus_exp_beta

    @nn.compact
    def log_alpha_prime(self, t):
        return self.alpha_prime(t) / (self.alpha(t) + self.config.eps)

    @nn.compact
    def get_sigma_params(self):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_min, self.config.eps, 1.0 - self.config.eps))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_max, self.config.eps, 1.0 - self.config.eps))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
            sigma_max = jnp.maximum(sigma_max, sigma_min + self.config.eps)
        else:
            sigma_min = self.config.sigma_min
            sigma_max = self.config.sigma_max
        
        return sigma_min, sigma_max

    @nn.compact
    def sigma(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        beta = self.get_beta_param() # Use same beta for symmetry? Or separate?
        # Usually symmetric logic.
        
        # For sigma (decreasing 1 -> 0):
        # sigma(t) = sigma_min + (sigma_max - sigma_min) * softplus_norm(1.0 - t)
        # Note: softplus_norm(0) > 0.
        # At t=1 (argument 0): sigma(1) = sigma_min + (diff) * small > sigma_min.
        # At t=0 (argument 1): sigma(0) = sigma_min + (diff) * 1 = sigma_max.
        
        log_1_plus_exp_beta_t_rev = jnp.logaddexp(0.0, beta * (1.0 - t))
        log_1_plus_exp_beta = jnp.logaddexp(0.0, beta)
        
        softplus_norm_rev = log_1_plus_exp_beta_t_rev / log_1_plus_exp_beta
        
        sigma_val = sigma_min + (sigma_max - sigma_min) * softplus_norm_rev
        return jnp.clip(sigma_val, 0.0, 1.0)

    @nn.compact
    def sigma_prime(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        beta = self.get_beta_param()
        
        # sigma(t) = sigma_min + (sigma_max - sigma_min) * f(1-t)
        # sigma'(t) = (sigma_max - sigma_min) * f'(1-t) * (-1)
        
        log_1_plus_exp_beta = jnp.logaddexp(0.0, beta)
        sigmoid_beta_t_rev = jax.nn.sigmoid(beta * (1.0 - t))
        
        return -(sigma_max - sigma_min) * (beta * sigmoid_beta_t_rev) / log_1_plus_exp_beta

    @nn.compact
    def log_sigma_prime(self, t):
        return self.sigma_prime(t) / (self.sigma(t) + self.config.eps)


class ExponentialFlowSchedule(FlowSchedule):
    """Exponential flow schedule with learnable parameters.
    
    Uses exponential interpolation with learnable rate and bounds.
    Mathematical form: f(t) = a + b * exp(beta * t)
    where a and b are determined by boundary conditions.
    """

    @nn.compact
    def get_alpha_params(self):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_min, 1e-6, 1.0 - 1e-6))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_max, 1e-6, 1.0 - 1e-6))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            alpha_max = jnp.maximum(alpha_max, alpha_min + self.config.eps)
        else:
            alpha_min = self.config.alpha_min
            alpha_max = self.config.alpha_max
        
        return alpha_min, alpha_max

    @nn.compact
    def get_beta_param(self):
        if self.learnable:
            # Initialize log_beta with log of absolute value of beta
            beta_val = self.config.beta if self.config.beta != 0 else 1.0
            beta_log = self.param('beta_log', nn.initializers.constant(jnp.log(jnp.abs(beta_val))), ())
            beta = jnp.exp(beta_log)
            # Preserve sign of original beta if it was negative (though default is 2.0)
            if self.config.beta < 0:
                beta = -beta
        else:
            beta = self.config.beta
        return beta

    @nn.compact
    def alpha(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        beta = self.get_beta_param()
        
        # Form: alpha(t) = alpha_min + (alpha_max - alpha_min) * (exp(beta*t) - 1) / (exp(beta) - 1)
        # Use expm1 for numerical stability when beta is small
        
        # Normalized exponential interpolation from 0 to 1
        # h(t) = (exp(beta*t) - 1) / (exp(beta) - 1)
        h_t = jnp.expm1(beta * t) / (jnp.expm1(beta) + 1e-8)
        
        alpha_val = alpha_min + (alpha_max - alpha_min) * h_t
        return jnp.clip(alpha_val, 0.0, 1.0)

    @nn.compact
    def get_sigma_params(self):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_min, 1e-6, 1.0 - 1e-6))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_max, 1e-6, 1.0 - 1e-6))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
            sigma_max = jnp.maximum(sigma_max, sigma_min + self.config.eps)
        else:
            sigma_min = self.config.sigma_min
            sigma_max = self.config.sigma_max
        
        return sigma_min, sigma_max

    @nn.compact
    def sigma(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        beta = self.get_beta_param()
        
        # Form: sigma(t) = sigma_max + (sigma_min - sigma_max) * (exp(beta*t) - 1) / (exp(beta) - 1)
        
        # Normalized exponential interpolation from 0 to 1
        h_t = jnp.expm1(beta * t) / (jnp.expm1(beta) + 1e-8)
        
        sigma_val = sigma_max + (sigma_min - sigma_max) * h_t
        return jnp.clip(sigma_val, 0.0, 1.0)

    @nn.compact
    def alpha_prime(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        beta = self.get_beta_param()
        
        # h(t) = (exp(beta*t) - 1) / (exp(beta) - 1)
        # h'(t) = beta * exp(beta*t) / (exp(beta) - 1)
        # alpha'(t) = (alpha_max - alpha_min) * h'(t)
        
        # Using expm1 for denominator stability
        h_prime = beta * jnp.exp(beta * t) / (jnp.expm1(beta) + 1e-8)
        return (alpha_max - alpha_min) * h_prime

    @nn.compact
    def log_alpha_prime(self, t):
        return self.alpha_prime(t) / (self.alpha(t) + self.config.eps)

    @nn.compact
    def sigma_prime(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        beta = self.get_beta_param()
        
        # h'(t) = beta * exp(beta*t) / (exp(beta) - 1)
        # sigma'(t) = (sigma_min - sigma_max) * h'(t)
        
        h_prime = beta * jnp.exp(beta * t) / (jnp.expm1(beta) + 1e-8)
        return (sigma_min - sigma_max) * h_prime

    @nn.compact
    def log_sigma_prime(self, t):
        return self.sigma_prime(t) / (self.sigma(t) + self.config.eps)

class PureExponentialFlowSchedule(FlowSchedule):
    """Exponential flow schedule with learnable parameters.
    
    Uses exponential interpolation with learnable rate and bounds.
    Mathematical form: f(t) = a + b * exp(beta * t)
    where a and b are determined by boundary conditions.
    """

    @nn.compact
    def get_alpha_params(self):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_min, 1e-6, 1.0 - 1e-6))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_max, 1e-6, 1.0 - 1e-6))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            alpha_max = jnp.maximum(alpha_max, alpha_min + self.config.eps)
        else:
            alpha_min = self.config.alpha_min
            alpha_max = self.config.alpha_max
        
        return alpha_min, alpha_max

    @nn.compact
    def alpha(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        return alpha_min*(alpha_max/alpha_min)**t
        
    @nn.compact
    def alpha_prime(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        return alpha_min*jnp.log(alpha_max/alpha_min)*(alpha_max/alpha_min)**t

    @nn.compact
    def log_alpha_prime(self, t):
        alpha_min, alpha_max = self.get_alpha_params()
        return jnp.log(alpha_max/alpha_min)

    @nn.compact
    def get_sigma_params(self):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_min, 1e-6, 1.0 - 1e-6))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_max, 1e-6, 1.0 - 1e-6))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
            sigma_max = jnp.maximum(sigma_max, sigma_min + self.config.eps)
        else:
            sigma_min = self.config.sigma_min
            sigma_max = self.config.sigma_max
        
        return sigma_min, sigma_max

    @nn.compact
    def sigma(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        return sigma_max*(sigma_min/sigma_max)**t

    @nn.compact
    def sigma_prime(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        return sigma_max*jnp.log(sigma_min/sigma_max)*(sigma_min/sigma_max)**t

    @nn.compact
    def log_sigma_prime(self, t):
        sigma_min, sigma_max = self.get_sigma_params()
        return jnp.log(sigma_min/sigma_max)


class CauchyFlowSchedule(FlowSchedule):
    """Cauchy distribution-based flow schedule with learnable parameters.
    
    Uses Cauchy CDF for interpolation with learnable location, scale, and bounds.
    """

    @nn.compact
    def get_alpha_params(self):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_min, 1e-6, 1.0 - 1e-6))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_max, 1e-6, 1.0 - 1e-6))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            loc = self.param('alpha_loc', nn.initializers.constant(self.config.loc), ())
            log_scale = self.param('alpha_log_scale', nn.initializers.constant(self.config.log_scale), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            alpha_max = jnp.maximum(alpha_max, alpha_min + self.config.eps)
            scale = jnp.exp(log_scale)  # Ensure positive
        else:
            alpha_min = self.config.alpha_min
            alpha_max = self.config.alpha_max
            loc = self.config.loc
            scale = jnp.exp(self.config.log_scale)
        
        return alpha_min, alpha_max, loc, scale

    @nn.compact
    def alpha(self, t):
        alpha_min, alpha_max, loc, scale = self.get_alpha_params()
        
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
    def get_sigma_params(self):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_min, 1e-6, 1.0 - 1e-6))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_max, 1e-6, 1.0 - 1e-6))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            loc = self.param('sigma_loc', nn.initializers.constant(self.config.loc), ())
            log_scale = self.param('sigma_log_scale', nn.initializers.constant(self.config.log_scale), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
            sigma_max = jnp.maximum(sigma_max, sigma_min + self.config.eps)
            scale = jnp.exp(log_scale)  # Ensure positive
        else:
            sigma_min = self.config.sigma_min
            sigma_max = self.config.sigma_max
            loc = self.config.loc
            scale = jnp.exp(self.config.log_scale)
        
        return sigma_min, sigma_max, loc, scale

    @nn.compact
    def sigma(self, t):
        sigma_min, sigma_max, loc, scale = self.get_sigma_params()
        
        # Reverse Cauchy CDF for decreasing function
        # Normalize to ensure proper boundary conditions: at t=0 -> sigma_max, at t=1 -> sigma_min
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

    @nn.compact
    def alpha_prime(self, t):
        alpha_min, alpha_max, loc, scale = self.get_alpha_params()
        
        # Cauchy CDF: F(z) = 0.5 + (1/pi) * arctan(z), z = (t - loc) / scale
        # PDF: f(z) = (1/pi) * (1 / (1 + z^2)) * (1/scale)
        
        z = (t - loc) / scale
        z_0 = (0.0 - loc) / scale
        z_1 = (1.0 - loc) / scale
        
        cauchy_cdf_0 = 0.5 + (1.0 / jnp.pi) * jnp.arctan(z_0)
        cauchy_cdf_1 = 0.5 + (1.0 / jnp.pi) * jnp.arctan(z_1)
        normalization_const = cauchy_cdf_1 - cauchy_cdf_0 + 1e-8
        
        pdf_val = (1.0 / jnp.pi) * (1.0 / (1.0 + z**2)) * (1.0 / scale)
        
        return (alpha_max - alpha_min) * pdf_val / normalization_const

    @nn.compact
    def log_alpha_prime(self, t):
        return self.alpha_prime(t) / (self.alpha(t) + self.config.eps)

    @nn.compact
    def sigma_prime(self, t):
        sigma_min, sigma_max, loc, scale = self.get_sigma_params()
        
        z = (t - loc) / scale
        z_0 = (0.0 - loc) / scale
        z_1 = (1.0 - loc) / scale
        
        cauchy_cdf_0 = 0.5 + (1.0 / jnp.pi) * jnp.arctan(z_0)
        cauchy_cdf_1 = 0.5 + (1.0 / jnp.pi) * jnp.arctan(z_1)
        normalization_const = cauchy_cdf_1 - cauchy_cdf_0 + 1e-8
        
        # sigma(t) = sigma_max + (sigma_min - sigma_max) * normalized_CDF
        # sigma'(t) = (sigma_min - sigma_max) * PDF / normalization_const
        
        pdf_val = (1.0 / jnp.pi) * (1.0 / (1.0 + z**2)) * (1.0 / scale)
        
        return (sigma_min - sigma_max) * pdf_val / normalization_const

    @nn.compact
    def log_sigma_prime(self, t):
        return self.sigma_prime(t) / (self.sigma(t) + self.config.eps)


class LaplaceFlowSchedule(FlowSchedule):
    """Laplace distribution-based flow schedule with learnable parameters.
    
    Uses Laplace CDF for interpolation with learnable location, scale, and bounds.
    """

    @nn.compact
    def get_alpha_params(self):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_min, self.config.eps, 1.0 - self.config.eps))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_max, self.config.eps, 1.0 - self.config.eps))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            loc = self.param('alpha_loc', nn.initializers.constant(self.config.loc), ())
            log_scale = self.param('alpha_log_scale', nn.initializers.constant(self.config.log_scale), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            alpha_max = jnp.maximum(alpha_max, alpha_min + self.config.eps)
            scale = jnp.exp(log_scale)  # Ensure positive
        else:
            alpha_min = self.config.alpha_min
            alpha_max = self.config.alpha_max
            loc = self.config.loc
            scale = jnp.exp(self.config.log_scale)
        
        return alpha_min, alpha_max, loc, scale

    @nn.compact
    def alpha(self, t):
        alpha_min, alpha_max, loc, scale = self.get_alpha_params()
        
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
    def get_sigma_params(self):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_min, self.config.eps, 1.0 - self.config.eps))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_max, self.config.eps, 1.0 - self.config.eps))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            loc = self.param('sigma_loc', nn.initializers.constant(self.config.loc), ())
            log_scale = self.param('sigma_log_scale', nn.initializers.constant(self.config.log_scale), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
            sigma_max = jnp.maximum(sigma_max, sigma_min + self.config.eps)
            scale = jnp.exp(log_scale)  # Ensure positive
        else:
            sigma_min = self.config.sigma_min
            sigma_max = self.config.sigma_max
            loc = self.config.loc
            scale = jnp.exp(self.config.log_scale)
        
        return sigma_min, sigma_max, loc, scale

    @nn.compact
    def sigma(self, t):
        sigma_min, sigma_max, loc, scale = self.get_sigma_params()
        
        # Reverse Laplace CDF for decreasing function
        # Normalize to ensure proper boundary conditions: at t=0 -> sigma_max, at t=1 -> sigma_min
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

    @nn.compact
    def alpha_prime(self, t):
        alpha_min, alpha_max, loc, scale = self.get_alpha_params()
        
        # Laplace CDF: F(z) = 0.5 * (1 + sign(z) * (1 - exp(-|z|))), z = (t - loc) / scale
        # PDF: f(z) = (1 / (2 * scale)) * exp(-|z|)
        
        z = (t - loc) / scale
        z_0 = (0.0 - loc) / scale
        z_1 = (1.0 - loc) / scale
        
        laplace_cdf_0 = 0.5 * (1.0 + jnp.sign(z_0) * (1.0 - jnp.exp(-jnp.abs(z_0))))
        laplace_cdf_1 = 0.5 * (1.0 + jnp.sign(z_1) * (1.0 - jnp.exp(-jnp.abs(z_1))))
        normalization_const = laplace_cdf_1 - laplace_cdf_0 + 1e-8
        
        pdf_val = (1.0 / (2.0 * scale)) * jnp.exp(-jnp.abs(z))
        
        return (alpha_max - alpha_min) * pdf_val / normalization_const

    @nn.compact
    def log_alpha_prime(self, t):
        return self.alpha_prime(t) / (self.alpha(t) + self.config.eps)

    @nn.compact
    def sigma_prime(self, t):
        sigma_min, sigma_max, loc, scale = self.get_sigma_params()
        
        z = (t - loc) / scale
        z_0 = (0.0 - loc) / scale
        z_1 = (1.0 - loc) / scale
        
        laplace_cdf_0 = 0.5 * (1.0 + jnp.sign(z_0) * (1.0 - jnp.exp(-jnp.abs(z_0))))
        laplace_cdf_1 = 0.5 * (1.0 + jnp.sign(z_1) * (1.0 - jnp.exp(-jnp.abs(z_1))))
        normalization_const = laplace_cdf_1 - laplace_cdf_0 + 1e-8
        
        # sigma(t) = sigma_max + (sigma_min - sigma_max) * normalized_CDF
        # sigma'(t) = (sigma_min - sigma_max) * PDF / normalization_const
        
        pdf_val = (1.0 / (2.0 * scale)) * jnp.exp(-jnp.abs(z))
        
        return (sigma_min - sigma_max) * pdf_val / normalization_const

    @nn.compact
    def log_sigma_prime(self, t):
        return self.sigma_prime(t) / (self.sigma(t) + self.config.eps)


class PolynomialFlowSchedule(FlowSchedule):
    """Polynomial flow schedule with learnable parameters.
    
    Uses polynomial interpolation: t^power with learnable power and bounds.
    """

    @nn.compact
    def get_alpha_params(self):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_min, self.config.eps, 1.0 - self.config.eps))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_max, self.config.eps, 1.0 - self.config.eps))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            log_power = self.param('alpha_log_power', nn.initializers.constant(self.config.log_power), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            alpha_max = jnp.maximum(alpha_max, alpha_min + self.config.eps)
            power = jnp.exp(log_power)  # Ensure positive, typically >= 1.0
            power = jnp.clip(power, 0.1, 10.0)  # Reasonable bounds
        else:
            alpha_min = self.config.alpha_min
            alpha_max = self.config.alpha_max
            power = jnp.exp(self.config.log_power)
        
        return alpha_min, alpha_max, power

    @nn.compact
    def alpha(self, t):
        alpha_min, alpha_max, power = self.get_alpha_params()
        # Polynomial interpolation: t^power maps [0,1] to [0,1]
        poly_val = t ** power
        return alpha_min + (alpha_max - alpha_min) * poly_val

    @nn.compact
    def get_sigma_params(self):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_min, self.config.eps, 1.0 - self.config.eps))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_max, self.config.eps, 1.0 - self.config.eps))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            log_power = self.param('sigma_log_power', nn.initializers.constant(self.config.log_power), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
            sigma_max = jnp.maximum(sigma_max, sigma_min + self.config.eps)
            power = jnp.exp(log_power)  # Ensure positive, typically >= 1.0
            power = jnp.clip(power, 0.1, 10.0)  # Reasonable bounds
        else:
            sigma_min = self.config.sigma_min
            sigma_max = self.config.sigma_max
            power = jnp.exp(self.config.log_power)
        
        return sigma_min, sigma_max, power

    @nn.compact
    def sigma(self, t):
        sigma_min, sigma_max, power = self.get_sigma_params()
        # Polynomial for decreasing function: use t^power which goes from 0 to 1
        # We want sigma to go from sigma_max (at t=0) to sigma_min (at t=1)
        # poly_val goes from 0 to 1 as t goes from 0 to 1
        poly_val = t ** power  
        return sigma_max + (sigma_min - sigma_max) * poly_val

    @nn.compact
    def alpha_prime(self, t):
        alpha_min, alpha_max, power = self.get_alpha_params()
        # alpha(t) = alpha_min + (alpha_max - alpha_min) * t^power
        # alpha'(t) = (alpha_max - alpha_min) * power * t^(power-1)
        return (alpha_max - alpha_min) * power * (t ** (power - 1.0))

    @nn.compact
    def log_alpha_prime(self, t):
        return self.alpha_prime(t) / (self.alpha(t) + self.config.eps)

    @nn.compact
    def sigma_prime(self, t):
        sigma_min, sigma_max, power = self.get_sigma_params()
        # sigma(t) = sigma_max + (sigma_min - sigma_max) * t^power
        # sigma'(t) = (sigma_min - sigma_max) * power * t^(power-1)
        return (sigma_min - sigma_max) * power * (t ** (power - 1.0))

    @nn.compact
    def log_sigma_prime(self, t):
        return self.sigma_prime(t) / (self.sigma(t) + self.config.eps)




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
    """
    
    monotonic_network: nn.Module = SimpleMonotonicNetwork

    @nn.compact
    def get_alpha_params(self):
        if self.learnable:
            alpha_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_min, self.config.eps, 1.0 - self.config.eps))
            alpha_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.alpha_max, self.config.eps, 1.0 - self.config.eps))
            
            alpha_min_logit = self.param('alpha_min_logit', 
                                        nn.initializers.constant(alpha_min_logit_val), ())
            alpha_max_logit = self.param('alpha_max_logit',
                                         nn.initializers.constant(alpha_max_logit_val), ())
            
            alpha_min = jax.nn.sigmoid(alpha_min_logit)
            alpha_max = jax.nn.sigmoid(alpha_max_logit)
            alpha_max = jnp.maximum(alpha_max, alpha_min + self.config.eps)
        else:
            alpha_min = self.config.alpha_min
            alpha_max = self.config.alpha_max
        
        return alpha_min, alpha_max

    @nn.compact
    def alpha(self, t):
        """Compute alpha(t) using a monotonic neural network."""
        alpha_min, alpha_max = self.get_alpha_params()
        
        # Create the monotonic network for alpha
        alpha_network = self.monotonic_network(hidden_dims=self.config.hidden_dims, name='alpha_net')
        
        def alpha_fn_scalar(t_input):
            """Scalar function for alpha that ensures boundary conditions."""
            # Network output (monotonic increasing)
            f_t = alpha_network(t_input)
            # Get values at boundaries for normalization
            f_0 = alpha_network(jnp.asarray(0.0))
            f_1 = alpha_network(jnp.asarray(1.0))
            # Normalize to [0, 1] range
            normalized = (f_t - f_0) / (f_1 - f_0 + self.config.eps)
            normalized = jnp.clip(normalized, 0.0, 1.0)
            # Scale to [alpha_min, alpha_max]
            alpha_val = alpha_min + (alpha_max - alpha_min) * normalized
            return alpha_val
        
        # Vectorize over t
        t = jnp.asarray(t)
        t_flat = t.reshape(-1)
        alpha_vals = jax.vmap(alpha_fn_scalar)(t_flat)
        alpha_vals = alpha_vals.reshape(t.shape)
        
        # Ensure values are in valid range
        alpha_vals = jnp.clip(alpha_vals, self.config.eps, 1.0 - self.config.eps)
        return alpha_vals

    @nn.compact
    def get_sigma_params(self):
        if self.learnable:
            sigma_min_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_min, self.config.eps, 1.0 - self.config.eps))
            sigma_max_logit_val = jax.scipy.special.logit(jnp.clip(self.config.sigma_max, self.config.eps, 1.0 - self.config.eps))
            
            sigma_min_logit = self.param('sigma_min_logit', 
                                        nn.initializers.constant(sigma_min_logit_val), ())
            sigma_max_logit = self.param('sigma_max_logit',
                                         nn.initializers.constant(sigma_max_logit_val), ())
            
            sigma_min = jax.nn.sigmoid(sigma_min_logit)
            sigma_max = jax.nn.sigmoid(sigma_max_logit)
            sigma_max = jnp.maximum(sigma_max, sigma_min + self.config.eps)
        else:
            sigma_min = self.config.sigma_min
            sigma_max = self.config.sigma_max
        
        return sigma_min, sigma_max

    @nn.compact
    def sigma(self, t):
        """Compute sigma(t) using a monotonic neural network for 1-sigma."""
        sigma_min, sigma_max = self.get_sigma_params()
        
        # Create the monotonic network for 1-sigma (so sigma is decreasing)
        # Use a distinct network for sigma
        sigma_network = self.monotonic_network(hidden_dims=self.config.hidden_dims, name='sigma_net')
        
        def sigma_fn_scalar(t_input):
            """Scalar function for sigma via 1-sigma network."""
            # Network output for 1-sigma (monotonic increasing)
            f_t = sigma_network(t_input)
            # Get values at boundaries for normalization
            f_0 = sigma_network(jnp.asarray(0.0))
            f_1 = sigma_network(jnp.asarray(1.0))
            # Normalize to [0, 1] range
            normalized = (f_t - f_0) / (f_1 - f_0 + self.config.eps)
            normalized = jnp.clip(normalized, 0.0, 1.0)
            # The normalized value represents how much 1-sigma has increased
            # At t=0: normalized=0, we want sigma=sigma_max
            # At t=1: normalized=1, we want sigma=sigma_min
            # So: sigma = sigma_max + (sigma_min - sigma_max) * normalized
            sigma_val = sigma_max + (sigma_min - sigma_max) * normalized
            return sigma_val
        
        # Vectorize over t
        t = jnp.asarray(t)
        t_flat = t.reshape(-1)
        sigma_vals = jax.vmap(sigma_fn_scalar)(t_flat)
        sigma_vals = sigma_vals.reshape(t.shape)
        
        # Ensure values are in valid range
        sigma_vals = jnp.clip(sigma_vals, self.config.eps, 1.0 - self.config.eps)
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
        latent_shape = config.get("latent_shape")
        
        if ndims is None and latent_shape is not None:
            ndims = len(latent_shape)
        
        if schedule_type is None or ndims is None:
            raise ValueError("config must contain 'schedule_type' and 'ndims' (or 'latent_shape')")
        
        learnable = config.get("learnable", False)
        alpha_min = config.get("alpha_min", 0.05)
        alpha_max = config.get("alpha_max", 0.95)
        sigma_min = config.get("sigma_min", 0.05)
        sigma_max = config.get("sigma_max", 0.95)
        k = config.get("k", 10.0)
        beta = config.get("beta", 2.0)
        softplus_beta = config.get("softplus_beta", 50.0)
        loc = config.get("loc", 0.5)
        log_scale = config.get("log_scale", -1.0)
        log_power = config.get("log_power", 0.69)
        hidden_dims = config.get("hidden_dims", (64, 64))
        eps = config.get("eps", 1e-8)
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
        softplus_beta = config.softplus_beta
        loc = config.loc
        log_scale = config.log_scale
        log_power = config.log_power
        hidden_dims = config.hidden_dims
        eps = config.eps
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
    softplus_beta = kwargs.get("softplus_beta", softplus_beta)
    loc = kwargs.get("loc", loc)
    log_scale = kwargs.get("log_scale", log_scale)
    log_power = kwargs.get("log_power", log_power)
    hidden_dims = kwargs.get("hidden_dims", hidden_dims)
    eps = kwargs.get("eps", eps)
    
    # Create the appropriate schedule based on schedule_type
    schedule_type = schedule_type.lower()
    
    # Create common config for all schedules
    # Note: we need to pass FlowScheduleConfig to the constructor, not individual args
    # The individual classes inherit from FlowSchedule which expects config
    
    # First, ensure latent_shape is set based on ndims if not already set in config
    latent_shape = getattr(config, "latent_shape", (ndims,) if isinstance(ndims, int) else tuple(ndims))
    if isinstance(ndims, int) and not latent_shape:
        latent_shape = (ndims,)
        
    schedule_config = FlowScheduleConfig(
        schedule_type=schedule_type,
        latent_shape=latent_shape,
        learnable=learnable,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        k=k,
        beta=beta,
        softplus_beta=softplus_beta,
        loc=loc,
        log_scale=log_scale,
        log_power=log_power,
        hidden_dims=hidden_dims,
        eps=eps
    )
    
    if schedule_type == "linear":
        return LinearFlowSchedule(config=schedule_config)
    elif schedule_type == "cosine":
        return CosineFlowSchedule(config=schedule_config)
    elif schedule_type == "sigmoid":
        return SigmoidFlowSchedule(config=schedule_config)
    elif schedule_type == "softplus":
        return SoftplusFlowSchedule(config=schedule_config)
    elif schedule_type == "exponential":
        return ExponentialFlowSchedule(config=schedule_config)
    elif schedule_type == "pure_exponential":
        return PureExponentialFlowSchedule(config=schedule_config)
    elif schedule_type == "cauchy":
        return CauchyFlowSchedule(config=schedule_config)
    elif schedule_type == "laplace":
        return LaplaceFlowSchedule(config=schedule_config)
    elif schedule_type == "polynomial":
        return PolynomialFlowSchedule(config=schedule_config)
    elif schedule_type in ["network", "neural", "learnable"]:
        return FlowScheduleNetwork(config=schedule_config)
    else:
        raise ValueError(
            f"Unknown schedule_type: {schedule_type}. "
            f"Options: linear, cosine, sigmoid, softplus, exponential, cauchy, laplace, polynomial, network/neural/learnable"
        )
