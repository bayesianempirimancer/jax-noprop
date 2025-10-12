"""
NoProp-CT: Continuous-time NoProp implementation.

This module implements the continuous-time variant of NoProp using neural ODEs.
The key idea is to model the denoising process as a continuous-time dynamical system.
"""

from typing import Any, Dict, Optional, Tuple, Callable
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from scipy.integrate import odeint
import optax

from .noise_schedules import NoiseSchedule, LearnableNoiseSchedule, CosineNoiseSchedule, LinearNoiseSchedule
from .utils.ode_integration import integrate_ode


class NoPropCT(nn.Module):
    """Continuous-time NoProp implementation.
    
    This class implements the continuous-time variant where the denoising
    process is modeled as a neural ODE. The model learns a vector field
    that transforms noisy targets to clean ones over continuous time.
    """
    
    target_dim: int  # Dimension of target z
    model: nn.Module
    noise_schedule: NoiseSchedule = LinearNoiseSchedule()
    num_timesteps: int = 20
    integration_method: str = "euler"  # "euler" or "heun"
    reg_weight: float = 0.0  # Hyperparameter from the paper
        
    @nn.compact
    def __call__(
        self, 
        z: jnp.ndarray, 
        x: jnp.ndarray, 
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Initialize all parameters by calling all @nn.compact methods once.
        
        This method is only called during init() to set up the parameter tree.
        During training, we use apply(params, method=...) to call specific methods.
        
        Args:
            z: Noisy target [batch_size, z_dim]
            x: Input data [batch_size, height, width, channels]
            t: Time step [batch_size]
            
        Returns:
            Model output [batch_size, z_dim]
        """
        # Get model output
        model_output = self._get_model_output(z, x, t)
        gamma_t, gamma_prime_t = self._get_gamma_gamma_prime(t)
        
        # Compute alpha_t and tau_inverse from gamma values using utility functions
        alpha_t = self.noise_schedule.get_alpha_from_gamma(gamma_t)
        tau_inverse = self.noise_schedule.get_tau_inverse_from_gamma(gamma_t, gamma_prime_t)
        
        # Reshape for broadcasting: [batch_size] -> [batch_size, 1]
        alpha_t = alpha_t[..., None]
        tau_inverse = tau_inverse[..., None]
        
        # Compute dz/dt = tau_inverse(t) * (sqrt(alpha(t))*target - (1+alpha(t))/2*z)
        return tau_inverse * (jnp.sqrt(alpha_t) * model_output - (1 + alpha_t) / 2.0 * z)
    
    @nn.compact
    def _get_model_output(
        self, 
        z: jnp.ndarray, 
        x: jnp.ndarray, 
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Get model output - @nn.compact method for parameter initialization."""
        return self.model(z, x, t)
    
    @nn.compact
    def _get_gamma_gamma_prime(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get gamma values - @nn.compact method for parameter initialization."""
        return self.noise_schedule(t)
    
    def dz_dt(
        self,
        params: Dict[str, Any],
        z: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Public interface for dz_dt computation.
        
        Args:
            params: Model parameters
            z: Current state [batch_size, num_classes]
            x: Input data [batch_size, height, width, channels]
            t: Current time [batch_size]
            
        Returns:
            Vector field dz/dt [batch_size, num_classes]
        """
        return self.apply(params, z, x, t)
    
    @partial(jax.jit, static_argnums=(0,))  # self is static
    def compute_loss(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        target: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute the NoProp-CT loss according to the paper and PyTorch implementation.
        
        Based on the PyTorch implementation, the loss should be:
        L_CT = E[(1/SNR(t)) * ||model_output - z_t||²]
        
        where model_output is the predicted denoised target and SNR(t) is the signal-to-noise ratio.
        
        Args:
            params: Model parameters
            x: Input data [batch_size, height, width, channels]
            target: Clean target [batch_size, num_classes]
            key: Random key for sampling t and z_t
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Infer batch size from input tensor
        batch_shape = x.shape[:-1]
        
        # Sample random timesteps
        key, t_key = jax.random.split(key)
        t = jax.random.uniform(t_key, batch_shape, minval=0.0, maxval=1.0)
        
        # Sample z_t from backward process
        key, noise_key = jax.random.split(key)
        z_t = self.sample_zt(noise_key, params, target, t)
        # Get model output using apply with method
        model_output = self.apply(params, z_t, x, t, method=self._get_model_output)
        
        # Get gamma values using apply with method
        gamma_t, gamma_prime_t = self.apply(params, t, method=self._get_gamma_gamma_prime)
        
        # Compute SNR and SNR derivative from gamma values
        snr = jnp.exp(gamma_t)  # SNR = exp(γ(t))
        snr_prime = gamma_prime_t * snr  # SNR' = γ'(t) * exp(γ(t))
        
        # Reshape for broadcasting: [batch_size] -> [batch_size, 1]
        snr_prime = snr_prime[..., None]
        
        # Main loss: SNR-weighted MSE between model output and noisy input
        # This follows the NoProp paper where loss is weighted by SNR derivative
        reg_loss = jnp.mean(model_output ** 2)
        squared_error = (model_output - target) ** 2
        mse = jnp.mean(squared_error)

        # Compute SNR-weighted loss
        snr_weighted_loss = jnp.mean(snr_prime * squared_error)
        
        # Normalize by expected SNR_prime to stabilize learning rate
        expected_snr_prime = jnp.mean(snr_prime)
        ct_loss = snr_weighted_loss / expected_snr_prime
#        ct_loss = snr_weighted_loss
        
        # total loss
        total_loss = ct_loss + self.reg_weight * reg_loss

        # Compute additional metrics
        metrics = {
            "ct_loss": ct_loss,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
            "mse": mse,
            "snr_weighted_loss": snr_weighted_loss,  # Before normalization
            "snr_prime_mean": expected_snr_prime,  # SNR derivative
        }
        
        return total_loss, metrics
    
    def sample_zt(
        self,
        key: jax.random.PRNGKey,
        params: Dict[str, Any],
        z_target: jnp.ndarray,
        t: jnp.ndarray
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
            params: Model parameters
            
        Returns:
            Sampled z_t [batch_size, ...]
        """
        # Get gamma_t and compute alpha_t from it
        gamma_t, _ = self.apply(params, t, method=self._get_gamma_gamma_prime)
        alpha_t = self.noise_schedule.get_alpha_from_gamma(gamma_t)
        
        # Compute z_t = sqrt(ᾱ(t)) * z_target + sqrt(1-ᾱ(t)) * ε
        alpha_t = alpha_t[..., None]
        return jnp.sqrt(alpha_t) * z_target + jnp.sqrt(1.0 - alpha_t) * jax.random.normal(key, z_target.shape)
    
    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))  # self, output_dim, num_steps, integration_method, output_type are static arguments    
    def predict(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        output_dim: int,
        num_steps: int,
        integration_method: str = "euler",
        output_type: str = "end_point"
    ) -> jnp.ndarray:
        """Generate predictions using the trained NoProp-CT neural ODE.
        
        This integrates the learned vector field from zeros (t=0)
        to the final prediction (t=1), following the paper's approach.
        Uses scan-based integration for better performance.
        
        Args:
            params: Model parameters
            x: Input data [batch_size, height, width, channels]
            output_dim: Output dimension
            num_steps: Number of integration steps
            integration_method: Integration method to use ("euler", "heun", "rk4", "adaptive")
            output_type: Type of output ("end_point" or "trajectory")
            
        Returns:
            If output_type="end_point": Final prediction [batch_size, output_dim]
            If output_type="trajectory": Full trajectory [batch_size, num_steps+1, output_dim]
        """
        # Disable gradient tracking through parameters for inference
        params_no_grad = jax.lax.stop_gradient(params)
        
        # Infer batch size from input tensor
        batch_shape = x.shape[:-1]
        
        # Start with zeros at t=0
        z0 = jnp.zeros(batch_shape + (output_dim,))
        
        # Create a static vector field function to avoid recompilation
        def vector_field(params, z, x, t):
            return self.dz_dt(params, z, x, t)
        
        # Use the main integrate_ode function which handles method dispatch
        result = integrate_ode(
            vector_field=vector_field,
            params=params_no_grad,
            z0=z0,
            x=x,
            time_span=(0.0, 1.0),
            num_steps=num_steps,
            method=integration_method,
            output_type=output_type
        )
        
        return result
    
    def predict_trajectory(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        integration_method: str,
        output_dim: int,
        num_steps: int
    ) -> jnp.ndarray:
        """Generate prediction trajectories using the trained NoProp-CT neural ODE.
        
        This is a wrapper around the predict method with output_type="trajectory".
        It integrates the learned vector field from zeros (t=0) to the final prediction (t=1),
        returning the full time course.
        
        Args:
            params: Model parameters
            x: Input data [batch_size, height, width, channels]
            integration_method: Integration method to use ("euler", "heun", "rk4", "adaptive")
            output_dim: Output dimension
            num_steps: Number of integration steps
            
        Returns:
            Full trajectory [num_steps + 1, batch_size, output_dim] (includes initial state)
        """
        return self.predict(
            params=params,
            x=x,
            output_dim=output_dim,
            num_steps=num_steps,
            integration_method=integration_method,
            output_type="trajectory"
        )
    
    @partial(jax.jit, static_argnums=(0, 6))  # self and optimizer are static arguments
    def train_step(
        self,
        params: Dict[str, Any],
        opt_state: optax.OptState,
        x: jnp.ndarray,
        target: jnp.ndarray,
        key: jax.random.PRNGKey,
        optimizer: optax.GradientTransformation
    ) -> Tuple[Dict[str, Any], optax.OptState, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Single training step for NoProp-CT.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            x: Input data [batch_size, height, width, channels]
            target: Clean target [batch_size, num_classes]
            key: Random key
            optimizer: Optax optimizer
            
        Returns:
            Tuple of (updated_params, updated_opt_state, loss, metrics)
        """
        # Compute loss and gradients (t and z_t are sampled inside compute_loss)
        # compute_loss is already JIT-compiled, so this will be fast
        (loss, metrics), grads = jax.value_and_grad(
            self.compute_loss, has_aux=True)(params, x, target, key)
        
        # Update parameters using optimizer
        updates, updated_opt_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        
        return updated_params, updated_opt_state, loss, metrics
    