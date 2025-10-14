"""
NoProp-CT: Continuous-time NoProp implementation.

This module implements the continuous-time variant of NoProp using neural ODEs.
The key idea is to model the denoising process as a continuous-time dynamical system.
"""

from typing import Any, Dict, Optional, Tuple, Callable
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
from scipy.integrate import odeint
import optax

from .embeddings.noise_schedules import NoiseSchedule, LearnableNoiseSchedule, CosineNoiseSchedule, LinearNoiseSchedule
from .utils.ode_integration import integrate_ode
from .utils.jacobian_utils import trace_jacobian


class NoPropCT(nn.Module):
    """Continuous-time NoProp implementation.
    
    This class implements the continuous-time variant where the denoising
    process is modeled as a neural ODE. The model learns a vector field
    that transforms noisy targets to clean ones over continuous time.
    """
    
    z_shape: Tuple[int, ...]  # Shape of target z (excluding batch dimensions)
    x_shape: Tuple[int, ...]  # Shape of input x (excluding batch dimensions)
    model: nn.Module
    noise_schedule: NoiseSchedule = LinearNoiseSchedule()
    num_timesteps: int = 20
    integration_method: str = "euler"  # "euler" or "heun"
    reg_weight: float = 0.0  # Hyperparameter from the paper

    def _get_z_dim(self) -> int:
        """Calculate the flattened dimension from z_shape."""
        z_dim = 1
        for dim in self.z_shape:
            z_dim *= dim
        return z_dim

    def _flatten_z(self, z: jnp.ndarray) -> jnp.ndarray:
        """Flatten the z tensor."""
        if len(self.z_shape)>1:
            return z.reshape(z.shape[:-1] + (self._get_z_dim(),))
        else:
            return z

    def _unflatten_z(self, z: jnp.ndarray) -> jnp.ndarray:
        """Unflatten the z tensor."""
        if len(self.z_shape)>1:
            return z.reshape(z.shape[:-1] + self.z_shape)
        else:
            return z

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
            z: Noisy target [batch_shape + z_shape]
            x: Input data [batch_shape, ...]
            t: Time step [batch_shape]
            
        Returns:
            Model output [batch_shape + z_shape]
        """
        # Get model output
        z = self._unflatten_z(z)
        model_output = self._get_model_output(z, x, t)
        model_output = self._flatten_z(model_output)
        gamma_t, gamma_prime_t = self._get_gamma_gamma_prime(t)
        
        # Compute alpha_t and tau_inverse from gamma values using utility functions
        alpha_t = self.noise_schedule.get_alpha_from_gamma(gamma_t)
        tau_inverse = self.noise_schedule.get_tau_inverse_from_gamma(gamma_t, gamma_prime_t)
        
        # Broadcast to the correct shape for tensor operations
        batch_shape = x.shape[:-len(self.x_shape)]
                
        # Compute dz/dt = tau_inverse(t) * (sqrt(alpha(t))*model_output - (1+alpha(t))/2*z)
        # The model_output should predict the target, so we use it in place of target
        return tau_inverse[...,None] * (jnp.sqrt(alpha_t[...,None]) * model_output - 0.5*(1 + alpha_t[...,None]) * z)
    
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
        key: jr.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute the NoProp-CT loss according to the paper and PyTorch implementation.
        
        Based on the PyTorch implementation, the loss should be:
        L_CT = E[(1/SNR(t)) * ||model_output - z_t||²]
        
        where model_output is the predicted denoised target and SNR(t) is the signal-to-noise ratio.
        
        Args:
            params: Model parameters
            x: Input data [batch_shape, ...]
            target: Clean target [batch_shape + z_shape]
            key: Random key for sampling t and z_t
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Infer batch size from input tensor
        batch_shape = x.shape[:-len(self.x_shape)]
        target = self._flatten_z(target)

        # Sample random timesteps
        key, t_key, z_t_key = jr.split(key, 3)
        t = jr.uniform(t_key, batch_shape, minval=0.0, maxval=1.0)

        gamma_t, gamma_prime_t = self.apply(params, t, method=self._get_gamma_gamma_prime)
        alpha_t = self.noise_schedule.get_alpha_from_gamma(gamma_t)
        z_t = jnp.sqrt(alpha_t[...,None]) * target + jnp.sqrt(1.0 - alpha_t[...,None]) * jr.normal(key, target.shape)
        
        # Sample z_t from backward process
        model_output = self.apply(params, z_t, x, t, method=self._get_model_output)

        # Compute SNR and SNR derivative from gamma values
        snr = jnp.exp(gamma_t)  # SNR = exp(γ(t))
        snr_prime = gamma_prime_t * snr  # SNR' = γ'(t) * exp(γ(t))

        # Main loss: SNR-weighted MSE between model output and noisy input
        # This follows the NoProp paper where loss is weighted by SNR derivative
        reg_loss = jnp.mean(model_output ** 2)
        squared_error = (model_output - target) ** 2
        mse = jnp.mean(squared_error)

        # Compute SNR-weighted loss
        snr_weighted_loss = jnp.mean(snr_prime[...,None] * squared_error)
        
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
        
    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))  # self, num_steps, integration_method, output_type, with_logp are static arguments    
    def predict(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        num_steps: int,
        integration_method: str = "euler",
        output_type: str = "end_point",
        with_logp: bool = False,
        key: jr.PRNGKey = None
    ) -> jnp.ndarray:
        """Generate predictions using the trained NoProp-CT neural ODE.
        
        This integrates the learned vector field from zeros (t=0)
        to the final prediction (t=1), following the paper's approach.
        Uses scan-based integration for better performance.
        
        Args:
            params: Model parameters
            x: Input data [batch_shape, ...]
            num_steps: Number of integration steps
            integration_method: Integration method to use ("euler", "heun", "rk4", "adaptive")
            output_type: Type of output ("end_point" or "trajectory")
            key: Random key
            
        Returns:
            If output_type="end_point": Final prediction [batch_shape + z_shape]
            If output_type="trajectory": Full trajectory [num_steps+1, batch_shape + z_shape]
        """
        # Disable gradient tracking through parameters for inference
        params_no_grad = jax.lax.stop_gradient(params)
        
        # Infer batch size from input tensor
        batch_shape = x.shape[:-len(self.x_shape)]
        if key is None:
            z0 = jnp.zeros(batch_shape + self.z_shape)
        else:
            z0 = jr.normal(key, batch_shape + self.z_shape)  # check for sensitivity to initial conditions
        
        if with_logp:
            # Combined vector field for z and logp integration
            def vector_field(params, state, x, t):
                z = state[..., :-1]  # All but last dimension
                logp = state[..., -1:]  # Last dimension
                dz_dt = self.dz_dt(params, z, x, t)
                # Compute Jacobian trace for logp evolution
                # Broadcast t to match batch size
                t_broadcast = jnp.broadcast_to(t, z.shape[:-1])
                trace_jac = trace_jacobian(self.dz_dt, params, z, x, t_broadcast)
                dlogp_dt = trace_jac
                return jnp.concatenate([dz_dt, dlogp_dt[..., None]], axis=-1)
            
            # Initialize combined state
            logp0 = jnp.zeros(batch_shape)
            state0 = jnp.concatenate([z0, logp0[..., None]], axis=-1)
            
            result = integrate_ode(
                vector_field=vector_field,
                params=params_no_grad,
                z0=state0,
                x=x,
                time_span=(0.0, 1.0),
                num_steps=num_steps,
                method=integration_method,
                output_type=output_type
            )
        else:
            # Standard vector field for z only
            def vector_field(params, z, x, t):
                return self.dz_dt(params, z, x, t)

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

    @partial(jax.jit, static_argnums=(0, 5))  # self and optimizer are static arguments
    def train_step(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        target: jnp.ndarray,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        key: jr.PRNGKey,
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
        
        # Update parameters using optimizer (outside JIT)
        updates, updated_opt_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        
        return updated_params, updated_opt_state, loss, metrics
    