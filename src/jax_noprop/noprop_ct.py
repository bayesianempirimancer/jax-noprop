"""
NoProp-CT: Continuous-time NoProp implementation.

This module implements the continuous-time variant of NoProp using neural ODEs.
The key idea is to model the denoising process as a continuous-time dynamical system.
"""

from typing import Any, Dict, Optional, Tuple, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from scipy.integrate import odeint
import numpy as np
import optax

from .noise_schedules import NoiseSchedule, LinearNoiseSchedule, LearnableNoiseSchedule
from .ode_integration import euler_step, heun_step, integrate_ode


@struct.dataclass
class NoPropCT:
    """Continuous-time NoProp implementation.
    
    This class implements the continuous-time variant where the denoising
    process is modeled as a neural ODE. The model learns a vector field
    that transforms noisy targets to clean ones over continuous time.
    """
    
    model: nn.Module
    num_timesteps: int = 1000
    noise_schedule: NoiseSchedule = struct.field(default_factory=LinearNoiseSchedule)
    integration_method: str = "euler"  # "euler" or "heun"
    eta: float = 1.0  # Hyperparameter from the paper
    
    def __post_init__(self):
        # Create timestep values for continuous time
        object.__setattr__(self, 'timesteps', jnp.linspace(0.0, 1.0, self.num_timesteps + 1))
    
    def dz_dt(
        self,
        params: Dict[str, Any],
        z: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the vector field dz/dt = f(z, x, t).
        
        Based on the PyTorch implementation, the model directly predicts the denoised target, but the forward
        process moves toward the target is a manner specified by the noise schedule.  In the original paper, 
        the forward process was incorrectly specified for continuous time approximations, and in the torch 
        implementation a completely different approach seems to have been used.  Here, we note that the forward 
        is simply given by: 

        dz/dt = alpha'(t)/alpha(t)/(1-alpha(t))*(sqrt(alpha(t))*target - (1+alpha(t))/2*z)
         
        The vector field is then computed as the difference between the predicted target and current state.
        
        Args:
            params: Model parameters
            z: Current state [batch_size, num_classes]
            x: Input data [batch_size, height, width, channels]
            t: Current time [batch_size]
            
        Returns:
            Vector field dz/dt [batch_size, num_classes]
        """
        # The model predicts the denoised target directly
        predicted_target = self.model.apply(params, z, x, t)
        
        alpha_t = self.noise_schedule.get_alpha_t(t, params)
        tau_inverse = self.noise_schedule.get_tau_inverse(t, params)
        
        return tau_inverse * (jnp.sqrt(alpha_t) * predicted_target - (1 + alpha_t) / 2.0 * z)
    
    def integrate_ode(
        self,
        params: Dict[str, Any],
        z0: jnp.ndarray,
        x: jnp.ndarray,
        t_span: Tuple[float, float],
        num_steps: int
    ) -> jnp.ndarray:
        """Integrate the neural ODE from t_start to t_end.
        
        Args:
            params: Model parameters
            z0: Initial state [batch_size, num_classes]
            x: Input data [batch_size, height, width, channels]
            t_span: Time span (t_start, t_end)
            num_steps: Number of integration steps
            
        Returns:
            Final state [batch_size, num_classes]
        """
        return integrate_ode(
            vector_field=self.dz_dt,
            params=params,
            z0=z0,
            x=x,
            time_span=t_span,
            num_steps=num_steps,
            method=self.integration_method
        )
    
    def compute_loss(
        self,
        params: Dict[str, Any],
        z_t: jnp.ndarray,
        x: jnp.ndarray,
        target: jnp.ndarray,
        t: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute the NoProp-CT loss according to the paper and PyTorch implementation.
        
        Based on the PyTorch implementation, the loss should be:
        L_CT = E[(1/SNR(t)) * ||model_output - z_t||²]
        
        where model_output is the predicted denoised target and SNR(t) is the signal-to-noise ratio.
        
        Args:
            params: Model parameters
            z_t: Noisy target at timestep t [batch_size, num_classes]
            x: Input data [batch_size, height, width, channels]
            target: Clean target [batch_size, num_classes]
            t: Timesteps [batch_size]
            key: Random key
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Compute the model output (predicted denoised target)
        model_output = self.model.apply(params, z_t, x, t)
        
        # Compute SNR weighting factor using SNR derivative
        snr_weight = self.noise_schedule.get_snr_prime(t, params)
        
        # Main loss: SNR-weighted MSE between model output and noisy input
        # This follows the NoProp paper where loss is weighted by SNR derivative
        squared_error = (model_output - target) ** 2
        mse = jnp.mean(squared_error)
        reg_loss = jnp.mean(model_output ** 2)

        ct_loss = jnp.mean(snr_weight[..., None] * squared_error)
        
        # total loss
        total_loss = ct_loss + self.eta * reg_loss

        # Compute additional metrics
        metrics = {
            "ct_loss": ct_loss,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
            "mse": mse,
            "snr_weight_mean": jnp.mean(snr_weight),
        }
        
        return total_loss, metrics
    
    
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
        batch_size = x.shape[0]
        
        # Sample random timesteps
        key, t_key = jax.random.split(key)
        t = jax.random.uniform(t_key, (batch_size,), minval=0.0, maxval=1.0)
        
        # Sample z_t from backward process
        key, noise_key = jax.random.split(key)
        z_t = self.noise_schedule.sample_zt(noise_key, target, t)
        
        # Compute loss and gradients
        (loss, metrics), grads = jax.value_and_grad(
            self.compute_loss, has_aux=True)(params, z_t, x, target, t, key)
        
        # Update parameters using optimizer
        updates, updated_opt_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        
        return updated_params, updated_opt_state, loss, metrics
    
    def generate(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        key: jax.random.PRNGKey,
        num_steps: Optional[int] = None
    ) -> jnp.ndarray:
        """Generate predictions using the trained NoProp-CT neural ODE.
        
        This integrates the learned vector field from pure noise (t=1)
        to the final prediction (t=0), following the paper's approach.
        
        Args:
            params: Model parameters
            x: Input data [batch_size, height, width, channels]
            key: Random key
            num_steps: Number of integration steps (default: 40 for evaluation)
            
        Returns:
            Final prediction [batch_size, num_classes]
        """
        if num_steps is None:
            num_steps = 40  # Default for evaluation as in the paper
            
        batch_size = x.shape[0]
        
        # Start with pure noise at t=1
        key, noise_key = jax.random.split(key)
        z0 = jax.random.normal(noise_key, (batch_size, self.model.num_classes))
        
        # Integrate the neural ODE from t=1 to t=0
        # The vector field was trained to point from z_t to the clean target
        z_final = self.integrate_ode(params, z0, x, (1.0, 0.0), num_steps)
        
        return z_final
    
    def evaluate(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        target: jnp.ndarray,
        key: jax.random.PRNGKey,
        num_steps: Optional[int] = None
    ) -> Dict[str, jnp.ndarray]:
        """Evaluate the model on a batch of data.
        
        Args:
            params: Model parameters
            x: Input data [batch_size, height, width, channels]
            target: Clean target [batch_size, num_classes]
            key: Random key
            num_steps: Number of integration steps for generation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Generate predictions
        pred = self.generate(params, x, key, num_steps)
        
        # Compute metrics
        mse = jnp.mean((pred - target) ** 2)
        
        # For classification, compute accuracy
        if target.ndim > 1 and target.shape[-1] > 1:
            pred_class = jnp.argmax(pred, axis=-1)
            target_class = jnp.argmax(target, axis=-1)
            accuracy = jnp.mean(pred_class == target_class)
        else:
            accuracy = jnp.nan
        
        return {
            "mse": mse,
            "accuracy": accuracy,
            "pred_norm": jnp.mean(jnp.linalg.norm(pred, axis=-1)),
            "target_norm": jnp.mean(jnp.linalg.norm(target, axis=-1)),
        }
