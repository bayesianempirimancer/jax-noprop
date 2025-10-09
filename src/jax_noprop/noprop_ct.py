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

from .noise_schedules import NoiseSchedule, LinearNoiseSchedule, add_noise, sample_noise
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
    
    def sample_timestep(self, key: jax.random.PRNGKey, batch_size: int) -> jnp.ndarray:
        """Sample random timesteps for training."""
        return jax.random.uniform(key, (batch_size,), minval=0.0, maxval=1.0)
    
    def get_noise_params(self, t: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Get noise parameters for given timesteps."""
        return self.noise_schedule.get_noise_params(t)
    
    def add_noise_to_target(
        self, 
        target: jnp.ndarray, 
        key: jax.random.PRNGKey,
        t: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Add noise to the target according to the schedule.
        
        Args:
            target: Clean target [batch_size, num_classes]
            key: Random key
            t: Timesteps [batch_size]
            
        Returns:
            Tuple of (noisy_target, noise)
        """
        noise = sample_noise(key, target.shape)
        noise_params = self.get_noise_params(t)
        
        noisy_target = add_noise(
            target, 
            noise, 
            noise_params["alpha_t"], 
            noise_params["sigma_t"]
        )
        
        return noisy_target, noise
    
    def vector_field(
        self,
        params: Dict[str, Any],
        z: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the vector field dz/dt = f(z, x, t).
        
        Based on the PyTorch implementation, the model directly predicts the denoised target.
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
        
        # The vector field is the difference between predicted target and current state
        # This represents the direction the state should move
        return predicted_target - z
    
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
            vector_field=self.vector_field,
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
        
        # Compute SNR weighting factor (1/SNR(t))
        snr_weight = self._compute_snr_inverse(t)
        
        # Main loss: SNR-weighted MSE between model output and noisy input
        # This follows the PyTorch implementation pattern
        main_loss = jnp.mean(snr_weight * (model_output - z_t) ** 2)
        
        # Regularization term: L2 norm of the model output
        reg_loss = jnp.mean(model_output ** 2)
        
        # Total loss with regularization weight η
        total_loss = main_loss + self.eta * reg_loss
        
        # Compute additional metrics
        metrics = {
            "loss": total_loss,
            "main_loss": main_loss,
            "reg_loss": reg_loss,
            "model_output_mse": jnp.mean((model_output - z_t) ** 2),
            "target_mse": jnp.mean((model_output - target) ** 2),
            "snr_weight_mean": jnp.mean(snr_weight),
            "pred_norm": jnp.mean(jnp.linalg.norm(model_output, axis=-1)),
            "target_norm": jnp.mean(jnp.linalg.norm(target, axis=-1)),
        }
        
        return total_loss, metrics
    
    def _compute_snr_inverse(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute the inverse SNR weighting factor for the loss.
        
        Based on the PyTorch implementation, the loss should be weighted by 1/SNR(t).
        For the linear noise schedule: α_t = 1 - t, σ_t = sqrt(t)
        SNR(t) = α_t² / σ_t² = (1-t)² / t
        1/SNR(t) = t / (1-t)²
        
        Args:
            t: Timesteps [batch_size]
            
        Returns:
            Inverse SNR weight [batch_size]
        """
        # Avoid division by zero and ensure t < 1
        t_safe = jnp.maximum(jnp.minimum(t, 0.999), 1e-8)
        
        # For linear schedule: 1/SNR(t) = t / (1-t)²
        snr_inverse = t_safe / ((1 - t_safe) ** 2)
        
        # Normalize to have mean 1 for stability
        snr_inverse = snr_inverse / jnp.mean(snr_inverse)
        
        return snr_inverse
    
    def train_step(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        target: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[Dict[str, Any], jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Single training step for NoProp-CT.
        
        Args:
            params: Model parameters
            x: Input data [batch_size, height, width, channels]
            target: Clean target [batch_size, num_classes]
            key: Random key
            
        Returns:
            Tuple of (updated_params, loss, metrics)
        """
        batch_size = x.shape[0]
        
        # Sample random timesteps
        key, t_key = jax.random.split(key)
        t = self.sample_timestep(t_key, batch_size)
        
        # Add noise to target
        key, noise_key = jax.random.split(key)
        z_t, noise = self.add_noise_to_target(target, noise_key, t)
        
        # Compute loss and gradients
        (loss, metrics), grads = jax.value_and_grad(
            self.compute_loss, has_aux=True
        )(params, z_t, x, target, t, key)
        
        # Update parameters (this would typically be done by an optimizer)
        # For now, we just return the gradients
        updated_params = grads
        
        return updated_params, loss, metrics
    
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
        z0 = sample_noise(noise_key, (batch_size, self.model.num_classes))
        
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
