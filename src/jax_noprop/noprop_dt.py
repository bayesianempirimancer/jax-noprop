"""
NoProp-DT: Discrete-time NoProp implementation.

This module implements the discrete-time variant of NoProp as described in the paper.
The key idea is to train each layer independently to denoise a noisy target,
without relying on back-propagation through the network.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct

from .noise_schedules import NoiseSchedule, LinearNoiseSchedule, add_noise, sample_noise


@struct.dataclass
class NoPropDT:
    """Discrete-time NoProp implementation.
    
    This class implements the discrete-time variant where we have a fixed
    number of timesteps and each layer learns to denoise independently.
    """
    
    model: nn.Module
    num_timesteps: int = 10
    noise_schedule: NoiseSchedule = struct.field(default_factory=LinearNoiseSchedule)
    eta: float = 0.1  # Hyperparameter from the paper
    
    def __post_init__(self):
        # Create timestep values
        object.__setattr__(self, 'timesteps', jnp.linspace(0.0, 1.0, self.num_timesteps + 1))
    
    def sample_timestep(self, key: jax.random.PRNGKey, batch_size: int) -> jnp.ndarray:
        """Sample random timesteps for training."""
        return jax.random.choice(
            key, 
            self.timesteps[1:],  # Exclude t=0
            shape=(batch_size,)
        )
    
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
    
    def compute_loss(
        self,
        params: Dict[str, Any],
        z_t: jnp.ndarray,
        x: jnp.ndarray,
        target: jnp.ndarray,
        t: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute the NoProp-DT loss according to the paper and PyTorch implementation.
        
        Based on the PyTorch implementation, the model should predict the noise ε
        that was added to the clean target to create the noisy target z_t.
        
        The loss is: L_DT = E[||ε_pred - ε_true||²]
        where ε_true = z_t - target (the actual noise added)
        
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
        # Forward pass through the model - predicts the noise
        noise_pred = self.model.apply(params, z_t, x)
        
        # The true noise is the difference between noisy and clean target
        noise_true = z_t - target
        
        # Main loss: MSE between predicted and true noise
        main_loss = jnp.mean((noise_pred - noise_true) ** 2)
        
        # Regularization term: L2 norm of the predicted noise
        reg_loss = jnp.mean(noise_pred ** 2)
        
        # Total loss with regularization weight η
        total_loss = main_loss + self.eta * reg_loss
        
        # Compute additional metrics
        metrics = {
            "loss": total_loss,
            "main_loss": main_loss,
            "reg_loss": reg_loss,
            "noise_mse": main_loss,
            "pred_norm": jnp.mean(jnp.linalg.norm(noise_pred, axis=-1)),
            "true_noise_norm": jnp.mean(jnp.linalg.norm(noise_true, axis=-1)),
        }
        
        return total_loss, metrics
    
    def train_step(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        target: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[Dict[str, Any], jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Single training step for NoProp-DT.
        
        Key insight from the paper: NoProp does NOT require a forward pass during training.
        Each layer is trained independently to denoise a noisy target.
        
        Args:
            params: Model parameters
            x: Input data [batch_size, height, width, channels]
            target: Clean target [batch_size, num_classes]
            key: Random key
            
        Returns:
            Tuple of (updated_params, loss, metrics)
        """
        batch_size = x.shape[0]
        
        # Sample random timesteps (excluding t=0)
        key, t_key = jax.random.split(key)
        t = self.sample_timestep(t_key, batch_size)
        
        # Add noise to target according to the noise schedule
        key, noise_key = jax.random.split(key)
        z_t, noise = self.add_noise_to_target(target, noise_key, t)
        
        # Compute loss and gradients
        # Note: No forward pass through the network is needed!
        # The model directly learns to map (z_t, x) -> target
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
        """Generate predictions using the trained NoProp-DT model.
        
        This implements the denoising process by iteratively predicting and removing noise,
        following the PyTorch implementation approach.
        
        Args:
            params: Model parameters
            x: Input data [batch_size, height, width, channels]
            key: Random key
            num_steps: Number of denoising steps (default: num_timesteps)
            
        Returns:
            Final prediction [batch_size, num_classes]
        """
        if num_steps is None:
            num_steps = self.num_timesteps
            
        batch_size = x.shape[0]
        
        # Start with pure noise (t=1)
        key, noise_key = jax.random.split(key)
        z = sample_noise(noise_key, (batch_size, self.model.num_classes))
        
        # Iterative denoising from t=1 to t=0
        for i in range(num_steps):
            # Current timestep (going backwards from 1 to 0)
            t = jnp.full((batch_size,), 1.0 - (i + 1) / num_steps)
            
            # Predict the noise
            noise_pred = self.model.apply(params, z, x)
            
            # Remove the predicted noise to get cleaner target
            z = z - noise_pred
        
        return z
    
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
            num_steps: Number of denoising steps for generation
            
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
