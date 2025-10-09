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
import optax

from .noise_schedules import NoiseSchedule, LinearNoiseSchedule


@struct.dataclass
class NoPropDT:
    """Discrete-time NoProp implementation.
    
    This class implements the discrete-time variant where we have a fixed
    number of timesteps and each layer learns to denoise independently.
    """
    
    model: nn.Module
    num_timesteps: int = 20
    noise_schedule: NoiseSchedule = struct.field(default_factory=LinearNoiseSchedule)
    eta: float = 0.1  # Hyperparameter from the paper
    
    def __post_init__(self):
        # Create timestep values
        object.__setattr__(self, 'timesteps', jnp.linspace(0.0, 1.0, self.num_timesteps + 1))

    
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
        opt_state: optax.OptState,
        x: jnp.ndarray,
        target: jnp.ndarray,
        key: jax.random.PRNGKey,
        optimizer: optax.GradientTransformation
    ) -> Tuple[Dict[str, Any], optax.OptState, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Single training step for NoProp-DT.
        
        Key insight from the paper: NoProp does NOT require a forward pass during training.
        Each layer is trained independently to denoise a noisy target.
        
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
        
        # Sample random timesteps (excluding t=0)
        key, t_key = jax.random.split(key)
        t = jax.random.choice(t_key, self.timesteps[1:], shape=(batch_size,))
        
        # Add noise to target according to the noise schedule
        key, noise_key = jax.random.split(key)
        z_t = self.noise_schedule.sample_zt(noise_key, target, t)
        
        # Compute loss and gradients
        # Note: No forward pass through the network is needed!
        # The model directly learns to map (z_t, x) -> target
        (loss, metrics), grads = jax.value_and_grad(
            self.compute_loss, has_aux=True
        )(params, z_t, x, target, t, key)
        
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
        z = jax.random.normal(noise_key, (batch_size, self.model.num_classes))
        
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
