"""
NoProp-FM: Flow Matching NoProp implementation.

This module implements the flow matching variant of NoProp, which is inspired
by flow matching methods. The key idea is to learn a vector field that
transforms a simple distribution (e.g., Gaussian) to the target distribution.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import optax

from .noise_schedules import NoiseSchedule, LinearNoiseSchedule
from .ode_integration import euler_step, heun_step, integrate_flow


@struct.dataclass
class NoPropFM:
    """Flow Matching NoProp implementation.
    
    This class implements the flow matching variant where we learn a vector field
    that transforms a simple base distribution to the target distribution.
    The flow is parameterized by a neural network that takes (z, x, t) as input.
    """
    
    model: nn.Module
    num_timesteps: int = 1000
    noise_schedule: NoiseSchedule = struct.field(default_factory=LinearNoiseSchedule)
    integration_method: str = "euler"  # "euler" or "heun"
    eta: float = 1.0  # Hyperparameter from the paper
    
    def __post_init__(self):
        # Create timestep values for continuous time
        object.__setattr__(self, 'timesteps', jnp.linspace(0.0, 1.0, self.num_timesteps + 1))
    
    
    
    def sample_base_distribution(
        self, 
        key: jax.random.PRNGKey, 
        shape: Tuple[int, ...]
    ) -> jnp.ndarray:
        """Sample from the base distribution (Gaussian noise).
        
        Args:
            key: Random key
            shape: Shape of the samples
            
        Returns:
            Samples from base distribution
        """
        return jax.random.normal(key, shape)
    
    def interpolate_path(
        self,
        z0: jnp.ndarray,
        z1: jnp.ndarray,
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Interpolate between z0 and z1 at time t.
        
        This implements the linear interpolation path used in flow matching:
        z_t = (1 - t) * z0 + t * z1
        
        Args:
            z0: Base distribution samples [batch_size, ...]
            z1: Target samples [batch_size, ...]
            t: Time values [batch_size]
            
        Returns:
            Interpolated samples [batch_size, ...]
        """
        # Ensure proper broadcasting
        if t.ndim == 0:
            t = t[None]
        elif t.ndim == 1:
            t = t[:, None]
        
        # Reshape t for broadcasting across all dimensions
        while t.ndim < z0.ndim:
            t = t[..., None]
        
        return (1 - t) * z0 + t * z1
    
    def compute_flow_matching_loss(
        self,
        params: Dict[str, Any],
        z0: jnp.ndarray,
        z1: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute the NoProp-FM loss according to the paper and PyTorch implementation.
        
        Based on the PyTorch implementation, the model should predict the noise ε
        that transforms the base distribution z0 to the target z1.
        
        The loss is: L_FM = E[||ε_pred - ε_true||²]
        where ε_true = z1 - z0 (the transformation from base to target)
        
        Args:
            params: Model parameters
            z0: Base distribution samples [batch_size, num_classes]
            z1: Target samples [batch_size, num_classes]
            x: Input data [batch_size, height, width, channels]
            t: Timesteps [batch_size]
            key: Random key
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Interpolate between z0 and z1 at time t
        z_t = self.interpolate_path(z0, z1, t)
        
        # Compute the predicted noise/transformation
        noise_pred = self.model.apply(params, z_t, x, t)
        
        # The true transformation is the difference between target and base
        noise_true = z1 - z0
        
        # Main loss: MSE between predicted and true transformation
        main_loss = jnp.mean((noise_pred - noise_true) ** 2)
        
        # Regularization term: L2 norm of the predicted transformation
        reg_loss = jnp.mean(noise_pred ** 2)
        
        # Total loss with regularization weight η
        total_loss = main_loss + self.eta * reg_loss
        
        # Compute additional metrics
        metrics = {
            "loss": total_loss,
            "main_loss": main_loss,
            "reg_loss": reg_loss,
            "transformation_mse": main_loss,
            "pred_norm": jnp.mean(jnp.linalg.norm(noise_pred, axis=-1)),
            "true_norm": jnp.mean(jnp.linalg.norm(noise_true, axis=-1)),
        }
        
        return total_loss, metrics
    
    def integrate_flow(
        self,
        params: Dict[str, Any],
        z0: jnp.ndarray,
        x: jnp.ndarray,
        num_steps: int
    ) -> jnp.ndarray:
        """Integrate the flow from base distribution to target.
        
        Args:
            params: Model parameters
            z0: Initial state (base distribution) [batch_size, num_classes]
            x: Input data [batch_size, height, width, channels]
            num_steps: Number of integration steps
            
        Returns:
            Final state [batch_size, num_classes]
        """
        # Define vector field function for flow matching
        def vector_field(params, z, x, t):
            return self.model.apply(params, z, x, t)
        
        return integrate_flow(
            vector_field=vector_field,
            params=params,
            z0=z0,
            x=x,
            time_span=(0.0, 1.0),
            num_steps=num_steps,
            method=self.integration_method
        )
    
    def train_step(
        self,
        params: Dict[str, Any],
        opt_state: optax.OptState,
        x: jnp.ndarray,
        target: jnp.ndarray,
        key: jax.random.PRNGKey,
        optimizer: optax.GradientTransformation
    ) -> Tuple[Dict[str, Any], optax.OptState, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Single training step for NoProp-FM.
        
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
        
        # Sample from base distribution
        key, z0_key = jax.random.split(key)
        z0 = self.sample_base_distribution(z0_key, target.shape)
        
        # Use target as z1
        z1 = target
        
        # Compute loss and gradients
        (loss, metrics), grads = jax.value_and_grad(
            self.compute_flow_matching_loss, has_aux=True
        )(params, z0, z1, x, t, key)
        
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
        """Generate predictions using the trained flow matching model.
        
        This integrates the learned vector field from the base distribution
        to generate samples from the target distribution.
        
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
        
        # Start with samples from base distribution
        z0 = self.sample_base_distribution(key, (batch_size, self.model.num_classes))
        
        # Integrate the flow
        z_final = self.integrate_flow(params, z0, x, num_steps)
        
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
