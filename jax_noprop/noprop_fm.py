"""NoProp Flow Matching (FM) implementation for conditional ResNets.

This module implements the flow matching version of NoProp for networks
that learn to match velocity fields in continuous normalizing flows.
"""

from typing import Callable, Any, Tuple
import jax
import jax.numpy as jnp
from jax_noprop.base import NoPropBase


class NoPropFM(NoPropBase):
    """NoProp wrapper for flow matching with conditional ResNets.
    
    This wrapper is designed for networks that learn velocity fields for
    continuous normalizing flows. The network takes (z, x, t) as input
    and produces a velocity v(z, x, t), where:
    - z: current state
    - x: conditioning information (e.g., target/data)
    - t: time in [0, 1]
    - v: velocity field
    
    Flow matching trains the network to match a target velocity field that
    interpolates between a base distribution and the data distribution.
    """
    
    def __init__(
        self,
        model: Callable,
        noise_scale: float = 0.01,
        learning_rate: float = 0.001,
        num_noise_samples: int = 2,
        time_steps: int = 10,
    ):
        """Initialize NoPropFM wrapper.
        
        Args:
            model: Callable that takes (params, z, x, t) and returns velocity v
            noise_scale: Scale of noise for gradient estimation
            learning_rate: Learning rate for updates
            num_noise_samples: Number of noise samples for gradient estimation
            time_steps: Number of time steps for flow integration
        """
        super().__init__(model, noise_scale, learning_rate)
        self.num_noise_samples = num_noise_samples
        self.time_steps = time_steps
    
    def forward_with_noise(
        self,
        params: Any,
        inputs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, dict]:
        """Forward pass with noise injection.
        
        Args:
            params: Model parameters
            inputs: Tuple of (z, x, t) where z is current state, x is target/data,
                   and t is time
            rng: Random key for noise generation
            
        Returns:
            Tuple of (velocity, info_dict) with intermediate information
        """
        z, x, t = inputs
        rng_noise = jax.random.split(rng, 1)[0]
        
        # Add noise to current state
        noise = jax.random.normal(rng_noise, z.shape) * self.noise_scale
        z_noisy = z + noise
        
        # Predict velocity field
        velocity = self.model(params, z_noisy, x, t)
        
        info = {
            'noise': noise,
            'z_noisy': z_noisy,
            'z_clean': z,
            'time': t,
        }
        
        return velocity, info
    
    def sample_conditional_flow(
        self,
        params: Any,
        z0: jnp.ndarray,
        x: jnp.ndarray,
        rng: jax.random.PRNGKey = None,
    ) -> jnp.ndarray:
        """Sample from the conditional flow.
        
        Integrates the learned velocity field from t=0 to t=1 to generate
        samples conditioned on x.
        
        Args:
            params: Model parameters
            z0: Initial state (typically from base distribution)
            x: Conditioning information
            rng: Optional random key for noise injection
            
        Returns:
            Final state z(t=1)
        """
        if rng is not None:
            noise = jax.random.normal(rng, z0.shape) * self.noise_scale
            z = z0 + noise
        else:
            z = z0
        
        # Integrate flow from t=0 to t=1
        dt = 1.0 / self.time_steps
        t = 0.0
        
        for _ in range(self.time_steps):
            # Get velocity at current time
            v = self.model(params, z, x, t)
            # Euler step
            z = z + dt * v
            t = t + dt
        
        return z
    
    def compute_target_velocity(
        self,
        z_t: jnp.ndarray,
        x: jnp.ndarray,
        t: float,
    ) -> jnp.ndarray:
        """Compute target velocity for flow matching.
        
        For conditional flow matching, the target velocity is typically:
        v_t(z_t) = (x - z_t) / (1 - t)
        
        This creates a flow that interpolates linearly from z_0 to x.
        
        Args:
            z_t: State at time t
            x: Target state (data)
            t: Current time in [0, 1]
            
        Returns:
            Target velocity
        """
        # Avoid division by zero at t=1
        eps = 1e-5
        return (x - z_t) / jnp.maximum(1.0 - t, eps)
    
    def compute_synthetic_gradient(
        self,
        params: Any,
        inputs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        loss_fn: Callable = None,
        rng: jax.random.PRNGKey = None,
    ) -> Any:
        """Compute synthetic gradients for flow matching.
        
        This method estimates gradients by:
        1. Sampling time points and interpolated states
        2. Computing predicted vs target velocities
        3. Estimating parameter gradients via perturbations
        
        Args:
            params: Model parameters
            inputs: Tuple of (z0, x, t) where z0 is base state, x is target
            loss_fn: Optional custom loss function (default: velocity MSE)
            rng: Random key
            
        Returns:
            Synthetic gradients for params
        """
        z0, x, t = inputs
        
        # Compute interpolated state at time t
        # Linear interpolation: z_t = (1 - t) * z0 + t * x
        z_t = (1.0 - t) * z0 + t * x
        
        # Default loss: MSE between predicted and target velocity
        if loss_fn is None:
            def default_loss(velocity):
                target_velocity = self.compute_target_velocity(z_t, x, t)
                return jnp.mean((velocity - target_velocity) ** 2)
            loss_fn = default_loss
        
        def loss_with_noise(params, noise):
            """Evaluate loss with noise perturbation."""
            z_t_noisy = z_t + noise
            velocity = self.model(params, z_t_noisy, x, t)
            return loss_fn(velocity)
        
        # Generate noise samples
        rngs = jax.random.split(rng, self.num_noise_samples)
        noises = jax.vmap(
            lambda r: jax.random.normal(r, z_t.shape) * self.noise_scale
        )(rngs)
        
        # Compute gradient estimates
        def compute_gradient_estimate(noise):
            grad_params = jax.grad(loss_with_noise, argnums=0)(params, noise)
            return grad_params
        
        # Average over noise samples
        grad_estimates = jax.vmap(compute_gradient_estimate)(noises)
        
        # Average the gradients
        synthetic_grad = jax.tree.map(
            lambda x: jnp.mean(x, axis=0),
            grad_estimates
        )
        
        return synthetic_grad
    
    def train_step(
        self,
        params: Any,
        batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        loss_fn: Callable = None,
        rng: jax.random.PRNGKey = None,
    ) -> Tuple[Any, dict]:
        """Perform a single training step.
        
        Args:
            params: Current parameters
            batch: Batch of (z0, x, t) data where z0 is base distribution sample,
                   x is target/data, and t is time
            loss_fn: Optional custom loss function
            rng: Random key
            
        Returns:
            Tuple of (updated_params, metrics_dict)
        """
        # Compute synthetic gradients
        gradients = self.compute_synthetic_gradient(params, batch, loss_fn, rng)
        
        # Update parameters
        new_params = self.update_params(params, gradients)
        
        # Compute loss for monitoring
        z0, x, t = batch
        z_t = (1.0 - t) * z0 + t * x
        velocity = self.model(params, z_t, x, t)
        target_velocity = self.compute_target_velocity(z_t, x, t)
        loss = jnp.mean((velocity - target_velocity) ** 2)
        
        metrics = {
            'loss': loss,
            'velocity_norm': jnp.mean(jnp.linalg.norm(velocity, axis=-1)),
        }
        
        return new_params, metrics
