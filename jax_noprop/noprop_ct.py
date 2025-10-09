"""NoProp Continuous Time (CT) implementation for conditional ResNets.

This module implements the continuous time version of NoProp for networks
that have layers taking (z, x, t) and outputting z'.
"""

from typing import Callable, Any, Tuple
import jax
import jax.numpy as jnp
from jax_noprop.base import NoPropBase


class NoPropCT(NoPropBase):
    """NoProp wrapper for continuous time conditional ResNets.
    
    This wrapper is designed for networks with layers that take (z, x, t) as input
    and produce z' as output, where:
    - z: hidden state
    - x: conditioning information
    - t: time variable
    - z': next hidden state
    
    The continuous time version treats the network as an ODE and uses noise
    injection to estimate gradients along the continuous trajectory.
    """
    
    def __init__(
        self,
        model: Callable,
        noise_scale: float = 0.01,
        learning_rate: float = 0.001,
        num_noise_samples: int = 2,
        time_steps: int = 10,
    ):
        """Initialize NoPropCT wrapper.
        
        Args:
            model: Callable that takes (params, z, x, t) and returns z'
            noise_scale: Scale of noise for gradient estimation
            learning_rate: Learning rate for updates
            num_noise_samples: Number of noise samples for gradient estimation
            time_steps: Number of time steps for integration
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
        """Forward pass with noise injection along time trajectory.
        
        Args:
            params: Model parameters
            inputs: Tuple of (z, x, t) where z is hidden state, x is conditioning,
                   and t is time
            rng: Random key for noise generation
            
        Returns:
            Tuple of (output, info_dict) with intermediate information
        """
        z, x, t = inputs
        rng_noise = jax.random.split(rng, 1)[0]
        
        # Add noise to input state
        noise = jax.random.normal(rng_noise, z.shape) * self.noise_scale
        z_noisy = z + noise
        
        # Forward pass through model
        output = self.model(params, z_noisy, x, t)
        
        info = {
            'noise': noise,
            'z_noisy': z_noisy,
            'z_clean': z,
            'time': t,
        }
        
        return output, info
    
    def integrate_trajectory(
        self,
        params: Any,
        z0: jnp.ndarray,
        x: jnp.ndarray,
        t0: float = 0.0,
        t1: float = 1.0,
        noise: jnp.ndarray = None,
        rng: jax.random.PRNGKey = None,
    ) -> jnp.ndarray:
        """Integrate the continuous dynamics from t0 to t1.
        
        Args:
            params: Model parameters
            z0: Initial state
            x: Conditioning information
            t0: Start time
            t1: End time
            noise: Optional noise to add to initial state
            rng: Optional random key for noise generation
            
        Returns:
            Final state z(t1)
        """
        if noise is not None:
            z = z0 + noise
        elif rng is not None:
            noise = jax.random.normal(rng, z0.shape) * self.noise_scale
            z = z0 + noise
        else:
            z = z0
        
        # Simple Euler integration
        dt = (t1 - t0) / self.time_steps
        t = t0
        
        for _ in range(self.time_steps):
            # Compute derivative dz/dt
            dz_dt = self.model(params, z, x, t)
            # Update state
            z = z + dt * dz_dt
            t = t + dt
        
        return z
    
    def compute_synthetic_gradient(
        self,
        params: Any,
        inputs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        loss_fn: Callable,
        rng: jax.random.PRNGKey,
    ) -> Any:
        """Compute synthetic gradients for continuous time dynamics.
        
        This method estimates gradients by:
        1. Integrating trajectories with noise perturbations
        2. Computing loss sensitivity to initial perturbations
        3. Estimating parameter gradients via finite differences
        
        Args:
            params: Model parameters
            inputs: Tuple of (z0, x, t) where z0 is initial state
            loss_fn: Loss function that takes final state and returns scalar loss
            rng: Random key
            
        Returns:
            Synthetic gradients for params
        """
        z0, x, t = inputs
        
        def loss_with_noise(params, noise):
            """Evaluate loss with specific noise perturbation."""
            # Integrate trajectory with noise
            z_final = self.integrate_trajectory(
                params, z0, x, t0=0.0, t1=t, noise=noise
            )
            return loss_fn(z_final)
        
        # Generate multiple noise samples
        rngs = jax.random.split(rng, self.num_noise_samples)
        noises = jax.vmap(
            lambda r: jax.random.normal(r, z0.shape) * self.noise_scale
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
        loss_fn: Callable,
        rng: jax.random.PRNGKey,
    ) -> Tuple[Any, dict]:
        """Perform a single training step.
        
        Args:
            params: Current parameters
            batch: Batch of (z0, x, t) data
            loss_fn: Loss function
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
        z_final = self.integrate_trajectory(params, z0, x, t0=0.0, t1=t)
        loss = loss_fn(z_final)
        
        metrics = {
            'loss': loss,
        }
        
        return new_params, metrics
