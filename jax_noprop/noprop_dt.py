"""NoProp Discrete Time (DT) implementation for conditional ResNets.

This module implements the discrete time version of NoProp for networks
that have layers taking (z, x) and outputting z'.
"""

from typing import Callable, Any, Tuple
import jax
import jax.numpy as jnp
from jax_noprop.base import NoPropBase


class NoPropDT(NoPropBase):
    """NoProp wrapper for discrete time conditional ResNets.
    
    This wrapper is designed for networks with layers that take (z, x) as input
    and produce z' as output, where:
    - z: hidden state
    - x: conditioning information
    - z': next hidden state
    
    The discrete time version processes the network layer-by-layer, injecting
    noise at each layer and computing synthetic gradients based on the loss
    sensitivity to these perturbations.
    """
    
    def __init__(
        self,
        model: Callable,
        noise_scale: float = 0.01,
        learning_rate: float = 0.001,
        num_noise_samples: int = 2,
    ):
        """Initialize NoPropDT wrapper.
        
        Args:
            model: Callable that takes (params, z, x) and returns z'
            noise_scale: Scale of noise for gradient estimation
            learning_rate: Learning rate for updates
            num_noise_samples: Number of noise samples for gradient estimation
        """
        super().__init__(model, noise_scale, learning_rate)
        self.num_noise_samples = num_noise_samples
    
    def forward_with_noise(
        self,
        params: Any,
        inputs: Tuple[jnp.ndarray, jnp.ndarray],
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, dict]:
        """Forward pass with noise injection at each layer.
        
        Args:
            params: Model parameters
            inputs: Tuple of (z, x) where z is hidden state and x is conditioning
            rng: Random key for noise generation
            
        Returns:
            Tuple of (output, info_dict) where info_dict contains intermediate states
        """
        z, x = inputs
        rng_noise = jax.random.split(rng, 1)[0]
        
        # Add noise to input state
        noise = jax.random.normal(rng_noise, z.shape) * self.noise_scale
        z_noisy = z + noise
        
        # Forward pass through model
        output = self.model(params, z_noisy, x)
        
        info = {
            'noise': noise,
            'z_noisy': z_noisy,
            'z_clean': z,
        }
        
        return output, info
    
    def compute_synthetic_gradient(
        self,
        params: Any,
        inputs: Tuple[jnp.ndarray, jnp.ndarray],
        loss_fn: Callable,
        rng: jax.random.PRNGKey,
    ) -> Any:
        """Compute synthetic gradients using finite differences with noise.
        
        This method estimates gradients by:
        1. Evaluating loss with multiple noise perturbations
        2. Computing finite differences of loss w.r.t. noise
        3. Propagating these to parameter gradients
        
        Args:
            params: Model parameters
            inputs: Tuple of (z, x)
            loss_fn: Loss function that takes model output and returns scalar loss
            rng: Random key
            
        Returns:
            Synthetic gradients for params
        """
        z, x = inputs
        
        def loss_with_noise(params, noise):
            """Evaluate loss with specific noise perturbation."""
            z_noisy = z + noise
            output = self.model(params, z_noisy, x)
            return loss_fn(output)
        
        # Generate multiple noise samples
        rngs = jax.random.split(rng, self.num_noise_samples)
        noises = jax.vmap(
            lambda r: jax.random.normal(r, z.shape) * self.noise_scale
        )(rngs)
        
        # Compute loss for positive and negative perturbations
        def compute_gradient_estimate(noise):
            # Forward difference approximation
            loss_pos = loss_with_noise(params, noise)
            loss_neg = loss_with_noise(params, -noise)
            
            # Gradient estimate via finite differences
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
        batch: Tuple[jnp.ndarray, jnp.ndarray],
        loss_fn: Callable,
        rng: jax.random.PRNGKey,
    ) -> Tuple[Any, dict]:
        """Perform a single training step.
        
        Args:
            params: Current parameters
            batch: Batch of (z, x) data
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
        z, x = batch
        output = self.model(params, z, x)
        loss = loss_fn(output)
        
        metrics = {
            'loss': loss,
        }
        
        return new_params, metrics
