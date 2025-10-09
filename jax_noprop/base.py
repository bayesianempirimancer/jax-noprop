"""Base class for NoProp wrappers"""

from typing import Callable, Any, Tuple
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from flax import linen as nn


class NoPropBase(ABC):
    """Base class for NoProp algorithm wrappers.
    
    NoProp is a gradient-free training approach that works by:
    1. Adding noise to layer activations during forward pass
    2. Computing synthetic gradients based on the noise perturbations
    3. Using these synthetic gradients to update parameters
    
    This base class provides common functionality for all NoProp variations.
    """
    
    def __init__(
        self,
        model: Callable,
        noise_scale: float = 0.01,
        learning_rate: float = 0.001,
    ):
        """Initialize NoProp wrapper.
        
        Args:
            model: The neural network model to wrap
            noise_scale: Scale of noise to inject for gradient estimation
            learning_rate: Learning rate for parameter updates
        """
        self.model = model
        self.noise_scale = noise_scale
        self.learning_rate = learning_rate
    
    @abstractmethod
    def forward_with_noise(
        self,
        params: Any,
        inputs: Tuple[jnp.ndarray, ...],
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, Any]:
        """Forward pass with noise injection.
        
        Args:
            params: Model parameters
            inputs: Input data tuple
            rng: Random key for noise generation
            
        Returns:
            Tuple of (output, auxiliary_info)
        """
        pass
    
    @abstractmethod
    def compute_synthetic_gradient(
        self,
        params: Any,
        inputs: Tuple[jnp.ndarray, ...],
        loss_fn: Callable,
        rng: jax.random.PRNGKey,
    ) -> Any:
        """Compute synthetic gradients using noise perturbations.
        
        Args:
            params: Model parameters
            inputs: Input data tuple
            loss_fn: Loss function to optimize
            rng: Random key for noise generation
            
        Returns:
            Synthetic gradients for parameters
        """
        pass
    
    def update_params(
        self,
        params: Any,
        gradients: Any,
    ) -> Any:
        """Update parameters using synthetic gradients.
        
        Args:
            params: Current parameters
            gradients: Synthetic gradients
            
        Returns:
            Updated parameters
        """
        return jax.tree.map(
            lambda p, g: p - self.learning_rate * g,
            params,
            gradients
        )
