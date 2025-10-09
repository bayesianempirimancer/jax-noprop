"""
NoProp Model Wrapper for integrating learnable noise schedules.

This module provides a wrapper that combines the main model and learnable noise schedule
into a single parameter tree, ensuring all parameters are trained together.
"""

from typing import Any, Dict, Optional

import jax.numpy as jnp
import flax.linen as nn
from flax import struct

from .models import ConditionalResNet
from .noise_schedules import NoiseSchedule, LearnableNoiseSchedule


class NoPropModelWrapper(nn.Module):
    """Wrapper that combines model and learnable noise schedule parameters.
    
    This wrapper ensures that when using a learnable noise schedule, its parameters
    are included in the main model's parameter tree so they can be trained together.
    """
    
    model: nn.Module  # The main model (e.g., ConditionalResNet)
    noise_schedule: NoiseSchedule  # Noise schedule (can be learnable or fixed)
    
    def setup(self):
        """Setup the wrapper."""
        # If using a learnable noise schedule, create the network as a submodule
        if isinstance(self.noise_schedule, LearnableNoiseSchedule):
            # Create the gamma network as a submodule so its parameters are included
            self.gamma_network = self.noise_schedule.gamma_network
    
    @nn.compact
    def __call__(
        self, 
        z: jnp.ndarray, 
        x: jnp.ndarray, 
        t: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Forward pass through the model.
        
        Args:
            z: Noisy target [batch_size, z_dim]
            x: Input data [batch_size, height, width, channels]
            t: Time step [batch_size] (optional, for continuous-time variants)
            
        Returns:
            Model output [batch_size, z_dim]
        """
        # If using a learnable noise schedule, we need to use it in the forward pass
        # so its parameters are included in the parameter tree
        if isinstance(self.noise_schedule, LearnableNoiseSchedule):
            # Use the gamma network to ensure its parameters are included
            # This is a dummy call that doesn't affect the output but includes the params
            _ = self.gamma_network(t)
        
        # Forward pass through the main model
        model_output = self.model(z, x, t)
        return model_output
    
    def get_noise_schedule_params(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract noise schedule parameters from the combined parameter tree.
        
        Args:
            params: Combined parameter tree containing both model and noise schedule params
            
        Returns:
            Noise schedule parameters in the correct format for the noise schedule,
            or None if using a fixed schedule
        """
        if isinstance(self.noise_schedule, LearnableNoiseSchedule):
            # The noise schedule parameters will be under the gamma_network key
            # Format them correctly for the noise schedule
            noise_params = params["params"]["gamma_network"]
            return {"params": noise_params}
        else:
            return None
    
    def get_model_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model parameters from the combined parameter tree.
        
        Args:
            params: Combined parameter tree containing both model and noise schedule params
            
        Returns:
            Model parameters
        """
        return params["params"]["model"]
    
    def get_combined_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get the full combined parameter tree.
        
        Args:
            params: Combined parameter tree
            
        Returns:
            Full parameter tree
        """
        return params


def create_no_prop_model(
    model: nn.Module,
    noise_schedule: NoiseSchedule
) -> NoPropModelWrapper:
    """Factory function to create a NoProp model wrapper.
    
    Args:
        model: The main model (e.g., ConditionalResNet)
        noise_schedule: Noise schedule (can be learnable or fixed)
        
    Returns:
        NoPropModelWrapper instance
    """
    return NoPropModelWrapper(
        model=model,
        noise_schedule=noise_schedule
    )
