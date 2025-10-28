"""
Gradient wrapper models for Conditional ResNets.

This module provides simplified gradient wrapper models that wrap CRN models from crn_wip.py
to create different types of flows (potential, geometric, natural, hamiltonian).
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from src.layers.gradnet_utils import GradNetUtility, GradAndHessNetUtility
from src.flow_models.crn import create_conditional_resnet


# ============================================================================
# UNIFIED FACTORY FUNCTION
# ============================================================================

def create_gradient_wrapper(wrapper_type: str, config_dict: dict, latent_shape: Tuple[int, ...], input_shape: Tuple[int, ...], output_shape: Optional[Tuple[int, ...]] = None) -> nn.Module:
    """
    Unified factory function for creating gradient wrapper models.
    
    Args:
        wrapper_type: Type of wrapper ("potential", "geometric", "natural", "hamiltonian")
        config_dict: Configuration dictionary for the underlying CRN
        latent_shape: Shape of the latent state (z)
        input_shape: Shape of the conditional input (x)
        output_shape: Shape of the output (optional, defaults to latent_shape)
        
    Returns:
        Instantiated gradient wrapper model
    """
    # Create the underlying CRN using the factory from crn_wip.py
    crn_model = create_conditional_resnet(config_dict, latent_shape=latent_shape, input_shape=input_shape, output_shape=output_shape)
    
    # Create the appropriate wrapper
    if wrapper_type == "potential":
        return PotentialFlow(crn_model=crn_model, latent_shape=latent_shape, input_shape=input_shape, output_shape=output_shape)
    elif wrapper_type == "geometric":
        return GeometricFlow(crn_model=crn_model, latent_shape=latent_shape, input_shape=input_shape, output_shape=output_shape)
    elif wrapper_type == "natural":
        return NaturalFlow(crn_model=crn_model, latent_shape=latent_shape, input_shape=input_shape, output_shape=output_shape)
    elif wrapper_type == "hamiltonian":
        return HamiltonianFlow(crn_model=crn_model, latent_shape=latent_shape, input_shape=input_shape, output_shape=output_shape)
    else:
        raise ValueError(f"Unknown wrapper_type: {wrapper_type}. Supported types: potential, geometric, natural, hamiltonian")


# ============================================================================
# GRADIENT WRAPPER CLASSES
# ============================================================================

class PotentialFlow(nn.Module):
    """
    Wrapper that converts any conditional ResNet into a potential flow.
    
    The flow is computed as dz/dt = -∇_z V(z,x,t), where V is the potential function
    defined by the underlying CRN.
    
    Args:
        crn_model: The underlying CRN model instance
    """
    crn_model: nn.Module
    latent_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    
    def setup(self):
        """Initialize the gradient utility."""
        # Pre-compute dimensions to avoid JAX tracing issues
        self.latent_dim = int(np.prod(self.latent_shape))
        self.input_dim = int(np.prod(self.input_shape))
        
        # Create ResNet factory function that uses the wrapper's parameters
        def resnet_factory(z_input, x_input, t_input, training=True):
            output = self.crn_model(z_input, x_input, t_input, training=training)
            # Reduce output to scalar for gradient computation
            return jnp.sum(output, axis=-1)
        
        # Create gradient utility for computing gradients
        self.grad_utility = GradNetUtility(resnet_factory, reduction_method="sum")
    
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, 
                 training: bool = True, rngs=None) -> jnp.ndarray:
        """Compute the potential flow dz/dt = -∇_z V(z,x,t).
        
        Args:
            z: Current state [batch_size, ...latent_shape]
            x: Conditional input [batch_size, ...input_shape] 
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Flow dz/dt [batch_size, ...latent_shape]
        """
        # Flatten inputs for gradient computation
        batch_shape_z = z.shape[:-len(self.latent_shape)]
        batch_shape_x = x.shape[:-len(self.input_shape)]
        batch_shape = jnp.broadcast_shapes(batch_shape_z, batch_shape_x, t.shape if t is not None else ())
        
        z_flat = z.reshape(-1, self.latent_dim)
        x_flat = x.reshape(-1, self.input_dim)
        
        # Compute gradients using the pre-created utility class
        gradients_flat = self.grad_utility(self.variables, z_flat, x_flat, t, training=training, rngs=rngs)
        
        # Reshape gradients back to original shape
        gradients = gradients_flat.reshape(batch_shape + self.latent_shape)
        
        # Return negative gradient (potential flow)
        return -gradients


class GeometricFlow(nn.Module):
    """
    Wrapper that converts any conditional ResNet into a geometric flow.
    
    The flow is computed as dz/dt = Hessian @ x, where the Hessian is computed
    from the underlying CRN.
    
    Args:
        crn_model: The underlying CRN model instance
    """
    crn_model: nn.Module
    latent_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    
    def setup(self):
        """Initialize the Hessian utility."""
        # Pre-compute dimensions to avoid JAX tracing issues
        self.latent_dim = int(np.prod(self.latent_shape))
        self.input_dim = int(np.prod(self.input_shape))
        
        # Create ResNet factory function that uses the wrapper's parameters
        def resnet_factory(z_input, x_input, t_input, training=True):
            output = self.crn_model(z_input, x_input, t_input, training=training)
            # Reduce output to scalar for gradient computation
            return jnp.sum(output, axis=-1)
        
        # Create Hessian utility for computing Hessians
        self.hess_utility = GradAndHessNetUtility(resnet_factory, reduction_method="sum")
    
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, 
                 training: bool = True, rngs=None) -> jnp.ndarray:
        """Compute the geometric flow dz/dt = Hessian @ x.
        
        Args:
            z: Current state [batch_size, ...latent_shape]
            x: Conditional input [batch_size, ...input_shape] 
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Flow dz/dt [batch_size, ...latent_shape]
        """
        # Check dimension consistency - input_dim must match latent_dim for geometric flow
        assert self.input_dim == self.latent_dim, f"input_dim ({self.input_dim}) must match latent_dim ({self.latent_dim}) for geometric flow"
        
        # Flatten inputs for gradient computation
        batch_shape_z = z.shape[:-len(self.latent_shape)]
        batch_shape_x = x.shape[:-len(self.input_shape)]
        batch_shape = jnp.broadcast_shapes(batch_shape_z, batch_shape_x, t.shape if t is not None else ())
        
        z_flat = z.reshape(-1, self.latent_dim)
        x_flat = x.reshape(-1, self.input_dim)
        
        # Compute Hessians using the pre-created utility class
        _, hessians = self.hess_utility(self.variables, z_flat, x_flat, t, training=training, rngs=rngs)
        
        # Compute geometric flow: dz/dt = Hessian @ x
        dz_dt_flat = jnp.einsum("...ij, ...j -> ...i", hessians, x_flat)
        
        # Reshape back to original shape
        dz_dt = dz_dt_flat.reshape(batch_shape + self.latent_shape)
        
        return dz_dt


class NaturalFlow(nn.Module):
    """
    Wrapper that converts any conditional ResNet into a natural flow.
    
    The flow is computed using natural gradient information.
    
    Args:
        crn_model: The underlying CRN model instance
    """
    crn_model: nn.Module
    latent_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]

    
    def setup(self):
        """Initialize the gradient utility."""
        # Pre-compute dimensions to avoid JAX tracing issues
        self.latent_dim = int(np.prod(self.latent_shape))
        self.input_dim = int(np.prod(self.input_shape))
        
        # Create ResNet factory function that uses the wrapper's parameters
        def resnet_factory(z_input, x_input, t_input, training=True):
            output = self.crn_model(z_input, x_input, t_input, training=training)
            # Reduce output to scalar for gradient computation
            return jnp.sum(output, axis=-1)
        
        # Create gradient utility for computing gradients
        self.grad_utility = GradNetUtility(resnet_factory, reduction_method="sum")
    
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, 
                 training: bool = True, rngs=None) -> jnp.ndarray:
        """Compute the natural flow.
        
        Args:
            z: Current state [batch_size, ...latent_shape]
            x: Conditional input [batch_size, ...input_shape] 
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Flow dz/dt [batch_size, ...latent_shape]
        """
        # Flatten inputs for gradient computation
        batch_shape_z = z.shape[:-len(self.latent_shape)]
        batch_shape_x = x.shape[:-len(self.input_shape)]
        batch_shape = jnp.broadcast_shapes(batch_shape_z, batch_shape_x, t.shape if t is not None else ())
        
        z_flat = z.reshape(-1, self.latent_dim)
        x_flat = x.reshape(-1, self.input_dim)
        
        # For now, natural flow is implemented as potential flow
        # This can be extended with proper natural gradient computation
        gradients_flat = self.grad_utility(self.variables, z_flat, x_flat, t, training=training, rngs=rngs)
        
        # Reshape gradients back to original shape
        gradients = gradients_flat.reshape(batch_shape + self.latent_shape)
        
        return -gradients


class HamiltonianFlow(nn.Module):
    """
    Wrapper that converts any conditional ResNet into a Hamiltonian flow.
    
    The flow is computed using Hamiltonian dynamics.
    
    Args:
        crn_model: The underlying CRN model instance
    """
    crn_model: nn.Module
    latent_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]

    
    def setup(self):
        """Initialize the gradient utility."""
        # Pre-compute dimensions to avoid JAX tracing issues
        self.latent_dim = int(np.prod(self.latent_shape))
        self.input_dim = int(np.prod(self.input_shape))
        
        # Create ResNet factory function that uses the wrapper's parameters
        def resnet_factory(z_input, x_input, t_input, training=True):
            output = self.crn_model(z_input, x_input, t_input, training=training)
            # Reduce output to scalar for gradient computation
            return jnp.sum(output, axis=-1)
        
        # Create gradient utility for computing gradients
        self.grad_utility = GradNetUtility(resnet_factory, reduction_method="sum")
    
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, 
                 training: bool = True, rngs=None) -> jnp.ndarray:
        """Compute the Hamiltonian flow.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim] 
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Flow dz/dt [batch_size, z_dim]
        """
        # Flatten inputs for gradient computation
        batch_shape_z = z.shape[:-len(self.latent_shape)]
        batch_shape_x = x.shape[:-len(self.input_shape)]
        batch_shape = jnp.broadcast_shapes(batch_shape_z, batch_shape_x, t.shape if t is not None else ())
        
        z_flat = z.reshape(-1, self.latent_dim)
        x_flat = x.reshape(-1, self.input_dim)
        
        # For now, Hamiltonian flow is implemented as potential flow
        # This can be extended with proper Hamiltonian dynamics
        gradients_flat = self.grad_utility(self.variables, z_flat, x_flat, t, training=training, rngs=rngs)
        
        dHdq, dHdp = jnp.split(gradients_flat, 2, axis=-1)
        
        # Reshape back to original shape
        gradients = jnp.concatenate([dHdp, -dHdq], axis=-1)
        return gradients.reshape(batch_shape + self.latent_shape)
