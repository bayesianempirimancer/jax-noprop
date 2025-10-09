"""Example conditional ResNet implementations for use with NoProp wrappers."""

from typing import Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn


class ConditionalResNetDT(nn.Module):
    """Example conditional ResNet for discrete time NoProp.
    
    Takes (z, x) as input and produces z' as output.
    """
    hidden_dims: Sequence[int] = (64, 64)
    output_dim: int = 32
    
    @nn.compact
    def __call__(self, z, x):
        """Forward pass.
        
        Args:
            z: Hidden state, shape (batch, z_dim)
            x: Conditioning information, shape (batch, x_dim)
            
        Returns:
            z': Next hidden state, shape (batch, output_dim)
        """
        # Concatenate z and x
        h = jnp.concatenate([z, x], axis=-1)
        
        # Process through ResNet blocks
        for dim in self.hidden_dims:
            # Store residual
            residual = h
            
            # Dense layer with activation
            h = nn.Dense(dim)(h)
            h = nn.relu(h)
            h = nn.Dense(dim)(h)
            
            # Residual connection (with projection if needed)
            if residual.shape[-1] != h.shape[-1]:
                residual = nn.Dense(dim)(residual)
            h = h + residual
            h = nn.relu(h)
        
        # Output layer
        z_next = nn.Dense(self.output_dim)(h)
        
        return z_next


class ConditionalResNetCT(nn.Module):
    """Example conditional ResNet for continuous time NoProp.
    
    Takes (z, x, t) as input and produces dz/dt as output.
    """
    hidden_dims: Sequence[int] = (64, 64)
    output_dim: int = 32
    
    @nn.compact
    def __call__(self, z, x, t):
        """Forward pass.
        
        Args:
            z: Hidden state, shape (batch, z_dim)
            x: Conditioning information, shape (batch, x_dim)
            t: Time variable, shape (batch, 1) or scalar
            
        Returns:
            dz/dt: Time derivative of state, shape (batch, output_dim)
        """
        # Ensure t is a JAX array
        t = jnp.asarray(t)
        
        # Ensure t has batch dimension
        if t.ndim == 0:
            t = jnp.expand_dims(t, 0)
        if t.shape[-1] != 1:
            t = jnp.expand_dims(t, -1)
        
        # Broadcast t to match batch size if needed
        if t.shape[0] == 1 and z.shape[0] > 1:
            t = jnp.broadcast_to(t, (z.shape[0], 1))
        
        # Concatenate z, x, and t
        h = jnp.concatenate([z, x, t], axis=-1)
        
        # Process through ResNet blocks
        for dim in self.hidden_dims:
            residual = h
            
            h = nn.Dense(dim)(h)
            h = nn.relu(h)
            h = nn.Dense(dim)(h)
            
            if residual.shape[-1] != h.shape[-1]:
                residual = nn.Dense(dim)(residual)
            h = h + residual
            h = nn.relu(h)
        
        # Output layer - represents dz/dt
        dz_dt = nn.Dense(self.output_dim)(h)
        
        return dz_dt


class ConditionalResNetFM(nn.Module):
    """Example conditional ResNet for flow matching NoProp.
    
    Takes (z, x, t) as input and produces velocity v as output.
    """
    hidden_dims: Sequence[int] = (64, 64)
    output_dim: int = 32
    
    @nn.compact
    def __call__(self, z, x, t):
        """Forward pass.
        
        Args:
            z: Current state, shape (batch, z_dim)
            x: Target/conditioning information, shape (batch, x_dim)
            t: Time in [0, 1], shape (batch, 1) or scalar
            
        Returns:
            v: Velocity field, shape (batch, output_dim)
        """
        # Ensure t is a JAX array
        t = jnp.asarray(t)
        
        # Ensure t has batch dimension
        if t.ndim == 0:
            t = jnp.expand_dims(t, 0)
        if t.shape[-1] != 1:
            t = jnp.expand_dims(t, -1)
        
        # Broadcast t to match batch size if needed
        if t.shape[0] == 1 and z.shape[0] > 1:
            t = jnp.broadcast_to(t, (z.shape[0], 1))
        
        # Time embedding (sinusoidal)
        freq = 2.0 * jnp.pi
        t_emb = jnp.concatenate([
            jnp.sin(freq * t),
            jnp.cos(freq * t)
        ], axis=-1)
        
        # Concatenate z, x, and time embedding
        h = jnp.concatenate([z, x, t_emb], axis=-1)
        
        # Process through ResNet blocks
        for dim in self.hidden_dims:
            residual = h
            
            h = nn.Dense(dim)(h)
            h = nn.relu(h)
            h = nn.Dense(dim)(h)
            
            if residual.shape[-1] != h.shape[-1]:
                residual = nn.Dense(dim)(residual)
            h = h + residual
            h = nn.relu(h)
        
        # Output layer - represents velocity
        velocity = nn.Dense(self.output_dim)(h)
        
        return velocity
