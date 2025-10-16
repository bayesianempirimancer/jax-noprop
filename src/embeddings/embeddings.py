"""
Embedding utilities for NoProp implementations.

This module provides various embedding functions used in the NoProp algorithm,
time embeddings assume t is in [0, 1].
"""

from typing import Optional

import jax.numpy as jnp
import flax.linen as nn


def sinusoidal_time_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Create sinusoidal time embeddings as used in the NoProp paper.
    
    This implements the sinusoidal positional encoding used in transformer models
    and adapted for time embeddings in diffusion models. The embedding uses
    multiple frequency bands with logarithmic spacing to capture temporal
    information effectively.
    
    Args:
        t: Time values [batch_size] or [batch_size, 1]
        dim: Embedding dimension (must be even for proper sin/cos pairing)
        
    Returns:
        Time embeddings [batch_size, dim]
        
    Example:
        >>> t = jnp.array([0.0, 0.5, 1.0])
        >>> emb = sinusoidal_time_embedding(t, 64)
        >>> print(emb.shape)  # (3, 64)
    """
    # Ensure t is 2D
    if t.ndim == 0:
        t = jnp.array([t])[:, None] # has shape [1, 1]
    elif t.ndim == 1:
        t = t[:, None]
    
    # Create frequency bands with logarithmic spacing
    half_dim = dim // 2
    emb = jnp.log(10000.0) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)*2*jnp.pi
    
    # Apply frequencies to time
    emb = t * emb[None, :]  # [batch_size, half_dim]
    
    # Create sinusoidal embeddings
    sin_emb = jnp.sin(emb)
    cos_emb = jnp.cos(emb)
    
    # Concatenate sin and cos embeddings
    time_emb = jnp.concatenate([sin_emb, cos_emb], axis=-1)
    
    # Pad if dim is odd
    if dim % 2 == 1:
        time_emb = jnp.pad(time_emb, ((0, 0), (0, 1)), mode='constant')
    
    return time_emb

def linear_time_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Create linear time embeddings.
    
    This is a simple linear embedding that scales time values to fill
    the embedding dimension. It's the most basic form of time embedding.
    
    Args:
        t: Time values [batch_size]
        dim: Embedding dimension
        
    Returns:
        Time embeddings [batch_size, dim]
    """
    # Ensure t is 2D
    if t.ndim == 0:
        t = jnp.array([t])[:, None] # has shape [1, 1]
    elif t.ndim == 1:
        t = t[:, None]
    
    # Create linear scaling across the embedding dimension
    # Scale time from [0, 1] to [0, dim-1] and create a linear ramp
    
    # Create linear embeddings by repeating the scaled time
    # and adding a linear ramp across the dimension
    
    # Add a linear ramp across the embedding dimension
    ramp = jnp.linspace(0, 1, dim)[None, :]
    time_emb = nn.relu(t - ramp)
    
    return time_emb

def fourier_time_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Create Fourier time embeddings.
    
    This is a Fourier embedding that projects time values to a higher
    dimensional space using random frequencies.
    """
    if t.ndim == 0:
        t = t[:, None]
    elif t.ndim == 1:
        t = t[:, None]
        
    freqs = jnp.linspace(0, dim//2, dim)
    sin_embed = jnp.sin(jnp.pi*t * freqs)
    cos_embed = jnp.cos(jnp.pi*t * freqs)
    return jnp.concatenate([sin_embed, cos_embed], axis=-1)

def get_time_embedding(t: jnp.ndarray, dim: int, method: str = "sinusoidal") -> jnp.ndarray:
    """Get time embedding using the specified method.
    
    This is a convenience function that allows switching between
    different time embedding methods.
    
    Args:
        t: Time values [batch_size]
        dim: Embedding dimension
        method: Embedding method ("sinusoidal", "fourier", "linear", "learnable", "gaussian")
        
    Returns:
        Time embeddings [batch_size, dim]
        
    Raises:
        ValueError: If method is not supported
    """
    if method == "sinusoidal":
        return sinusoidal_time_embedding(t, dim)
    elif method == "fourier":
        return fourier_time_embedding(t[:, None], dim)
    elif method == "linear":
        return linear_time_embedding(t, dim)
    elif method == "learnable":
        return learnable_time_embedding(t, dim)
    elif method == "gaussian":
        return gaussian_time_embedding(t, dim)
    else:
        raise ValueError(f"Unsupported embedding method: {method}. "
                        f"Supported methods: sinusoidal, fourier, linear, learnable, gaussian")


def gaussian_time_embedding(t: jnp.ndarray, dim: int, sigma: float = 1.0) -> jnp.ndarray:
    """Create Gaussian time embeddings.
    
    This creates time embeddings using Gaussian basis functions
    centered at different time points. This can be useful for
    capturing temporal smoothness.
    
    Args:
        t: Time values [batch_size] or [batch_size, 1]
        dim: Embedding dimension
        sigma: Standard deviation of Gaussian basis functions
        
    Returns:
        Time embeddings [batch_size, dim]
    """
    # Ensure t is 2D
    if t.ndim == 0:
        t = jnp.array([t])[:, None]
    elif t.ndim == 1:
        t = t[:, None]
    
    # Create Gaussian centers
    centers = jnp.linspace(0, 1, dim)
    
    # Compute Gaussian activations
    t_expanded = t  # [batch_size, 1]
    centers_expanded = centers[None, :]  # [1, dim]
    
    # Gaussian: exp(-(t - center)^2 / (2 * sigma^2))
    diff = t_expanded - centers_expanded
    gaussian_emb = jnp.exp(-(diff ** 2) / (2 * sigma ** 2))
    
    return gaussian_emb
