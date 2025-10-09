"""
Embedding utilities for NoProp implementations.

This module provides various embedding functions used in the NoProp algorithm,
including time embeddings and positional encodings.
"""

from typing import Optional

import jax
import jax.numpy as jnp


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
    if t.ndim == 1:
        t = t[:, None]
    
    # Create frequency bands with logarithmic spacing
    half_dim = dim // 2
    emb = jnp.log(10000.0) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    
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


def fourier_features(x: jnp.ndarray, num_features: int, scale: float = 1.0) -> jnp.ndarray:
    """Create Fourier features for input x.
    
    This is an alternative embedding approach that can be used for
    encoding continuous variables. It projects the input to a higher
    dimensional space using random Fourier features.
    
    Args:
        x: Input values [batch_size, input_dim]
        num_features: Number of Fourier features to generate
        scale: Scaling factor for the random frequencies
        
    Returns:
        Fourier features [batch_size, num_features]
    """
    # Generate random frequencies
    key = jax.random.PRNGKey(42)  # Fixed key for reproducibility
    freqs = jax.random.normal(key, (x.shape[-1], num_features // 2)) * scale
    
    # Apply frequencies
    x_proj = 2 * jnp.pi * x @ freqs
    
    # Create sin and cos features
    sin_features = jnp.sin(x_proj)
    cos_features = jnp.cos(x_proj)
    
    # Concatenate and return
    return jnp.concatenate([sin_features, cos_features], axis=-1)


def positional_encoding(seq_len: int, dim: int) -> jnp.ndarray:
    """Create positional encoding for sequences.
    
    This implements the standard transformer positional encoding
    that can be used for sequence-based inputs in NoProp variants.
    
    Args:
        seq_len: Length of the sequence
        dim: Embedding dimension
        
    Returns:
        Positional encoding [seq_len, dim]
    """
    pos = jnp.arange(seq_len)[:, None]
    dim_indices = jnp.arange(dim)[None, :]
    
    # Create frequency bands
    div_term = jnp.exp(dim_indices // 2 * -jnp.log(10000.0) / dim)
    
    # Apply frequencies
    pe = pos * div_term
    
    # Create sin and cos encodings
    pe = pe.at[:, 0::2].set(jnp.sin(pe[:, 0::2]))
    pe = pe.at[:, 1::2].set(jnp.cos(pe[:, 1::2]))
    
    return pe


def learnable_time_embedding(t: jnp.ndarray, dim: int, max_timesteps: int = 1000) -> jnp.ndarray:
    """Create learnable time embeddings.
    
    This is an alternative to sinusoidal embeddings that uses
    learnable parameters. It can be useful when the time dynamics
    are complex and need to be learned rather than fixed.
    
    Args:
        t: Time values [batch_size]
        dim: Embedding dimension
        max_timesteps: Maximum number of timesteps for the lookup table
        
    Returns:
        Time embeddings [batch_size, dim]
        
    Note:
        This function requires a learnable embedding table to be
        defined in the model. It's provided here as an alternative
        approach but requires model modifications to use.
    """
    # This would require a learnable embedding table in the model
    # For now, we'll use a simple linear projection as a placeholder
    t_norm = t / max_timesteps
    t_expanded = jnp.tile(t_norm[:, None], (1, dim))
    
    # Apply learnable transformation (this would be a model parameter)
    # For demonstration, we use a simple scaling
    return t_expanded * jnp.linspace(0, 1, dim)[None, :]


def get_time_embedding(t: jnp.ndarray, dim: int, method: str = "sinusoidal") -> jnp.ndarray:
    """Get time embedding using the specified method.
    
    This is a convenience function that allows switching between
    different time embedding methods.
    
    Args:
        t: Time values [batch_size]
        dim: Embedding dimension
        method: Embedding method ("sinusoidal", "fourier", "learnable")
        
    Returns:
        Time embeddings [batch_size, dim]
        
    Raises:
        ValueError: If method is not supported
    """
    if method == "sinusoidal":
        return sinusoidal_time_embedding(t, dim)
    elif method == "fourier":
        return fourier_features(t[:, None], dim)
    elif method == "learnable":
        return learnable_time_embedding(t, dim)
    else:
        raise ValueError(f"Unsupported embedding method: {method}")


# Default embedding configurations
DEFAULT_TIME_EMBED_DIMS = {
    "resnet": 128,
    "simple_cnn": 64,
    "transformer": 256,
}

DEFAULT_EMBEDDING_METHOD = "sinusoidal"
