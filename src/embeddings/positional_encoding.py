"""
Positional encoding functions for transformer and sequence models.

This module contains various positional encoding implementations used in
transformer architectures and other sequence models.
"""

import jax.numpy as jnp
from typing import Optional


def positional_encoding(seq_len: int, d_model: int) -> jnp.ndarray:
    """Create positional encoding for transformer models.
    
    This implements the standard sinusoidal positional encoding used in
    transformer architectures. It creates unique position embeddings
    for each position in a sequence.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        
    Returns:
        Positional encoding [seq_len, d_model]
        
    Example:
        >>> pe = positional_encoding(10, 64)
        >>> print(pe.shape)  # (10, 64)
    """
    pos = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
    
    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(pos * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(pos * div_term))
    
    return pe


def relative_positional_encoding(seq_len: int, d_model: int, max_relative_position: int = 32) -> jnp.ndarray:
    """Create relative positional encoding.
    
    This implements relative positional encoding where the encoding
    depends on the relative distance between positions rather than
    absolute positions.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        max_relative_position: Maximum relative position to encode
        
    Returns:
        Relative positional encoding [seq_len, seq_len, d_model]
    """
    # Create relative position matrix
    relative_positions = jnp.arange(seq_len)[:, None] - jnp.arange(seq_len)[None, :]
    
    # Clip to max_relative_position
    relative_positions = jnp.clip(relative_positions, -max_relative_position, max_relative_position)
    
    # Create encoding for each relative position
    pe = jnp.zeros((seq_len, seq_len, d_model))
    
    for i in range(seq_len):
        for j in range(seq_len):
            rel_pos = relative_positions[i, j]
            pos = jnp.abs(rel_pos)
            div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
            
            pe = pe.at[i, j, 0::2].set(jnp.sin(pos * div_term))
            pe = pe.at[i, j, 1::2].set(jnp.cos(pos * div_term))
    
    return pe


def rotary_positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> jnp.ndarray:
    """Create rotary positional encoding (RoPE).
    
    This implements rotary positional encoding which rotates the query
    and key vectors based on their positions.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        base: Base for the frequency calculation
        
    Returns:
        Rotary positional encoding [seq_len, d_model]
    """
    # Create frequency tensor
    inv_freq = 1.0 / (base ** (jnp.arange(0, d_model, 2) / d_model))
    
    # Create position tensor
    pos = jnp.arange(seq_len)
    
    # Create angle tensor
    angle = pos[:, None] * inv_freq[None, :]
    
    # Create encoding
    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(angle))
    pe = pe.at[:, 1::2].set(jnp.cos(angle))
    
    return pe


def get_positional_encoding(seq_len: int, d_model: int, method: str = "sinusoidal") -> jnp.ndarray:
    """Get positional encoding using specified method.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        method: Method to use ("sinusoidal", "learnable", "relative", "rotary")
        
    Returns:
        Positional encoding [seq_len, d_model] or [seq_len, seq_len, d_model] for relative
    """
    if method == "sinusoidal":
        return positional_encoding(seq_len, d_model)
    elif method == "relative":
        return relative_positional_encoding(seq_len, d_model)
    elif method == "rotary":
        return rotary_positional_encoding(seq_len, d_model)
    else:
        raise ValueError(f"Unknown positional encoding method: {method}")
