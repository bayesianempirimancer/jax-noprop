"""
Time embedding implementations for neural networks.

This module provides standardized time embedding functions for continuous-time
neural networks, particularly useful for flow-based models and NoProp training.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Union, Callable

class ConstantTimeEmbedding(nn.Module):
    """
    Constant time embedding that returns a constant value.
    
    This effectively disables temporal information by returning
    the same embedding regardless of the input time value.
    """
    
    embed_dim: int
    
    def __call__(self, t: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Create constant time embedding.
        
        Args:
            t: Time value(s) ∈ [0, 1] (ignored)
            
        Returns:
            Constant embedding [embed_dim,] or [batch_shape..., embed_dim]
        """
        # Convert scalar to (1,) array to unify handling
        is_scalar = isinstance(t, float) or (hasattr(t, 'ndim') and t.ndim == 0)
        if is_scalar:
            t = jnp.array(t)[None]
        
        # Create constant embeddings: batch_shape + (embed_dim,)
        embeddings = jnp.ones(t.shape + (self.embed_dim,))
        
        # Squeeze out the batch dimension if input was scalar
        return embeddings

class LinearTimeEmbedding(nn.Module):
    embed_dim: int
    
    def __call__(self, t: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Create linear time embedding by relu of t-thresh[0:embed_dim]
        
        Args:
            t: Time value(s) ∈ [0, 1]
            
        Returns:
            Time embedding [embed_dim,] or [batch_shape..., embed_dim]
        """
        # Convert scalar to (1,) array to unify handling
        is_scalar = isinstance(t, float) or (hasattr(t, 'ndim') and t.ndim == 0)
        if is_scalar:
            t = jnp.array(t)[None]
        
        # Repeat t along the last dimension to create embeddings
        thresh  = jnp.linspace(0, 1.0-1.0/self.embed_dim, self.embed_dim)
        embeddings = jnp.repeat(t[..., None]-thresh, self.embed_dim, axis=-1)
        embedding = nn.relu(embeddings)
        
        # Squeeze out the batch dimension if input was scalar
        return embeddings

class CyclicalFourierTimeEmbedding(nn.Module):
    """
    Fourier-based time embedding using sinusoidal functions integer frequencies.  
    This assumes that there is a max period of the signal, T_max
    
    Creates embeddings using sin and cos functions with different frequencies.
    """

    embed_dim: int
    T_max: float = 1.0

    def __call__(self, t: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Create cyclical Fourier time embedding.
        
        Args:
            t: Time value(s) ∈ [0, 1]
            
        Returns:
            Time embedding [embed_dim,] or [batch_shape..., embed_dim]
        """
        if self.embed_dim%2 != 0:
            raise ValueError("Cyclical Fourier time embedding requires embed_dim to be even")
        
        # Convert scalar to (1,) array to unify handling
        is_scalar = isinstance(t, float) or (hasattr(t, 'ndim') and t.ndim == 0)
        if is_scalar:
            t = jnp.array(t)
        
        n_freqs = self.embed_dim//2
        freqs = jnp.linspace(0, 2*jnp.pi/self.T_max*n_freqs, n_freqs)
        sin_embeddings = jnp.sin(2 * jnp.pi * freqs * t[..., None])
        cos_embeddings = jnp.cos(2 * jnp.pi * freqs * t[..., None])
        embeddings = jnp.concatenate([sin_embeddings, cos_embeddings], axis=-1)
        
        # Squeeze out the batch dimension if input was scalar
        return embeddings

class SinusoidalTimeEmbedding(nn.Module):
    embed_dim: int

    def __call__(self, t: Union[float, jnp.ndarray]) -> jnp.ndarray:

        is_scalar = isinstance(t, float) or (hasattr(t, 'ndim') and t.ndim == 0)
        if is_scalar:
            t = jnp.array(t)

        half = self.embed_dim // 2
        log_freqs = -jnp.log(10000) * jnp.linspace(0, 1, half)        
        freqs = 0.5*jnp.pi*jnp.exp(log_freqs)
        return jnp.concatenate([jnp.sin(t[..., None] * freqs), jnp.cos(t[..., None] * freqs)], axis=-1)

class LogFreqTimeEmbedding(nn.Module):
    """
    Fourier-based time embedding using sinusoidal functions.
    
    Creates embeddings using sin and cos functions with different frequencies.
    """

    embed_dim: int
    min_freq: Optional[float] = 0.1
    max_freq: Optional[float] = 10

    def __call__(self, t: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Create Fourier time embedding.
        
        Args:
            t: Time value(s) ∈ [0, 1]
            
        Returns:
            Time embedding [embed_dim,] or [batch_shape..., embed_dim]
        """
        if self.embed_dim%2 != 0:
            raise ValueError("Fourier time embedding requires embed_dim to be even")

        if self.max_freq is None: 
            self.max_freq = self.embed_dim//2

        # Convert scalar to (1,) array to unify handling
        is_scalar = isinstance(t, float) or (hasattr(t, 'ndim') and t.ndim == 0)
        if is_scalar:
            t = jnp.array(t)
        
        # Create frequency schedule
        n_freqs = self.embed_dim//2
        log_freqs = jnp.linspace(jnp.log(self.min_freq), jnp.log(self.max_freq), n_freqs)
        freqs = jnp.exp(log_freqs)
        
        # Create sin and cos embeddings
        # freqs has shape (n_freqs,), t has shape (batch_shape,)
        # We need to broadcast: (n_freqs,) * (batch_shape,) -> (batch_shape, n_freqs)
        sin_embeddings = jnp.sin(2 * jnp.pi * freqs * t[..., None])
        cos_embeddings = jnp.cos(2 * jnp.pi * freqs * t[..., None])        
        # Combine sin and cos: (batch_shape, n_freqs) -> (batch_shape, 2*n_freqs)
        embeddings = jnp.concatenate([sin_embeddings, cos_embeddings], axis=-1)
        
        # Squeeze out the batch dimension if input was scalar
        return embeddings
        

# Convenience functions for creating time embeddings
def create_time_embedding(embed_dim: int, 
                         method: str,
                         min_freq: float = 0.1,
                         max_freq: float = 10.0,
                         T_max: float = 1.0):
    """
    Create a time embedding instance.
    
    Args:
        embed_dim: Dimension of the time embedding
        method: Method to use ("fourier", "log_freq", "cyclical_fourier", "sinusoidal", "linear", "constant")
        min_freq: Minimum frequency for log frequency embeddings
        max_freq: Maximum frequency for Fourier embeddings
        T_max: Maximum period for cyclical Fourier embeddings
        
    Returns:
        TimeEmbedding instance
    """
    if method == "fourier" or method == "log_freq":
        return LogFreqTimeEmbedding(
            embed_dim=embed_dim,
            min_freq=min_freq,
            max_freq=max_freq
        )
    elif method == "cyclical_fourier":
        return CyclicalFourierTimeEmbedding(
            embed_dim=embed_dim,
            T_max=T_max
        )
    elif method == "sinusoidal":
        return SinusoidalTimeEmbedding(
            embed_dim=embed_dim
        )
    elif method == "linear" or method == "simple":
        return LinearTimeEmbedding(
            embed_dim=embed_dim
        )
    elif method == "constant":
        return ConstantTimeEmbedding(
            embed_dim=embed_dim
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fourier', 'log_freq', 'cyclical_fourier', 'sinusoidal', 'linear', 'simple', or 'constant'")

# Example usage and testing
if __name__ == "__main__":
    import jax
    
    print("Testing time embeddings:")
    
    # Test with single time value
    t_single = 0.5
    print(f"\nSingle time value: {t_single}")
    
    # Test Fourier embedding
    fourier_embed = fourier_time_embedding(t_single, embed_dim=8, max_freq=5.0)
    print(f"Fourier embedding shape: {fourier_embed.shape}")
    print(f"Fourier embedding values: {fourier_embed}")
    
    # Test simple embedding
    simple_embed = simple_time_embedding(t_single, embed_dim=8)
    print(f"Simple embedding shape: {simple_embed.shape}")
    print(f"Simple embedding values: {simple_embed}")
    
    # Test with batch of time values
    t_batch = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"\nBatch time values: {t_batch}")
    
    fourier_batch = fourier_time_embedding(t_batch, embed_dim=6, max_freq=3.0)
    print(f"Fourier batch shape: {fourier_batch.shape}")
    print(f"Fourier batch values:\n{fourier_batch}")
    
    # Test class-based usage
    print(f"\nClass-based usage:")
    embedding = create_time_embedding(embed_dim=4, method="fourier")
    class_embed = embedding(t_single)
    print(f"Class embedding shape: {class_embed.shape}")
    print(f"Class embedding values: {class_embed}")
