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

class FourierTimeEmbedding(nn.Module):
    """
    Simple Fourier time embedding using evenly spaced frequencies.
    
    This creates embeddings using sin and cos functions with evenly spaced
    frequencies from 0 to dim//2.
    """
    
    embed_dim: int
    
    def __call__(self, t: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Create Fourier time embedding.
        
        Args:
            t: Time value(s) ∈ [0, 1]
            
        Returns:
            Time embedding [embed_dim,] or [batch_shape..., embed_dim]
        """
        # Convert scalar to (1,) array to unify handling
        # Match original embeddings.py behavior: expects t to be 2D [batch, 1]
        is_scalar = isinstance(t, float) or (hasattr(t, 'ndim') and t.ndim == 0)
        if is_scalar:
            t = jnp.array([t])[:, None]  # shape [1, 1]
        elif t.ndim == 1:
            t = t[:, None]  # shape [batch, 1]
        
        # Create evenly spaced frequencies (matching embeddings.py behavior)
        # Note: This creates 2*embed_dim output (sin + cos), matching original implementation
        freqs = jnp.linspace(0, self.embed_dim // 2, self.embed_dim)
        
        # Apply frequencies to time: t has shape (batch_shape, 1), freqs has shape (embed_dim,)
        # Result: (batch_shape, embed_dim) for sin, (batch_shape, embed_dim) for cos
        # Final: (batch_shape, 2*embed_dim)
        sin_embed = jnp.sin(jnp.pi * t * freqs)
        cos_embed = jnp.cos(jnp.pi * t * freqs)
        
        embeddings = jnp.concatenate([sin_embed, cos_embed], axis=-1)
        
        # Squeeze out the batch dimension if input was scalar
        return embeddings

class GaussianTimeEmbedding(nn.Module):
    """
    Gaussian time embedding using Gaussian basis functions.
    
    This creates time embeddings using Gaussian basis functions centered
    at different time points. This can be useful for capturing temporal smoothness.
    """
    
    embed_dim: int
    sigma: float = 1.0
    
    def __call__(self, t: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Create Gaussian time embedding.
        
        Args:
            t: Time value(s) ∈ [0, 1]
            
        Returns:
            Time embedding [embed_dim,] or [batch_shape..., embed_dim]
        """
        # Convert scalar to (1,) array to unify handling
        is_scalar = isinstance(t, float) or (hasattr(t, 'ndim') and t.ndim == 0)
        if is_scalar:
            t = jnp.array(t)[None]
        elif t.ndim == 1:
            t = t[:, None]
        
        # Create Gaussian centers evenly spaced from 0 to 1
        centers = jnp.linspace(0, 1, self.embed_dim)
        
        # Compute Gaussian activations
        # t has shape (batch_shape, 1), centers has shape (embed_dim,)
        # Broadcast: (batch_shape, 1) - (1, embed_dim) -> (batch_shape, embed_dim)
        t_expanded = t  # [batch_shape, 1]
        centers_expanded = centers[None, :]  # [1, embed_dim]
        
        # Gaussian: exp(-(t - center)^2 / (2 * sigma^2))
        diff = t_expanded - centers_expanded
        gaussian_emb = jnp.exp(-(diff ** 2) / (2 * self.sigma ** 2))
        
        # Squeeze out the batch dimension if input was scalar
        return gaussian_emb
        

# Convenience functions for creating time embeddings
def create_time_embedding(embed_dim: int, 
                         method: str,
                         min_freq: float = 0.1,
                         max_freq: float = 10.0,
                         T_max: float = 1.0,
                         sigma: float = 1.0):
    """
    Create a time embedding instance.
    
    Args:
        embed_dim: Dimension of the time embedding
        method: Method to use ("fourier", "log_freq", "cyclical_fourier", "sinusoidal", "linear", "constant", "gaussian")
        min_freq: Minimum frequency for log frequency embeddings
        max_freq: Maximum frequency for Fourier embeddings
        T_max: Maximum period for cyclical Fourier embeddings
        sigma: Standard deviation for Gaussian embeddings
        
    Returns:
        TimeEmbedding instance
    """
    if method == "log_freq":
        return LogFreqTimeEmbedding(
            embed_dim=embed_dim,
            min_freq=min_freq,
            max_freq=max_freq
        )
    elif method == "fourier":
        return FourierTimeEmbedding(
            embed_dim=embed_dim
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
    elif method == "gaussian":
        return GaussianTimeEmbedding(
            embed_dim=embed_dim,
            sigma=sigma
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fourier', 'log_freq', 'cyclical_fourier', 'sinusoidal', 'linear', 'simple', 'constant', or 'gaussian'")

# Backward compatibility wrapper functions (matching embeddings.py API)
def sinusoidal_time_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Create sinusoidal time embeddings as used in the NoProp paper.
    
    This is a wrapper function for backward compatibility.
    See SinusoidalTimeEmbedding class for details.
    
    Args:
        t: Time values [batch_size] or [batch_size, 1]
        dim: Embedding dimension (must be even for proper sin/cos pairing)
        
    Returns:
        Time embeddings [batch_size, dim]
    """
    embedding = SinusoidalTimeEmbedding(embed_dim=dim)
    return embedding(t)

def linear_time_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Create linear time embeddings.
    
    This is a wrapper function for backward compatibility.
    See LinearTimeEmbedding class for details.
    
    Args:
        t: Time values [batch_size]
        dim: Embedding dimension
        
    Returns:
        Time embeddings [batch_size, dim]
    """
    embedding = LinearTimeEmbedding(embed_dim=dim)
    return embedding(t)

def fourier_time_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Create Fourier time embeddings.
    
    This is a wrapper function for backward compatibility.
    See FourierTimeEmbedding class for details.
    
    Note: This returns 2*dim dimensions (sin + cos concatenated).
    
    Args:
        t: Time values [batch_size] or [batch_size, 1]
        dim: Embedding dimension
        
    Returns:
        Time embeddings [batch_size, 2*dim]
    """
    embedding = FourierTimeEmbedding(embed_dim=dim)
    return embedding(t)

def gaussian_time_embedding(t: jnp.ndarray, dim: int, sigma: float = 1.0) -> jnp.ndarray:
    """Create Gaussian time embeddings.
    
    This is a wrapper function for backward compatibility.
    See GaussianTimeEmbedding class for details.
    
    Args:
        t: Time values [batch_size] or [batch_size, 1]
        dim: Embedding dimension
        sigma: Standard deviation of Gaussian basis functions
        
    Returns:
        Time embeddings [batch_size, dim]
    """
    embedding = GaussianTimeEmbedding(embed_dim=dim, sigma=sigma)
    return embedding(t)

def get_time_embedding(t: jnp.ndarray, dim: int, method: str = "sinusoidal", **kwargs) -> jnp.ndarray:
    """Get time embedding using the specified method.
    
    This is a convenience function that allows switching between
    different time embedding methods.
    
    Args:
        t: Time values [batch_size]
        dim: Embedding dimension
        method: Embedding method ("sinusoidal", "fourier", "linear", "gaussian")
        **kwargs: Additional arguments passed to embedding classes (e.g., sigma for gaussian)
        
    Returns:
        Time embeddings [batch_size, dim] (or [batch_size, 2*dim] for fourier)
        
    Raises:
        ValueError: If method is not supported
    """
    # Handle fourier method which expects 2D input in original implementation
    if method == "fourier":
        if t.ndim == 1:
            t = t[:, None]
        embedding = create_time_embedding(embed_dim=dim, method=method, **kwargs)
        return embedding(t)
    else:
        embedding = create_time_embedding(embed_dim=dim, method=method, **kwargs)
        return embedding(t)
# Example usage and testing
if __name__ == "__main__":
    import jax
    
    print("Testing time embeddings:")
    
    # Test with single time value
    t_single = 0.5
    print(f"\nSingle time value: {t_single}")
    
    # Test with batch of time values
    t_batch = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"\nBatch time values: {t_batch}")
    
    # Test class-based usage
    print(f"\nClass-based usage:")
    
    # Test Fourier embedding
    fourier_embedding = create_time_embedding(embed_dim=8, method="fourier")
    fourier_embed = fourier_embedding(t_single)
    print(f"Fourier embedding shape: {fourier_embed.shape}")
    print(f"Fourier embedding values: {fourier_embed}")
    
    # Test linear embedding
    linear_embedding = create_time_embedding(embed_dim=8, method="linear")
    linear_embed = linear_embedding(t_single)
    print(f"Linear embedding shape: {linear_embed.shape}")
    print(f"Linear embedding values: {linear_embed}")
    
    # Test sinusoidal embedding
    sinusoidal_embedding = create_time_embedding(embed_dim=8, method="sinusoidal")
    sinusoidal_embed = sinusoidal_embedding(t_batch)
    print(f"Sinusoidal batch shape: {sinusoidal_embed.shape}")
    print(f"Sinusoidal batch values:\n{sinusoidal_embed}")
    
    # Test Gaussian embedding
    gaussian_embedding = create_time_embedding(embed_dim=6, method="gaussian", sigma=1.0)
    gaussian_embed = gaussian_embedding(t_batch)
    print(f"Gaussian batch shape: {gaussian_embed.shape}")
    print(f"Gaussian batch values:\n{gaussian_embed}")

