"""
Model architectures for NoProp implementations.

This module provides ResNet wrappers that can be used with the NoProp algorithm.
The wrappers handle the specific input/output requirements for each NoProp variant.
"""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct

from .embeddings.embeddings import sinusoidal_time_embedding, fourier_time_embedding, linear_time_embedding, get_time_embedding
from .layers.concatsquash import ConcatSquash

class ConditionalResNet_CNNx(nn.Module):
    """Simple CNN Conditional ResNet for smaller datasets like MNIST."""
    
    hidden_dims: Tuple[int, ...] = (64, 128, 64)
    time_embed_dim: int = 64
    
    def setup(self):
        pass
    
    @nn.compact
    def __call__(
        self, 
        z: jnp.ndarray, 
        x: jnp.ndarray, 
        t: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        batch_size = z.shape[0]
        z_dim = z.shape[-1]
        
        # CNN for x
        x = nn.Conv(32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape(batch_size, -1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        # Time embedding
        t = sinusoidal_time_embedding(t, self.time_embed_dim)

        # Fusion
        x = ConcatSquash(self.hidden_dims[0])(x, z, t)

        # Processing layers
        for hidden_dim in self.hidden_dims[1:]:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)

        # Output projection to match input
        return nn.Dense(z_dim)(x)

class ConditionalResnet_MLP(nn.Module):
    """Simple MLP for NoProp with vector inputs.
    
    A standard multi-layer perceptron that follows the 5-component structure:
    1. x features processing: Dense layer
    2. z features processing: Dense layer  
    3. time embeddings: Sinusoidal time embedding
    4. fusion: Concatenation
    5. processing layers: Additional dense layers
    
    Args:
        hidden_dims: Tuple of hidden layer dimensions
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation: Activation function to use
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
    """
    
    hidden_dims: Tuple[int, ...] = (256, 256, 128)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    activation_fn: Callable = nn.swish
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through simple MLP.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim]
            t: Time values [batch_size] or scalar
            
        Returns:
            Updated state [batch_size, z_dim] (same shape as input z)
        """        

        output_dim = z.shape[-1]
        # 1. time embeddings
        t = get_time_embedding(t, self.time_embed_dim, self.time_embed_method)
        # 2. fusion
        x = ConcatSquash(self.hidden_dims[0])(x, z, t)

        # 3. processing layers
        for hidden_dim in self.hidden_dims[1:]:
            x = nn.Dense(hidden_dim)(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=True)(x)
            x = self.activation_fn(x)            
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=True)(x)
                
        # 4. output projection to match input
        return nn.Dense(output_dim)(x)


