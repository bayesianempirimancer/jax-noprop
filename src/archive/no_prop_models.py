"""
Model architectures for NoProp implementations.

This module provides ResNet wrappers that can be used with the NoProp algorithm.
The wrappers handle the specific input/output requirements for each NoProp variant.
"""

from typing import Callable, Optional, Tuple

import jax.numpy as jnp
import flax.linen as nn

from ..embeddings.embeddings import sinusoidal_time_embedding, fourier_time_embedding, linear_time_embedding, get_time_embedding
from ..layers.concatsquash import ConcatSquash
from ..layers.builders import get_act

class ConditionalResNet_CNNx(nn.Module):
    """Simple CNN Conditional ResNet for smaller datasets like MNIST."""
    
    hidden_dims: Tuple[int, ...] = (64, 128, 64)
    time_embed_dim: int = 64
    dropout_rate: float = 0.1
    
    def setup(self):
        pass
    
    @nn.compact
    def __call__(
        self, 
        z: jnp.ndarray, 
        x: jnp.ndarray, 
        t: Optional[jnp.ndarray] = None, 
        training: bool = True
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
        z = ConcatSquash(self.hidden_dims[0])(z, x, t)

        # Processing layers
        for hidden_dim in self.hidden_dims[1:]:
            z = nn.Dense(hidden_dim)(z)
            z = nn.relu(z)
            z = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z)

        # Output projection to match input
        return nn.Dense(z_dim)(z)

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
    activation_fn: Callable = get_act("swish")
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> jnp.ndarray:
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
        z = ConcatSquash(self.hidden_dims[0])(z, x, t)

        # 3. processing layers
        for hidden_dim in self.hidden_dims[1:]:
            z = nn.Dense(hidden_dim)(z)
            if self.use_batch_norm:
                z = nn.BatchNorm(use_running_average=True)(z)
            z = self.activation_fn(z)            
            if self.dropout_rate > 0:
                z = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z)
                
        # 4. output projection to match input
        return nn.Dense(output_dim)(z)


