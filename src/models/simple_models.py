"""
Simple NoProp models for vector inputs.

This module contains standard neural network architectures designed for
NoProp-CT and NoProp-FM when both z and x are vectors with no known structure.
These models are primarily for testing purposes and small demos.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Any, Dict, Callable
from ..embeddings.embeddings import get_time_embedding
from ..embeddings.positional_encoding import positional_encoding


class SimpleMLP(nn.Module):
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
        # 1. x features processing
        x_features = nn.Dense(self.hidden_dims[0] // 2)(x)
        x_features = self.activation_fn(x_features)
        
        # 2. z features processing  
        z_features = nn.Dense(self.hidden_dims[0] // 2)(z)
        z_features = self.activation_fn(z_features)
        
        # 3. time embeddings
        t_features = get_time_embedding(t, self.time_embed_dim, self.time_embed_method)
        
        # 4. fusion: combine all features
        h = jnp.concatenate([z_features, x_features, t_features], axis=-1)
        
        # 5. additional processing layers
        for hidden_dim in self.hidden_dims:
            h = nn.Dense(hidden_dim)(h)
            
            if self.use_batch_norm:
                h = nn.BatchNorm(use_running_average=True)(h)
            
            h = self.activation_fn(h)
            
            if self.dropout_rate > 0:
                h = nn.Dropout(rate=self.dropout_rate, deterministic=True)(h)
        
        # Output projection to match input z shape
        output = nn.Dense(z.shape[-1])(h)
        
        return output
    

class SimpleResNet(nn.Module):
    """Simple ResNet for NoProp with vector inputs.
    
    A ResNet-style architecture with residual connections that follows the 5-component structure:
    1. x features processing: Dense layer
    2. z features processing: Dense layer  
    3. time embeddings: Sinusoidal time embedding
    4. fusion: Concatenation
    5. processing layers: ResNet blocks with residual connections
    
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
    use_batch_norm: bool = True
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through simple ResNet.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim]
            t: Time values [batch_size] or scalar
            
        Returns:
            Updated state [batch_size, z_dim] (same shape as input z)
        """
        # 1. x features processing
        x_features = nn.Dense(self.hidden_dims[0] // 2)(x)
        x_features = self.activation_fn(x_features)
        
        # 2. z features processing  
        z_features = nn.Dense(self.hidden_dims[0] // 2)(z)
        z_features = self.activation_fn(z_features)
        
        # 3. time embeddings
        t_features = get_time_embedding(t, self.time_embed_dim, self.time_embed_method)
        
        # 4. fusion: combine all features
        h = jnp.concatenate([z_features, x_features, t_features], axis=-1)
        
        # 5. additional processing layers with residual connections
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Store input for residual connection
            residual = h if h.shape[-1] == hidden_dim else None
            
            # Linear transformation
            h = nn.Dense(hidden_dim)(h)
            
            if self.use_batch_norm:
                h = nn.BatchNorm(use_running_average=True)(h)
            
            h = self.activation_fn(h)
            
            if self.dropout_rate > 0:
                h = nn.Dropout(rate=self.dropout_rate, deterministic=True)(h)
            
            # Residual connection if dimensions match
            if residual is not None:
                h = h + residual
        
        # Output projection to match input z shape
        output = nn.Dense(z.shape[-1])(h)
        
        return output


class SimpleTransformer(nn.Module):
    """Simple Transformer for NoProp with vector inputs.
    
    A transformer-based architecture that follows the 5-component structure:
    1. x features processing: Dense layer
    2. z features processing: Dense layer  
    3. time embeddings: Sinusoidal time embedding
    4. fusion: Create sequence [z, x, t]
    5. processing layers: Transformer layers with self-attention
    
    Args:
        hidden_dim: Hidden dimension of the transformer
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        dropout_rate: Dropout rate for regularization
    """
    
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    time_embed_dim: int = 64
    activation_fn: Callable = nn.swish
    time_embed_method: str = "sinusoidal"
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through simple transformer.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim]
            t: Time values [batch_size] or scalar
            
        Returns:
            Updated state [batch_size, z_dim] (same shape as input z)
        """
        # 1. x features processing
        x_features = nn.Dense(self.hidden_dim)(x)
        
        # 2. z features processing  
        z_features = nn.Dense(self.hidden_dim)(z)
        
        # 3. time embeddings
        t_embed = get_time_embedding(t, self.time_embed_dim, self.time_embed_method)
        t_features = nn.Dense(self.hidden_dim)(t_embed)
        
        # 4. fusion: create sequence [z, x, t]
        sequence = jnp.stack([z_features, x_features, t_features], axis=1)  # [batch, 3, hidden_dim]
        
        # Add positional encoding
        pos_encoding = positional_encoding(sequence.shape[1], self.hidden_dim)
        sequence = sequence + pos_encoding[None, :, :]
        
        # 5. transformer processing layers
        for _ in range(self.num_layers):
            # Self-attention
            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=True
            )(sequence, sequence)
            
            # Add & norm
            sequence = sequence + attn_output
            sequence = nn.LayerNorm()(sequence)
            
            # Feed-forward
            ff_output = nn.Dense(self.hidden_dim * 4)(sequence)
            ff_output = self.activation_fn(ff_output)
            ff_output = nn.Dense(self.hidden_dim)(ff_output)
            
            # Add & norm
            sequence = sequence + ff_output
            sequence = nn.LayerNorm()(sequence)
        
        # Extract z representation (first token)
        z_repr = sequence[:, 0, :]  # [batch, hidden_dim]
        
        # Output projection to match input z shape
        output = nn.Dense(z.shape[-1])(z_repr)
        
        return output
    
