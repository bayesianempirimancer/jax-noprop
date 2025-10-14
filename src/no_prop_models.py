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

from .embeddings.embeddings import sinusoidal_time_embedding, fourier_time_embedding, linear_time_embedding


def swiglu(x: jnp.ndarray) -> jnp.ndarray:
    """SwiGLU activation function.
    
    SwiGLU is a gated activation function that combines Swish and GLU:
    SwiGLU(x) = Swish(x) * sigmoid(x)
    
    Args:
        x: Input tensor
        
    Returns:
        SwiGLU activated tensor
    """
    # SwiGLU: x * sigmoid(x)^2
    (x1, x2) = jnp.split(x, 2, axis=-1)
    return nn.swish(x1) * nn.sigmoid(x2)


class SimpleConditionalResnet(nn.Module):
    """Simple Conditional ResNet for NoProp-CT.
    
    This is a lightweight model that concatenates z, x, and t_embed
    and processes them through multiple dense layers with residual connections.
    """
    hidden_dims: Tuple[int, ...] = (64,)
    activation: Callable = jax.nn.swish
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the simple conditional ResNet.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Input data [batch_size, x_dim]
            t: Time values [batch_size]
            
        Returns:
            dz/dt prediction [batch_size, z_dim] - same shape as z
        """
        # Get the output dimension from the input z shape
        output_dim = z.shape[-1]
        
        # Sinusoidal time embedding
        x = nn.Dense(self.hidden_dims[0])(x)
        x = x + sinusoidal_time_embedding(t, self.hidden_dims[0])
        x = x + nn.Dense(self.hidden_dims[0])(z)

        x = self.activation(x)
        for hidden_dim in self.hidden_dims[1:]:
            x = nn.Dense(hidden_dim)(x)
            x = self.activation(x)
        x = nn.Dense(output_dim)(x)
        return x


class ResNetBlock(nn.Module):
    """Basic ResNet block."""
    
    features: int
    stride: int = 1
    use_projection: bool = False
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        
        # First conv layer
        x = nn.Conv(
            self.features, 
            kernel_size=(3, 3), 
            strides=(self.stride, self.stride),
            padding="SAME"
        )(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        
        # Second conv layer
        x = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME"
        )(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        
        # Projection shortcut if needed
        if self.use_projection:
            residual = nn.Conv(
                self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="SAME"
            )(residual)
            residual = nn.BatchNorm(use_running_average=True)(residual)
        
        x = x + residual
        x = nn.relu(x)
        return x


class ResNet(nn.Module):
    """ResNet backbone architecture."""
    
    num_classes: int
    depth: int = 18  # 18, 50, 152
    width: int = 64
    
    def get_num_blocks(self):
        """Get number of blocks per stage based on depth."""
        if self.depth == 18:
            return [2, 2, 2, 2]
        elif self.depth == 50:
            return [3, 4, 6, 3]
        elif self.depth == 152:
            return [3, 8, 36, 3]
        else:
            raise ValueError(f"Unsupported ResNet depth: {self.depth}")
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Initial conv layer
        x = nn.Conv(
            self.width,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME"
        )(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        
        # ResNet stages
        num_blocks_list = self.get_num_blocks()
        for stage_idx, num_blocks in enumerate(num_blocks_list):
            features = self.width * (2 ** stage_idx)
            for block_idx in range(num_blocks):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                use_projection = stage_idx > 0 and block_idx == 0
                x = ResNetBlock(
                    features=features,
                    stride=stride,
                    use_projection=use_projection
                )(x)
        
        # Global average pooling and classifier
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(self.num_classes)(x)
        return x


class ConditionalResNet(nn.Module):
    """Conditional ResNet that handles NoProp-specific inputs.
    
    This wrapper takes the noisy target z and input x, and outputs
    the denoised prediction with the same shape as input z. For 
    continuous-time variants, it also takes the time step t.
    """
    
    num_classes: int
    depth: int = 18
    width: int = 64
    z_dim: Optional[int] = None  # If None, uses num_classes
    time_embed_dim: int = 128
    
    def setup(self):
        pass
    
    @nn.compact
    def __call__(
        self, 
        z: jnp.ndarray, 
        x: jnp.ndarray, 
        t: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            z: Noisy target [batch_size, z_dim]
            x: Input data [batch_size, height, width, channels]
            t: Time step [batch_size] (optional, for continuous-time variants)
            
        Returns:
            Denoised prediction [batch_size, z_dim] (same shape as input z)
        """
        batch_size = z.shape[0]
        z_dim = self.z_dim if self.z_dim is not None else self.num_classes
        
        # Process input x through backbone
        backbone = ResNet(
            num_classes=self.num_classes,
            depth=self.depth,
            width=self.width
        )
        x_features = backbone(x)  # [batch_size, num_classes]
        
        # Project x_features to match z_dim
        x_proj = nn.Dense(z_dim)
        x_features_proj = x_proj(x_features)  # [batch_size, z_dim]
        
        # Process z
        z_proj = nn.Dense(self.width)
        z_features = z_proj(z)  # [batch_size, width]
        
        # Add time embedding if provided (for continuous-time variants)
        if t is not None:
            # Create sinusoidal time embedding as used in the paper
            t_embed = sinusoidal_time_embedding(t, self.time_embed_dim)
            
            # Project time embedding to match z_features dimension
            time_proj = nn.Dense(self.width)
            t_proj = time_proj(t_embed)
            
            # Combine z and time features
            combined_features = z_features + t_proj  # Additive combination
        else:
            combined_features = z_features
        
        # Project combined features to z_dim
        combined_proj = nn.Dense(z_dim)
        combined_features_proj = combined_proj(combined_features)  # [batch_size, z_dim]
        
        # Fuse x and z features
        # For simplicity, we add them together, but more sophisticated
        # fusion strategies could be implemented
        fused_features = x_features_proj + combined_features_proj
        
        # Final output projection (identity if already correct shape)
        output = nn.Dense(z_dim)(fused_features)
        
        return output


class SimpleCNN(nn.Module):
    """Simple CNN for smaller datasets like MNIST."""
    
    num_classes: int
    z_dim: Optional[int] = None
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
        z_dim = self.z_dim if self.z_dim is not None else self.num_classes
        
        # Simple CNN for x
        x = nn.Conv(32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape(batch_size, -1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x_features = nn.Dense(self.num_classes)(x)
        
        # Process z
        z_proj = nn.Dense(64)
        z_features = z_proj(z)
        
        # Add time embedding if provided
        if t is not None:
            # Create sinusoidal time embedding as used in the paper
            t_embed = sinusoidal_time_embedding(t, self.time_embed_dim)
            
            # Project time embedding to match z_features dimension
            time_proj = nn.Dense(64)
            t_proj = time_proj(t_embed)
            
            # Combine z and time features
            combined_features = z_features + t_proj  # Additive combination
        else:
            combined_features = z_features
        
        # Fuse features
        fusion_layer = nn.Dense(self.num_classes)
        fused_features = x_features + fusion_layer(combined_features)
        
        return fused_features


class ImageNoPropModel(nn.Module):
    """Image-based model for NoProp that handles image-like inputs and outputs.
    
    This model is designed to work with z and x that have image-like shapes:
    - z: [batch_shape, height, width, channels]
    - x: [batch_shape, height, width, channels] 
    - t: [batch_shape]
    
    The model outputs the same shape as z.
    """
    
    hidden_dims: Tuple[int, ...] = (64, 64, 64)
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Process image-like inputs and output same shape as z.
        
        Args:
            z: Current state [batch_shape, height, width, channels]
            x: Input data [batch_shape, height, width, channels]
            t: Time values [batch_shape]
            
        Returns:
            dz/dt prediction [batch_shape, height, width, channels] - same shape as z
        """
        # Get the output shape from z
        output_shape = z.shape
        
        # Time embedding
        # Assume batch_shape = (x.shape[0],) - just the first dimension
        batch_size = x.shape[0]
        
        # Handle different t shapes that might come from ODE integration
        if t.ndim == 0:  # scalar
            t_batch = jnp.full((batch_size,), t)
        elif t.shape == (batch_size,):  # already correct shape
            t_batch = t
        else:  # other shapes - take the first element and broadcast
            t_batch = jnp.full((batch_size,), t.flatten()[0])
            
        t_emb = sinusoidal_time_embedding(t_batch, self.hidden_dims[0])  # (batch_size, hidden_dim)
        
        # Process z through conv layers
        z_features = z
        for hidden_dim in self.hidden_dims:
            z_features = nn.Conv(hidden_dim, (3, 3), padding='SAME')(z_features)
            z_features = self.activation(z_features)
        
        # Process x through conv layers  
        x_features = x
        for hidden_dim in self.hidden_dims:
            x_features = nn.Conv(hidden_dim, (3, 3), padding='SAME')(x_features)
            x_features = self.activation(x_features)
        
        # Add time embedding to z_features
        # t_emb has shape [batch_size, hidden_dim], we need to broadcast it
        # z_features has shape [batch_size, height, width, hidden_dim]
        t_emb_expanded = t_emb[:, None, None, :]  # [batch_size, 1, 1, hidden_dim]
        z_features = z_features + t_emb_expanded
        
        # Combine z and x features
        combined = z_features + x_features
        
        # Final conv layer to output same shape as input z
        output = nn.Conv(output_shape[-1], (3, 3), padding='SAME')(combined)
        
        return output
