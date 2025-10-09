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

from .embeddings import sinusoidal_time_embedding


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
