"""
Standard image processing layers and blocks.

This module contains well-known, well-studied image processing architectures
that can be used as building blocks for NoProp image models.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Any, Dict, Callable
from ..embeddings.positional_encoding import positional_encoding


class ResNetBlock(nn.Module):
    """Standard ResNet block with batch normalization and ReLU activation.
    
    This implements the standard ResNet block used in ResNet-50 and similar architectures.
    
    Args:
        features: Number of output features
        stride: Stride for the first convolution
        use_1x1_conv: Whether to use 1x1 convolution for dimension matching
    """
    
    features: int
    stride: int = 1
    use_1x1_conv: bool = False
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through ResNet block."""
        residual = x
        
        # First convolution
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME"
        )(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        
        # Second convolution
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            padding="SAME"
        )(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        
        # Skip connection with 1x1 conv if needed
        if self.use_1x1_conv:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride)
            )(residual)
            residual = nn.BatchNorm(use_running_average=True)(residual)
        
        x = x + residual
        x = nn.relu(x)
        
        return x


class ResNet50(nn.Module):
    """ResNet-50 architecture for image feature extraction.
    
    Standard ResNet-50 implementation that can be used as a feature extractor
    for image inputs in NoProp models.
    
    Args:
        num_classes: Number of output classes (if used for classification)
        include_top: Whether to include the final classification layer
    """
    
    num_classes: int = 1000
    include_top: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through ResNet-50."""
        # Initial convolution and pooling
        x = nn.Conv(features=64, kernel_size=(7, 7), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        
        # ResNet blocks
        # Stage 1: 64 features, 3 blocks
        for _ in range(3):
            x = ResNetBlock(features=64)(x)
        
        # Stage 2: 128 features, 4 blocks
        x = ResNetBlock(features=128, stride=2, use_1x1_conv=True)(x)
        for _ in range(3):
            x = ResNetBlock(features=128)(x)
        
        # Stage 3: 256 features, 6 blocks
        x = ResNetBlock(features=256, stride=2, use_1x1_conv=True)(x)
        for _ in range(5):
            x = ResNetBlock(features=256)(x)
        
        # Stage 4: 512 features, 3 blocks
        x = ResNetBlock(features=512, stride=2, use_1x1_conv=True)(x)
        for _ in range(2):
            x = ResNetBlock(features=512)(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # [batch_size, 512]
        
        # Classification head (optional)
        if self.include_top:
            x = nn.Dense(self.num_classes)(x)
        
        return x


class ResNetDecoder(nn.Module):
    """ResNet-style decoder for image generation.
    
    A decoder architecture that can generate images from feature vectors,
    commonly used in VAEs and other generative models.
    
    Args:
        output_channels: Number of output image channels
        initial_size: Initial spatial size (e.g., 7 for 7x7)
        final_size: Final spatial size (e.g., 224 for 224x224)
    """
    
    output_channels: int = 3
    initial_size: int = 7
    final_size: int = 224
    activation_fn: Callable = nn.relu
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through ResNet decoder."""
        # Initial dense layer to expand features
        x = nn.Dense(512 * self.initial_size * self.initial_size)(x)
        x = self.activation_fn(x)
        x = x.reshape(x.shape[0], self.initial_size, self.initial_size, 512)
        
        # Upsampling blocks
        # 7x7 -> 14x14
        x = nn.ConvTranspose(features=256, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = self.activation_fn(x)
        
        # 14x14 -> 28x28
        x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = self.activation_fn(x)
        
        # 28x28 -> 56x56
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = self.activation_fn(x)
        
        # 56x56 -> 112x112
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = self.activation_fn(x)
        
        # 112x112 -> 224x224
        x = nn.ConvTranspose(features=self.output_channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        
        return x


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 architecture for image feature extraction.
    
    A more efficient alternative to ResNet-50, using compound scaling
    and mobile inverted bottleneck convolutions.
    
    Args:
        num_classes: Number of output classes
        include_top: Whether to include the final classification layer
    """
    
    num_classes: int = 1000
    include_top: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through EfficientNet-B0."""
        # Simplified EfficientNet-B0 implementation
        # Initial convolution
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.swish(x)
        
        # MBConv blocks (simplified)
        # Stage 1: 16 features
        x = self._mbconv_block(x, 16, 1, 1)
        
        # Stage 2: 24 features
        x = self._mbconv_block(x, 24, 2, 6)
        x = self._mbconv_block(x, 24, 1, 6)
        
        # Stage 3: 40 features
        x = self._mbconv_block(x, 40, 2, 6)
        x = self._mbconv_block(x, 40, 1, 6)
        
        # Stage 4: 80 features
        x = self._mbconv_block(x, 80, 2, 6)
        x = self._mbconv_block(x, 80, 1, 6)
        x = self._mbconv_block(x, 80, 1, 6)
        
        # Stage 5: 112 features
        x = self._mbconv_block(x, 112, 1, 6)
        x = self._mbconv_block(x, 112, 1, 6)
        x = self._mbconv_block(x, 112, 1, 6)
        
        # Stage 6: 192 features
        x = self._mbconv_block(x, 192, 2, 6)
        x = self._mbconv_block(x, 192, 1, 6)
        x = self._mbconv_block(x, 192, 1, 6)
        x = self._mbconv_block(x, 192, 1, 6)
        
        # Stage 7: 320 features
        x = self._mbconv_block(x, 320, 1, 6)
        
        # Final convolution
        x = nn.Conv(features=1280, kernel_size=(1, 1), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.swish(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # [batch_size, 1280]
        
        # Classification head (optional)
        if self.include_top:
            x = nn.Dense(self.num_classes)(x)
        
        return x
    
    def _mbconv_block(self, x: jnp.ndarray, features: int, stride: int, expand_ratio: int) -> jnp.ndarray:
        """Mobile inverted bottleneck convolution block."""
        residual = x
        
        # Expansion phase
        if expand_ratio > 1:
            expanded_features = x.shape[-1] * expand_ratio
            x = nn.Conv(features=expanded_features, kernel_size=(1, 1), padding="SAME")(x)
            x = nn.BatchNorm(use_running_average=True)(x)
            x = nn.swish(x)
        
        # Depthwise convolution
        x = nn.Conv(
            features=x.shape[-1],
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding="SAME",
            feature_group_count=x.shape[-1]
        )(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.swish(x)
        
        # Projection phase
        x = nn.Conv(features=features, kernel_size=(1, 1), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        
        # Skip connection
        if stride == 1 and residual.shape[-1] == features:
            x = x + residual
        
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image feature extraction.
    
    A transformer-based architecture that treats images as sequences of patches.
    
    Args:
        patch_size: Size of image patches
        num_patches: Number of patches per image
        hidden_dim: Hidden dimension of the transformer
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        num_classes: Number of output classes
        include_top: Whether to include the final classification layer
    """
    
    patch_size: int = 16
    num_patches: int = 196  # For 224x224 images with 16x16 patches
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    num_classes: int = 1000
    include_top: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through Vision Transformer."""
        batch_size = x.shape[0]
        
        # Patch embedding
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size)
        )(x)
        x = x.reshape(batch_size, self.num_patches, self.hidden_dim)
        
        # Add positional encoding
        pos_encoding = positional_encoding(self.num_patches, self.hidden_dim)
        x = x + pos_encoding[None, :, :]
        
        # Add class token
        cls_token = self.param('cls_token', nn.initializers.normal(0.02), (1, 1, self.hidden_dim))
        cls_token = jnp.tile(cls_token, (batch_size, 1, 1))
        x = jnp.concatenate([cls_token, x], axis=1)
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Self-attention
            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads
            )(x, x)
            x = x + attn_output
            x = nn.LayerNorm()(x)
            
            # Feed-forward
            ff_output = nn.Dense(self.hidden_dim * 4)(x)
            ff_output = nn.gelu(ff_output)
            ff_output = nn.Dense(self.hidden_dim)(ff_output)
            x = x + ff_output
            x = nn.LayerNorm()(x)
        
        # Extract class token
        x = x[:, 0, :]  # [batch_size, hidden_dim]
        
        # Classification head (optional)
        if self.include_top:
            x = nn.Dense(self.num_classes)(x)
        
        return x
    