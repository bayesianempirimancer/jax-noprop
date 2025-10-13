"""
Point cloud processing building blocks for NoProp.

This module contains neural network building blocks specifically designed
for point cloud processing tasks, including components that can be used
within NoProp-compatible models.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Any, Dict, Callable
from ..embeddings.positional_encoding import positional_encoding


class PointNet(nn.Module):
    """PointNet architecture for point cloud processing.
    
    A permutation-invariant neural network for point cloud classification
    and segmentation tasks.
    
    Args:
        num_classes: Number of output classes
        hidden_dims: Tuple of hidden layer dimensions
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
    """
    
    num_classes: int = 10
    hidden_dims: Tuple[int, ...] = (64, 128, 1024)
    activation_fn: Callable = nn.swish
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, points: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass through PointNet.
        
        Args:
            points: Point cloud [batch_size, num_points, 3]
            training: Whether in training mode
            
        Returns:
            Class predictions [batch_size, num_classes]
        """
        batch_size, num_points, _ = points.shape
        
        # Input transformation (T-Net)
        input_transform = self._transformation_net(points, 3, training)
        points_transformed = jnp.matmul(points, input_transform)
        
        # Point feature extraction
        x = points_transformed
        for hidden_dim in self.hidden_dims[:-1]:
            x = nn.Dense(hidden_dim)(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not training)(x)
            x = self.activation_fn(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Feature transformation (T-Net)
        feature_transform = self._transformation_net(x, self.hidden_dims[-2], training)
        x_transformed = jnp.matmul(x, feature_transform)
        
        # Final feature extraction
        x = nn.Dense(self.hidden_dims[-1])(x_transformed)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        
        # Global max pooling (permutation invariant)
        x = jnp.max(x, axis=1)  # [batch_size, hidden_dims[-1]]
        
        # Classification head
        x = nn.Dense(512)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = nn.Dense(256)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        output = nn.Dense(self.num_classes)(x)
        
        return output
    
    def _transformation_net(self, x: jnp.ndarray, k: int, training: bool) -> jnp.ndarray:
        """T-Net for learning spatial transformations.
        
        Args:
            x: Input features [batch_size, num_points, feature_dim]
            k: Output dimension of transformation matrix
            training: Whether in training mode
            
        Returns:
            Transformation matrix [batch_size, k, k]
        """
        # Feature extraction
        x = nn.Dense(64)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        
        x = nn.Dense(128)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        
        x = nn.Dense(1024)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        
        # Global max pooling
        x = jnp.max(x, axis=1)  # [batch_size, 1024]
        
        # Transformation matrix prediction
        x = nn.Dense(512)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        
        x = nn.Dense(256)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        
        # Initialize as identity matrix
        transform = nn.Dense(k * k)(x)
        transform = transform.reshape(-1, k, k)
        
        # Add identity matrix
        identity = jnp.eye(k)[None, :, :]
        transform = transform + identity
        
        return transform


class PointNetSegmentation(nn.Module):
    """PointNet for point cloud segmentation.
    
    A variant of PointNet designed for per-point classification
    (segmentation) tasks.
    
    Args:
        num_classes: Number of segmentation classes
        hidden_dims: Tuple of hidden layer dimensions
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
    """
    
    num_classes: int = 10
    hidden_dims: Tuple[int, ...] = (64, 128, 1024)
    activation_fn: Callable = nn.swish
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, points: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass through PointNet segmentation.
        
        Args:
            points: Point cloud [batch_size, num_points, 3]
            training: Whether in training mode
            
        Returns:
            Per-point predictions [batch_size, num_points, num_classes]
        """
        batch_size, num_points, _ = points.shape
        
        # Input transformation (T-Net)
        input_transform = self._transformation_net(points, 3, training)
        points_transformed = jnp.matmul(points, input_transform)
        
        # Point feature extraction
        x = points_transformed
        for hidden_dim in self.hidden_dims[:-1]:
            x = nn.Dense(hidden_dim)(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not training)(x)
            x = self.activation_fn(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Feature transformation (T-Net)
        feature_transform = self._transformation_net(x, self.hidden_dims[-2], training)
        x_transformed = jnp.matmul(x, feature_transform)
        
        # Final feature extraction
        x = nn.Dense(self.hidden_dims[-1])(x_transformed)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        
        # Global feature (max pooling)
        global_features = jnp.max(x, axis=1, keepdims=True)  # [batch_size, 1, hidden_dims[-1]]
        global_features = jnp.tile(global_features, (1, num_points, 1))  # [batch_size, num_points, hidden_dims[-1]]
        
        # Concatenate point features with global features
        x = jnp.concatenate([x, global_features], axis=-1)
        
        # Segmentation head
        x = nn.Dense(512)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = nn.Dense(256)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = nn.Dense(128)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        output = nn.Dense(self.num_classes)(x)
        
        return output
    
    def _transformation_net(self, x: jnp.ndarray, k: int, training: bool) -> jnp.ndarray:
        """T-Net for learning spatial transformations.
        
        Args:
            x: Input features [batch_size, num_points, feature_dim]
            k: Output dimension of transformation matrix
            training: Whether in training mode
            
        Returns:
            Transformation matrix [batch_size, k, k]
        """
        # Feature extraction
        x = nn.Dense(64)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        
        x = nn.Dense(128)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        
        x = nn.Dense(1024)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        
        # Global max pooling
        x = jnp.max(x, axis=1)  # [batch_size, 1024]
        
        # Transformation matrix prediction
        x = nn.Dense(512)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        
        x = nn.Dense(256)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation_fn(x)
        
        # Initialize as identity matrix
        transform = nn.Dense(k * k)(x)
        transform = transform.reshape(-1, k, k)
        
        # Add identity matrix
        identity = jnp.eye(k)[None, :, :]
        transform = transform + identity
        
        return transform


class PointCNN(nn.Module):
    """PointCNN architecture for point cloud processing.
    
    A convolutional neural network that operates directly on point clouds
    using X-conv operations for hierarchical feature learning.
    
    Args:
        num_classes: Number of output classes
        num_points: Number of points in input
        k: Number of nearest neighbors for X-conv
        dropout_rate: Dropout rate for regularization
    """
    
    num_classes: int = 10
    num_points: int = 1024
    k: int = 16
    activation_fn: Callable = nn.swish
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, points: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass through PointCNN.
        
        Args:
            points: Point cloud [batch_size, num_points, 3]
            training: Whether in training mode
            
        Returns:
            Class predictions [batch_size, num_classes]
        """
        # X-Conv layers with hierarchical downsampling
        x = self._x_conv_layer(points, 64, 8, training)  # [batch_size, num_points//8, 64]
        x = self._x_conv_layer(x, 128, 4, training)      # [batch_size, num_points//32, 128]
        x = self._x_conv_layer(x, 256, 2, training)      # [batch_size, num_points//64, 256]
        x = self._x_conv_layer(x, 512, 1, training)      # [batch_size, num_points//64, 512]
        
        # Global max pooling
        x = jnp.max(x, axis=1)  # [batch_size, 512]
        
        # Classification head
        x = nn.Dense(256)(x)
        x = self.activation_fn(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = nn.Dense(128)(x)
        x = self.activation_fn(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        output = nn.Dense(self.num_classes)(x)
        
        return output
    
    def _x_conv_layer(self, points: jnp.ndarray, out_channels: int, stride: int, training: bool) -> jnp.ndarray:
        """X-Conv layer for point cloud convolution.
        
        Args:
            points: Input points [batch_size, num_points, in_channels]
            out_channels: Number of output channels
            stride: Downsampling stride
            training: Whether in training mode
            
        Returns:
            Convolved features [batch_size, num_points//stride, out_channels]
        """
        batch_size, num_points, in_channels = points.shape
        
        # Downsample points
        if stride > 1:
            indices = jnp.arange(0, num_points, stride)
            points = points[:, indices, :]
        
        # For simplicity, we'll use a basic MLP instead of full X-Conv
        # In practice, X-Conv involves finding k-nearest neighbors and
        # applying convolution-like operations
        x = nn.Dense(out_channels)(points)
        x = self.activation_fn(x)
        
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        return x


class PointTransformer(nn.Module):
    """Point Transformer for point cloud processing.
    
    A transformer-based architecture that uses self-attention
    to model relationships between points in a point cloud.
    
    Args:
        num_classes: Number of output classes
        hidden_dim: Hidden dimension of the transformer
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout_rate: Dropout rate for regularization
    """
    
    num_classes: int = 10
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    activation_fn: Callable = nn.swish
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, points: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass through Point Transformer.
        
        Args:
            points: Point cloud [batch_size, num_points, 3]
            training: Whether in training mode
            
        Returns:
            Class predictions [batch_size, num_classes]
        """
        batch_size, num_points, _ = points.shape
        
        # Input projection
        x = nn.Dense(self.hidden_dim)(points)
        
        # Positional encoding
        pos_encoding = positional_encoding(num_points, self.hidden_dim)
        x = x + pos_encoding[None, :, :]
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Self-attention
            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not training
            )(x, x)
            
            # Add & norm
            x = x + attn_output
            x = nn.LayerNorm()(x)
            
            # Feed-forward
            ff_output = nn.Dense(self.hidden_dim * 4)(x)
            ff_output = self.activation_fn(ff_output)
            ff_output = nn.Dense(self.hidden_dim)(ff_output)
            
            # Add & norm
            x = x + ff_output
            x = nn.LayerNorm()(x)
        
        # Global max pooling
        x = jnp.max(x, axis=1)  # [batch_size, hidden_dim]
        
        # Classification head
        x = nn.Dense(256)(x)
        x = self.activation_fn(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = nn.Dense(128)(x)
        x = self.activation_fn(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        output = nn.Dense(self.num_classes)(x)
        
        return output
    


class PointCloudVAE(nn.Module):
    """Variational Autoencoder for point cloud generation.
    
    A VAE architecture that can encode point clouds into latent
    representations and decode them back to point clouds.
    
    Args:
        latent_dim: Dimension of latent space
        num_points: Number of points in point cloud
        hidden_dims: Tuple of hidden layer dimensions
        dropout_rate: Dropout rate for regularization
    """
    
    latent_dim: int = 128
    num_points: int = 1024
    hidden_dims: Tuple[int, ...] = (256, 512, 1024)
    activation_fn: Callable = nn.swish
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, points: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass through Point Cloud VAE.
        
        Args:
            points: Input point cloud [batch_size, num_points, 3]
            training: Whether in training mode
            
        Returns:
            Tuple of (reconstructed_points, mean, log_var)
        """
        # Encode
        mean, log_var = self._encode(points, training)
        
        # Sample from latent space
        if training:
            std = jnp.exp(0.5 * log_var)
            eps = jax.random.normal(self.make_rng('sample'), mean.shape)
            z = mean + std * eps
        else:
            z = mean
        
        # Decode
        reconstructed = self._decode(z, training)
        
        return reconstructed, mean, log_var
    
    def _encode(self, points: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode point cloud to latent space.
        
        Args:
            points: Input point cloud [batch_size, num_points, 3]
            training: Whether in training mode
            
        Returns:
            Tuple of (mean, log_var) [batch_size, latent_dim]
        """
        # Flatten point cloud
        x = points.reshape(points.shape[0], -1)  # [batch_size, num_points * 3]
        
        # Encoder layers
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = self.activation_fn(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Mean and log variance
        mean = nn.Dense(self.latent_dim)(x)
        log_var = nn.Dense(self.latent_dim)(x)
        
        return mean, log_var
    
    def _decode(self, z: jnp.ndarray, training: bool) -> jnp.ndarray:
        """Decode latent representation to point cloud.
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            training: Whether in training mode
            
        Returns:
            Reconstructed point cloud [batch_size, num_points, 3]
        """
        # Decoder layers
        x = z
        for hidden_dim in reversed(self.hidden_dims):
            x = nn.Dense(hidden_dim)(x)
            x = self.activation_fn(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output layer
        x = nn.Dense(self.num_points * 3)(x)
        
        # Reshape to point cloud
        points = x.reshape(-1, self.num_points, 3)
        
        return points


