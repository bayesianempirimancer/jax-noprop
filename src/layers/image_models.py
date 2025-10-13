"""
NoProp image models with CNN-based architectures.

This module contains CNN-based architectures for NoProp when x and/or z are images.
Three variants are supported:
1. Image-to-Label: x=image, z=vector (logits)
2. Label-to-Image: x=vector, z=image  
3. Image-to-Image: x=image, z=image

All models follow the 5-component structure using well-known image processing blocks.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Any, Dict, Callable
from ..embeddings.embeddings import get_time_embedding
from ..blocks.image_blocks import ResNet50, ResNetDecoder, EfficientNetB0, VisionTransformer


class ImageToLabelModel(nn.Module):
    """Image-to-Label model: x=image, z=vector (logits).
    
    This model processes image inputs x and outputs vector logits z.
    Uses CNN-based feature extraction for x and standard dense layers for z.
    
    Architecture:
    1. x features processing: ResNet-50 or EfficientNet-B0
    2. z features processing: Dense layer
    3. time embeddings: Sinusoidal time embedding
    4. fusion: Concatenation
    5. processing layers: Dense layers + output projection
    
    Args:
        backbone: CNN backbone ("resnet50", "efficientnet_b0", "vit")
        num_classes: Number of output classes
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        hidden_dims: Tuple of hidden layer dimensions for processing
        activation: Activation function to use
        dropout_rate: Dropout rate for regularization
    """
    
    backbone: str = "resnet50"
    num_classes: int = 1000
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    hidden_dims: Tuple[int, ...] = (512, 256)
    activation_fn: Callable = nn.swish
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through image-to-label model.
        
        Args:
            z: Current state [batch_size, num_classes] (logits)
            x: Conditional input [batch_size, height, width, channels] (image)
            t: Time values [batch_size] or scalar
            
        Returns:
            Updated state [batch_size, num_classes] (same shape as input z)
        """
        # 1. x features processing (image -> features)
        x_features = self._process_image_features(x)
        
        # 2. z features processing (logits -> features)
        z_features = nn.Dense(self.hidden_dims[0] // 2)(z)
        z_features = self.activation_fn(z_features)
        
        # 3. time embeddings
        t_features = get_time_embedding(t, self.time_embed_dim, self.time_embed_method)
        
        # 4. fusion: combine all features
        h = jnp.concatenate([z_features, x_features, t_features], axis=-1)
        
        # 5. additional processing layers
        for hidden_dim in self.hidden_dims:
            h = nn.Dense(hidden_dim)(h)
            h = self.activation_fn(h)
            
            if self.dropout_rate > 0:
                h = nn.Dropout(rate=self.dropout_rate, deterministic=True)(h)
        
        # Output projection to match input z shape
        output = nn.Dense(z.shape[-1])(h)
        
        return output
    
    def _process_image_features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Process image features using CNN backbone."""
        if self.backbone == "resnet50":
            # Use ResNet-50 without classification head
            backbone = ResNet50(num_classes=1000, include_top=False)
            features = backbone(x)  # [batch_size, 512]
        elif self.backbone == "efficientnet_b0":
            # Use EfficientNet-B0 without classification head
            backbone = EfficientNetB0(num_classes=1000, include_top=False)
            features = backbone(x)  # [batch_size, 1280]
        elif self.backbone == "vit":
            # Use Vision Transformer without classification head
            backbone = VisionTransformer(num_classes=1000, include_top=False)
            features = backbone(x)  # [batch_size, 768]
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        # Project to consistent dimension
        features = nn.Dense(self.hidden_dims[0] // 2)(features)
        features = self.activation_fn(features)
        
        return features
    

class LabelToImageModel(nn.Module):
    """Label-to-Image model: x=vector, z=image.
    
    This model processes vector inputs x and outputs images z.
    Uses dense layers for x and CNN-based decoder for z.
    
    Architecture:
    1. x features processing: Dense layers
    2. z features processing: CNN decoder
    3. time embeddings: Sinusoidal time embedding
    4. fusion: Concatenation
    5. processing layers: Dense layers + image decoder
    
    Args:
        image_size: Output image size (e.g., 224 for 224x224)
        image_channels: Number of output image channels
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        hidden_dims: Tuple of hidden layer dimensions for processing
        activation: Activation function to use
        dropout_rate: Dropout rate for regularization
    """
    
    image_size: int = 224
    image_channels: int = 3
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    hidden_dims: Tuple[int, ...] = (512, 256)
    activation_fn: Callable = nn.swish
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through label-to-image model.
        
        Args:
            z: Current state [batch_size, height, width, channels] (image)
            x: Conditional input [batch_size, x_dim] (vector encoding of image query)
            t: Time values [batch_size] or scalar
            
        Returns:
            Updated state [batch_size, height, width, channels] (same shape as input z)
        """
        # 1. x features processing (vector -> features)
        x_features = nn.Dense(self.hidden_dims[0] // 2)(x)
        x_features = self.activation_fn(x_features)
        
        # 2. z features processing (image -> features)
        z_features = self._process_image_to_features(z)
        
        # 3. time embeddings
        t_features = get_time_embedding(t, self.time_embed_dim, self.time_embed_method)
        
        # 4. fusion: combine all features
        h = jnp.concatenate([z_features, x_features, t_features], axis=-1)
        
        # 5. additional processing layers
        for hidden_dim in self.hidden_dims:
            h = nn.Dense(hidden_dim)(h)
            h = self.activation_fn(h)
            
            if self.dropout_rate > 0:
                h = nn.Dropout(rate=self.dropout_rate, deterministic=True)(h)
        
        # Output projection to image
        output = self._features_to_image(h, z.shape)
        
        return output
    
    def _process_image_to_features(self, z: jnp.ndarray) -> jnp.ndarray:
        """Process image to features using CNN encoder."""
        # Simple CNN encoder
        x = nn.Conv(features=64, kernel_size=(7, 7), strides=(2, 2), padding="SAME")(z)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=256, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # [batch_size, 256]
        
        # Project to consistent dimension
        features = nn.Dense(self.hidden_dims[0] // 2)(x)
        features = self.activation_fn(features)
        
        return features
    
    def _features_to_image(self, features: jnp.ndarray, target_shape: Tuple[int, ...]) -> jnp.ndarray:
        """Convert features back to image using CNN decoder."""
        # Use ResNet decoder
        decoder = ResNetDecoder(
            output_channels=target_shape[-1],
            initial_size=7,
            final_size=target_shape[1]  # Assuming square images
        )
        
        return decoder(features)


class ImageToImageModel(nn.Module):
    """Image-to-Image model: x=image, z=image.
    
    This model processes image inputs x and outputs images z.
    Uses CNN-based feature extraction for x and CNN-based decoder for z.
    
    Architecture:
    1. x features processing: ResNet-50 or EfficientNet-B0
    2. z features processing: CNN encoder
    3. time embeddings: Sinusoidal time embedding
    4. fusion: Concatenation
    5. processing layers: Dense layers + image decoder
    
    Args:
        backbone: CNN backbone for x processing ("resnet50", "efficientnet_b0", "vit")
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        hidden_dims: Tuple of hidden layer dimensions for processing
        activation: Activation function to use
        dropout_rate: Dropout rate for regularization
    """
    
    backbone: str = "resnet50"
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    hidden_dims: Tuple[int, ...] = (512, 256)
    activation: str = "relu"
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through image-to-image model.
        
        Args:
            z: Current state [batch_size, height, width, channels] (image)
            x: Conditional input [batch_size, height, width, channels] (image)
            t: Time values [batch_size] or scalar
            
        Returns:
            Updated state [batch_size, height, width, channels] (same shape as input z)
        """
        # 1. x features processing (image -> features)
        x_features = self._process_image_features(x)
        
        # 2. z features processing (image -> features)
        z_features = self._process_image_to_features(z)
        
        # 3. time embeddings
        t_features = get_time_embedding(t, self.time_embed_dim, self.time_embed_method)
        
        # 4. fusion: combine all features
        h = jnp.concatenate([z_features, x_features, t_features], axis=-1)
        
        # 5. additional processing layers
        for hidden_dim in self.hidden_dims:
            h = nn.Dense(hidden_dim)(h)
            h = self.activation_fn(h)
            
            if self.dropout_rate > 0:
                h = nn.Dropout(rate=self.dropout_rate, deterministic=True)(h)
        
        # Output projection to image
        output = self._features_to_image(h, z.shape)
        
        return output
    
    def _process_image_features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Process image features using CNN backbone."""
        if self.backbone == "resnet50":
            # Use ResNet-50 without classification head
            backbone = ResNet50(num_classes=1000, include_top=False)
            features = backbone(x)  # [batch_size, 512]
        elif self.backbone == "efficientnet_b0":
            # Use EfficientNet-B0 without classification head
            backbone = EfficientNetB0(num_classes=1000, include_top=False)
            features = backbone(x)  # [batch_size, 1280]
        elif self.backbone == "vit":
            # Use Vision Transformer without classification head
            backbone = VisionTransformer(num_classes=1000, include_top=False)
            features = backbone(x)  # [batch_size, 768]
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        # Project to consistent dimension
        features = nn.Dense(self.hidden_dims[0] // 2)(features)
        features = self.activation_fn(features)
        
        return features
    
    def _process_image_to_features(self, z: jnp.ndarray) -> jnp.ndarray:
        """Process image to features using CNN encoder."""
        # Simple CNN encoder
        x = nn.Conv(features=64, kernel_size=(7, 7), strides=(2, 2), padding="SAME")(z)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=256, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # [batch_size, 256]
        
        # Project to consistent dimension
        features = nn.Dense(self.hidden_dims[0] // 2)(x)
        features = self.activation_fn(features)
        
        return features
    
    def _features_to_image(self, features: jnp.ndarray, target_shape: Tuple[int, ...]) -> jnp.ndarray:
        """Convert features back to image using CNN decoder."""
        # Use ResNet decoder
        decoder = ResNetDecoder(
            output_channels=target_shape[-1],
            initial_size=7,
            final_size=target_shape[1]  # Assuming square images
        )
        
        return decoder(features)
    
