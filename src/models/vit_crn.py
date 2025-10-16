"""
ViT-based Conditional ResNet using DinoV2-ViT-S14 as frozen feature extractor.

This module implements a conditional ResNet that uses DinoV2-ViT-S14 to extract
rich image features and combines them with state (z) and time (t) information
to predict vector fields for NoProp training.

Key Features:
- Uses pretrained DinoV2-ViT-S14 as frozen feature extractor
- Combines image features with state and time embeddings
- Outputs vector field dz/dt for NoProp integration
- Supports both NoProp-CT and NoProp-FM variants
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Callable

try:
    # Try relative imports first (when used as a module)
    from ..embeddings.embeddings import get_time_embedding
    from ..layers.concatsquash import ConcatSquash
    from .weights import load_pretrained_dinov2_vits14
    from .vit import DinoV2
except ImportError:
    # Fall back to absolute imports (when run directly)
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.embeddings.embeddings import get_time_embedding
    from src.layers.concatsquash import ConcatSquash
    from src.models.weights import load_pretrained_dinov2_vits14
    from src.models.vit import DinoV2


class ViTCRN(nn.Module):
    """ViT-based Conditional ResNet for image-to-vector tasks.
    
    This model is designed for image-to-vector tasks where:
    - x: Input images [batch_size, height, width, channels] (518x518x3 for DinoV2)
    - z: Target vectors [batch_size, z_dim] (e.g., class logits, embeddings)
    - t: Time values [batch_size] or scalar
    
    The model uses DinoV2-ViT-S14 to extract rich image features and combines
    them with z and t information to predict the vector field dz/dt.
    This is a conditional ResNet architecture that can be used as a model
    in NoProp settings (both CT and FM variants).
    
    Note: ViT weights are frozen by default to preserve pretrained features.
    """
    
    # Model configuration
    z_dim: int = 10
    time_embed_dim: int = 128
    hidden_dims: Tuple[int, ...] = (512, 256, 128)
    activation_fn: Callable = nn.swish
    dropout_rate: float = 0.1


    def setup(self):
        """Initialize the ViT model and parameters once during setup."""
        # Load pretrained DinoV2 model and parameters once
        self.vit_model, self.vit_params = load_pretrained_dinov2_vits14()
        
        # Define all submodules in setup()
        self.concat_squash = ConcatSquash(
            features=self.hidden_dims[0], 
            use_bias=True,
            name='concat_squash'
        )
        
        # Create fusion layers
        self.fusion_layers = [
            nn.Dense(hidden_dim, name=f'fusion_{i}') 
            for i, hidden_dim in enumerate(self.hidden_dims)
        ]
        
        # Create dropout layers
        self.dropout_layers = [
            nn.Dropout(self.dropout_rate, name=f'dropout_{i}') 
            for i in range(len(self.hidden_dims))
        ]
        
        # Output layer
        self.output_layer = nn.Dense(self.z_dim, name='output')
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, 
                 training: bool = False) -> jnp.ndarray:
        """Forward pass through the ViT-Conditional ResNet model.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Input images [batch_size, height, width, channels] (518x518x3)
            t: Time values [batch_size] or scalar
            training: Whether in training mode (for dropout)
            
        Returns:
            Vector field dz/dt [batch_size, z_dim]
        """
        # Extract image features using cached DinoV2-ViT-S14
        # DinoV2 outputs [batch_size, 1370, 384] where 1370 = 1 class token + 1369 patch tokens
        x_features = self.vit_model.apply(self.vit_params, x)
        
        # Use class token (first token) as global image representation
        # Shape: [batch_size, 384]
        x_features = x_features[:, 0, :]  # Extract class token
        
        # Process time embedding
        t_embed = get_time_embedding(t, self.time_embed_dim, method="sinusoidal")
                
        # Fuse x, z, and t features using ConcatSquash
        # Input dimensions: x_features (384), z (z_dim), t_features (hidden_dims[0])
        fused = self.concat_squash(x_features, z, t_embed)
        
        # Process through fusion layers with residual connections
        for i, hidden_dim in enumerate(self.hidden_dims):
            
            # Main processing layer
            fused = self.fusion_layers[i](fused)
            fused = self.activation_fn(fused)
            
            
            # Optional dropout
            if training and self.dropout_rate > 0:
                fused = self.dropout_layers[i](fused, deterministic=False)
        
        # Output vector field
        dz_dt = self.output_layer(fused)
        return dz_dt
    

def create_vit_crn_model(z_dim: int = 10, 
                        hidden_dims: Tuple[int, ...] = (512, 256, 128),
                        dropout_rate: float = 0.1) -> ViTCRN:
    """Create a ViT-based Conditional ResNet model with specified configuration.
    
    Args:
        z_dim: Dimension of the target vector z
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Configured ViTCRN model
    """
    return ViTCRN(
        z_dim=z_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    )

