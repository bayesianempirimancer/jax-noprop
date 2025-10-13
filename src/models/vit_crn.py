"""
ViT-based NoProp-FM model using Big Vision ViT-Small feature extractor.

This module demonstrates how to integrate Big Vision pretrained models
into NoProp-FM training for image-based tasks.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Dict, Any, Tuple, Optional, Callable

try:
    # Try relative imports first (when used as a module)
    from ..embeddings.embeddings import get_time_embedding
    from ..layers.concatsquash import ConcatSquash
    # Note: VitSmallFeatureExtractor is not available in current structure
    # from .pretrained.vit_small_feature_extractor import (
    #     VitSmallFeatureExtractor,
    #     create_vit_gradient_stop_fn
    # )
except ImportError:
    # Fall back to absolute imports (when run directly)
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.embeddings.embeddings import get_time_embedding
    from src.layers.concatsquash import ConcatSquash
    # Note: VitSmallFeatureExtractor is not available in current structure
    # from src.models.pretrained.vit_small_feature_extractor import (
    #     VitSmallFeatureExtractor,
    #     create_vit_gradient_stop_fn
    # )


class ViTCRN(nn.Module):
    """ViT-based Conditional ResNet for image-to-vector tasks.
    
    This model is designed for image-to-vector tasks where:
    - x: Input images [batch_size, height, width, channels]
    - z: Target vectors [batch_size, z_dim] (e.g., class logits, embeddings)
    - t: Time values [batch_size] or scalar
    
    The model uses ViT-Small to extract rich image features and combines
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
    freeze_vit: bool = True  # Whether to freeze ViT gradients
    use_pretrained_vit: bool = True  # Whether to use pretrained ViT weights
    
    # ViT configuration
    vit_patch_size: int = 16
    vit_num_layers: int = 12
    vit_hidden_dim: int = 384
    vit_num_heads: int = 6
    vit_mlp_dim: int = 1536
    
    def setup(self):
        """Setup the model components."""
        # ViT feature extractor for x (images)
        self.vit_extractor = VitSmallFeatureExtractor(pretrained=self.use_pretrained_vit)
        
        # Time embedding
        self.time_embed = nn.Dense(self.time_embed_dim)
        
        # ConcatSquash layer to combine x_features, z, and t_features
        # Output dimension should match the last hidden_dim (which would be z_processed shape)
        concat_output_dim = self.hidden_dims[-1] if self.hidden_dims else self.z_dim
        self.concat_squash = ConcatSquash(features=concat_output_dim, use_bias=True)
        
        # Fusion layers
        self.fusion_layers = [nn.Dense(dim, name=f'fusion_dense_{i}') for i, dim in enumerate(self.hidden_dims)]
        
        # Output layer
        self.output_layer = nn.Dense(self.z_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, 
                 training: bool = False) -> jnp.ndarray:
        """Forward pass through the ViT-NoProp-FM model.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Input images [batch_size, height, width, channels]
            t: Time values [batch_size] or scalar
            training: Whether in training mode (for dropout)
            
        Returns:
            Vector field dz/dt [batch_size, z_dim]
        """
        # Extract image features using ViT
        x_features = self.vit_extractor(x)  # [batch_size, vit_hidden_dim]
        
        # Stop gradients at ViT output if freeze_vit is True
        # This prevents gradient computation through the ViT entirely
        if self.freeze_vit:
            x_features = jax.lax.stop_gradient(x_features)
        
        # Process time embedding
        t_embed = get_time_embedding(t, self.time_embed_dim, method="sinusoidal")
        t_features = self.time_embed(t_embed)
        
        # Fuse x, z, and t features using ConcatSquash
        # This replaces the z processing layers and concatenation
        combined = self.concat_squash(x_features, z, t_features)
        
        # Process through fusion layers
        fused = combined
        for layer in self.fusion_layers:
            fused = layer(fused)
            fused = self.activation_fn(fused)
            # Note: Dropout disabled for now to avoid RNG issues
            # if training:
            #     fused = self.dropout(fused, deterministic=False)
        
        # Output vector field
        dz_dt = self.output_layer(fused)
        return dz_dt
    

def create_vit_crn_model(z_dim: int = 10, 
                        hidden_dims: Tuple[int, ...] = (512, 256, 128),
                        dropout_rate: float = 0.1,
                        freeze_vit: bool = True) -> ViTCRN:
    """Create a ViT-based Conditional ResNet model with specified configuration.
    
    Args:
        z_dim: Dimension of the target vector z
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate for regularization
        freeze_vit: Whether to freeze ViT gradients
        
    Returns:
        Configured ViTCRN model
    """
    return ViTCRN(
        z_dim=z_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        freeze_vit=freeze_vit
    )


# Example usage and testing functions
def test_vit_crn():
    """Test the ViT-based Conditional ResNet model."""
    print("Testing ViT-based Conditional ResNet model...")
    
    # Create model
    model = create_vit_crn_model(z_dim=10, hidden_dims=(256, 128))
    
    # Create test data
    batch_size = 4
    z = jnp.ones((batch_size, 10))
    x = jnp.ones((batch_size, 224, 224, 3))
    t = jnp.ones(batch_size)
    
    # Initialize model
    key = jax.random.PRNGKey(42)
    params = model.init(key, z, x, t, training=False)
    
    # Forward pass
    dz_dt = model.apply(params, z, x, t, training=False)
    
    print(f"✓ Input z shape: {z.shape}")
    print(f"✓ Input x shape: {x.shape}")
    print(f"✓ Input t shape: {t.shape}")
    print(f"✓ Output dz/dt shape: {dz_dt.shape}")
    
    return model, params


def test_vit_crn_with_jacobian():
    """Test the ViT-based Conditional ResNet model with Jacobian trace computation."""
    print("\nTesting ViT-based Conditional ResNet model with Jacobian trace...")
    
    # Create model
    model = create_vit_crn_model(z_dim=10, hidden_dims=(256, 128))
    
    # Create test data
    batch_size = 2  # Smaller batch for Jacobian computation
    z = jnp.ones((batch_size, 10))
    x = jnp.ones((batch_size, 224, 224, 3))
    t = jnp.ones(batch_size)
    
    # Initialize model
    key = jax.random.PRNGKey(42)
    params = model.init(key, z, x, t, training=False)
    
    # Forward pass
    dz_dt = model.apply(params, z, x, t, training=False)
    
    # Compute Jacobian trace using centralized function
    from ..utils.jacobian_utils import trace_jacobian
    jacobian_trace = trace_jacobian(model.apply, params, z, x, t)
    
    print(f"✓ Input z shape: {z.shape}")
    print(f"✓ Input x shape: {x.shape}")
    print(f"✓ Input t shape: {t.shape}")
    print(f"✓ Output dz/dt shape: {dz_dt.shape}")
    print(f"✓ Jacobian trace shape: {jacobian_trace.shape}")
    
    return model, params


if __name__ == "__main__":
    print("ViT-based Conditional ResNet Model Test")
    print("=" * 50)
    
    # Test basic model
    model, params = test_vit_crn()
    
    # Test model with Jacobian trace computation
    model_jac, params_jac = test_vit_crn_with_jacobian()
    
    print("\n" + "=" * 50)
    print("ViT-based Conditional ResNet Model Test Complete!")
    print("=" * 50)
    print("\nThe model is ready for integration with NoProp training.")
    print("Key features:")
    print("- Uses ViT-Small for rich image feature extraction")
    print("- Freezes ViT weights to preserve pretrained features")
    print("- Combines image features with z and t information")
    print("- Outputs vector field dz/dt for NoProp integration")
    print("- Can be used with both NoProp-CT and NoProp-FM")
