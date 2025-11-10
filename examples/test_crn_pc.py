"""
Test script for PointCloudConditionalResnet.

This script tests the point cloud CRN implementation to ensure it works correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.flow_models.crn_pc import PointCloudConditionalResnet, Config
from flax.core import FrozenDict


def test_point_cloud_crn():
    """Test the PointCloudConditionalResnet model."""
    print("="*60)
    print("Testing PointCloudConditionalResnet")
    print("="*60)
    
    # Test parameters
    batch_size = 2
    point_dim = 3  # 3D points
    z_num_points = 50
    feature_dim = 16  # Feature dimension for both x and z
    x_num_points = 30  # Variable number of points
    embed_dim = 64
    
    # Create config
    config_dict = FrozenDict({
        "point_dim": point_dim,
        "z_num_points": z_num_points,
        "feature_dim": feature_dim,
        "embed_dim": embed_dim,
        "fourier_num_frequencies": 8,
        "fourier_include_original": True,
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "activation_fn": "swish",
        "dropout_rate": 0.1,
        "num_layers": 2,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "time_conditioning_method": "film",
    })
    
    # Create model
    model = PointCloudConditionalResnet(config=config_dict)
    
    # Compute Fourier feature dimension
    fourier_feature_dim = point_dim * 2 * config_dict["fourier_num_frequencies"]
    if config_dict["fourier_include_original"]:
        fourier_feature_dim += point_dim
    
    print(f"\nModel configuration:")
    print(f"  - Point dimension: {point_dim}D")
    print(f"  - z: {z_num_points} points, {feature_dim} features each")
    print(f"  - x: variable points, {feature_dim} features each")
    print(f"  - Embedding dimension: {embed_dim}")
    print(f"  - Fourier feature dim: {fourier_feature_dim}")
    
    # Create test inputs
    key = jax.random.PRNGKey(42)
    key_z, key_x, key_t = jax.random.split(key, 3)
    
    # z: [batch, z_num_points, point_dim + feature_dim]
    z = jax.random.normal(
        key_z, 
        (batch_size, z_num_points, point_dim + feature_dim)
    )
    
    # x: [batch, x_num_points, point_dim + feature_dim]
    x = jax.random.normal(
        key_x,
        (batch_size, x_num_points, point_dim + feature_dim)
    )
    
    # t: [batch]
    t = jax.random.uniform(key_t, (batch_size,), minval=0.0, maxval=1.0)
    
    print(f"\nInput shapes:")
    print(f"  - z: {z.shape}")
    print(f"  - x: {x.shape}")
    print(f"  - t: {t.shape}")
    
    # Initialize model
    print(f"\nInitializing model...")
    params = model.init(key, z, x, t, training=True)
    
    # Forward pass
    print(f"\nRunning forward pass...")
    key_forward = jax.random.PRNGKey(123)
    output = model.apply(params, z, x, t, training=True, rngs={'dropout': key_forward})
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: {z.shape}")
    
    # Verify output shape matches input z shape
    assert output.shape == z.shape, (
        f"Output shape {output.shape} does not match input z shape {z.shape}"
    )
    print("✅ Output shape matches input z shape!")
    
    # Test without x
    print(f"\n" + "="*60)
    print("Testing without x (z only)")
    print("="*60)
    key_forward = jax.random.PRNGKey(123)
    output_no_x = model.apply(params, z, x=None, t=t, training=True, rngs={'dropout': key_forward})
    assert output_no_x.shape == z.shape
    print(f"✅ Works without x! Output shape: {output_no_x.shape}")
    
    # Test without t
    print(f"\n" + "="*60)
    print("Testing without t (no time conditioning)")
    print("="*60)
    key_forward = jax.random.PRNGKey(123)
    output_no_t = model.apply(params, z, x=x, t=None, training=True, rngs={'dropout': key_forward})
    assert output_no_t.shape == z.shape
    print(f"✅ Works without t! Output shape: {output_no_t.shape}")
    
    # Test with x_mask
    print(f"\n" + "="*60)
    print("Testing with x_mask")
    print("="*60)
    x_mask = jnp.ones((batch_size, x_num_points), dtype=bool)
    x_mask = x_mask.at[:, :x_num_points//2].set(False)  # Mask first half
    key_forward = jax.random.PRNGKey(123)
    output_masked = model.apply(params, z, x=x, t=t, x_mask=x_mask, training=True, rngs={'dropout': key_forward})
    assert output_masked.shape == z.shape
    print(f"✅ Works with x_mask! Output shape: {output_masked.shape}")
    
    # Test 2D points
    print(f"\n" + "="*60)
    print("Testing 2D points")
    print("="*60)
    config_2d = FrozenDict({
        "point_dim": 2,
        "z_num_points": z_num_points,
        "feature_dim": feature_dim,
        "embed_dim": embed_dim,
        "fourier_num_frequencies": 8,
        "fourier_include_original": True,
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "activation_fn": "swish",
        "dropout_rate": 0.1,
        "num_layers": 2,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "time_conditioning_method": "film",
    })
    model_2d = PointCloudConditionalResnet(config=config_2d)
    
    z_2d = jax.random.normal(key_z, (batch_size, z_num_points, 2 + feature_dim))
    x_2d = jax.random.normal(key_x, (batch_size, x_num_points, 2 + feature_dim))
    
    params_2d = model_2d.init(key, z_2d, x_2d, t, training=True)
    key_forward = jax.random.PRNGKey(123)
    output_2d = model_2d.apply(params_2d, z_2d, x_2d, t, training=True, rngs={'dropout': key_forward})
    
    assert output_2d.shape == z_2d.shape
    print(f"✅ 2D points work! Output shape: {output_2d.shape}")
    
    # Test standard attention with time conditioning
    print(f"\n" + "="*60)
    print("Testing standard attention with time conditioning")
    print("="*60)
    config_standard = FrozenDict({
        "point_dim": point_dim,
        "z_num_points": z_num_points,
        "feature_dim": feature_dim,
        "embed_dim": embed_dim,
        "fourier_num_frequencies": 8,
        "fourier_include_original": True,
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "activation_fn": "swish",
        "dropout_rate": 0.1,
        "num_layers": 2,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "time_conditioning_method": "film",  # Use FiLM time conditioning
    })
    model_standard = PointCloudConditionalResnet(config=config_standard)
    
    params_standard = model_standard.init(key, z, x, t, training=True)
    key_forward = jax.random.PRNGKey(123)
    output_standard = model_standard.apply(params_standard, z, x, t, training=True, rngs={'dropout': key_forward})
    
    assert output_standard.shape == z.shape
    print(f"✅ Standard attention works! Output shape: {output_standard.shape}")
    
    # Test standard attention without time (should work fine)
    output_standard_no_t = model_standard.apply(params_standard, z, x, t=None, training=True, rngs={'dropout': key_forward})
    assert output_standard_no_t.shape == z.shape
    print(f"✅ Standard attention works without time! Output shape: {output_standard_no_t.shape}")
    
    print(f"\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_point_cloud_crn()

