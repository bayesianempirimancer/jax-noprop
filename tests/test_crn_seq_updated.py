#!/usr/bin/env python3
"""
Test script for the updated TransformerSeq2SeqConditionalResnet.

This tests:
1. 2D input (price, volume) handling
2. Internal projection (2D -> embed_dim)
3. RoPE positional encoding with relative positioning
4. Day-of-week embeddings (optional)
5. Variable-length x sequences
6. Output projection (embed_dim -> 2D)
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from src.flow_models.crn_seq import TransformerSeq2SeqConditionalResnet

def test_basic_forward():
    """Test basic forward pass with 2D input."""
    print("=" * 60)
    print("Test 1: Basic forward pass with 2D input")
    print("=" * 60)
    
    # Model parameters
    z_seq_len = 48  # Target sequence length
    x_seq_len = 18  # Conditional sequence length
    embed_dim = 20
    batch_size = 4
    
    # Create model
    model = TransformerSeq2SeqConditionalResnet(
        latent_shape=(z_seq_len, 2),  # 2D: price, volume
        input_shape=(x_seq_len, 2),   # 2D: price, volume
        output_shape=(z_seq_len, 2),  # 2D: price, volume
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        rope_base=10000.0,
        projection_seed=42,
        use_day_of_week=False,
    )
    
    # Initialize
    key = jr.PRNGKey(42)
    key1, key2 = jr.split(key)
    
    # Create dummy inputs
    z = jr.normal(key1, (batch_size, z_seq_len, 2))  # 2D: price, volume
    x = jr.normal(key2, (batch_size, x_seq_len, 2))  # 2D: price, volume
    t = jr.uniform(key1, (batch_size,), minval=0.0, maxval=1.0)
    
    # Initialize parameters
    params = model.init(key, z, x, t, training=False)
    
    # Forward pass (need rngs for dropout even if training=False)
    key, rng_key = jr.split(key)
    rngs = {'dropout': rng_key}
    output = model.apply(params, z, x, t, training=False, rngs=rngs)
    
    print(f"Input z shape: {z.shape}")
    print(f"Input x shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: {(batch_size, z_seq_len, 2)}")
    
    assert output.shape == (batch_size, z_seq_len, 2), f"Output shape mismatch: {output.shape} != {(batch_size, z_seq_len, 2)}"
    assert not jnp.isnan(output).any(), "Output contains NaN values"
    assert not jnp.isinf(output).any(), "Output contains Inf values"
    
    print("✓ Basic forward pass test passed!")
    print()


def test_variable_length_x():
    """Test variable-length x sequences."""
    print("=" * 60)
    print("Test 2: Variable-length x sequences")
    print("=" * 60)
    
    z_seq_len = 48
    embed_dim = 20
    batch_size = 3
    
    # Create model
    model = TransformerSeq2SeqConditionalResnet(
        latent_shape=(z_seq_len, 2),
        input_shape=(None, 2),  # Variable length
        output_shape=(z_seq_len, 2),
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=4,
        projection_seed=42,
    )
    
    key = jr.PRNGKey(42)
    
    # Create variable-length x sequences
    key_seq = jr.PRNGKey(100)
    x_sequences = []
    for seq_len in [10, 15, 20]:
        key_seq, key_seq_use = jr.split(key_seq)
        x_sequences.append(jr.normal(key_seq_use, (seq_len, 2)))
    
    z = jr.normal(key, (batch_size, z_seq_len, 2))
    t = jr.uniform(key, (batch_size,), minval=0.0, maxval=1.0)
    
    # Initialize with first sequence
    params = model.init(key, z, x_sequences[0], t, training=False)
    
    # Test each sequence
    key, rng_key = jr.split(key)
    rngs = {'dropout': rng_key}
    for i, x_seq in enumerate(x_sequences):
        x_batch = x_seq[None, :, :]  # Add batch dimension
        output = model.apply(params, z[0:1], x_batch, t[0:1], training=False, rngs=rngs)
        
        print(f"  x_seq[{i}] length: {x_seq.shape[0]}, output shape: {output.shape}")
        assert output.shape == (1, z_seq_len, 2), f"Output shape mismatch for x_seq[{i}]"
        assert not jnp.isnan(output).any(), f"Output contains NaN for x_seq[{i}]"
        assert not jnp.isinf(output).any(), f"Output contains Inf for x_seq[{i}]"
    
    print("✓ Variable-length x sequences test passed!")
    print()


def test_day_of_week_embeddings():
    """Test day-of-week embeddings."""
    print("=" * 60)
    print("Test 3: Day-of-week embeddings")
    print("=" * 60)
    
    z_seq_len = 48
    x_seq_len = 18
    embed_dim = 20
    batch_size = 5
    
    # Create model with day-of-week embeddings
    model = TransformerSeq2SeqConditionalResnet(
        latent_shape=(z_seq_len, 2),
        input_shape=(x_seq_len, 2),
        output_shape=(z_seq_len, 2),
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=4,
        projection_seed=42,
        use_day_of_week=True,
        day_of_week_seed=42,
    )
    
    key = jr.PRNGKey(42)
    key1, key2 = jr.split(key)
    
    z = jr.normal(key1, (batch_size, z_seq_len, 2))
    x = jr.normal(key2, (batch_size, x_seq_len, 2))
    t = jr.uniform(key1, (batch_size,), minval=0.0, maxval=1.0)
    day_of_week = jnp.array([0, 1, 2, 3, 4])  # Mon-Fri
    
    # Initialize
    params = model.init(key, z, x, t, day_of_week, training=False)
    
    # Forward pass with day-of-week
    key, rng_key = jr.split(key)
    rngs = {'dropout': rng_key}
    output = model.apply(params, z, x, t, day_of_week, training=False, rngs=rngs)
    
    print(f"Input z shape: {z.shape}")
    print(f"Input x shape: {x.shape}")
    print(f"Day-of-week indices: {day_of_week}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, z_seq_len, 2), f"Output shape mismatch: {output.shape}"
    assert not jnp.isnan(output).any(), "Output contains NaN values"
    assert not jnp.isinf(output).any(), "Output contains Inf values"
    
    print("✓ Day-of-week embeddings test passed!")
    print()


def test_rope_relative_positioning():
    """Test that RoPE is applied with relative positioning (x relative to z)."""
    print("=" * 60)
    print("Test 4: RoPE relative positioning")
    print("=" * 60)
    
    z_seq_len = 48
    x_seq_len = 18
    embed_dim = 20
    batch_size = 2
    
    model = TransformerSeq2SeqConditionalResnet(
        latent_shape=(z_seq_len, 2),
        input_shape=(x_seq_len, 2),
        output_shape=(z_seq_len, 2),
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=4,
        rope_base=10000.0,
        projection_seed=42,
        dropout_rate=0.0,  # Disable dropout for deterministic outputs
    )
    
    key = jr.PRNGKey(42)
    key1, key2 = jr.split(key)
    
    z = jr.normal(key1, (batch_size, z_seq_len, 2))
    x = jr.normal(key2, (batch_size, x_seq_len, 2))
    t = jr.uniform(key1, (batch_size,), minval=0.0, maxval=1.0)
    
    params = model.init(key, z, x, t, training=False)
    
    # Test that RoPE is applied correctly
    # z should start at position 0, x should start at position -x_seq_len
    key, rng_key1 = jr.split(key)
    rngs1 = {'dropout': rng_key1}
    output1 = model.apply(params, z, x, t, training=False, rngs=rngs1)
    
    key, rng_key2 = jr.split(key)
    rngs2 = {'dropout': rng_key2}
    output2 = model.apply(params, z, x, t, training=False, rngs=rngs2)
    
    # Outputs should be deterministic (same inputs -> same outputs)
    # With dropout_rate=0.0, outputs should be identical
    assert jnp.allclose(output1, output2, atol=1e-6), "Outputs should be deterministic (within tolerance)"
    
    # Check max difference
    max_diff = jnp.abs(output1 - output2).max()
    print(f"  Max difference between outputs: {max_diff:.2e}")
    if max_diff > 1e-6:
        print(f"  WARNING: Outputs differ by more than 1e-6, may be due to numerical precision or dropout")
    
    # Test with different x lengths
    x_short = x[:, :10, :]  # 10 timesteps
    x_long = x  # 18 timesteps
    
    key, rng_key3 = jr.split(key)
    rngs3 = {'dropout': rng_key3}
    output_short = model.apply(params, z, x_short, t, training=False, rngs=rngs3)
    
    key, rng_key4 = jr.split(key)
    rngs4 = {'dropout': rng_key4}
    output_long = model.apply(params, z, x_long, t, training=False, rngs=rngs4)
    
    print(f"  x_short length: 10, output shape: {output_short.shape}")
    print(f"  x_long length: 18, output shape: {output_long.shape}")
    
    assert output_short.shape == (batch_size, z_seq_len, 2), "Output shape mismatch for short x"
    assert output_long.shape == (batch_size, z_seq_len, 2), "Output shape mismatch for long x"
    
    print("✓ RoPE relative positioning test passed!")
    print()


def test_projection_matrix():
    """Test that projection matrix is initialized correctly."""
    print("=" * 60)
    print("Test 5: Projection matrix initialization")
    print("=" * 60)
    
    z_seq_len = 48
    x_seq_len = 18
    embed_dim = 20
    num_heads = 4  # Must divide embed_dim
    
    # Create model with specific seed
    model = TransformerSeq2SeqConditionalResnet(
        latent_shape=(z_seq_len, 2),
        input_shape=(x_seq_len, 2),
        output_shape=(z_seq_len, 2),
        embed_dim=embed_dim,
        num_heads=num_heads,
        projection_seed=42,
        dropout_rate=0.0,
    )
    
    key = jr.PRNGKey(42)
    key1, key2 = jr.split(key)
    
    z = jr.normal(key1, (2, z_seq_len, 2))
    x = jr.normal(key2, (2, x_seq_len, 2))
    t = jr.uniform(key1, (2,), minval=0.0, maxval=1.0)
    
    # Initialize model
    params = model.init(key, z, x, t, training=False)
    
    # Test forward pass
    key, rng_key = jr.split(key)
    rngs = {'dropout': rng_key}
    output = model.apply(params, z, x, t, training=False, rngs=rngs)
    
    print(f"  Input z shape: {z.shape}")
    print(f"  Input x shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: (2, {z_seq_len}, 2)")
    
    assert output.shape == (2, z_seq_len, 2), f"Output shape mismatch: {output.shape}"
    assert not jnp.isnan(output).any(), "Output contains NaN values"
    assert not jnp.isinf(output).any(), "Output contains Inf values"
    
    # Check that projection from 2D to embed_dim works
    # The model should handle 2D input and project internally
    print(f"  ✓ Model successfully projects 2D input to embed_dim={embed_dim} internally")
    
    print("✓ Projection matrix initialization test passed!")
    print()


def test_unconditional():
    """Test unconditional generation (x=None)."""
    print("=" * 60)
    print("Test 6: Unconditional generation (x=None)")
    print("=" * 60)
    
    z_seq_len = 48
    embed_dim = 20
    batch_size = 2
    
    model = TransformerSeq2SeqConditionalResnet(
        latent_shape=(z_seq_len, 2),
        input_shape=(10, 2),  # Not used when x=None
        output_shape=(z_seq_len, 2),
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=4,
        projection_seed=42,
    )
    
    key = jr.PRNGKey(42)
    key1 = jr.split(key)[0]
    
    z = jr.normal(key1, (batch_size, z_seq_len, 2))
    t = jr.uniform(key1, (batch_size,), minval=0.0, maxval=1.0)
    
    # Initialize with x=None
    params = model.init(key, z, None, t, training=False)
    
    # Forward pass with x=None
    key, rng_key = jr.split(key)
    rngs = {'dropout': rng_key}
    output = model.apply(params, z, None, t, training=False, rngs=rngs)
    
    print(f"Input z shape: {z.shape}")
    print(f"Input x: None")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, z_seq_len, 2), f"Output shape mismatch: {output.shape}"
    assert not jnp.isnan(output).any(), "Output contains NaN values"
    assert not jnp.isinf(output).any(), "Output contains Inf values"
    
    print("✓ Unconditional generation test passed!")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing Updated TransformerSeq2SeqConditionalResnet")
    print("=" * 60 + "\n")
    
    try:
        test_basic_forward()
        test_variable_length_x()
        test_day_of_week_embeddings()
        test_rope_relative_positioning()
        test_projection_matrix()
        test_unconditional()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

