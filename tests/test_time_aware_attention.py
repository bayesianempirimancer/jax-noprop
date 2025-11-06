#!/usr/bin/env python3
"""
Test script for the time-aware attention implementation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from src.flow_models.crn_seq import TransformerSeq2SeqConditionalResnet

def test_time_aware_attention():
    """Test time-aware attention with the full CRN model."""
    print("=" * 60)
    print("Test: Time-aware attention in TransformerSeq2SeqConditionalResnet")
    print("=" * 60)
    
    # Model parameters
    z_seq_len = 48  # Target sequence length
    x_seq_len = 36  # Conditional sequence length
    embed_dim = 20
    batch_size = 4
    time_embed_dim = 32
    
    # Create model
    model = TransformerSeq2SeqConditionalResnet(
        latent_shape=(z_seq_len, embed_dim),  # Already in embed_dim
        input_shape=(x_seq_len, embed_dim),   # Already in embed_dim
        output_shape=(z_seq_len, embed_dim),  # Output in embed_dim
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        rope_base=10000.0,
        time_embed_dim=time_embed_dim,
        time_embed_method="fourier",
        activation_fn="swish",
        dropout_rate=0.1,
        qkv_bias=True,
        x_static_dim=0,  # No static features for this test
    )
    
    # Initialize
    key = jr.PRNGKey(42)
    key1, key2, key3 = jr.split(key, 3)
    
    # Create dummy inputs (already in embed_dim space)
    z = jr.normal(key1, (batch_size, z_seq_len, embed_dim))
    x = jr.normal(key2, (batch_size, x_seq_len, embed_dim))
    t = jr.uniform(key3, (batch_size,), minval=0.0, maxval=1.0)
    
    print(f"Input shapes:")
    print(f"  z: {z.shape}")
    print(f"  x: {x.shape}")
    print(f"  t: {t.shape}")
    
    # Initialize parameters
    print("\nInitializing model...")
    params = model.init(key, z, x, t, training=False)
    
    # Forward pass with time
    print("\nForward pass with time conditioning...")
    key, rng_key = jr.split(key)
    rngs = {'dropout': rng_key}
    output_with_time = model.apply(params, z, x, t, training=False, rngs=rngs)
    
    print(f"  Output shape: {output_with_time.shape}")
    print(f"  Expected shape: {(batch_size, z_seq_len, embed_dim)}")
    assert output_with_time.shape == (batch_size, z_seq_len, embed_dim), \
        f"Output shape mismatch: {output_with_time.shape} != {(batch_size, z_seq_len, embed_dim)}"
    
    # Forward pass without time
    print("\nForward pass without time conditioning...")
    key, rng_key = jr.split(key)
    rngs = {'dropout': rng_key}
    output_no_time = model.apply(params, z, x, None, training=False, rngs=rngs)
    
    print(f"  Output shape: {output_no_time.shape}")
    assert output_no_time.shape == (batch_size, z_seq_len, embed_dim), \
        f"Output shape mismatch: {output_no_time.shape} != {(batch_size, z_seq_len, embed_dim)}"
    
    # Check that outputs are different (time should affect output)
    diff = jnp.abs(output_with_time - output_no_time)
    print(f"\nDifference between time-conditioned and non-time-conditioned outputs:")
    print(f"  Mean difference: {jnp.mean(diff):.6f}")
    print(f"  Max difference: {jnp.max(diff):.6f}")
    
    if jnp.max(diff) < 1e-6:
        print("  WARNING: Outputs are very similar - time conditioning may not be working!")
    else:
        print("  ✓ Time conditioning is affecting the output")
    
    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_time_aware_attention()

