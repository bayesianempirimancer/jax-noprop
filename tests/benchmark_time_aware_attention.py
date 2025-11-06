#!/usr/bin/env python3
"""
Benchmark script to compare runtime costs of time-aware attention 
with and without time conditioning.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from src.flow_models.crn_seq import TransformerSeq2SeqConditionalResnet

def benchmark_forward_pass(forward_fn, z, x, t, rngs, num_warmup=20, num_trials=50):
    """Benchmark a forward pass using a JIT-compiled function, averaging over multiple trials."""
    # Warmup to ensure compilation is complete
    for _ in range(num_warmup):
        if t is not None:
            _ = forward_fn(z, x, t, rngs)
        else:
            _ = forward_fn(z, x, rngs)
        _ = jax.block_until_ready(_)
    
    # Time the forward pass (compilation already done)
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        if t is not None:
            output = forward_fn(z, x, t, rngs)
        else:
            output = forward_fn(z, x, rngs)
        _ = jax.block_until_ready(output)  # Ensure computation completes
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times), output

def benchmark_time_aware_attention():
    """Benchmark time-aware attention vs no time conditioning."""
    print("=" * 80)
    print("Benchmark: Time-aware attention vs no time conditioning")
    print("=" * 80)
    
    # Model parameters
    z_seq_len = 48  # Target sequence length
    x_seq_len = 36  # Conditional sequence length
    batch_size = 32  # Fixed batch size
    time_embed_dim = 32
    
    # Test different embedding dimensions
    embed_dims = [32, 64, 128, 192, 256]
    
    results = []
    
    for embed_dim in embed_dims:
        print(f"\n{'='*80}")
        print(f"Embedding dimension: {embed_dim} (Batch size: {batch_size})")
        print(f"{'='*80}")
        
        # Create model
        model = TransformerSeq2SeqConditionalResnet(
            latent_shape=(z_seq_len, embed_dim),
            input_shape=(x_seq_len, embed_dim),
            output_shape=(z_seq_len, embed_dim),
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
            x_static_dim=0,
        )
        
        # Initialize
        key = jr.PRNGKey(42)
        key1, key2, key3 = jr.split(key, 3)
        
        # Create dummy inputs
        z = jr.normal(key1, (batch_size, z_seq_len, embed_dim))
        x = jr.normal(key2, (batch_size, x_seq_len, embed_dim))
        t = jr.uniform(key3, (batch_size,), minval=0.0, maxval=1.0)
        
        # Initialize parameters
        params = model.init(key, z, x, t, training=False)
        
        # JIT compile
        print("JIT compiling...")
        key, rng_key = jr.split(key)
        rngs = {'dropout': rng_key}
        
        # JIT compile both versions
        @jax.jit
        def forward_with_time(z, x, t, rngs):
            return model.apply(params, z, x, t, training=False, rngs=rngs)
        
        @jax.jit
        def forward_no_time(z, x, rngs):
            return model.apply(params, z, x, None, training=False, rngs=rngs)
        
        # Do initial compilation and warmup (this may include compilation time)
        print("  Compiling and warming up...")
        _ = forward_with_time(z, x, t, rngs)
        _ = forward_no_time(z, x, rngs)
        _ = jax.block_until_ready(_)
        
        # Additional warmup to ensure compilation is complete
        for _ in range(10):
            _ = forward_with_time(z, x, t, rngs)
            _ = forward_no_time(z, x, rngs)
        _ = jax.block_until_ready(_)
        
        print("Benchmarking...")
        
        # Benchmark with time (using pre-compiled function)
        key, rng_key = jr.split(key)
        rngs = {'dropout': rng_key}
        time_mean, time_std, _ = benchmark_forward_pass(
            forward_with_time, z, x, t, rngs, num_warmup=0, num_trials=50
        )
        
        # Benchmark without time (using pre-compiled function)
        key, rng_key = jr.split(key)
        rngs = {'dropout': rng_key}
        no_time_mean, no_time_std, _ = benchmark_forward_pass(
            forward_no_time, z, x, None, rngs, num_warmup=0, num_trials=50
        )
        
        # Calculate overhead
        overhead_ms = (time_mean - no_time_mean) * 1000
        overhead_std_ms = np.sqrt(time_std**2 + no_time_std**2) * 1000
        overhead_pct = (time_mean / no_time_mean - 1.0) * 100
        
        print(f"\nResults:")
        print(f"  With time:    {time_mean*1000:.3f} ± {time_std*1000:.3f} ms")
        print(f"  Without time: {no_time_mean*1000:.3f} ± {no_time_std*1000:.3f} ms")
        print(f"  Overhead:     {overhead_ms:.3f} ± {overhead_std_ms:.3f} ms ({overhead_pct:.1f}%)")
        
        results.append({
            'embed_dim': embed_dim,
            'with_time_ms': time_mean * 1000,
            'without_time_ms': no_time_mean * 1000,
            'overhead_ms': overhead_ms,
            'overhead_pct': overhead_pct
        })
    
    # Summary table
    print(f"\n{'='*80}")
    print("Summary Table")
    print(f"{'='*80}")
    print(f"{'Embed Dim':<12} {'With Time (ms)':<18} {'Without Time (ms)':<20} {'Overhead (ms)':<15} {'Overhead (%)':<15}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['embed_dim']:<12} {r['with_time_ms']:<18.3f} {r['without_time_ms']:<20.3f} {r['overhead_ms']:<15.3f} {r['overhead_pct']:<15.1f}")
    
    print(f"\n{'='*80}")
    print("Benchmark complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    benchmark_time_aware_attention()

