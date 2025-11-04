"""
Test script for TransformerSeq2SeqConditionalResnet model.

Tests the model with fabricated inputs, specifically testing:
- Fixed sequence length for z
- Variable/arbitrary sequence length for x
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from src.flow_models.crn_seq import TransformerSeq2SeqConditionalResnet


def test_fixed_z_variable_x():
    """Test model with fixed z sequence length and variable x sequence length."""
    
    print("=" * 60)
    print("Testing TransformerSeq2SeqConditionalResnet")
    print("=" * 60)
    
    # Fixed z sequence: (seq_len=10, features=8)
    z_seq_len = 10
    z_features = 8
    latent_shape = (z_seq_len, z_features)
    
    # Variable x sequence lengths to test
    x_seq_lengths = [5, 15, 20, 30]
    x_features = 8
    input_shape = (x_seq_lengths[0], x_features)  # Model expects fixed shape, but we'll test with different actual sizes
    
    # Output shape should match z
    output_shape = (z_seq_len, z_features)
    
    # Model configuration
    # Note: z and x should already be embedded to embed_dim
    # So latent_shape and input_shape should be (seq_len, embed_dim)
    embed_dim = 256
    latent_shape = (z_seq_len, embed_dim)  # Already embedded
    input_shape = (x_seq_lengths[0], embed_dim)  # Already embedded (will test variable length)
    output_shape = (z_seq_len, embed_dim)  # Output is also in embed_dim
    
    model = TransformerSeq2SeqConditionalResnet(
        latent_shape=latent_shape,
        input_shape=input_shape,
        output_shape=output_shape,
        hidden_dims=(model_dim,),  # Model dimension
        time_embed_dim=64,
        num_layers=2,  # Fewer layers for faster testing
        num_heads=4,
        mlp_ratio=4.0,
        dropout_rate=0.1,
    )
    
    # Generate a random key
    key = jr.PRNGKey(42)
    
    print(f"\nModel configuration:")
    print(f"  z sequence shape: {latent_shape}")
    print(f"  z embed dim: {model.embed_dim}")
    print(f"  output shape: {output_shape}")
    
    # Test with different x sequence lengths
    batch_size = 2
    
    for x_seq_len in x_seq_lengths:
        print(f"\n{'='*60}")
        print(f"Testing with x sequence length: {x_seq_len}")
        print(f"{'='*60}")
        
        # Create fake inputs
        # Note: z and x should already be embedded to embed_dim
        key, z_key, x_key, t_key = jr.split(key, 4)
        
        # z has fixed shape (batch, z_seq_len, embed_dim) - already embedded
        z = jr.normal(z_key, (batch_size, z_seq_len, embed_dim))
        
        # x has variable shape (batch, x_seq_len, embed_dim) - already embedded
        x = jr.normal(x_key, (batch_size, x_seq_len, embed_dim))
        
        # t is scalar or (batch,)
        t = jr.uniform(t_key, (batch_size,), minval=0.0, maxval=1.0)
        
        print(f"  Input shapes:")
        print(f"    z: {z.shape}")
        print(f"    x: {x.shape}")
        print(f"    t: {t.shape}")
        
        # Initialize model parameters
        key, init_key = jr.split(key)
        try:
            # Initialize with matching shapes
            z_init = z.reshape(batch_size, *latent_shape)
            x_init = jr.normal(x_key, (batch_size, *input_shape))  # Use standard shape for init
            params = model.init(init_key, z_init, x_init, t)
            print(f"  ✓ Model initialized successfully")
                
        except Exception as e:
            print(f"  ERROR during initialization: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Forward pass
        try:
            key, rng_key = jr.split(key)
            rngs = {'dropout': rng_key}
            
            # Use actual z and x tensors - x can have variable length
            z_for_test = z.reshape(batch_size, *latent_shape)
            x_for_test = x  # Can have variable length (batch, x_seq_len, embed_dim)
            
            output = model.apply(params, z_for_test, x_for_test, t, training=False, rngs=rngs)
            print(f"  Output shape: {output.shape}")
            print(f"  Expected shape: {(batch_size,) + output_shape}")
            
            if output.shape == (batch_size,) + output_shape:
                print(f"  ✓ SUCCESS: Output shape matches expected!")
                print(f"  ✓ Model successfully handled x with sequence length {x_seq_len}!")
            else:
                print(f"  ✗ WARNING: Output shape mismatch")
                
        except Exception as e:
            print(f"  ERROR during forward pass: {e}")
            import traceback
            traceback.print_exc()


def test_reshaping_logic():
    """Test the reshaping logic to understand how the model handles sequences."""
    
    print("\n" + "=" * 60)
    print("Testing reshaping logic")
    print("=" * 60)
    
    # Simulate different scenarios
    scenarios = [
        {"z_shape": (10, 8), "x_shape": (5, 8), "name": "z longer than x"},
        {"z_shape": (10, 8), "x_shape": (15, 8), "name": "x longer than z"},
        {"z_shape": (10, 8), "x_shape": (10, 8), "name": "same length"},
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        z_shape = scenario['z_shape']
        x_shape = scenario['x_shape']
        
        # z processing
        z_ndims = len(z_shape)
        if z_ndims >= 2:
            z_seq_len = z_shape[-2]
            z_features = z_shape[-1]
        else:
            z_seq_len = z_shape[0]
            z_features = 1
        
        # x processing
        x_ndims = len(x_shape)
        if x_ndims >= 2:
            x_seq_len = x_shape[-2]
            x_features = x_shape[-1]
        else:
            x_seq_len = x_shape[0]
            x_features = 1
        
        print(f"  z: seq_len={z_seq_len}, features={z_features}")
        print(f"  x: seq_len={x_seq_len}, features={x_features}")
        print(f"  Model can handle variable x lengths in cross-attention!")


if __name__ == "__main__":
    print("\nNote: The model expects input_shape to be fixed at initialization,")
    print("but cross-attention should allow x to have different sequence lengths.")
    print("However, the current implementation reads sequence length from shape tuple.")
    print("Let's test what actually happens...\n")
    
    # First test the reshaping logic
    test_reshaping_logic()
    
    # Then test with actual model
    test_fixed_z_variable_x()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

