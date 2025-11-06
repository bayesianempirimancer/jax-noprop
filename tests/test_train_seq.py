#!/usr/bin/env python3
"""
Test script for sequence training with synthetic data.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax.core import FrozenDict

from src.flow_models.trainer_seq import SequenceTrainer
from src.flow_models.fm import VAEFlowConfig as FMConfig
from src.flow_models.df import VAEFlowConfig as DFConfig
from src.flow_models.ct import VAEFlowConfig as CTConfig


def build_config(model: str,
                 input_shape,
                 output_shape,
                 latent_shape,
                 hidden_dims,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0):
    main = FrozenDict({
        'input_shape': input_shape,
        'output_shape': output_shape,
        'latent_shape': latent_shape,
        'recon_loss_type': 'mse',
        'recon_weight': 1.0,
        'reg_weight': 0.0,
        'use_snr_weight': False if model == 'flow_matching' else True,
        'integration_method': 'midpoint' if model in ('ct', 'diffusion') else 'euler',
        'sigma': 0.02,
        'noise_schedule': 'exponential',
    })

    crn = FrozenDict({
        'model_type': 'vanilla',
        'network_type': 'transformer_seq2seq',
        'hidden_dims': tuple(hidden_dims),
        'time_embed_dim': 32,
        'time_embed_method': 'sinusoidal',
        'activation_fn': 'swish',
        'use_batch_norm': False,
        'dropout_rate': 0.0,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'qkv_bias': True,
    })
    
    # For sequences, use identity encoder since y and z have the same shape
    encoder_model_type = 'identity'
    decoder_model_type = 'identity'
    decoder_output_type = 'none'
    
    enc = FrozenDict({
        'model_type': encoder_model_type,
        'encoder_type': 'deterministic',
        'input_shape': input_shape,
        'latent_shape': latent_shape,
        'hidden_dims': (16, 16),
        'activation': 'swish',
        'dropout_rate': 0.0,
    })
    dec = FrozenDict({
        'model_type': decoder_model_type,
        'decoder_type': decoder_output_type,
        'latent_shape': latent_shape,
        'output_shape': output_shape,
        'hidden_dims': (32, 16),
        'activation': 'swish',
        'dropout_rate': 0.0,
    })

    noise_schedule_config = FrozenDict({
        'schedule_type': 'exponential',
        'learnable': False,
    })
    
    if model == 'diffusion':
        return DFConfig(main=main, noise_schedule=noise_schedule_config, crn=crn, encoder=enc, decoder=dec)
    if model == 'ct':
        return CTConfig(main=main, noise_schedule=noise_schedule_config, crn=crn, encoder=enc, decoder=dec)
    return FMConfig(main=main, noise_schedule=noise_schedule_config, crn=crn, encoder=enc, decoder=dec)


def generate_synthetic_sequences(key, x_seq_len, y_seq_len, n_samples, embed_dim):
    """Generate synthetic sequence data.
    
    Args:
        key: Random key
        x_seq_len: Sequence length for x (conditional input)
        y_seq_len: Sequence length for y (output/target)
        n_samples: Number of samples
        embed_dim: Embedding dimension
    """
    key, x_key, y_key = jr.split(key, 3)
    
    # Generate x sequences (conditional input)
    x = jr.normal(x_key, (n_samples, x_seq_len, embed_dim))
    
    # Generate y sequences (output) - make it correlated with x but with different length
    # y is roughly x transformed with some noise
    y_base = jnp.mean(x, axis=1, keepdims=True)  # [n_samples, 1, embed_dim]
    y_trend = jnp.linspace(0, 1, y_seq_len)[None, :, None]  # [1, y_seq_len, 1]
    y = y_base * (1 + 0.5 * y_trend) + 0.1 * jr.normal(y_key, (n_samples, y_seq_len, embed_dim))
    
    return x, y


def test_model(model_type='flow_matching'):
    """Test a specific model type."""
    print(f"\n{'='*60}")
    print(f"Testing {model_type.upper()} Model")
    print(f"{'='*60}")
    
    # Configuration
    z_seq_len = 10
    x_seq_len = 20
    embed_dim = 128  # Smaller for faster testing
    batch_size = 16
    n_train = 100
    n_val = 20
    
    key = jr.PRNGKey(42)
    
    # Generate data
    key, train_key, val_key = jr.split(key, 3)
    x_train, y_train = generate_synthetic_sequences(train_key, x_seq_len, z_seq_len, n_train, embed_dim)
    x_val, y_val = generate_synthetic_sequences(val_key, x_seq_len, z_seq_len, n_val, embed_dim)
    
    print(f"Generated data:")
    print(f"  x_train: {x_train.shape} (variable length)")
    print(f"  y_train: {y_train.shape} (must match z shape: {z_seq_len}, {embed_dim})")
    print(f"  x_val: {x_val.shape} (variable length)")
    print(f"  y_val: {y_val.shape} (must match z shape: {z_seq_len}, {embed_dim})")
    
    # Verify y matches z shape
    assert y_train.shape[1:] == (z_seq_len, embed_dim), f"y_train shape {y_train.shape[1:]} doesn't match z shape ({z_seq_len}, {embed_dim})"
    assert y_val.shape[1:] == (z_seq_len, embed_dim), f"y_val shape {y_val.shape[1:]} doesn't match z shape ({z_seq_len}, {embed_dim})"
    
    # Build config
    # Note: input_shape is for x (can be variable length, but we need to specify a default)
    # output_shape is for y (must match z shape)
    # latent_shape is for z (same as output_shape since encoder maps y -> z)
    input_shape = (x_seq_len, embed_dim)  # x can be variable length
    output_shape = (z_seq_len, embed_dim)  # y must match z shape
    latent_shape = (z_seq_len, embed_dim)  # z shape
    
    config = build_config(
        model=model_type,
        input_shape=input_shape,
        output_shape=output_shape,
        latent_shape=latent_shape,
        hidden_dims=[embed_dim],
        num_layers=2,  # Fewer layers for faster testing
        num_heads=4,
        mlp_ratio=2.0,
    )
    
    # Create trainer
    trainer = SequenceTrainer(
        config=config,
        learning_rate=1e-3,
        optimizer_name='adam',
        seed=42,
        unconditional=False
    )
    
    # Initialize
    print("\nInitializing model...")
    key, init_key = jr.split(key)
    bs = min(batch_size, y_train.shape[0])
    x_sample = x_train[:bs]
    y_sample = y_train[:bs]
    z_sample = jr.normal(jr.PRNGKey(43), (bs, z_seq_len, embed_dim))
    t_sample = jr.uniform(jr.PRNGKey(44), (bs,), minval=0.0, maxval=1.0)
    
    try:
        trainer.initialize(x_sample, y_sample, z_sample, t_sample)
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Train for a few epochs
    print("\nTraining for 3 epochs...")
    try:
        history = trainer.train(
            x_data=x_train,
            y_data=y_train,
            num_epochs=3,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            dropout_epochs=0,  # No dropout for testing
            verbose=True,
        )
        print("✓ Training completed successfully")
        print(f"  Final train loss: {history['train_losses'][-1]:.6f}")
        if history.get('val_losses'):
            print(f"  Final val loss: {history['val_losses'][-1]:.6f}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test generation
    print("\nTesting generation...")
    key, gen_key = jr.split(key)
    try:
        # Conditional generation
        num_gen = min(5, y_val.shape[0])
        cond_x = x_val[:num_gen]
        y_gen = trainer.conditional_generate(cond_x, num_steps=10)
        print(f"✓ Conditional generation successful")
        print(f"  Generated shape: {y_gen.shape}")
        print(f"  Expected shape: ({num_gen}, {z_seq_len}, {embed_dim})")
        
        # Compute metrics
        y_real = y_val[:num_gen]
        metrics = trainer.compute_sequence_metrics(jnp.array(y_gen), y_real)
        print(f"  Metrics: {metrics}")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"✓ {model_type.upper()} model test passed!")
    return True


def main():
    print("="*60)
    print("Testing Sequence Training with Synthetic Data")
    print("="*60)
    
    results = {}
    
    # Test each model type
    for model_type in ['flow_matching', 'ct', 'diffusion']:
        try:
            results[model_type] = test_model(model_type)
        except Exception as e:
            print(f"\n✗ {model_type} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for model_type, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {model_type:20s}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

