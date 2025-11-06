#!/usr/bin/env python3
"""
Train sequence model on stock market data.

NOTE: This script should be called from the project root directory:
    python examples/stock_prediction/train_stock_sequences.py [args]

All paths (data_path, save_dir) are relative to the project root directory.
"""

import sys
from pathlib import Path
# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pickle
import jax.numpy as jnp
import numpy as np
import argparse

from src.flow_models.trainer_seq import SequenceTrainer
from src.flow_models.fm import VAEFlowConfig as FMConfig
from src.flow_models.df import VAEFlowConfig as DFConfig
from src.flow_models.ct import VAEFlowConfig as CTConfig
from flax.core import FrozenDict


def build_config(model: str,
                 input_shape,
                 output_shape,
                 latent_shape,
                 hidden_dims,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 encoder_config=None,
                 decoder_config=None,
                 crn_embed_dim: int = 20):
    main = FrozenDict({
        'input_shape': input_shape,
        'output_shape': output_shape,
        'latent_shape': latent_shape,
        'recon_loss_type': 'mse',
        'recon_weight': 1.0,
        'reg_weight': 0.0,
        'vae_weight': 0.0,
        'use_snr_weight': False if model == 'flow_matching' else True,
        'integration_method': 'midpoint' if model in ('ct', 'diffusion') else 'euler',
        'sigma': 0.02,
        'noise_schedule': 'exponential',
        'encode_x': True,  # Enable x encoding for sequence models
    })

    crn = FrozenDict({
        'model_type': 'vanilla',
        'network_type': 'transformer_seq2seq',
        'embed_dim': crn_embed_dim,  # Embedding dimension (should match latent_dim)
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
        'rope_base': 10000.0,  # Base for RoPE frequency calculation
        'projection_seed': 42,  # Seed for 2D->embed_dim projection matrix
        'x_static_dim': 0,  # Dimension of static features (0 means no static features)
    })
    
    # Use provided encoder/decoder configs or create defaults
    if encoder_config is None:
        encoder_config = FrozenDict({
            'model_type': 'mlp',
            'encoder_type': 'deterministic',
            'input_shape': input_shape,
            'latent_shape': latent_shape,
            'hidden_dims': (64, 32, 16),
            'activation': 'swish',
            'dropout_rate': 0.0,
        })
    
    if decoder_config is None:
        decoder_config = FrozenDict({
            'model_type': 'mlp',
            'decoder_type': 'none',
            'latent_shape': latent_shape,
            'output_shape': output_shape,
            'hidden_dims': (64, 32, 16),
            'activation': 'swish',
            'dropout_rate': 0.0,
        })

    noise_schedule_config = FrozenDict({
        'schedule_type': 'exponential',
        'learnable': False,
    })
    
    if model == 'diffusion':
        return DFConfig(main=main, noise_schedule=noise_schedule_config, crn=crn, encoder=encoder_config, decoder=decoder_config)
    if model == 'ct':
        return CTConfig(main=main, noise_schedule=noise_schedule_config, crn=crn, encoder=encoder_config, decoder=decoder_config)
    return FMConfig(main=main, noise_schedule=noise_schedule_config, crn=crn, encoder=encoder_config, decoder=decoder_config)


def main():
    parser = argparse.ArgumentParser(description='Train sequence model on stock data')
    parser.add_argument('--data_path', type=str, default='data/stock_sequences_full_day_2d.pkl', help='Path to 2D preprocessed data pickle file (no positional embeddings)')
    parser.add_argument('--model_type', type=str, default='flow_matching', choices=['flow_matching', 'diffusion', 'ct'])
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # Default learning rate
    parser.add_argument('--embed_dim', type=int, default=20, help='Embedding dimension (after projection from 2D inside CRN)')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='artifacts/stock_sequences')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    # New format: full-day sequences (no pre-split x/y)
    train_sequences = data['train']['sequences']  # List of full-day sequences
    val_sequences = data['val']['sequences']  # List of full-day sequences
    
    # Get metadata
    metadata = data.get('metadata', {})
    y_seq_len = metadata.get('y_seq_len', 12)  # Default: 12 = 1 hour
    
    print(f"Data loaded:")
    print(f"  train_sequences: {len(train_sequences)} full-day sequences")
    print(f"  val_sequences: {len(val_sequences)} full-day sequences")
    print(f"  y_seq_len: {y_seq_len} (target sequence length)")
    
    # Get sequence lengths from first sequence
    if len(train_sequences) > 0:
        first_seq = train_sequences[0]
        current_embed_dim = first_seq.shape[1] if len(first_seq.shape) > 1 else 2
        seq_lengths = [len(seq) for seq in train_sequences]
        print(f"  Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, median={int(np.median(seq_lengths))}")
    else:
        current_embed_dim = 2
    
    # For config, we need to estimate max x_seq_len (will be dynamically padded during training)
    # Use a reasonable estimate based on median sequence length
    if len(train_sequences) > 0:
        estimated_max_x_len = int(np.percentile([len(seq) for seq in train_sequences], 90))  # 90th percentile
        x_seq_len = estimated_max_x_len  # For config purposes
    else:
        x_seq_len = 50  # Default estimate
    
    print(f"\nSequence info:")
    print(f"  x_seq_len: {x_seq_len}")
    print(f"  y_seq_len: {y_seq_len}")
    print(f"  current embed_dim: {current_embed_dim}")
    print(f"  requested embed_dim: {args.embed_dim}")
    
    # Data should be 2D (price, volume)
    if current_embed_dim != 2:
        print(f"\nWARNING: Data has embed_dim={current_embed_dim} but expected 2D (price, volume).")
        print("The CRN will project 2D->embed_dim internally, so data should be 2D.")
        if current_embed_dim == args.embed_dim:
            print("If data is already projected, you may need to adjust the pipeline.")
    
    # Adjust num_heads if embed_dim is too small
    # Need embed_dim >= num_heads for attention to work
    if args.num_heads > args.embed_dim:
        print(f"\nNOTE: num_heads={args.num_heads} > embed_dim={args.embed_dim}. Adjusting num_heads to {args.embed_dim}.")
        args.num_heads = args.embed_dim
    
    # Data is 2D (price, volume), but we'll use 10D latent space
    # Encoder: 2D -> 10D, Decoder: 10D -> 2D
    # Note: Encoder/decoder operate only on feature dimension (last dim), not sequence length
    latent_dim = 10  # Expanded latent dimension
    input_shape = (x_seq_len, 2)  # Full input shape for main config
    latent_shape = (y_seq_len, latent_dim)  # Full latent shape for main config
    output_shape = (y_seq_len, 2)  # Full output shape for main config
    
    # Encoder/decoder configs use only feature dimensions (no sequence length)
    encoder_input_shape = (2,)  # Feature dimension only
    encoder_latent_shape = (latent_dim,)  # Feature dimension only
    decoder_latent_shape = (latent_dim,)  # Feature dimension only
    decoder_output_shape = (2,)  # Feature dimension only
    
    print(f"\n✓ Data shapes:")
    print(f"  Input (x): {input_shape} (2D: price, volume)")
    print(f"  Latent (z): {latent_shape} ({latent_dim}D latent space)")
    print(f"  Output: {output_shape} (2D: price, volume)")
    print(f"  Encoder: {encoder_input_shape} -> {encoder_latent_shape} (operates per timestep)")
    print(f"  Decoder: {decoder_latent_shape} -> {decoder_output_shape} (operates per timestep)")
    print(f"  CRN internal embed_dim: {latent_dim} (should match latent_dim)")
    
    # Adjust num_heads to ensure embed_dim is divisible by num_heads
    # For attention to work properly, embed_dim must be divisible by num_heads
    if latent_dim % args.num_heads != 0:
        # Adjust num_heads to be a divisor of latent_dim
        # Find the largest divisor of latent_dim that is <= args.num_heads
        adjusted_num_heads = args.num_heads
        while latent_dim % adjusted_num_heads != 0 and adjusted_num_heads > 1:
            adjusted_num_heads -= 1
        if adjusted_num_heads != args.num_heads:
            print(f"\nNOTE: Adjusted num_heads from {args.num_heads} to {adjusted_num_heads} "
                  f"to ensure embed_dim ({latent_dim}) is divisible by num_heads.")
        args.num_heads = adjusted_num_heads
    
    # Build CRN config with embed_dim matching latent_dim
    crn_config = FrozenDict({
        'model_type': 'vanilla',
        'network_type': 'transformer_seq2seq',
        'embed_dim': latent_dim,  # Embedding dimension (should match latent_dim)
        'hidden_dims': (),
        'time_embed_dim': 32,
        'time_embed_method': 'sinusoidal',
        'activation_fn': 'swish',
        'use_batch_norm': False,
        'dropout_rate': 0.0,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'mlp_ratio': args.mlp_ratio,
        'qkv_bias': True,
        'rope_base': 10000.0,  # Base for RoPE frequency calculation
        'projection_seed': 42,  # Seed for 2D->embed_dim projection matrix
        'x_static_dim': 0,  # Dimension of static features (0 means no static features)
    })
    
    # Create encoder/decoder configs with feature dimensions only
    enc_config = FrozenDict({
        'model_type': 'mlp',
        'encoder_type': 'deterministic',
        'input_shape': encoder_input_shape,  # Only feature dimension: (2,)
        'latent_shape': encoder_latent_shape,  # Only feature dimension: (10,)
        'hidden_dims': (32, 32),
        'activation': 'swish',
        'dropout_rate': 0.0,
    })
    dec_config = FrozenDict({
        'model_type': 'mlp',
        'decoder_type': 'none',
        'latent_shape': decoder_latent_shape,  # Only feature dimension: (10,)
        'output_shape': decoder_output_shape,  # Only feature dimension: (2,)
        'hidden_dims': (32, 32),
        'activation': 'swish',
        'dropout_rate': 0.0,
    })
    
    config = build_config(
        model=args.model_type,
        input_shape=input_shape,
        output_shape=output_shape,
        latent_shape=latent_shape,
        hidden_dims=[],
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        encoder_config=enc_config,
        decoder_config=dec_config,
        crn_embed_dim=latent_dim,  # Pass latent_dim to build_config
    )
    
    # Create trainer
    trainer = SequenceTrainer(
        config=config,
        learning_rate=args.learning_rate,
        optimizer_name='adam',
        seed=args.seed,
        unconditional=False
    )
    
    # Initialize
    print("\nInitializing model...")
    import jax.random as jr
    key = jr.PRNGKey(args.seed)
    key, init_key = jr.split(key)
    
    # Create a sample batch for initialization (with random splits)
    bs = min(args.batch_size, len(train_sequences))
    x_sample, y_sample = trainer._create_minibatch_with_random_splits(
        train_sequences, jnp.arange(bs), y_seq_len=y_seq_len
    )
    
    # Sample z_0 in latent space (10D)
    z_sample = jr.normal(jr.PRNGKey(args.seed), (bs, y_seq_len, latent_dim))
    t_sample = jr.uniform(jr.PRNGKey(args.seed+1), (bs,), minval=0.0, maxval=1.0)
    
    try:
        trainer.initialize(x_sample, y_sample, z_sample, t_sample)
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Train
    print(f"\nTraining for {args.num_epochs} epochs...")
    history = trainer.train(
        sequences_data=train_sequences,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        y_seq_len=y_seq_len,
        validation_data=val_sequences,
        dropout_epochs=0,
        verbose=args.verbose,
    )
    
    print(f"\nTraining completed!")
    print(f"  Final train loss: {history['train_losses'][-1]:.6f}")
    if history.get('val_losses'):
        print(f"  Final val loss: {history['val_losses'][-1]:.6f}")
    
    # Test generation
    print("\nTesting generation...")
    key, gen_key = jr.split(key)
    num_gen = min(100, len(val_sequences))
    # Create a sample batch with random splits for generation
    eval_indices = jnp.arange(num_gen)
    cond_x, y_real = trainer._create_minibatch_with_random_splits(
        val_sequences, eval_indices, y_seq_len=y_seq_len
    )
    y_gen = np.array(trainer.conditional_generate(cond_x, num_steps=20))
    print(f"✓ Generated sequences: {y_gen.shape}")
    
    # Compute metrics
    y_real_np = np.array(y_real)
    metrics = trainer.compute_sequence_metrics(jnp.array(y_gen), jnp.array(y_real_np))
    print(f"  Metrics: {metrics}")
    if 'percent_variance_explained' in metrics:
        pve = metrics['percent_variance_explained']
        if np.isfinite(pve):
            print(f"  Percent Variance Explained: {pve:.2f}%")
        else:
            print(f"  Percent Variance Explained: N/A (insufficient variance in data)")
    
    # Save results
    # Always save (default to artifacts/stock_sequences if not specified)
    from datetime import datetime
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create results dictionary with all metrics
    results = {
        'history': history,
        'metrics': metrics,
        'config': config.__dict__ if hasattr(config, '__dict__') else config,
        'model_type': args.model_type,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    }
    
    with open(Path(args.save_dir) / 'training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    with open(Path(args.save_dir) / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    trainer.save_params(str(Path(args.save_dir) / 'model_params.pkl'))
    
    # Generate plots
    print("\nGenerating plots...")
    
    # 1. Loss trends plot
    trainer.save_loss_trends_plot(history, output_dir=str(args.save_dir))
    
    # 2. Direct comparison plot (in model input/output space)
    trainer.save_direct_comparison_plot(
        y_real=y_real_np,
        y_pred=y_gen,
        output_dir=str(args.save_dir),
        num_samples=100
    )
    
    # 3. Trajectory comparison plot (raw predictions vs ground truth)
    trainer.save_trajectory_comparison_plot(
        y_real=y_real_np,
        y_pred=y_gen,
        output_dir=str(args.save_dir),
        num_samples=20
    )
    
    print(f"\nSaved results and plots to {args.save_dir}")


if __name__ == '__main__':
    main()

