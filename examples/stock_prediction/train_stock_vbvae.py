#!/usr/bin/env python3
"""
Train Variational Bayesian VAE (VBVAE) model on stock market data.

This script trains a VBVAE on 1-hour stock price and volume trajectories (12x2 data).
The VBVAE uses a Gaussian Mixture Model with Variational Bayesian EM for clustering.

NOTE: This script should be called from the project root directory:
    python examples/stock_prediction/train_stock_vbvae.py [args]

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
from flax.core import FrozenDict

from src.models.vae.vb_vae import VBVAEConfig
from src.models.vae.vb_vae_trainer import VBVAETrainer


def main():
    parser = argparse.ArgumentParser(description='Train VBVAE model on stock data')
    parser.add_argument('--data_path', type=str, default='data/stock_sequences_full_day_2d.pkl', 
                       help='Path to 2D preprocessed data pickle file')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--latent_dim', type=int, default=6, help='Latent dimension (z_dim)')
    parser.add_argument('--num_clusters', type=int, default=64, help='Number of GMM clusters')
    parser.add_argument('--N_eff', type=float, default=None, help='Effective number of data points. If None, uses total training data points.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='artifacts/stock_vbvae')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract sequences
    train_sequences = data['train']['sequences']  # List of full-day sequences
    val_sequences = data['val']['sequences']  # List of full-day sequences
    
    # Get metadata
    metadata = data.get('metadata', {})
    y_seq_len = metadata.get('y_seq_len', 12)  # Default: 12 = 1 hour
    
    print(f"Data loaded:")
    print(f"  train_sequences: {len(train_sequences)} full-day sequences")
    print(f"  val_sequences: {len(val_sequences)} full-day sequences")
    print(f"  y_seq_len: {y_seq_len} (target sequence length)")
    
    # Get sequence shape from first sequence
    if len(train_sequences) > 0:
        first_seq = train_sequences[0]
        feature_dim = first_seq.shape[1] if len(first_seq.shape) > 1 else 2
        print(f"  Feature dimension: {feature_dim}")
    else:
        feature_dim = 2
    
    # Extract 1-hour sequences (12 timesteps) from full-day sequences
    # Each sequence should be (12, 2) for 1 hour of price and volume
    print(f"\nExtracting {y_seq_len}-timestep sequences...")
    
    train_hour_sequences = []
    for seq in train_sequences:
        # Extract all possible 1-hour windows from the sequence
        for i in range(len(seq) - y_seq_len + 1):
            hour_seq = seq[i:i+y_seq_len]
            train_hour_sequences.append(hour_seq)
    
    val_hour_sequences = []
    for seq in val_sequences:
        # Extract all possible 1-hour windows from the sequence
        for i in range(len(seq) - y_seq_len + 1):
            hour_seq = seq[i:i+y_seq_len]
            val_hour_sequences.append(hour_seq)
    
    # Convert to numpy arrays
    train_data = np.array(train_hour_sequences)  # [n_samples, 12, 2]
    val_data = np.array(val_hour_sequences)  # [n_val_samples, 12, 2]
    
    print(f"Extracted sequences:")
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")
    print(f"  Sample sequence shape: {train_data[0].shape}")
    
    # Convert to JAX arrays
    train_data = jnp.array(train_data)
    val_data = jnp.array(val_data)
    
    # Build VBVAE config
    input_shape = (y_seq_len, feature_dim)  # (12, 2)
    latent_shape = (args.latent_dim,)  # (6,) - vector
    output_shape = (y_seq_len, feature_dim)  # (12, 2)
    
    print(f"\n✓ VBVAE Configuration:")
    print(f"  Input shape: {input_shape} (12 timesteps x 2 features: price, volume)")
    print(f"  Latent shape: {latent_shape} ({args.latent_dim}D vector)")
    print(f"  Output shape: {output_shape} (12 timesteps x 2 features: price, volume)")
    print(f"  Number of clusters: {args.num_clusters}")
    
    # Create config
    config = VBVAEConfig(
        model_name="stock_vbvae",
        main=FrozenDict({
            "input_shape": input_shape,
            "latent_shape": latent_shape,
            "output_shape": output_shape,
            "num_clusters": args.num_clusters,
            "recon_loss_type": "mse",
            "recon_weight": 1.0,
            "gmm_weight": 1.0,
            "prior_mu": 0.0,
            "prior_alpha": 1.5,
            "prior_beta": 0.5,
            "prior_alpha_mix": 1.0,
        }),
        encoder=FrozenDict({
            "model_type": "mlp",
            "encoder_type": "none",
            "input_shape": "NA",  # Will be set from main
            "latent_shape": "NA",  # Will be set from main
            "hidden_dims": (64, 32),
            "activation": "swish",
            "dropout_rate": 0.1,
        }),
        decoder=FrozenDict({
            "model_type": "mlp",
            "decoder_type": "none",
            "latent_shape": "NA",  # Will be set from main
            "output_shape": "NA",  # Will be set from main
            "hidden_dims": (32, 64),
            "activation": "swish",
            "dropout_rate": 0.1,
        }),
    )
    
    # Create trainer
    trainer = VBVAETrainer(
        config=config,
        learning_rate=args.learning_rate,
        optimizer_name="adam",
        N_eff=args.N_eff,
        seed=args.seed
    )
    
    # Initialize model
    print(f"\nInitializing model...")
    sample_batch = train_data[:args.batch_size]
    trainer.initialize(sample_batch)
    
    # Train model
    print(f"\nStarting training...")
    import time
    start_time = time.time()
    history = trainer.train(
        x_data=train_data,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=val_data,
        verbose=args.verbose or True
    )
    total_runtime = time.time() - start_time
    print(f"\n✓ Total training runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    
    # Save results
    print(f"\nSaving results to {args.save_dir}...")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_results(history, args.save_dir)
    
    # Test reconstruction
    print(f"\nTesting reconstruction on a few samples...")
    test_samples = train_data[:5]
    reconstructions = trainer.reconstruct(test_samples)
    
    # Compute reconstruction error
    recon_error = jnp.mean((test_samples - reconstructions) ** 2)
    print(f"  Reconstruction MSE: {float(recon_error):.6f}")
    
    # Test encoding
    z_e = trainer.encode(test_samples)
    print(f"  Encoded z_e shape: {z_e.shape}")
    print(f"  Decoded output shape: {reconstructions.shape}")
    
    print(f"\n✓ Training complete!")
    print(f"  Final train loss: {history['train_losses'][-1]:.4f}")
    if len(history.get('val_losses', [])) > 0:
        print(f"  Final val loss: {history['val_losses'][-1]:.4f}")


if __name__ == "__main__":
    main()




