#!/usr/bin/env python3
"""
Training script for VAE_flow (NoProp-CT) implementation.

This script demonstrates how to train the VAE_flow model on the two moons dataset.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import argparse
import os
import pickle
from flax.core import FrozenDict

from .fm_wip import VAEFlowConfig
from .trainer_wip import VAEFlowTrainer


def load_two_moons_data(data_path: str = "data/two_moons_formatted.pkl") -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Load the two moons dataset.
    
    Args:
        data_path: Path to the two moons dataset file
        
    Returns:
        Tuple of (x_train, y_train, x_val, y_val)
    """
    print(f"Loading two moons dataset from {data_path}...")
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract train and validation data
        x_train = jnp.array(data['train']['x'])
        y_train = jnp.array(data['train']['y'])
        x_val = jnp.array(data['val']['x'])
        y_val = jnp.array(data['val']['y'])
        
        print(f"Loaded dataset:")
        print(f"  Training: x={x_train.shape}, y={y_train.shape}")
        print(f"  Validation: x={x_val.shape}, y={y_val.shape}")
        
        return x_train, y_train, x_val, y_val
        
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {data_path}")
        print("Please ensure the two moons dataset is available.")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def generate_synthetic_data(
    num_samples: int = 1000,
    input_dim: int = 10,
    output_dim: int = 10,
    latent_dim: int = 5,
    seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate synthetic data for training (fallback option).
    
    Args:
        num_samples: Number of samples to generate
        input_dim: Input dimension
        output_dim: Output dimension
        latent_dim: Latent dimension
        seed: Random seed
        
    Returns:
        Tuple of (x_data, y_data)
    """
    rng = jr.PRNGKey(seed)
    
    # Generate input data (x) - random normal
    rng, x_rng = jr.split(rng)
    x_data = jr.normal(x_rng, (num_samples, input_dim))
    
    # Generate target data (y) - some function of x with noise
    # For simplicity, let's make y a linear transformation of x with noise
    rng, y_rng, noise_rng = jr.split(rng, 3)
    
    # Create a random transformation matrix
    transform_matrix = jr.normal(y_rng, (input_dim, output_dim))
    
    # Generate y as a function of x
    y_data = jnp.dot(x_data, transform_matrix) + 0.1 * jr.normal(noise_rng, (num_samples, output_dim))
    
    return x_data, y_data


def split_data(x_data: jnp.ndarray, y_data: jnp.ndarray, train_ratio: float = 0.8) -> Tuple:
    """
    Split data into training and validation sets.
    
    Args:
        x_data: Input data
        y_data: Target data
        train_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (x_train, y_train, x_val, y_val)
    """
    num_samples = x_data.shape[0]
    num_train = int(num_samples * train_ratio)
    
    # Shuffle indices
    indices = jnp.arange(num_samples)
    indices = jr.permutation(jr.PRNGKey(42), indices)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    x_train = x_data[train_indices]
    y_train = y_data[train_indices]
    x_val = x_data[val_indices]
    y_val = y_data[val_indices]
    
    return x_train, y_train, x_val, y_val


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Total loss
    axes[0].plot(history['train_loss'], label='Train', color='blue')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='Validation', color='red')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Flow loss
    axes[1].plot(history['train_flow_loss'], label='Train', color='blue')
    if 'val_flow_loss' in history and history['val_flow_loss']:
        axes[1].plot(history['val_flow_loss'], label='Validation', color='red')
    axes[1].set_title('Flow Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # Reconstruction loss
    axes[2].plot(history['train_recon_loss'], label='Train', color='blue')
    if 'val_recon_loss' in history and history['val_recon_loss']:
        axes[2].plot(history['val_recon_loss'], label='Validation', color='red')
    axes[2].set_title('Reconstruction Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def create_vae_config(
    input_shape: Tuple[int, ...] = (2,),
    output_shape: Tuple[int, ...] = (2,),
    latent_shape: Tuple[int, ...] = (4,),
    crn_type: str = "vanilla",
    network_type: str = "mlp",
    hidden_dims: Tuple[int, ...] = (64, 64, 64),
    learning_rate: float = 1e-3,
    flow_loss_weight: float = 0.01
) -> VAEFlowConfig:
    """Create VAE flow configuration for two moons dataset."""
    return VAEFlowConfig(
        config=FrozenDict({
            "input_shape": input_shape,
            "output_shape": output_shape,
            "latent_shape": latent_shape,
            "loss_type": "mse",
            "flow_loss_weight": flow_loss_weight,
            "reg_weight": 0.0
        }),
        crn_config=FrozenDict({
            "model_type": crn_type,
            "network_type": network_type,
            "hidden_dims": hidden_dims,
            "time_embed_dim": 32,
            "time_embed_method": "sinusoidal",
            "dropout_rate": 0.1,
            "activation_fn": "swish",
            "use_batch_norm": False
        }),
        encoder_config=FrozenDict({
            "model_type": "mlp",
            "encoder_type": "deterministic",
            "input_shape": input_shape,
            "latent_shape": latent_shape,
            "hidden_dims": (128, 64, 32),
            "activation": "swish",
            "dropout_rate": 0.1
        }),
        decoder_config=FrozenDict({
            "model_type": "mlp",
            "decoder_type": "deterministic",
            "latent_shape": latent_shape,
            "output_shape": output_shape,
            "hidden_dims": (32, 64, 128),
            "activation": "swish",
            "dropout_rate": 0.1
        })
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train VAE_flow model on two moons dataset')
    parser.add_argument('--data_path', type=str, default='data/two_moons_formatted.pkl', 
                       help='Path to two moons dataset')
    parser.add_argument('--use_synthetic', action='store_true', 
                       help='Use synthetic data instead of two moons dataset')
    parser.add_argument('--num_samples', type=int, default=1000, 
                       help='Number of training samples (for synthetic data)')
    parser.add_argument('--input_dim', type=int, default=2, 
                       help='Input dimension (2 for two moons)')
    parser.add_argument('--output_dim', type=int, default=2, 
                       help='Output dimension (2 for two moons)')
    parser.add_argument('--latent_dim', type=int, default=4, 
                       help='Latent dimension')
    parser.add_argument('--crn_type', type=str, default='vanilla', 
                       choices=['vanilla', 'geometric', 'potential', 'natural', 'hamiltonian'],
                       help='CRN flow type')
    parser.add_argument('--network_type', type=str, default='mlp', 
                       choices=['mlp', 'bilinear', 'convex'],
                       help='Network backbone type')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64, 64], 
                       help='Hidden dimensions for CRN')
    parser.add_argument('--num_epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, 
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, 
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'sgd', 'adagrad'],
                       help='Optimizer')
    parser.add_argument('--flow_loss_weight', type=float, default=0.01, 
                       help='Flow loss weight')
    parser.add_argument('--dropout_epochs', type=int, default=None, 
                       help='Number of epochs to use dropout (default: all epochs)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='artifacts/two_moons_training', 
                       help='Directory to save results')
    parser.add_argument('--verbose', action='store_true', 
                       help='Verbose training output')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("VAE_flow Training Script - Two Moons Dataset")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Dataset: {'Synthetic' if args.use_synthetic else 'Two Moons'}")
    if not args.use_synthetic:
        print(f"  Data path: {args.data_path}")
    print(f"  Input dim: {args.input_dim}")
    print(f"  Output dim: {args.output_dim}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  CRN type: {args.crn_type}")
    print(f"  Network type: {args.network_type}")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Num epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Flow loss weight: {args.flow_loss_weight}")
    print(f"  Dropout epochs: {args.dropout_epochs if args.dropout_epochs else args.num_epochs}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)
    
    # Load data
    if args.use_synthetic:
        print("Generating synthetic data...")
        x_data, y_data = generate_synthetic_data(
            num_samples=args.num_samples,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            latent_dim=args.latent_dim,
            seed=args.seed
        )
        x_train, y_train, x_val, y_val = split_data(x_data, y_data, train_ratio=0.8)
    else:
        print("Loading two moons dataset...")
        x_train, y_train, x_val, y_val = load_two_moons_data(args.data_path)
    
    print(f"Data shapes:")
    print(f"  Train: x={x_train.shape}, y={y_train.shape}")
    print(f"  Val: x={x_val.shape}, y={y_val.shape}")
    
    # Create model configuration
    # For geometric flows, input_dim must match latent_dim
    if args.crn_type == "geometric":
        effective_input_dim = args.latent_dim
        print(f"Note: Using input_dim={effective_input_dim} for geometric flow (must match latent_dim)")
        
        # Pad input data to match latent_dim
        if x_train.shape[1] != effective_input_dim:
            print(f"Padding input data from {x_train.shape[1]} to {effective_input_dim} dimensions")
            pad_width = effective_input_dim - x_train.shape[1]
            x_train = jnp.pad(x_train, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            x_val = jnp.pad(x_val, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            print(f"  Padded train: x={x_train.shape}, y={y_train.shape}")
            print(f"  Padded val: x={x_val.shape}, y={y_val.shape}")
    else:
        effective_input_dim = args.input_dim
    
    config = create_vae_config(
        input_shape=(effective_input_dim,),
        output_shape=(args.output_dim,),
        latent_shape=(args.latent_dim,),
        crn_type=args.crn_type,
        network_type=args.network_type,
        hidden_dims=tuple(args.hidden_dims),
        flow_loss_weight=args.flow_loss_weight
    )
    
    # Create trainer
    trainer = VAEFlowTrainer(
        config=config,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        seed=args.seed
    )
    
    # Initialize model
    print("Initializing model...")
    # Create sample data for initialization
    batch_size = min(args.batch_size, x_train.shape[0])
    x_sample = x_train[:batch_size]
    y_sample = y_train[:batch_size]
    z_sample = jr.normal(jr.PRNGKey(args.seed), (batch_size, args.latent_dim))
    t_sample = jr.uniform(jr.PRNGKey(args.seed + 1), (batch_size,), minval=0.0, maxval=1.0)
    
    trainer.initialize(x_sample, y_sample, z_sample, t_sample)
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        x_data=x_train,
        y_data=y_train,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        dropout_epochs=args.dropout_epochs,
        verbose=args.verbose
    )
    
    # Save results using trainer's built-in save functionality
    print("Saving results...")
    trainer.save_results(history, args.save_dir)
    
    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final training loss: {history['train_losses'][-1]:.6f}")
    if history['val_losses']:
        print(f"Final validation loss: {history['val_losses'][-1]:.6f}")
    print(f"Final flow loss: {history['train_flow_losses'][-1]:.6f}")
    print(f"Final reconstruction loss: {history['train_recon_losses'][-1]:.6f}")
    if history['train_accuracies']:
        print(f"Final training accuracy: {history['train_accuracies'][-1]:.4f}")
    if history['val_accuracies']:
        print(f"Final validation accuracy: {history['val_accuracies'][-1]:.4f}")
    
    print(f"\nResults saved to: {args.save_dir}")
    print("Files created:")
    for file in os.listdir(args.save_dir):
        file_path = os.path.join(args.save_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  - {file} ({size} bytes)")


if __name__ == "__main__":
    main()
