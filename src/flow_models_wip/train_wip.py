#!/usr/bin/env python3
"""
Training script for VAE_flow (NoProp-CT) implementation.

This script demonstrates how to train the VAE_flow model on synthetic data.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import argparse
import os

from .fm_wip import VAEFlowConfig
from .trainer_wip import VAEFlowTrainer


def generate_synthetic_data(
    num_samples: int = 1000,
    input_dim: int = 10,
    output_dim: int = 10,
    latent_dim: int = 5,
    seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate synthetic data for training.
    
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


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train VAE_flow model')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--input_dim', type=int, default=10, help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=10, help='Output dimension')
    parser.add_argument('--latent_dim', type=int, default=5, help='Latent dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 16], help='Hidden dimensions')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer (adam, adamw, sgd)')
    parser.add_argument('--vae_loss_weight', type=float, default=1.0, help='VAE loss weight')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./artifacts', help='Directory to save results')
    parser.add_argument('--plot', action='store_true', help='Plot training history')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 50)
    print("VAE_flow Training Script")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Input dim: {args.input_dim}")
    print(f"  Output dim: {args.output_dim}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Num epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  VAE loss weight: {args.vae_loss_weight}")
    print(f"  Seed: {args.seed}")
    print("=" * 50)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    x_data, y_data = generate_synthetic_data(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        latent_dim=args.latent_dim,
        seed=args.seed
    )
    
    # Split data
    x_train, y_train, x_val, y_val = split_data(x_data, y_data, train_ratio=0.8)
    
    print(f"Data shapes:")
    print(f"  Train: x={x_train.shape}, y={y_train.shape}")
    print(f"  Val: x={x_val.shape}, y={y_val.shape}")
    
    # Create model configuration
    config = VAEFlowConfig(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.hidden_dims),
        vae_loss_weight=args.vae_loss_weight
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
    # Create dummy data for initialization
    batch_size = min(args.batch_size, x_train.shape[0])
    x_init = x_train[:batch_size]
    y_init = y_train[:batch_size]
    z_init = jr.normal(jr.PRNGKey(args.seed), (batch_size, args.latent_dim))
    t_init = jr.uniform(jr.PRNGKey(args.seed + 1), (batch_size,), minval=0.0, maxval=1.0)
    
    trainer.initialize(x_init, y_init, z_init, t_init)
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        x_data=x_train,
        y_data=y_train,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        verbose=True
    )
    
    # Save results
    print("Saving results...")
    trainer.save_params(os.path.join(args.save_dir, 'vae_flow_params.pkl'))
    
    # Save training history
    np.save(os.path.join(args.save_dir, 'training_history.npy'), history)
    
    # Print final results
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Final flow loss: {history['train_flow_loss'][-1]:.4f}")
    print(f"Final reconstruction loss: {history['train_recon_loss'][-1]:.4f}")
    
    # Plot training history
    if args.plot:
        plot_training_history(history, save_path=os.path.join(args.save_dir, 'training_history.png'))
    
    print(f"\nResults saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
