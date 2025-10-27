#!/usr/bin/env python3
"""
Training script for VAE_flow on Two Moons classification problem.

This script loads the two moons dataset and trains the VAE_flow model
for the classification task.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
from typing import Tuple
from datetime import datetime
from dataclasses import replace
from flax.core import FrozenDict

try:
    from .fm_wip import VAEFlowConfig
    from .trainer_wip import VAEFlowTrainer
except ImportError:
    # Fallback for direct script execution
    from fm_wip import VAEFlowConfig
    from trainer_wip import VAEFlowTrainer


def load_two_moons_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the two moons dataset from pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Tuple of (x_data, y_data)
    """
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    
    x_data = dataset['x_data']
    y_data = dataset['y_data']
    
    print(f"Loaded dataset:")
    print(f"  Samples: {x_data.shape[0]}")
    print(f"  Input dim: {x_data.shape[1]}")
    print(f"  Classes: {len(np.unique(y_data))}")
    
    return x_data, y_data


def preprocess_data(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Preprocess the data for VAE_flow training.
    
    Args:
        x_data: Input features [n_samples, 2]
        y_data: Class labels [n_samples]
        
    Returns:
        Tuple of (x_processed, y_processed)
    """
    # Convert to JAX arrays
    x_processed = jnp.array(x_data, dtype=jnp.float32)
    y_processed = jnp.array(y_data, dtype=jnp.float32)
    
    # Normalize x_data to have zero mean and unit variance
    x_mean = jnp.mean(x_processed, axis=0)
    x_std = jnp.std(x_processed, axis=0)
    x_processed = (x_processed - x_mean) / (x_std + 1e-8)
    
    # Convert y to one-hot encoding for the decoder
    num_classes = len(jnp.unique(y_processed))
    y_onehot = jax.nn.one_hot(y_processed.astype(int), num_classes)
    
    print(f"Data preprocessing:")
    print(f"  X normalized: mean={jnp.mean(x_processed):.3f}, std={jnp.std(x_processed):.3f}")
    print(f"  Y one-hot shape: {y_onehot.shape}")
    
    return x_processed, y_onehot


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


def evaluate_classification(model, params: dict, x_data: jnp.ndarray, y_data: jnp.ndarray, 
                           batch_size: int = 100) -> dict:
    """
    Evaluate the model's classification performance.
    
    Args:
        model: VAE_flow model
        params: Model parameters
        x_data: Input data
        y_data: Target data (one-hot)
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    num_samples = x_data.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    all_predictions = []
    all_targets = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        x_batch = x_data[start_idx:end_idx]
        y_batch = y_data[start_idx:end_idx]
        
        # Generate random z and t for this batch
        batch_size_actual = x_batch.shape[0]
        rng = jr.PRNGKey(42 + i)
        z_rng, t_rng = jr.split(rng, 2)
        z_batch = jr.normal(z_rng, (batch_size_actual, model.config.latent_dim))
        t_batch = jr.uniform(t_rng, (batch_size_actual,), minval=0.0, maxval=1.0)
        
        # Get predictions from the decoder (now returns probabilities directly)
        predictions = model.apply(params, z_batch, method='decoder', training=False)
        
        all_predictions.append(predictions)
        all_targets.append(y_batch)
    
    # Concatenate all predictions and targets
    all_predictions = jnp.concatenate(all_predictions, axis=0)
    all_targets = jnp.concatenate(all_targets, axis=0)
    
    # Convert to class predictions
    pred_classes = jnp.argmax(all_predictions, axis=1)
    true_classes = jnp.argmax(all_targets, axis=1)
    
    # Calculate accuracy
    accuracy = jnp.mean(pred_classes == true_classes)
    
    return {
        'accuracy': float(accuracy),
        'predictions': all_predictions,
        'targets': all_targets,
        'pred_classes': pred_classes,
        'true_classes': true_classes
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train VAE_flow on Two Moons dataset')
    parser.add_argument('--dataset_path', type=str, default='./data/two_moons_dataset.pkl',
                       help='Path to the two moons dataset')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer (adam, adamw, sgd)')
    parser.add_argument('--vae_loss_weight', type=float, default=1.0, help='VAE loss weight')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None, 
                       help='Directory to save results (auto-generated if not specified)')
    parser.add_argument('--plot', action='store_true', help='Plot training history')
    
    args = parser.parse_args()
    
    # Create save directory with timestamp and configuration
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_str = f"ep{args.num_epochs}_bs{args.batch_size}_lr{args.learning_rate}"
        args.save_dir = f"./artifacts/vae_flow_two_moons_{timestamp}_{config_str}"
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("VAE_flow Training on Two Moons Dataset")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Num epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  VAE loss weight: {args.vae_loss_weight}")
    print(f"  Seed: {args.seed}")
    print(f"  Save directory: {args.save_dir}")
    print("=" * 60)
    
    # Load dataset
    print("Loading two moons dataset...")
    x_data, y_data = load_two_moons_dataset(args.dataset_path)
    
    # Preprocess data
    print("Preprocessing data...")
    x_processed, y_processed = preprocess_data(x_data, y_data)
    
    # Split data
    x_train, y_train, x_val, y_val = split_data(x_processed, y_processed, train_ratio=0.8)
    
    print(f"Data shapes:")
    print(f"  Train: x={x_train.shape}, y={y_train.shape}")
    print(f"  Val: x={x_val.shape}, y={y_val.shape}")
    
    # Create model configuration with nested dictionary structure
    config = VAEFlowConfig()
    
    # Update config with custom values using dataclass replace
    config = replace(
        config,
        config=FrozenDict({
            **config.config,
            "input_shape": (2,),
            "output_shape": (2,),
            "latent_shape": (6,),
            "flow_loss_weight": args.vae_loss_weight,
        }),
        crn_config=FrozenDict({
            **config.crn_config,
            "model_type": "potential",  # Use potential flow instead of vanilla
            "hidden_dims": (64, 32, 16)  # Default hidden dims
        }),
        encoder_config=FrozenDict({
            **config.encoder_config,
            "hidden_dims": (64, 32, 16)  # Default hidden dims
        }),
        decoder_config=FrozenDict({
            **config.decoder_config,
            "hidden_dims": (64, 32, 16)  # Default hidden dims
        })
    )
    
    # Create trainer (shapes are now computed from config)
    trainer = VAEFlowTrainer(
        config=config,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        seed=args.seed
    )
    
    # Initialize model
    print("Initializing model...")
    batch_size = min(args.batch_size, x_train.shape[0])
    x_init = x_train[:batch_size]
    y_init = y_train[:batch_size]
    z_init = jr.normal(jr.PRNGKey(args.seed), (batch_size, 8))  # Default 1D latent space
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
        dropout_epochs=80,  # Use dropout for first 80 epochs
        verbose=True
    )
    
    # Final accuracies are now computed during training and stored in history
    final_train_acc = history['train_accuracies'][-1] if history['train_accuracies'] else 0.0
    final_val_acc = history['val_accuracies'][-1] if history['val_accuracies'] else 0.0
    
    # Save results and generate plots
    print("Saving results and generating plots...")
    trainer.save_params(os.path.join(args.save_dir, 'vae_flow_two_moons_params.pkl'))
    
    # Use the new save_results method to generate comprehensive plots
    trainer.save_results(history, args.save_dir)
    
    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final training loss: {history['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {history['val_losses'][-1]:.4f}")
    print(f"Final flow loss: {history['train_flow_losses'][-1]:.4f}")
    print(f"Final reconstruction loss: {history['train_recon_losses'][-1]:.4f}")
    print(f"Training accuracy: {final_train_acc:.4f}")
    print(f"Validation accuracy: {final_val_acc:.4f}")
    
    print(f"\nResults and plots saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
