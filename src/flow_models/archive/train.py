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
import pickle
from datetime import datetime
from pathlib import Path
from flax.core import FrozenDict

# Import flow model configurations and trainer
from .fm import VAEFlowConfig as FlowMatchingConfig
from .df import VAEFlowConfig as DiffusionConfig
from .ct import VAEFlowConfig as CTConfig
from .trainer import VAEFlowTrainer


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
        
        # Convert binary labels {0, 1} to {-1, 1} for non-softmax/MSE training
        y_train = 2 * y_train - 1
        y_val = 2 * y_val - 1
        
        print(f"Loaded dataset:")
        print(f"  Training: x={x_train.shape}, y={y_train.shape}")
        print(f"  Validation: x={x_val.shape}, y={y_val.shape}")
        print(f"  Label range: [{jnp.min(y_train):.1f}, {jnp.max(y_train):.1f}]")
        
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
    latent_shape: Tuple[int, ...] = (2,),
    crn_type: str = "vanilla",
    network_type: str = "mlp",
    hidden_dims: Tuple[int, ...] = (64, 64, 64),
    learning_rate: float = 0.001,
    recon_weight: float = 0.0,
    decoder_type: str = "none",
    decoder_model: str = "identity",
    encoder_type: str = "identity",
    recon_loss_type: str = "mse",
    reg_weight: float = 0.0,
    model_type: str = "flow_matching",
    noise_schedule: str = "linear"
):
    """Create VAE flow configuration for two moons dataset."""
    base_config = FrozenDict({
        "input_shape": input_shape,
        "output_shape": output_shape,
        "latent_shape": latent_shape,
        "recon_loss_type": recon_loss_type,
        "recon_weight": recon_weight,
        "reg_weight": reg_weight,
        "integration_method": "euler",  # Options: "euler", "heun", "rk4", "adaptive", "midpoint"
        "num_timesteps": 20,
        "sigma": 0.02,
    })
    
    crn = FrozenDict({
        "model_type": crn_type,
        "network_type": network_type,
        "hidden_dims": hidden_dims,
        "time_embed_dim": 32,
        "time_embed_method": "sinusoidal",
        "dropout_rate": 0.1,
        "activation_fn": "swish",
        "use_batch_norm": False
    })
    
    encoder = FrozenDict({
        "model_type": encoder_type,
        "encoder_type": "deterministic",
        "input_shape": input_shape,
        "latent_shape": latent_shape,
        "hidden_dims": (16,32,16,),  # Very simple architecture
        "activation": "swish",
        "dropout_rate": 0.0  # No dropout for simplicity
    })
    
    decoder = FrozenDict({
        "model_type": decoder_model,
        "decoder_type": decoder_type,
        "latent_shape": latent_shape,
        "output_shape": output_shape,
        "hidden_dims": (16,32,16,),  # Very simple architecture
        "activation": "swish",
        "dropout_rate": 0.0  # No dropout for simplicity
    })
    
    if model_type == "diffusion":
        # Add diffusion-specific config
        diffusion_config = FrozenDict({
            **base_config,
            "recon_loss_type": recon_loss_type,  # Use passed recon_loss_type
            "reg_weight": 0.00,  # Add regularization for diffusion
        })
        return DiffusionConfig(
            main=diffusion_config, 
            crn=crn,
            encoder=encoder,
            decoder=decoder
        )
    elif model_type == "ct":
        # Add CT-specific config with noise schedule
        ct_config = FrozenDict({
            **base_config,
            "recon_loss_type": recon_loss_type,
            "reg_weight": reg_weight,
            "noise_schedule": noise_schedule,
        })
        return CTConfig(
            main=ct_config,
            crn=crn,
            encoder=encoder,
            decoder=decoder
        )
    else:  # flow_matching
        return FlowMatchingConfig(
            main=base_config, 
            crn=crn,
            encoder=encoder,
            decoder=decoder
        )


def main():
    """Main training function.

    Modified to remove external ResultsTracker and to train all three models
    (flow_matching, diffusion, ct) sequentially on the two moons dataset.
    """
    parser = argparse.ArgumentParser(description='Train VAE_flow model on two moons dataset')
    # Retain model_type arg for backward compatibility, but we'll train all three
    parser.add_argument('--model_type', type=str, default='all', 
                       choices=['all', 'flow_matching', 'diffusion', 'ct'],
                       help='Model type to train. "all" trains flow_matching, diffusion, and ct sequentially')
    parser.add_argument('--data_path', type=str, default='data/two_moons_formatted.pkl', 
                       help='Path to two moons dataset')
    parser.add_argument('--use_synthetic', action='store_true', 
                       help='Use synthetic data instead of two moons dataset')
    parser.add_argument('--input_dim', type=int, default=2, 
                       help='Input dimension (2 for two moons)')
    parser.add_argument('--output_dim', type=int, default=2, 
                       help='Output dimension (2 for two moons)')
    parser.add_argument('--latent_dim', type=int, default=2,
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
    parser.add_argument('--recon_weight', type=float, default=0.0,
                        help='Reconstruction loss weight (0.0 = no reconstruction, 1.0 = equal weight)')
    parser.add_argument('--decoder_type', type=str, default='none',
                        choices=['none', 'linear', 'softmax'],
                        help='Decoder output transformation: none, linear, or softmax')
    parser.add_argument('--decoder_model', type=str, default='identity',
                        choices=['identity', 'mlp', 'resnet'],
                        help='Decoder model architecture: identity, mlp, or resnet')
    parser.add_argument('--encoder_type', type=str, default='identity',
                        choices=['mlp', 'mlp_normal', 'resnet', 'resnet_normal', 'identity', 'linear'],
                        help='Encoder type: mlp, mlp_normal, resnet, resnet_normal, identity, or linear')
    parser.add_argument('--recon_loss_type', type=str, default='mse',
                        choices=['mse', 'cross_entropy', 'none'],
                        help='Reconstruction loss type')
    parser.add_argument('--reg_weight', type=float, default=0.0,
                        help='Regularization loss weight')
    parser.add_argument('--noise_schedule', type=str, default='linear',
                        choices=['linear', 'cosine', 'sigmoid', 'learnable', 'simple_learnable'],
                        help='Noise schedule for CT model (linear, cosine, sigmoid, learnable, simple_learnable)')
    parser.add_argument('--reverse', action='store_true',
                        help='Swap x and y: predict moon coordinates from labels (y -> x)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment root directory (auto-generated if not provided)')
    parser.add_argument('--dropout_epochs', type=int, default=None, 
                       help='Number of epochs to use dropout (default: all epochs)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None, 
                       help='Directory to save results (auto-generated if not provided)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Verbose training output')
    
    args = parser.parse_args()
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"two_moons_{timestamp}"
    
    # Generate unique root save directory if not provided
    if args.save_dir is None:
        args.save_dir = f"artifacts/{args.experiment_name}"
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("VAE_flow Training Script - Two Moons Dataset")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Configuration:")
    print(f"  Model type: {args.model_type}")
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
    print(f"  Reconstruction weight: {args.recon_weight}")
    if args.model_type == "ct":
        print(f"  Noise schedule: {args.noise_schedule}")
    print(f"  Dropout epochs: {args.dropout_epochs if args.dropout_epochs else args.num_epochs}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)
    
    # Load data
    if args.use_synthetic:
        print("Generating synthetic data...")
        x_data, y_data = generate_synthetic_data(
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
    
    # Helper to build config per model
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
    
    def build_config(model_type: str):
        return create_vae_config(
            input_shape=(args.output_dim,) if args.reverse else (effective_input_dim,),
            output_shape=(effective_input_dim,) if args.reverse else (args.output_dim,),
            latent_shape=(args.latent_dim,),
            crn_type=args.crn_type,
            network_type=args.network_type,
            hidden_dims=tuple(args.hidden_dims),
            recon_weight=args.recon_weight,
            decoder_type=args.decoder_type,
            decoder_model=args.decoder_model,
            encoder_type=args.encoder_type,
            recon_loss_type=args.recon_loss_type,
            reg_weight=args.reg_weight,
            model_type=model_type,
            noise_schedule=args.noise_schedule
        )
    
    # Determine which models to train
    model_list = ['flow_matching', 'diffusion', 'ct'] if args.model_type == 'all' else [args.model_type]

    # Swap x and y if reverse mode is enabled
    if args.reverse:
        print("Reverse mode: Predicting moon coordinates from labels (y -> x)")
        train_x, train_y = y_train, x_train
        val_x, val_y = y_val, x_val
    else:
        train_x, train_y = x_train, y_train
        val_x, val_y = x_val, y_val

    # Train each requested model
    for mtype in model_list:
        print("\n" + "-" * 60)
        print(f"Training model: {mtype}")
        print("-" * 60)

        config = build_config(mtype)

        # Sanity print key config fields
        try:
            print(f"Config[{mtype}]: recon_loss_type={config.main['recon_loss_type']}, decoder_type={config.decoder['decoder_type']}, encoder_type={config.encoder['model_type']}")
        except Exception:
            pass

        trainer = VAEFlowTrainer(
            config=config,
            learning_rate=args.learning_rate,
            optimizer_name=args.optimizer,
            seed=args.seed
        )

        # Initialize model
        print("Initializing model...")
        batch_size = min(args.batch_size, train_x.shape[0])
        x_sample = train_x[:batch_size]
        y_sample = train_y[:batch_size]
        z_sample = jr.normal(jr.PRNGKey(args.seed), (batch_size, args.latent_dim))
        t_sample = jr.uniform(jr.PRNGKey(args.seed + 1), (batch_size,), minval=0.0, maxval=1.0)
        trainer.initialize(x_sample, y_sample, z_sample, t_sample)

        # Train
        history = trainer.train(
            x_data=train_x,
            y_data=train_y,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            validation_data=(val_x, val_y),
            dropout_epochs=args.dropout_epochs,
            verbose=args.verbose
        )

        # Save outputs in per-model directory
        model_save_dir = f"{args.save_dir}/{mtype}"
        print(f"Saving results to {model_save_dir}...")
        trainer.save_results(history, model_save_dir)

        # Print brief summary
        print(f"{mtype} final train loss: {history['train_losses'][-1]:.6f}")
        if history['val_losses']:
            print(f"{mtype} final val loss: {history['val_losses'][-1]:.6f}")


if __name__ == "__main__":
    main()
