#!/usr/bin/env python3
"""
Training script for conditional generation on sequence data (x | y).

This script trains the selected model with sequence data where both inputs and outputs are sequences.
It uses reversed mapping (inputs=x, targets=y) for conditional sequence generation.

Supports loading config from YAML files with command-line overrides.
"""

import argparse
from pathlib import Path
import pickle
from typing import Optional, Any, Dict

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax.core import FrozenDict

from src.flow_models.trainer_seq import SequenceTrainer
from src.flow_models.config import Config
from src.configs.base_config import BaseConfig
from src.flow_models.training_utils import get_save_directory, save_training_artifacts


def load_data(data_path: str):
    """Load dataset from pickle file.
    
    Expected format: dict with 'train' and 'val' keys, each containing 'x' and 'y' arrays.
    For sequences, x and y should be 3D arrays: (N, seq_len, embed_dim)
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    x_train = jnp.array(data['train']['x'])
    y_train = jnp.array(data['train']['y'])
    x_val = jnp.array(data['val']['x']) if 'val' in data and data['val'] is not None else None
    y_val = jnp.array(data['val']['y']) if 'val' in data and data['val'] is not None else None
    
    return x_train, y_train, x_val, y_val


def main():
    parser = argparse.ArgumentParser(
        description='Conditional generation training on sequence data (x | y) with config file support'
    )
    
    # Config file argument
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to YAML config file. If provided, uses default Config from src.flow_models.config. '
                            'If not provided, uses default Config with default values.')
    parser.add_argument('--config_class', type=str, default=None,
                       help='Optional: Custom config class name with full module path (e.g., "examples.two_moons.config.Config"). '
                            'Only needed if you want to use a custom config class instead of the default Config from flow_models.')
    
    # Model and data arguments
    parser.add_argument('--model_type', type=str, default='flow_matching', 
                       choices=['flow_matching', 'diffusion', 'ct'],
                       help='Model type to train')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to data file (pickle format with train/val splits). If not provided, generates synthetic data.')
    
    # Sequence-specific dimensions
    parser.add_argument('--z_seq_len', type=int, default=None,
                       help='Sequence length for z (latent/output sequence). Mutually exclusive with --output_shape.')
    parser.add_argument('--x_seq_len', type=int, default=None,
                       help='Sequence length for x (conditional input sequence). Mutually exclusive with --input_shape.')
    parser.add_argument('--embed_dim', type=int, default=None,
                       help='Embedding dimension for sequences')
    parser.add_argument('--input_shape', type=int, nargs='+', default=None,
                       help='Input shape as tuple (e.g., --input_shape 20 256 for seq_len, embed_dim). Mutually exclusive with --x_seq_len/--embed_dim.')
    parser.add_argument('--output_shape', type=int, nargs='+', default=None,
                       help='Output shape as tuple (e.g., --output_shape 10 256 for seq_len, embed_dim). Mutually exclusive with --z_seq_len/--embed_dim.')
    parser.add_argument('--latent_shape', type=int, nargs='+', default=None,
                       help='Latent shape as tuple (e.g., --latent_shape 10 256). If not provided, uses output_shape.')
    
    # Architecture arguments (can override config)
    parser.add_argument('--crn_type', type=str, default=None,
                       help='CRN type (overrides config). Use transformer_seq2seq for sequences.')
    parser.add_argument('--network_type', type=str, default=None,
                       help='Network type (overrides config). Use transformer_seq2seq for sequences.')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=None,
                       help='Hidden dimensions (overrides config)')
    parser.add_argument('--num_layers', type=int, default=None,
                       help='Number of transformer layers (overrides config)')
    parser.add_argument('--num_heads', type=int, default=None,
                       help='Number of attention heads (overrides config)')
    parser.add_argument('--mlp_ratio', type=float, default=None,
                       help='MLP ratio for transformer (overrides config)')
    parser.add_argument('--encoder_model_type', type=str, default=None,
                       choices=['mlp', 'mlp_normal', 'resnet', 'resnet_normal', 'identity', 'linear'],
                       help='Encoder model type (overrides config)')
    parser.add_argument('--decoder_model_type', type=str, default=None,
                       choices=['mlp', 'resnet', 'identity'],
                       help='Decoder model type (overrides config)')
    parser.add_argument('--decoder_type', type=str, default=None,
                       choices=['linear', 'softmax', 'none'],
                       help='Decoder output type (overrides config)')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--dropout_epochs', type=int, default=None,
                       help='Number of epochs to use dropout. If None, defaults to num_epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Number of training steps for learning rate warmup (0 = no warmup)')
    parser.add_argument('--warmup_epochs', type=float, default=None,
                       help='Number of epochs for warmup (overrides warmup_steps if provided)')
    
    # Loss arguments (can override config)
    parser.add_argument('--recon_weight', type=float, default=None,
                       help='Reconstruction loss weight (overrides config)')
    parser.add_argument('--recon_loss_type', type=str, default=None,
                       choices=['mse', 'cross_entropy', 'none'],
                       help='Reconstruction loss type (overrides config)')
    parser.add_argument('--reg_weight', type=float, default=None,
                       help='Regularization weight (overrides config)')
    parser.add_argument('--vae_weight', type=float, default=None,
                       help='VAE loss weight (overrides config)')
    parser.add_argument('--use_snr_weight', action='store_const', const=True, default=None,
                       help='Apply SNR weighting (overrides config)')
    parser.add_argument('--no_snr_weight', dest='use_snr_weight', action='store_const', const=False,
                       help='Disable SNR weighting (overrides config)')
    
    # Noise schedule arguments (can override config)
    parser.add_argument('--noise_schedule', type=str, default=None,
                       choices=['linear', 'cosine', 'sigmoid', 'exponential', 'cauchy', 'laplace', 
                               'logistic', 'quadratic', 'polynomial', 'monotonic_nn', 'learnable', 'network'],
                       help='Noise schedule type (overrides config)')
    parser.add_argument('--noise_schedule_learnable', action='store_const', const=True, default=None,
                       help='Make noise schedule learnable (overrides config)')
    parser.add_argument('--noise_schedule_fixed', dest='noise_schedule_learnable', action='store_const', const=False,
                       help='Freeze noise schedule (overrides config)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save results (default: auto-generated)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--unconditional', action='store_true',
                       help='Train for unconditional generation (x=None)')
    
    args = parser.parse_args()
    
    # Validate shape arguments
    if args.input_shape is not None and (args.x_seq_len is not None or args.embed_dim is not None):
        raise ValueError("Cannot specify both --input_shape and --x_seq_len/--embed_dim. Use one or the other.")
    if args.output_shape is not None and (args.z_seq_len is not None or args.embed_dim is not None):
        raise ValueError("Cannot specify both --output_shape and --z_seq_len/--embed_dim. Use one or the other.")
    
    # Convert sequence dimensions to shapes if specified
    if args.x_seq_len is not None and args.embed_dim is not None:
        args.input_shape = (args.x_seq_len, args.embed_dim)
    if args.z_seq_len is not None and args.embed_dim is not None:
        args.output_shape = (args.z_seq_len, args.embed_dim)
        if args.latent_shape is None:
            args.latent_shape = (args.z_seq_len, args.embed_dim)
    
    # Load config: priority is config_class (with or without YAML), then YAML, then default
    if args.config_class:
        # Import the custom config class
        try:
            parts = args.config_class.split('.')
            if len(parts) < 2:
                raise ValueError(
                    f"config_class must be a full module path. "
                    f"Example: 'examples.two_moons.config.Config'"
                )
            module_path = '.'.join(parts[:-1])
            class_name = parts[-1]
            module = __import__(module_path, fromlist=[class_name])
            config_class = getattr(module, class_name)
            
            if not issubclass(config_class, BaseConfig):
                raise ValueError(f"{config_class.__name__} does not inherit from BaseConfig")
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not import config class '{args.config_class}': {e}")
        
        if args.config_file:
            # Load from YAML using custom class
            print(f"Loading config from {args.config_file} using custom class {config_class.__name__}...")
            loaded_config = config_class.load_yaml(args.config_file)
            # Merge with defaults to ensure all default values are preserved
            base_config = config_class.merge_with_defaults(loaded_config)
            print(f"Loaded config with custom class: {base_config.__class__.__name__}")
        else:
            # Instantiate custom class with default values
            print(f"Using custom config class {config_class.__name__} with default values...")
            print("NOTE: You may need to specify shapes via --input_shape/--x_seq_len+--embed_dim, --output_shape/--z_seq_len+--embed_dim, and --latent_shape")
            print("      (encoder and decoder shapes will be set automatically from main config)")
            base_config = config_class()
            print(f"Instantiated config: {base_config.__class__.__name__}")
    elif args.config_file:
        # Load from YAML using default Config class
        print(f"Loading config from {args.config_file}...")
        
        # Validate file extension
        config_path = Path(args.config_file)
        if config_path.suffix not in ['.yaml', '.yml']:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}. Use .yaml or .yml")
        
        loaded_config = Config.load_yaml(args.config_file)
        # Merge with defaults to ensure all default values are preserved
        base_config = Config.merge_with_defaults(loaded_config)
        print(f"Loaded config with default Config class: {base_config.__class__.__name__}")
    else:
        # Use default unified Config from flow_models with default values
        # NOTE: When no config file is provided, the user MUST specify:
        #   - input_shape/x_seq_len+embed_dim, output_shape/z_seq_len+embed_dim, latent_shape
        #     (these set the shapes which are "NA" in default config)
        # The encoder and decoder shapes will automatically be set from these main config values
        print("Using default Config from src.flow_models.config with default values...")
        print("NOTE: You must specify shapes via --input_shape/--x_seq_len+--embed_dim, --output_shape/--z_seq_len+--embed_dim, and --latent_shape")
        print("      (encoder and decoder shapes will be set automatically from main config)")
        base_config = Config()
        print(f"Using default config: {base_config.__class__.__name__}")
    
    # Override with command-line arguments
    # For sequences, we use override_from_args (like generation) since it's conditional generation
    config = base_config.override_from_args(args, args.model_type, args.unconditional)
    
    # Override sequence-specific CRN parameters if provided
    if args.num_layers is not None or args.num_heads is not None or args.mlp_ratio is not None:
        crn_dict = dict(config.crn)
        crn_updates = BaseConfig.filter_none({
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'mlp_ratio': args.mlp_ratio,
        })
        if crn_updates:
            updated_crn = config.merge_frozen_dict('crn', crn_updates)
            from dataclasses import replace
            config = replace(config, crn=updated_crn)
    
    # Set up save directory
    args.save_dir = get_save_directory(args.save_dir, 'seq', args.model_type)
    
    # Load or generate data
    if args.data_path:
        print(f"Loading data from {args.data_path}...")
        x_train, y_train, x_val, y_val = load_data(args.data_path)
    else:
        # Generate synthetic sequence data for testing
        print("No data_path provided, generating synthetic sequence data...")
        key = jr.PRNGKey(args.seed)
        key, x_key, y_key = jr.split(key, 3)
        
        # Get shapes from config
        output_shape = config.main['output_shape']
        input_shape = config.main['input_shape'] if not args.unconditional else ()
        
        if isinstance(output_shape, str) or len(output_shape) < 2:
            raise ValueError("For sequences, output_shape must be (seq_len, embed_dim). Please specify via --output_shape or --z_seq_len+--embed_dim")
        if not args.unconditional and (isinstance(input_shape, str) or len(input_shape) < 2):
            raise ValueError("For sequences, input_shape must be (seq_len, embed_dim). Please specify via --input_shape or --x_seq_len+--embed_dim")
        
        z_seq_len, embed_dim = output_shape[0], output_shape[1]
        x_seq_len = input_shape[0] if not args.unconditional else 20  # default if unconditional
        
        n_train = 1000
        n_val = 200
        x_train = jr.normal(x_key, (n_train, x_seq_len, embed_dim)) if not args.unconditional else None
        y_train = jr.normal(y_key, (n_train, z_seq_len, embed_dim))
        x_val = jr.normal(jr.PRNGKey(args.seed + 1), (n_val, x_seq_len, embed_dim)) if not args.unconditional else None
        y_val = jr.normal(jr.PRNGKey(args.seed + 2), (n_val, z_seq_len, embed_dim))
    
    print(f"Data shapes:")
    print(f"  Train: x={x_train.shape if x_train is not None else None}, y={y_train.shape}")
    if x_val is not None:
        print(f"  Val: x={x_val.shape}, y={y_val.shape}")
    else:
        print(f"  Val: x=None, y={y_val.shape if y_val is not None else None}")
    
    # For conditional generation: inputs=x (conditional), targets=y (output sequence)
    train_x, train_y = x_train, y_train
    val_x, val_y = (x_val, y_val) if x_val is not None else (None, y_val)
    
    # Calculate warmup_steps
    if args.warmup_epochs is not None:
        # Calculate number of batches per epoch
        num_samples = train_y.shape[0]
        batches_per_epoch = (num_samples + args.batch_size - 1) // args.batch_size
        warmup_steps = int(args.warmup_epochs * batches_per_epoch)
    else:
        warmup_steps = args.warmup_steps
    
    # Create trainer
    trainer = SequenceTrainer(
        config=config,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        seed=args.seed,
        unconditional=args.unconditional,
        warmup_steps=warmup_steps,
        model_type=args.model_type
    )
    
    # Initialize
    bs = min(args.batch_size, train_y.shape[0])
    if args.unconditional:
        x_sample = None
    else:
        x_sample = train_x[:bs]
    y_sample = train_y[:bs]
    latent_shape = config.main['latent_shape']
    if isinstance(latent_shape, str) or len(latent_shape) < 2:
        raise ValueError("For sequences, latent_shape must be (seq_len, embed_dim). Please specify via --latent_shape or --z_seq_len+--embed_dim")
    # For sequences, latent_shape is (seq_len, embed_dim)
    z_sample = jr.normal(jr.PRNGKey(args.seed), (bs, latent_shape[0], latent_shape[1]))
    t_sample = jr.uniform(jr.PRNGKey(args.seed+1), (bs,), minval=0.0, maxval=1.0)
    
    print("Initializing model...")
    trainer.initialize(x_sample, y_sample, z_sample, t_sample)
    
    # Train
    train_x_input = None if args.unconditional else train_x
    val_x_input = None if args.unconditional else (val_x if val_x is not None else None)
    dropout_epochs = args.dropout_epochs if args.dropout_epochs is not None else args.num_epochs
    
    # Convert JAX arrays to lists of sequences for the trainer
    # Each row becomes a sequence in the list
    train_sequences = [train_y[i] for i in range(train_y.shape[0])]
    val_sequences = [val_y[i] for i in range(val_y.shape[0])] if val_y is not None else None
    
    print(f"Starting training for {args.num_epochs} epochs...")
    history = trainer.train(
        sequences_data=train_sequences,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        y_seq_len=train_y.shape[1] if train_y.ndim >= 2 else 12,  # Extract sequence length from data
        validation_data=val_sequences,
        dropout_epochs=dropout_epochs,
    )
    
    # Save results
    save_training_artifacts(args.save_dir, history, trainer, config)
    
    # Generation
    num_gen = min(200, train_y.shape[0])
    prng = jr.PRNGKey(args.seed + 123)
    
    if args.unconditional:
        # Unconditional generation
        y_gen = np.array(trainer.unconditional_generate(
            batch_shape=(num_gen,),
            num_steps=20,
            prng_key=prng
        ))
        y_real = np.array(train_y[:num_gen])
        x_labels = None
        cond_x = None
    else:
        # Conditional generation
        cond_x = train_x[:num_gen]
        y_gen = np.array(trainer.conditional_generate(cond_x, num_steps=20, prng_key=prng))
        y_real = np.array(train_y[:num_gen])
        x_labels = np.array(cond_x)
    
    # Compute sequence metrics on generated samples
    seq_metrics = trainer.compute_sequence_metrics(jnp.array(y_gen), jnp.array(y_real))
    
    # Plot
    trainer.save_sequence_plot(y_real=y_real, x_labels=x_labels, y_gen=y_gen, output_dir=args.save_dir)
    trainer.save_loss_trends_plot(history, output_dir=args.save_dir)
    
    # Generate trajectory plot
    trajectory_prng = jr.PRNGKey(args.seed + 456)
    trainer.save_trajectory_plot(
        cond_x=cond_x,
        num_trajectories=40,
        num_steps=20,
        prng_key=trajectory_prng,
        output_dir=args.save_dir
    )
    
    if args.verbose:
        print(f"Final Sequence Metrics: {seq_metrics}")
        if history.get('val_seq_metrics') and len(history['val_seq_metrics']) > 0:
            print(f"Final Validation Sequence Metrics: {history['val_seq_metrics'][-1]}")
        print(f"Saved generation assets to {args.save_dir}")
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()

