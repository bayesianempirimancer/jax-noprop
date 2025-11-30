#!/usr/bin/env python3
"""
Unified training script for flow models (Flow Matching, Diffusion, CT).

This script trains the selected model using the unified FlowModel architecture.
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
from dataclasses import replace

from src.flow_models.trainers.trainer import Trainer
from src.flow_models.config import Config
from src.configs.base_config import BaseConfig
from src.flow_models.trainers.training_utils import get_save_directory, save_training_artifacts


def load_data(data_path: str):
    """Load dataset from pickle file.
    
    Expected format: dict with 'train' and 'val' keys, each containing 'x' and 'y' arrays.
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    x_train = jnp.array(data['train']['x'])
    y_train = jnp.array(data['train']['y'])
    x_val = jnp.array(data['val']['x'])
    y_val = jnp.array(data['val']['y'])
    
    return x_train, y_train, x_val, y_val


def main():
    parser = argparse.ArgumentParser(
        description='Unified flow model training (x -> y) with config file support'
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
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data file (pickle format with train/val splits)')
    
    # Data dimensions/shapes (can override config)
    parser.add_argument('--input_dim', type=int, default=None,
                       help='Input dimension (converted to shape (dim,)). Mutually exclusive with --input_shape.')
    parser.add_argument('--input_shape', type=int, nargs='+', default=None,
                       help='Input shape as tuple (e.g., --input_shape 2 3). Mutually exclusive with --input_dim.')
    parser.add_argument('--output_dim', type=int, default=None,
                       help='Output dimension (converted to shape (dim,)). Mutually exclusive with --output_shape.')
    parser.add_argument('--output_shape', type=int, nargs='+', default=None,
                       help='Output shape as tuple (e.g., --output_shape 2 3). Mutually exclusive with --output_dim.')
    parser.add_argument('--latent_dim', type=int, default=None,
                       help='Latent dimension (converted to shape (dim,)). Mutually exclusive with --latent_shape.')
    parser.add_argument('--latent_shape', type=int, nargs='+', default=None,
                       help='Latent shape as tuple (e.g., --latent_shape 4 4). Mutually exclusive with --latent_dim.')
    
    # Architecture arguments (can override config)
    parser.add_argument('--crn_type', type=str, default=None,
                       help='CRN type (overrides config)')
    parser.add_argument('--network_type', type=str, default=None,
                       help='Network type (overrides config)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=None,
                       help='Hidden dimensions (overrides config)')
    parser.add_argument('--encoder_model_type', type=str, default=None,
                       choices=['mlp', 'mlp_normal', 'resnet', 'resnet_normal', 'identity', 'linear'],
                       help='Encoder model type (overrides config)')
    parser.add_argument('--decoder_model_type', type=str, default=None,
                       choices=['mlp', 'resnet', 'identity'],
                       help='Decoder model type (overrides config)')
    parser.add_argument('--decoder_type', type=str, default=None,
                       choices=['linear', 'softmax', 'none'],
                       help='Decoder output type (overrides config)')
    
    # Training arguments (defaults are None to allow config file values to take precedence)
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs (overrides config if provided)')
    parser.add_argument('--dropout_epochs', type=int, default=None,
                       help='Number of epochs to use dropout. If None, defaults to num_epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config if provided)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config if provided)')
    parser.add_argument('--optimizer', type=str, default=None, choices=['adam', 'sgd', 'adagrad'],
                       help='Optimizer (overrides config if provided)')
    parser.add_argument('--warmup_steps', type=int, default=None,
                       help='Number of training steps for learning rate warmup (overrides config if provided)')
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
    parser.add_argument('--normalize_snr_weight', action='store_const', const=True, default=None,
                       help='Normalize SNR weights by their mean (overrides config)')
    parser.add_argument('--no_normalize_snr_weight', dest='normalize_snr_weight', action='store_const', const=False,
                       help='Disable SNR weight normalization (overrides config)')
    
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
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config if provided)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save results (default: auto-generated)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate that dim and shape arguments are not both specified
    if args.input_dim is not None and args.input_shape is not None:
        raise ValueError("Cannot specify both --input_dim and --input_shape. Use one or the other.")
    if args.output_dim is not None and args.output_shape is not None:
        raise ValueError("Cannot specify both --output_dim and --output_shape. Use one or the other.")
    if args.latent_dim is not None and args.latent_shape is not None:
        raise ValueError("Cannot specify both --latent_dim and --latent_shape. Use one or the other.")
    
    # Convert dims to shapes if specified
    if args.input_dim is not None:
        args.input_shape = (args.input_dim,)
    if args.output_dim is not None:
        args.output_shape = (args.output_dim,)
    if args.latent_dim is not None:
        args.latent_shape = (args.latent_dim,)
    
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
            print("NOTE: You may need to specify shapes via --input_shape/--input_dim, --output_shape/--output_dim, and --latent_shape/--latent_dim")
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
        print("Using default Config from src.flow_models.config with default values...")
        print("NOTE: You must specify shapes via --input_shape/--input_dim, --output_shape/--output_dim, and --latent_shape/--latent_dim")
        print("      (encoder and decoder shapes will be set automatically from main config)")
        base_config = Config()
        print(f"Using default config: {base_config.__class__.__name__}")
    
    # Override with command-line arguments
    # For regression, we don't reverse shapes (unlike generation)
    config = base_config.override_from_args_regression(args, args.model_type)
    
    # Set loss_type based on model_type
    loss_type_map = {
        'flow_matching': 'flow_loss',
        'diffusion': 'noise_loss',
        'ct': 'target_loss'
    }
    loss_type = loss_type_map.get(args.model_type, 'flow_loss')
    
    # Update config with loss_type
    main_updates = {'loss_type': loss_type}
    updated_main = config.merge_frozen_dict('main', main_updates)
    config = replace(config, main=updated_main)
    
    print(f"Configured for model type: {args.model_type} -> loss type: {loss_type}")
    
    # Get training parameters from config if not provided via command line
    def get_config_value(key, default, config_obj=config):
        """Get value from config.main dict, config attribute, or default."""
        if hasattr(config_obj, 'main') and key in config_obj.main:
            return config_obj.main[key]
        if hasattr(config_obj, key):
            return getattr(config_obj, key)
        return default
    
    num_epochs = args.num_epochs if args.num_epochs is not None else get_config_value('num_epochs', 50)
    batch_size = args.batch_size if args.batch_size is not None else get_config_value('batch_size', 256)
    learning_rate = args.learning_rate if args.learning_rate is not None else get_config_value('learning_rate', 0.0025)
    optimizer = args.optimizer if args.optimizer is not None else get_config_value('optimizer', 'adam')
    seed = args.seed if args.seed is not None else get_config_value('seed', 42)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else get_config_value('warmup_steps', 0)
    
    # Set up save directory
    args.save_dir = get_save_directory(args.save_dir, 'reg_unified', args.model_type)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    x_train, y_train, x_val, y_val = load_data(args.data_path)
    
    print(f"Data shapes:")
    print(f"  Train: x={x_train.shape}, y={y_train.shape}")
    print(f"  Val: x={x_val.shape}, y={y_val.shape}")
    
    # Calculate warmup_steps from warmup_epochs if provided
    if args.warmup_epochs is not None:
        # Calculate number of batches per epoch
        num_samples = y_train.shape[0]
        batches_per_epoch = (num_samples + batch_size - 1) // batch_size
        warmup_steps = int(args.warmup_epochs * batches_per_epoch)
    
    # Create trainer
    trainer = Trainer(
        config=config,
        learning_rate=learning_rate,
        optimizer_name=optimizer,
        seed=seed,
        warmup_steps=warmup_steps,
        model_type=args.model_type
    )
    
    # Initialize
    print("Initializing model...")
    # Use a single sample for initialization (model will add batch dimension internally)
    trainer.initialize(x_train[0], y_train[0])
    
    # Train
    dropout_epochs = args.dropout_epochs if args.dropout_epochs is not None else num_epochs
    
    print(f"Starting training for {num_epochs} epochs...")
    history = trainer.train(
        x_data=x_train,
        y_data=y_train,
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        dropout_epochs=dropout_epochs
    )
    
    # Save results
    save_training_artifacts(args.save_dir, history, trainer, config)
    
    # Create plots using trainer's save_results method
    trainer.save_results(history, args.save_dir)
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
