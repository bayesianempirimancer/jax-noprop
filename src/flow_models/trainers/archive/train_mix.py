#!/usr/bin/env python3
"""
Training script for regression/classification (x -> y) with VAE_flow_mix model.

This script trains the fm_mix model with forward mapping (inputs=x, targets=y)
for regression or classification tasks.

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

from src.flow_models.trainers.trainer_mix import MixTrainer
from src.flow_models.config_mix import Config
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
        description='Regression/classification training (x -> y) with fm_mix model and config file support'
    )
    
    # Config file argument
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to YAML config file. If provided, uses default Config from src.flow_models.config_mix. '
                            'If not provided, uses default Config with default values.')
    parser.add_argument('--config_class', type=str, default=None,
                       help='Optional: Custom config class name with full module path (e.g., "examples.two_moons.config.Config"). '
                            'Only needed if you want to use a custom config class instead of the default Config from flow_models.')
    
    # Data arguments
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
    
    # Flow planner arguments
    parser.add_argument('--sample_method', type=str, default=None,
                       choices=['mixture', 'normal'],
                       help='Flow planner sampling method: "mixture" (GMM) or "normal" (overrides config)')
    parser.add_argument('--num_clusters', type=int, default=None,
                       help='Number of GMM clusters (overrides config)')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Number of top clusters to sample from (overrides config)')
    parser.add_argument('--sinkhorn_refinement', action='store_const', const=True, default=None,
                       help='Enable sinkhorn refinement (overrides config)')
    parser.add_argument('--no_sinkhorn_refinement', dest='sinkhorn_refinement', action='store_const', const=False,
                       help='Disable sinkhorn refinement (overrides config)')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--dropout_epochs', type=int, default=None,
                       help='Number of epochs to use dropout. If None, defaults to num_epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0025,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Number of training steps for learning rate warmup (0 = no warmup)')
    parser.add_argument('--warmup_epochs', type=float, default=None,
                       help='Number of epochs for warmup (overrides warmup_steps if provided)')
    
    # GMM training arguments
    # Note: GMM updates are automatically enabled when sample_method='mixture'
    parser.add_argument('--gmm_lr', type=float, default=0.2,
                       help='Learning rate for GMM VBEM updates')
    parser.add_argument('--gmm_N_eff', type=float, default=2000.0,
                       help='Effective number of data points for GMM updates')
    
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
    parser.add_argument('--gmm_weight', type=float, default=None,
                       help='GMM loss weight (overrides config)')
    parser.add_argument('--normalize_snr_weight', action='store_const', const=True, default=None,
                       help='Normalize SNR weights by their mean (overrides config)')
    parser.add_argument('--no_normalize_snr_weight', dest='normalize_snr_weight', action='store_const', const=False,
                       help='Disable SNR weight normalization (overrides config)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
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
        args.output_shape = (args.output_shape,)
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
        # Load from YAML using default Config class from config_mix
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
        # Use default Config from config_mix with default values
        print("Using default Config from src.flow_models.config_mix with default values...")
        print("NOTE: You must specify shapes via --input_shape/--input_dim, --output_shape/--output_dim, and --latent_shape/--latent_dim")
        print("      (encoder and decoder shapes will be set automatically from main config)")
        base_config = Config()
        print(f"Using default config: {base_config.__class__.__name__}")
    
    # Override with command-line arguments
    # Check if this is a config_mix.Config (has flow_planner) or old config.Config (has noise_schedule)
    is_config_mix = hasattr(base_config, 'flow_planner')
    
    if is_config_mix and hasattr(base_config, 'override_from_args_regression'):
        # Use config_mix override_from_args_regression for regression
        config = base_config.override_from_args_regression(args, 'flow_matching')
    elif hasattr(base_config, 'override_from_args'):
        # Old config.py - might fail if it expects noise_schedule, so use fallback
        try:
            config = base_config.override_from_args(args, 'flow_matching', False)
        except AttributeError:
            # Fallback to manual override
            config = base_config
    else:
        # Fallback: manual override for config_mix
        config = base_config
        # Apply shape overrides if provided
        from dataclasses import replace
        main_dict = dict(config.main)
        if args.input_shape is not None:
            main_dict['input_shape'] = tuple(args.input_shape)
        if args.output_shape is not None:
            main_dict['output_shape'] = tuple(args.output_shape)
        if args.latent_shape is not None:
            main_dict['latent_shape'] = tuple(args.latent_shape)
        
        # Apply other main config overrides
        if args.recon_weight is not None:
            main_dict['recon_weight'] = args.recon_weight
        if args.reg_weight is not None:
            main_dict['reg_weight'] = args.reg_weight
        if args.vae_weight is not None:
            main_dict['vae_weight'] = args.vae_weight
        if args.gmm_weight is not None:
            main_dict['gmm_weight'] = args.gmm_weight
        if args.recon_loss_type is not None:
            main_dict['recon_loss_type'] = args.recon_loss_type
        if args.normalize_snr_weight is not None:
            main_dict['normalize_snr_weight'] = args.normalize_snr_weight
        
        config = replace(config, main=FrozenDict(main_dict))
        
        # Apply flow_planner overrides
        if args.sample_method is not None or args.top_k is not None or args.sinkhorn_refinement is not None or args.num_clusters is not None:
            flow_planner_dict = dict(config.flow_planner)
            if args.sample_method is not None:
                flow_planner_dict['sample_method'] = args.sample_method
            if args.top_k is not None:
                flow_planner_dict['top_k'] = args.top_k
            if args.sinkhorn_refinement is not None:
                flow_planner_dict['sinkhorn_refinement'] = args.sinkhorn_refinement
            if args.num_clusters is not None:
                gmm_dict = dict(flow_planner_dict.get('gmm', {}))
                gmm_dict['num_clusters'] = args.num_clusters
                flow_planner_dict['gmm'] = FrozenDict(gmm_dict)
            config = replace(config, flow_planner=FrozenDict(flow_planner_dict))
    
    # Set up save directory
    args.save_dir = get_save_directory(args.save_dir, 'reg_mix', 'flow_matching')
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    x_train, y_train, x_val, y_val = load_data(args.data_path)
    
    print(f"Data shapes:")
    print(f"  Train: x={x_train.shape}, y={y_train.shape}")
    print(f"  Val: x={x_val.shape}, y={y_val.shape}")
    
    # Calculate warmup_steps
    if args.warmup_epochs is not None:
        # Calculate number of batches per epoch
        num_samples = y_train.shape[0]
        batches_per_epoch = (num_samples + args.batch_size - 1) // args.batch_size
        warmup_steps = int(args.warmup_epochs * batches_per_epoch)
    else:
        warmup_steps = args.warmup_steps
    
    # Create trainer
    trainer = MixTrainer(
        config=config,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        seed=args.seed,
        warmup_steps=warmup_steps,
        gmm_lr=args.gmm_lr,
        gmm_N_eff=args.gmm_N_eff
    )
    
    # Initialize
    print("Initializing model...")
    # Use a single sample for initialization (model will add batch dimension internally)
    trainer.initialize(x_train[0], y_train[0])
    
    # Train
    dropout_epochs = args.dropout_epochs if args.dropout_epochs is not None else args.num_epochs
    
    print(f"Starting training for {args.num_epochs} epochs...")
    history = trainer.train(
        x_data=x_train,
        y_data=y_train,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        dropout_epochs=dropout_epochs
    )
    
    # Save results
    save_training_artifacts(args.save_dir, history, trainer, config)
    
    # Create plots using trainer's save_results method
    # This will generate training progress plots, data visualization, and trajectory plots
    trainer.save_results(history, args.save_dir)
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()

