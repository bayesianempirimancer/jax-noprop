#!/usr/bin/env python3
"""
Training script for conditional generation (x | y).

This script trains the selected model with reversed mapping (inputs=y, targets=x)
and evaluates conditional generation by sampling with PRNGKey.

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

from src.flow_models.trainer_gen import GenerationTrainer
from src.flow_models.config import Config
from src.configs.base_config import BaseConfig
from src.flow_models.training_utils import get_save_directory, save_training_artifacts
from dataclasses import replace


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
        description='Conditional generation training (x | y) with config file support'
    )
    
    # Config file argument
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to YAML config file. If provided, uses default Config from src.flow_models.config. '
                            'If not provided, uses default Config with default values.')
    parser.add_argument('--config_class', type=str, default=None,
                       help='Optional: Custom config class name with full module path (e.g., "examples.two_moons.config.TwoMoonsFlowConfig"). '
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
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--dropout_epochs', type=int, default=None,
                       help='Number of epochs to use dropout. If None, defaults to num_epochs')
    parser.add_argument('--batch_size', type=int, default=256,
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
    
    # Validate that dim and shape arguments are not both specified
    if args.input_dim is not None and args.input_shape is not None:
        raise ValueError("Cannot specify both --input_dim and --input_shape. Use one or the other.")
    if args.output_dim is not None and args.output_shape is not None:
        raise ValueError("Cannot specify both --output_dim and --output_shape. Use one or the other.")
    if args.latent_dim is not None and args.latent_shape is not None:
        raise ValueError("Cannot specify both --latent_dim and --latent_shape. Use one or the other.")
    
    # Convert dims to shapes if specified
    # Special case: input_dim=0 for unconditional generation should become empty tuple ()
    if args.input_dim is not None:
        if args.input_dim == 0 and args.unconditional:
            args.input_shape = ()  # Empty tuple for unconditional generation
        else:
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
        # NOTE: When no config file is provided, the user MUST specify:
        #   - input_shape/input_dim, output_shape/output_dim, latent_shape/latent_dim
        #     (these set the shapes which are "NA" in default config)
        # The encoder and decoder shapes will automatically be set from these main config values
        print("Using default Config from src.flow_models.config with default values...")
        print("NOTE: You must specify shapes via --input_shape/--input_dim, --output_shape/--output_dim, and --latent_shape/--latent_dim")
        print("      (encoder and decoder shapes will be set automatically from main config)")
        base_config = Config()
        print(f"Using default config: {base_config.__class__.__name__}")
    
    # Override with command-line arguments
    config = base_config.override_from_args(args, args.model_type, args.unconditional)
    
    # Set up save directory
    args.save_dir = get_save_directory(args.save_dir, 'gen', args.model_type, unconditional=args.unconditional)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    x_train, y_train, x_val, y_val = load_data(args.data_path)
    
    # Reverse mapping for generation: inputs=y, targets=x
    train_x, train_y = y_train, x_train
    val_x, val_y = y_val, x_val
    
    # Calculate warmup_steps
    if args.warmup_epochs is not None:
        # Calculate number of batches per epoch
        num_samples = train_y.shape[0]
        batches_per_epoch = (num_samples + args.batch_size - 1) // args.batch_size
        warmup_steps = int(args.warmup_epochs * batches_per_epoch)
    else:
        warmup_steps = args.warmup_steps
    
    # Create trainer
    trainer = GenerationTrainer(
        config=config,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        seed=args.seed,
        unconditional=args.unconditional,
        warmup_steps=warmup_steps,
        model_type=args.model_type
    )
    
    # Initialize
    print("Initializing model...")
    # Use a single sample for initialization (model will add batch dimension internally)
    if args.unconditional:
        x_sample = None
    else:
        x_sample = train_x[0] if train_x is not None else None
    y_sample = train_y[0]
    trainer.initialize(x_sample, y_sample)
    
    # Train
    train_x_input = None if args.unconditional else train_x
    val_x_input = None if args.unconditional else val_x
    dropout_epochs = args.dropout_epochs if args.dropout_epochs is not None else args.num_epochs
    
    print(f"Starting training for {args.num_epochs} epochs...")
    history = trainer.train(
        x_data=train_x_input,
        y_data=train_y,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=(val_x_input, val_y),
        dropout_epochs=dropout_epochs
    )
    
    # Save results
    save_training_artifacts(args.save_dir, history, trainer, config)
    
    # Generation
    num_gen = min(2000, val_y.shape[0])
    prng = jr.PRNGKey(args.seed + 123)
    
    if args.unconditional:
        x_gen = np.array(trainer.unconditional_generate(
            batch_shape=(num_gen,),
            num_steps=20,
            prng_key=prng
        ))
        x_real = np.array(val_y[:num_gen])
        y_labels = None
        cond_y = None
    else:
        cond_y = val_x[:num_gen]
        x_gen = np.array(trainer.conditional_generate(cond_y, num_steps=20, prng_key=prng))
        x_real = np.array(val_y[:num_gen])
        y_labels = np.array(cond_y)
    
    # Compute Chamfer Distance
    from src.utils.metrics import chamfer_distance
    chamfer_dist = chamfer_distance(jnp.array(x_gen), jnp.array(x_real))
    
    # Save results (includes all plots: generation, loss trends, trajectories)
    trainer.save_results(history, args.save_dir, x_real=x_real, x_gen=x_gen, y_labels=y_labels)
    
    if args.verbose:
        print(f"Final Chamfer Distance: {chamfer_dist:.6f}")
        if history.get('val_chamfer_distances') and len(history['val_chamfer_distances']) > 0:
            print(f"Final Validation Chamfer Distance: {history['val_chamfer_distances'][-1]:.6f}")
        print(f"Saved generation assets to {args.save_dir}")


if __name__ == '__main__':
    main()

