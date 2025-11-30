#!/usr/bin/env python3
"""
Unified training script for conditional and unconditional generation.

This script trains a FlowModel for generation tasks (x | y or x_unconditional).
It uses the unified FlowModel and Trainer infrastructure.
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
        description='Unified flow model training for generation (x | y) or unconditional (x)'
    )
    
    # Config file argument
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to YAML config file.')
    parser.add_argument('--config_class', type=str, default=None,
                       help='Optional: Custom config class name with full module path.')
    
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
                       help='Input shape as tuple. Mutually exclusive with --input_dim.')
    parser.add_argument('--output_dim', type=int, default=None,
                       help='Output dimension (converted to shape (dim,)). Mutually exclusive with --output_shape.')
    parser.add_argument('--output_shape', type=int, nargs='+', default=None,
                       help='Output shape as tuple. Mutually exclusive with --output_dim.')
    parser.add_argument('--latent_dim', type=int, default=None,
                       help='Latent dimension (converted to shape (dim,)). Mutually exclusive with --latent_shape.')
    parser.add_argument('--latent_shape', type=int, nargs='+', default=None,
                       help='Latent shape as tuple. Mutually exclusive with --latent_dim.')
    
    # Architecture arguments (can override config)
    parser.add_argument('--crn_type', type=str, default=None, help='CRN type (overrides config)')
    parser.add_argument('--network_type', type=str, default=None, help='Network type (overrides config)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=None, help='Hidden dimensions (overrides config)')
    parser.add_argument('--encoder_model_type', type=str, default=None, help='Encoder model type (overrides config)')
    parser.add_argument('--decoder_model_type', type=str, default=None, help='Decoder model type (overrides config)')
    parser.add_argument('--decoder_type', type=str, default=None, help='Decoder output type (overrides config)')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--dropout_epochs', type=int, default=None, help='Number of epochs to use dropout')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default=None, choices=['adam', 'sgd', 'adagrad', 'adamw'], help='Optimizer')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Number of training steps for learning rate warmup')
    parser.add_argument('--warmup_epochs', type=float, default=None, help='Number of epochs for warmup')
    
    # Loss arguments
    parser.add_argument('--recon_weight', type=float, default=None, help='Reconstruction loss weight')
    parser.add_argument('--recon_loss_type', type=str, default=None, help='Reconstruction loss type')
    parser.add_argument('--reg_weight', type=float, default=None, help='Regularization weight')
    parser.add_argument('--vae_weight', type=float, default=None, help='VAE loss weight')
    parser.add_argument('--use_snr_weight', action='store_const', const=True, default=None, help='Use SNR weighting')
    parser.add_argument('--no_use_snr_weight', dest='use_snr_weight', action='store_const', const=False, help='Disable SNR weighting')
    parser.add_argument('--use_recon_snr_weight', action='store_const', const=True, default=None, help='Use SNR weighting for reconstruction loss')
    parser.add_argument('--no_use_recon_snr_weight', dest='use_recon_snr_weight', action='store_const', const=False, help='Disable SNR weighting for reconstruction loss')
    parser.add_argument('--normalize_snr_weight', action='store_const', const=True, default=None, help='Normalize SNR weights')
    parser.add_argument('--no_normalize_snr_weight', dest='normalize_snr_weight', action='store_const', const=False, help='Disable SNR weight normalization')
    
    # Noise schedule arguments
    parser.add_argument('--noise_schedule', type=str, default=None, help='Noise schedule type')
    parser.add_argument('--noise_schedule_learnable', action='store_const', const=True, default=None, help='Make noise schedule learnable')
    parser.add_argument('--noise_schedule_fixed', dest='noise_schedule_learnable', action='store_const', const=False, help='Freeze noise schedule')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--unconditional', action='store_true', help='Train for unconditional generation (x=None)')
    parser.add_argument('--num_steps', type=int, default=None, help='Number of integration steps for generation')
    
    args = parser.parse_args()
    
    # Shape handling logic similar to train_gen.py
    if args.input_dim is not None:
        if args.input_dim == 0 and args.unconditional:
            args.input_shape = ()
        else:
            args.input_shape = (args.input_dim,)
    if args.output_dim is not None:
        args.output_shape = (args.output_dim,)
    if args.latent_dim is not None:
        args.latent_shape = (args.latent_dim,)
        
    # Load Config
    if args.config_class:
        # Custom config class loading (simplified for brevity, assuming standard import)
        try:
            parts = args.config_class.split('.')
            module_path = '.'.join(parts[:-1])
            class_name = parts[-1]
            module = __import__(module_path, fromlist=[class_name])
            config_class = getattr(module, class_name)
            if args.config_file:
                loaded_config = config_class.load_yaml(args.config_file)
                base_config = config_class.merge_with_defaults(loaded_config)
            else:
                base_config = config_class()
        except Exception as e:
            raise ValueError(f"Could not load config class {args.config_class}: {e}")
    elif args.config_file:
        loaded_config = Config.load_yaml(args.config_file)
        base_config = Config.merge_with_defaults(loaded_config)
    else:
        print("Using default Config...")
        base_config = Config()
        
    # Override config from args
    # Note: override_from_args handles the logic for generation (reversing shapes if needed)
    config = base_config.override_from_args(args, args.model_type, args.unconditional)
    
    # Set loss_type based on model_type (Unified Trainer requirement)
    loss_type_map = {
        'flow_matching': 'flow_loss',
        'diffusion': 'noise_loss',
        'ct': 'target_loss'
    }
    loss_type = loss_type_map.get(args.model_type, 'flow_loss')
    
    # Update config with loss_type
    # Use merge_frozen_dict which handles the FrozenDict nature of 'main'
    updated_main = config.merge_frozen_dict('main', {'loss_type': loss_type})
    config = replace(config, main=updated_main)
    
    print(f"Configured for model type: {args.model_type} -> loss type: {loss_type}")
    
    # Setup save directory
    args.save_dir = get_save_directory(args.save_dir, 'gen_unified', args.model_type, unconditional=args.unconditional)
    
    # Training hyperparameters
    def get_config_value(key, default):
        if hasattr(config, 'main') and key in config.main:
            return config.main[key]
        if hasattr(config, key):
            return getattr(config, key)
        return default

    num_epochs = args.num_epochs if args.num_epochs is not None else get_config_value('num_epochs', 50)
    batch_size = args.batch_size if args.batch_size is not None else get_config_value('batch_size', 256)
    learning_rate = args.learning_rate if args.learning_rate is not None else get_config_value('learning_rate', 0.0025)
    optimizer = args.optimizer if args.optimizer is not None else get_config_value('optimizer', 'adam')
    seed = args.seed if args.seed is not None else get_config_value('seed', 42)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else get_config_value('warmup_steps', 0)
    num_steps = args.num_steps if args.num_steps is not None else get_config_value('num_steps', 20)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    x_train, y_train, x_val, y_val = load_data(args.data_path)
    
    # Prepare data for generation task:
    # Generation: x | y (generate x given y)
    # Unified Trainer expects inputs as 'x_sample' and targets as 'y_sample' in initialize/train methods.
    # However, FlowModel usually maps x -> y (encoder(x) -> z -> decoder(z) -> y).
    # Wait, for generation we want to model P(x|y).
    # The Unified FlowModel (via override_from_args) sets input_shape=y.shape, output_shape=x.shape for generation.
    # So "input" to the model is y (condition), "target" is x (generated).
    # Therefore, we pass y as input, x as target to the trainer.
    
    if args.unconditional:
        # Unconditional: x (generate x from nothing/latent)
        # Input is None/Empty. Target is x.
        train_input = None # Or appropriate placeholder if handled
        train_target = y_train # In original code, data was in y_train/x_train variables
        # Wait, load_data returns x_train, y_train.
        # In train_gen.py: "train_x, train_y = y_train, x_train" (reverse mapping)
        # And for unconditional: "train_x_input = None"
        # So we should stick to: Target is x_train. Input is y_train (cond) or None (uncond).
        
        train_target = x_train # We want to generate x
        train_input = None
        val_target = x_val
        val_input = None
    else:
        # Conditional: x | y
        train_target = x_train # We want to generate x
        train_input = y_train # Condition on y
        val_target = x_val
        val_input = y_val

    # Handle batch dimensions for initialization
    # Trainer.initialize expects single samples
    if args.unconditional:
         # For unconditional, input_shape in config is likely empty tuple ()
         # Trainer.initialize expects x_sample to match input_shape.
         # If input_shape is (), x_sample should be array of shape () or (1,) ?
         # Actually FlowModel handles x=None for unconditional if configured.
         # But Trainer.initialize calls self.model.init(..., x_sample, ...)
         # If x_sample is None, FlowModel.init must handle it.
         # Let's pass a dummy input if needed, or None if FlowModel supports it.
         # flow_model.py: __call__ says "x: Optional[jnp.ndarray] = None"
         init_input = None
    else:
         init_input = train_input[0]
    
    init_target = train_target[0]

    # Calculate warmup
    if args.warmup_epochs is not None:
        num_samples = train_target.shape[0]
        batches_per_epoch = (num_samples + batch_size - 1) // batch_size
        warmup_steps = int(args.warmup_epochs * batches_per_epoch)
        
    # Create Trainer
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
    trainer.initialize(init_input, init_target)
    
    # Train
    dropout_epochs = args.dropout_epochs if args.dropout_epochs is not None else num_epochs
    
    print(f"Starting training for {num_epochs} epochs...")
    
    if args.unconditional:
        # Pass None for unconditional training
        # The trainer handles None inputs correctly now
        train_input_pass = None
        val_input_pass = None
    else:
        train_input_pass = train_input
        val_input_pass = val_input
        
    history = trainer.train(
        x_data=train_input_pass,
        y_data=train_target,
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(val_input_pass, val_target),
        dropout_epochs=dropout_epochs
    )
    
    # Save base artifacts
    save_training_artifacts(args.save_dir, history, trainer, config)
    
    # --- Generation & Plotting ---
    print("Generating samples and plots...")
    
    num_gen = min(2000, val_target.shape[0])
    prng = jr.PRNGKey(seed + 123)
    
    if args.unconditional:
        # Unconditional sampling
        x_gen = np.array(trainer.sample(
            num_samples=num_gen,
            num_steps=num_steps,
            prng_key=prng
        ))
        x_real = np.array(val_target[:num_gen])
        y_labels = None
        cond_y = None
    else:
        # Conditional sampling
        cond_y = val_input[:num_gen]
        # Use conditional_sample (which calls predict/sample/conditional_generate)
        x_gen = np.array(trainer.conditional_sample(cond_y, num_steps=num_steps, prng_key=prng))
        x_real = np.array(val_target[:num_gen])
        y_labels = np.array(cond_y)
        
    # Store in history
    history['x_gen'] = x_gen
    history['x_real'] = x_real
    
    # Compute Chamfer
    try:
        from src.utils.metrics import chamfer_distance
        chamfer_dist = chamfer_distance(jnp.array(x_gen), jnp.array(x_real))
        print(f"Final Chamfer Distance: {chamfer_dist:.6f}")
        history['final_chamfer'] = chamfer_dist
    except ImportError:
        print("Could not import chamfer_distance")

    # Create Plots
    try:
        from src.utils.plotting.plot_generation import create_generation_plot
        from src.utils.plotting.plot_loss_trends import create_loss_trends_plot
        from src.utils.plotting.plot_trajectories import plot_latent_trajectories
        
        # Loss trends
        create_loss_trends_plot(history, args.model_type, args.save_dir)
        
        # Generation plot
        create_generation_plot(
            np.array(x_real), 
            np.array(y_labels) if y_labels is not None else None, 
            np.array(x_gen), 
            args.save_dir, 
            args.unconditional
        )
        
        # Latent trajectories
        trainer.rng, traj_rng = jr.split(trainer.rng)
        cond_subset = None if args.unconditional else (y_labels[:20] if y_labels is not None else None)
        
        plot_latent_trajectories(
            model=trainer.model,
            params=trainer.params,
            model_type=args.model_type,
            unconditional=args.unconditional,
            output_dir=args.save_dir,
            cond_y=cond_subset,
            num_trajectories=20,
            num_steps=num_steps,
            prng_key=traj_rng,
            rng=trainer.rng
        )
    except ImportError as e:
        print(f"Could not create plots due to import error: {e}")
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()

    print(f"Saved generation results to {args.save_dir}")

if __name__ == '__main__':
    main()

