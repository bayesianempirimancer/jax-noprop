#!/usr/bin/env python3
"""
Training script for conditional generation (x | y) using VAE_flow_mix model.

This script trains the fm_mix model with reversed mapping (inputs=y, targets=x)
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
import optax
from flax.core import FrozenDict

from src.flow_models.fm_mix import VAE_flow_mix
from src.flow_models.trainers.trainer_mix_gen import train_epoch_fm_mix, save_results_fm_mix
from src.flow_models.config_mix import Config
from src.configs.base_config import BaseConfig
from src.flow_models.trainers.training_utils import get_save_directory
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
        description='Conditional generation training (x | y) with fm_mix model and config file support'
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
    
    if is_config_mix and hasattr(base_config, 'override_from_args'):
        # Use config_mix override_from_args for generation
        config = base_config.override_from_args(args, 'flow_matching', args.unconditional)
    elif hasattr(base_config, 'override_from_args'):
        # Old config.py - might fail if it expects noise_schedule, so use fallback
        try:
            config = base_config.override_from_args(args, 'flow_matching', args.unconditional)
        except AttributeError:
            # Fallback to manual override
            config = base_config
    else:
        # Fallback: manual override for config_mix
        config = base_config
        # Apply shape overrides if provided
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
    args.save_dir = get_save_directory(args.save_dir, 'gen_mix', 'flow_matching', unconditional=args.unconditional)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    x_train, y_train, x_val, y_val = load_data(args.data_path)
    
    # Reverse mapping for generation: inputs=y, targets=x
    train_x, train_y = y_train, x_train
    val_x, val_y = y_val, x_val
    
    # Set random seed
    key = jr.PRNGKey(args.seed)
    np.random.seed(args.seed)
    
    # Create model
    print("Creating VAE_flow_mix model...")
    model = VAE_flow_mix(config=config)
    
    # Initialize model
    print("Initializing model...")
    x_sample = train_x[0:1] if train_x is not None else None
    y_sample = train_y[0:1]
    key, init_key = jr.split(key)
    params = model.init(init_key, x_sample, y_sample, init_key)
    
    # Initialize GMM cluster means from encoded target data (if using mixture sampling)
    sample_method = config.flow_planner.get('sample_method', 'mixture')
    if sample_method == "mixture":
        print("Initializing GMM cluster means from encoded target data...")
        # Use a batch of training data for better initialization
        num_init_samples = min(1000, train_y.shape[0])
        y_init = train_y[:num_init_samples]
        
        key, encode_key = jr.split(key)
        # Encode target data to get latent representations
        mu_z_target, _ = model.apply(
            params, y_init, method='encode', training=False, rngs={'dropout': encode_key}
        )
        z_target_flat = mu_z_target.reshape(-1, model.z_dim)
        
        # Get GMM config
        from src.vae.vb_gmm import GMMVBEM
        num_clusters = config.flow_planner.get('gmm', {}).get('num_clusters', 8)
        latent_dim = model.z_dim
        
        # Initialize cluster means from data
        key, init_key = jr.split(key)
        mu_n_initialized = GMMVBEM.get_initial_cluster_means(
            num_clusters=num_clusters,
            latent_dim=latent_dim,
            x=z_target_flat,
            key=init_key
        )
        
        # Update GMM params with initialized cluster means
        from flax.core import unfreeze, freeze
        params_unfrozen = unfreeze(params)
        gmm_params = dict(params_unfrozen['params']['flow_planner']['gmm'])
        gmm_params['mu_n'] = mu_n_initialized
        params_unfrozen['params']['flow_planner']['gmm'] = gmm_params
        params = freeze(params_unfrozen)
        print(f"  Initialized {num_clusters} cluster means from {z_target_flat.shape[0]} encoded samples")
    
    # Create optimizer
    # Create optimizer
    # Note: GMM params are excluded from gradients via stop_gradient in extract_params(),
    # so they won't be updated by the optimizer even though they're in the params structure
    if args.optimizer.lower() == 'adam':
        optimizer = optax.adam(args.learning_rate)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optax.adamw(args.learning_rate)
    else:
        optimizer = optax.sgd(args.learning_rate)
    
    opt_state = optimizer.init(params)
    
    # GMM will be updated automatically during training when sample_method='mixture'
    # No initial fitting needed - the update routine in train_epoch handles it
    sample_method = config.flow_planner.get('sample_method', 'mixture')
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_flow_losses': [],
        'val_flow_losses': [],
        'train_recon_losses': [],
        'val_recon_losses': [],
        'train_reg_losses': [],
        'val_reg_losses': [],
        'train_vae_losses': [],
        'val_vae_losses': [],
        'train_gmm_losses': [],
        'val_gmm_losses': [],
    }
    
    dropout_epochs = args.dropout_epochs if args.dropout_epochs is not None else args.num_epochs
    
    for epoch in range(args.num_epochs):
        use_dropout = epoch < dropout_epochs
        
        # Training
        key, epoch_key = jr.split(key)
        # GMM updates are automatically enabled when sample_method='mixture'
        update_gmm_epoch = (sample_method == "mixture")
        params, opt_state, avg_train_loss, train_metrics = train_epoch_fm_mix(
            model, params, train_x, train_y, opt_state, optimizer, epoch_key,
            batch_size=args.batch_size, training=use_dropout, update_gmm=update_gmm_epoch,
            gmm_lr=args.gmm_lr, N_eff=args.gmm_N_eff
        )
        
        history['train_losses'].append(avg_train_loss)
        history['train_flow_losses'].append(train_metrics.get('flow_loss', 0.0))
        history['train_recon_losses'].append(train_metrics.get('recon_loss', 0.0))
        history['train_reg_losses'].append(train_metrics.get('reg_loss', 0.0))
        history['train_vae_losses'].append(train_metrics.get('vae_loss', 0.0))
        history['train_gmm_losses'].append(train_metrics.get('gmm_loss', 0.0))
        
        # Validation
        val_losses = []
        val_metrics_list = []
        num_val_batches = (val_y.shape[0] + args.batch_size - 1) // args.batch_size
        for i in range(num_val_batches):
            start_idx = i * args.batch_size
            end_idx = min(start_idx + args.batch_size, val_y.shape[0])
            x_batch = val_x[start_idx:end_idx] if val_x is not None else None
            y_batch = val_y[start_idx:end_idx]
            
            key, step_key = jr.split(key)
            loss, metrics = model.loss(params, x_batch, y_batch, step_key, training=False)
            val_losses.append(float(loss))
            val_metrics_list.append(metrics)
        
        avg_val_loss = np.mean(val_losses)
        history['val_losses'].append(avg_val_loss)
        
        if val_metrics_list:
            history['val_flow_losses'].append(np.mean([float(m.get('flow_loss', 0.0)) for m in val_metrics_list]))
            history['val_recon_losses'].append(np.mean([float(m.get('recon_loss', 0.0)) for m in val_metrics_list]))
            history['val_reg_losses'].append(np.mean([float(m.get('reg_loss', 0.0)) for m in val_metrics_list]))
            history['val_vae_losses'].append(np.mean([float(m.get('vae_loss', 0.0)) for m in val_metrics_list]))
            history['val_gmm_losses'].append(np.mean([float(m.get('gmm_loss', 0.0)) for m in val_metrics_list]))
        else:
            history['val_flow_losses'].append(0.0)
            history['val_recon_losses'].append(0.0)
            history['val_reg_losses'].append(0.0)
            history['val_vae_losses'].append(0.0)
            history['val_gmm_losses'].append(0.0)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.num_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
    
    # Generation
    print("\nGenerating samples...")
    num_gen = min(2000, val_y.shape[0])
    key, gen_key = jr.split(key)
    
    if args.unconditional:
        # Unconditional generation
        x_gen = model.sample(
            params,
            gen_key,
            batch_shape=(num_gen,),
            num_steps=20,
            integration_method="euler",
            output_type="end_point"
        )
        x_gen_np = np.array(x_gen)
        x_real = np.array(val_y[:num_gen])
        y_labels = None
    else:
        # Conditional generation
        cond_y = val_x[:num_gen] if val_x is not None else None
        x_gen = model.predict(
            params,
            cond_y,
            num_steps=20,
            integration_method="euler",
            output_type="end_point",
            prng_key=gen_key
        )
        x_gen_np = np.array(x_gen)
        x_real = np.array(val_y[:num_gen])
        # y_labels should match x_real - use the labels corresponding to x_real
        # Since val_x and val_y are aligned, cond_y (which is val_x[:num_gen]) should match x_real
        # IMPORTANT: y_labels must be in the same order as x_real and x_gen
        y_labels = np.array(cond_y) if cond_y is not None else None
        
        # Ensure x_gen, x_real, and y_labels all have the same length and are aligned
        min_len = min(len(x_gen_np), len(x_real), len(y_labels) if y_labels is not None else len(x_real))
        x_gen_np = x_gen_np[:min_len]
        x_real = x_real[:min_len]
        if y_labels is not None:
            y_labels = y_labels[:min_len]
        
        # Debug: verify alignment
        if y_labels is not None and len(y_labels) > 0:
            print(f"  Label alignment check: x_gen shape={x_gen_np.shape}, x_real shape={x_real.shape}, y_labels shape={y_labels.shape}")
            print(f"  First 5 class indices: {np.argmax(y_labels[:5], axis=1) if len(y_labels.shape) == 2 and y_labels.shape[1] > 1 else y_labels[:5]}")
    
    print(f"  Generated {x_gen_np.shape[0]} samples")
    
    # Store generation results in history
    history['x_gen'] = x_gen_np
    history['x_real'] = x_real
    
    # Compute Chamfer Distance
    from src.utils.metrics import chamfer_distance
    chamfer_dist = chamfer_distance(jnp.array(x_gen_np), jnp.array(x_real))
    
    # Save results (includes all plots: generation, loss trends, trajectories)
    key, save_key = jr.split(key)
    save_results_fm_mix(
        model=model,
        params=params,
        history=history,
        output_dir=args.save_dir,
        x_real=x_real,
        x_gen=x_gen_np,
        y_labels=y_labels,
        key=save_key
    )
    
    # Save config
    if hasattr(config, 'save_yaml'):
        config.save_yaml(f"{args.save_dir}/config.yaml")
    
    # Save params
    import pickle
    with open(f"{args.save_dir}/params.pkl", 'wb') as f:
        pickle.dump(jax.device_get(params), f)
    
    if args.verbose:
        print(f"Final Chamfer Distance: {chamfer_dist:.6f}")
        print(f"Saved generation assets to {args.save_dir}")


if __name__ == '__main__':
    main()

