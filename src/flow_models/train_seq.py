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


def extract_patches_from_long_sequences(
    long_sequences: jnp.ndarray,
    output_seq_len: int,
    input_seq_len: int,
    seed: int = 42,
    num_patches_per_sequence: int = 1
) -> tuple:
    """
    Extract patches from long sequences for sequence-to-sequence prediction.
    
    For each long sequence, extracts multiple (x, y) patch pairs where:
    - y: a patch of length output_seq_len starting at position i
    - x: previous input_seq_len timesteps (i - input_seq_len to i)
    - [x, y] concatenated gives a contiguous segment of the original time series
    
    Args:
        long_sequences: Long sequences array (N, long_seq_len, D)
        output_seq_len: Length of output patch y (T)
        input_seq_len: Length of input patch x (T_x)
        seed: Random seed for sampling patch positions
        num_patches_per_sequence: Number of patches to extract per sequence
    
    Returns:
        x_patches: List of input patches, each of shape (input_seq_len, D)
        y_patches: List of output patches, each of shape (output_seq_len, D)
    """
    N, long_seq_len, D = long_sequences.shape
    
    # Validate that we can extract patches
    if input_seq_len + output_seq_len > long_seq_len:
        raise ValueError(
            f"Cannot extract patches: input_seq_len ({input_seq_len}) + output_seq_len ({output_seq_len}) "
            f"= {input_seq_len + output_seq_len} > long_seq_len ({long_seq_len})"
        )
    
    x_patches = []
    y_patches = []
    
    rng = jr.PRNGKey(seed)
    
    for seq_idx in range(N):
        sequence = long_sequences[seq_idx]  # (long_seq_len, D)
        
        # Sample patch positions for this sequence
        # Valid positions: [input_seq_len, long_seq_len - output_seq_len]
        min_pos = input_seq_len
        max_pos = long_seq_len - output_seq_len
        
        for _ in range(num_patches_per_sequence):
            rng, pos_rng = jr.split(rng)
            # Sample a random position in the valid range
            i = int(jr.randint(pos_rng, (), minval=min_pos, maxval=max_pos + 1))
            
            # Extract y patch: sequence[i:i+output_seq_len]
            y_patch = sequence[i:i+output_seq_len]  # (output_seq_len, D)
            
            # Extract x patch: sequence[i-input_seq_len:i]
            x_patch = sequence[i-input_seq_len:i]  # (input_seq_len, D)
            
            x_patches.append(x_patch)
            y_patches.append(y_patch)
    
    return x_patches, y_patches


def load_data(
    data_path: str,
    output_seq_len: Optional[int] = None,
    input_seq_len: Optional[int] = None,
    extract_patches: bool = False,
    num_patches_per_sequence: int = 1,
    seed: int = 42
):
    """
    Load dataset from pickle file.
    
    Expected format: dict with 'train' and 'val' keys, each containing 'x' and 'y' arrays.
    
    If extract_patches=True:
        - Expects long sequences: (N, long_seq_len, D) where long_seq_len > output_seq_len
        - Extracts patches where y is (output_seq_len, D) and x is (input_seq_len, D)
        - Returns lists of patches (one per extracted patch)
    
    If extract_patches=False:
        - Expects pre-split sequences: x and y should be 3D arrays: (N, seq_len, embed_dim)
        - Returns arrays as-is
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    if extract_patches:
        # Extract patches from long sequences
        if output_seq_len is None or input_seq_len is None:
            raise ValueError("output_seq_len and input_seq_len must be provided when extract_patches=True")
        
        # Load long sequences (could be in 'x' or 'y' key - we'll use 'y' as the main sequence)
        train_long = jnp.array(data['train']['y'])
        val_y_data = data['val'].get('y') if 'val' in data and data['val'] is not None else None
        val_long = jnp.array(val_y_data) if val_y_data is not None else None
        
        # Extract patches
        train_x_patches, train_y_patches = extract_patches_from_long_sequences(
            train_long, output_seq_len, input_seq_len, seed=seed, num_patches_per_sequence=num_patches_per_sequence
        )
        
        if val_long is not None:
            val_x_patches, val_y_patches = extract_patches_from_long_sequences(
                val_long, output_seq_len, input_seq_len, seed=seed+1, num_patches_per_sequence=num_patches_per_sequence
            )
        else:
            val_x_patches, val_y_patches = None, None
        
        return train_x_patches, train_y_patches, val_x_patches, val_y_patches
    else:
        # Load pre-split sequences
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
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data file (pickle format with train/val splits)')
    
    # Data dimensions/shapes (can override config)
    parser.add_argument('--input_shape', type=int, nargs='+', default=None,
                       help='Input shape as tuple (e.g., --input_shape 20 256 for seq_len, embed_dim)')
    parser.add_argument('--output_shape', type=int, nargs='+', default=None,
                       help='Output shape as tuple (e.g., --output_shape 10 256 for seq_len, embed_dim)')
    parser.add_argument('--latent_shape', type=int, nargs='+', default=None,
                       help='Latent shape as tuple (e.g., --latent_shape 10 256). If not provided, uses output_shape.')
    
    # Architecture arguments (can override config)
    parser.add_argument('--crn_type', type=str, default=None,
                       help='CRN type (overrides config). Use transformer_seq2seq for sequences.')
    parser.add_argument('--network_type', type=str, default=None,
                       help='Network type (overrides config). Use transformer_seq2seq for sequences.')
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
    parser.add_argument('--learning_rate', type=float, default=0.0025,
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
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save results (default: auto-generated)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--unconditional', action='store_true',
                       help='Train for unconditional generation (x=None)')
    
    # Patch extraction arguments
    parser.add_argument('--extract_patches', action='store_true',
                       help='Extract patches from long sequences. If True, expects long sequences in data and extracts (x, y) patches.')
    parser.add_argument('--num_patches_per_sequence', type=int, default=1,
                       help='Number of patches to extract per long sequence (default: 1)')
    
    args = parser.parse_args()
    
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
            print("NOTE: You may need to specify shapes via --input_shape, --output_shape, and --latent_shape")
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
        #   - input_shape, output_shape, latent_shape
        #     (these set the shapes which are "NA" in default config)
        # The encoder and decoder shapes will automatically be set from these main config values
        print("Using default Config from src.flow_models.config with default values...")
        print("NOTE: You must specify shapes via --input_shape, --output_shape, and --latent_shape")
        print("      (encoder and decoder shapes will be set automatically from main config)")
        base_config = Config()
        print(f"Using default config: {base_config.__class__.__name__}")
    
    # Override with command-line arguments
    # For sequences, we use override_from_args_regression (like regression) since we want x->y (no shape reversal)
    config = base_config.override_from_args_regression(args, args.model_type)
    
    # Set up save directory
    args.save_dir = get_save_directory(args.save_dir, 'seq', args.model_type, unconditional=args.unconditional)
    
    # Get expected sequence lengths from config
    output_shape = config.main['output_shape']
    input_shape = config.main['input_shape']
    output_seq_len = output_shape[0] if isinstance(output_shape, (tuple, list)) else output_shape
    input_seq_len = input_shape[0] if isinstance(input_shape, (tuple, list)) else input_shape
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    if args.extract_patches:
        print(f"Extracting patches: output_seq_len={output_seq_len}, input_seq_len={input_seq_len}")
        x_train, y_train, x_val, y_val = load_data(
            args.data_path,
            output_seq_len=output_seq_len,
            input_seq_len=input_seq_len,
            extract_patches=True,
            num_patches_per_sequence=args.num_patches_per_sequence,
            seed=args.seed
        )
        # x_train, y_train, etc. are now lists of patches
        print(f"Extracted patches:")
        print(f"  Train: {len(y_train)} y patches, {len(x_train) if x_train is not None else 0} x patches")
        if y_val is not None:
            print(f"  Val: {len(y_val)} y patches, {len(x_val) if x_val is not None else 0} x patches")
        if len(y_train) > 0:
            print(f"  Sample patch shapes: y={y_train[0].shape}, x={x_train[0].shape if x_train is not None else None}")
    else:
        x_train, y_train, x_val, y_val = load_data(args.data_path)
        print(f"Data shapes:")
        print(f"  Train: x={x_train.shape if x_train is not None else None}, y={y_train.shape}")
        if x_val is not None:
            print(f"  Val: x={x_val.shape}, y={y_val.shape}")
        else:
            print(f"  Val: x=None, y={y_val.shape if y_val is not None else None}")
    
    # For conditional generation: inputs=x (conditional), targets=y (output sequence)
    # If we extracted patches, they're already lists; otherwise convert arrays to lists
    if args.extract_patches:
        train_x, train_y = x_train, y_train  # Already lists
        val_x, val_y = (x_val, y_val) if y_val is not None else (None, y_val)
    else:
        train_x, train_y = x_train, y_train
        val_x, val_y = (x_val, y_val) if x_val is not None else (None, y_val)
    
    # Calculate warmup_steps
    if args.warmup_epochs is not None:
        # Calculate number of batches per epoch
        num_samples = len(train_y) if isinstance(train_y, list) else train_y.shape[0]
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
    print("Initializing model...")
    # Use a single sample for initialization (model will add batch dimension internally)
    if args.unconditional:
        x_sample = None
    else:
        # Handle both list and array cases
        if isinstance(train_x, list):
            x_sample = train_x[0] if train_x is not None and len(train_x) > 0 else None
        else:
            x_sample = train_x[0] if train_x is not None else None
    if isinstance(train_y, list):
        y_sample = train_y[0]
    else:
        y_sample = train_y[0]
    latent_shape = config.main['latent_shape']
    if isinstance(latent_shape, str) or len(latent_shape) < 2:
        raise ValueError("For sequences, latent_shape must be (seq_len, embed_dim). Please specify via --latent_shape")
    # For sequences, latent_shape is (seq_len, embed_dim)
    # Create single samples for z and t
    z_sample = jr.normal(jr.PRNGKey(args.seed), (1, latent_shape[0], latent_shape[1]))
    t_sample = jr.uniform(jr.PRNGKey(args.seed+1), (1,), minval=0.0, maxval=1.0)
    trainer.initialize(x_sample, y_sample, z_sample, t_sample)
    
    # Train
    dropout_epochs = args.dropout_epochs if args.dropout_epochs is not None else args.num_epochs
    
    # Convert to lists of sequences for the trainer
    # If already lists (from patch extraction), use as-is; otherwise convert arrays to lists
    if isinstance(train_y, list):
        train_x_sequences = train_x  # Already a list
        train_y_sequences = train_y  # Already a list
        val_x_sequences = val_x if val_x is not None else None  # Already a list or None
        val_y_sequences = val_y if val_y is not None else None  # Already a list or None
    else:
        # Convert JAX arrays to lists of sequences for the trainer
        # Each row becomes a sequence in the list
        train_x_sequences = [train_x[i] for i in range(train_x.shape[0])] if train_x is not None else None
        train_y_sequences = [train_y[i] for i in range(train_y.shape[0])]
        val_x_sequences = [val_x[i] for i in range(val_x.shape[0])] if val_x is not None else None
        val_y_sequences = [val_y[i] for i in range(val_y.shape[0])] if val_y is not None else None
    
    validation_data = (val_x_sequences, val_y_sequences) if val_y_sequences is not None else None
    
    print(f"Starting training for {args.num_epochs} epochs...")
    mask_ratio = 0.0  # Disable masking (set to 0.0)
    print(f"Using masked training protocol (mask_ratio={mask_ratio}, min_visible_len=1)")
    
    import time
    training_start_time = time.time()
    history = trainer.train(
        x_sequences=train_x_sequences,
        y_sequences=train_y_sequences,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=validation_data,
        dropout_epochs=dropout_epochs,
        mask_ratio=mask_ratio,  # Mask ratio (0.0 = no masking)
        min_visible_len=1  # Keep at least 1 timestep visible
    )
    
    training_end_time = time.time()
    training_elapsed = training_end_time - training_start_time
    hours = int(training_elapsed // 3600)
    minutes = int((training_elapsed % 3600) // 60)
    seconds = int(training_elapsed % 60)
    print(f"\n=== TRAINING TIME ===")
    print(f"Total training time: {hours}h {minutes}m {seconds}s ({training_elapsed:.2f} seconds)")
    
    # Save results
    save_training_artifacts(args.save_dir, history, trainer, config)
    
    # Generation
    num_samples = len(train_y_sequences) if isinstance(train_y_sequences, list) else train_y_sequences.shape[0]
    num_val_samples = len(val_y_sequences) if val_y_sequences is not None and isinstance(val_y_sequences, list) else (val_y.shape[0] if val_y is not None else 0)
    num_gen = min(2000, num_val_samples if num_val_samples > 0 else num_samples)
    prng = jr.PRNGKey(args.seed + 123)
    
    if args.unconditional:
        # Unconditional generation
        y_gen = np.array(trainer.unconditional_generate(
            batch_shape=(num_gen,),
            num_steps=20,
            prng_key=prng
        ))
        # Get real y samples (prefer val if available)
        if isinstance(val_y_sequences, list) and len(val_y_sequences) > 0:
            y_real = np.array([val_y_sequences[i] for i in range(min(num_gen, len(val_y_sequences)))])
        elif isinstance(train_y_sequences, list):
            y_real = np.array([train_y_sequences[i] for i in range(min(num_gen, len(train_y_sequences)))])
        else:
            y_real = np.array((val_y if val_y is not None else train_y)[:num_gen])
        x_labels = None
        cond_x = None
    else:
        # Conditional generation (prefer val if available)
        if isinstance(val_x_sequences, list) and len(val_x_sequences) > 0:
            cond_x = jnp.array([val_x_sequences[i] for i in range(min(num_gen, len(val_x_sequences)))])
            y_real = np.array([val_y_sequences[i] for i in range(min(num_gen, len(val_y_sequences)))])
        elif isinstance(train_x_sequences, list):
            cond_x = jnp.array([train_x_sequences[i] for i in range(min(num_gen, len(train_x_sequences)))])
            y_real = np.array([train_y_sequences[i] for i in range(min(num_gen, len(train_y_sequences)))])
        else:
            cond_x = (val_x if val_x is not None else train_x)[:num_gen]
            y_real = np.array((val_y if val_y is not None else train_y)[:num_gen])
        y_gen = np.array(trainer.conditional_generate(cond_x, num_steps=20, prng_key=prng))
        x_labels = np.array(cond_x)
    
    # Compute sequence metrics on generated samples
    seq_metrics = trainer.compute_sequence_metrics(jnp.array(y_gen), jnp.array(y_real))
    
    # Save results (includes all plots: generation, loss trends, trajectories)
    trainer.save_results(history, args.save_dir, y_real=y_real, y_gen=y_gen, x_labels=x_labels)
    
    if args.verbose:
        print(f"Final Sequence Metrics: {seq_metrics}")
        if history.get('val_seq_metrics') and len(history['val_seq_metrics']) > 0:
            print(f"Final Validation Sequence Metrics: {history['val_seq_metrics'][-1]}")
        print(f"Saved generation assets to {args.save_dir}")
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()

