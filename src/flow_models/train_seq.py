#!/usr/bin/env python3
"""
Training script for conditional generation on sequence data (x | y).

This script trains the selected model (FM, CT, or DF) with sequence data
where both inputs and outputs are sequences.
"""

import argparse
from datetime import datetime
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax.core import FrozenDict

from src.flow_models.trainer_seq import SequenceTrainer
from src.flow_models.fm import VAEFlowConfig as FMConfig
from src.flow_models.df import VAEFlowConfig as DFConfig
from src.flow_models.ct import VAEFlowConfig as CTConfig


def build_config(model: str,
                 input_shape,
                 output_shape,
                 latent_shape,
                 crn_type: str,
                 network_type: str,
                 hidden_dims,
                 recon_loss_type: str,
                 reg_weight: float,
                 recon_weight: float,
                 vae_weight: float,
                 noise_schedule: str,
                 noise_schedule_learnable: bool = False,
                 use_snr_weight: bool = None,
                 encoder_type: str = None,
                 decoder_type: str = None,
                 decoder_model_type: str = None,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0):
    # Default use_snr_weight based on model type
    if use_snr_weight is None:
        # Default to False for flow_matching, True for others
        use_snr_weight = False if model == 'flow_matching' else True
    main = FrozenDict({
        'input_shape': input_shape,
        'output_shape': output_shape,
        'latent_shape': latent_shape,
        'recon_loss_type': recon_loss_type,
        'recon_weight': recon_weight,
        'reg_weight': reg_weight,
        'vae_weight': vae_weight,
        'use_snr_weight': use_snr_weight,
        'integration_method': 'midpoint' if model in ('ct', 'diffusion') else 'euler',
        'noise_schedule': noise_schedule,  # Legacy support
    })

    crn = FrozenDict({
        'model_type': crn_type,
        'network_type': network_type,
        'hidden_dims': tuple(hidden_dims),
        'time_embed_dim': 32,
        'time_embed_method': 'sinusoidal',
        'activation_fn': 'swish',
        'use_batch_norm': False,
        'dropout_rate': 0.1,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'qkv_bias': True,
    })
    # Determine encoder/decoder model types
    # Default to MLP for encoder and decoder
    latent_dim = latent_shape[-1] if len(latent_shape) >= 2 else latent_shape[0] if len(latent_shape) > 0 else 256
    encoder_model_type = encoder_type if encoder_type is not None else 'mlp'
    decoder_model_type = decoder_model_type if decoder_model_type is not None else 'mlp'
    decoder_output_type = decoder_type if decoder_type is not None else 'none'
    
    enc = FrozenDict({
        'model_type': encoder_model_type,
        'encoder_type': 'deterministic',
        'input_shape': input_shape,
        'latent_shape': latent_shape,
        'hidden_dims': (32,32),
        'activation': 'swish',
        'dropout_rate': 0.1,
    })
    dec = FrozenDict({
        'model_type': decoder_model_type,
        'decoder_type': decoder_output_type,
        'latent_shape': latent_shape,
        'output_shape': output_shape,
        'hidden_dims': (32, 32),
        'activation': 'swish',
        'dropout_rate': 0.1,
    })

    # Create noise schedule config for all models (default: linear, not learnable)
    noise_schedule_config = FrozenDict({
        'schedule_type': noise_schedule,
        'learnable': noise_schedule_learnable,
    })
    
    if model == 'diffusion':
        return DFConfig(main=main, noise_schedule=noise_schedule_config, crn=crn, encoder=enc, decoder=dec)
    if model == 'ct':
        return CTConfig(main=main, noise_schedule=noise_schedule_config, crn=crn, encoder=enc, decoder=dec)
    # Flow matching also uses noise schedule (default: linear)
    return FMConfig(main=main, noise_schedule=noise_schedule_config, crn=crn, encoder=enc, decoder=dec)


def main():
    parser = argparse.ArgumentParser(description='Conditional generation training on sequence data (x | y)')
    parser.add_argument('--model_type', type=str, default='flow_matching', choices=['flow_matching', 'diffusion', 'ct'])
    parser.add_argument('--data_path', type=str, default=None, help='Path to sequence data pickle file')
    parser.add_argument('--z_seq_len', type=int, default=10, help='Sequence length for z (latent)')
    parser.add_argument('--x_seq_len', type=int, default=20, help='Sequence length for x (conditional input, can be variable)')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension for sequences')
    parser.add_argument('--crn_type', type=str, default='transformer_seq2seq', help='CRN type (use transformer_seq2seq for sequences)')
    parser.add_argument('--network_type', type=str, default='transformer_seq2seq', help='Network type for CRN')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256], help='Hidden dimensions (first is used as embed_dim if not in shape)')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio for transformer')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--dropout_epochs', type=int, default=None,
                        help='Number of epochs to use dropout. If None, defaults to num_epochs (dropout for all epochs)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--recon_loss_type', type=str, default='mse', choices=['mse', 'cross_entropy', 'none'])
    parser.add_argument('--use_snr_weight', action='store_const', const=True, default=True,
                        help='Apply SNR weighting to reconstruction loss (default: True)')
    parser.add_argument('--no_snr_weight', dest='use_snr_weight', action='store_const', const=False,
                        help='Disable SNR weighting for reconstruction loss')
    parser.add_argument('--reg_weight', type=float, default=0.0)
    parser.add_argument('--vae_weight', type=float, default=1.0)
    parser.add_argument('--noise_schedule', type=str, default='exponential',
                        choices=['linear', 'cosine', 'sigmoid', 'exponential', 'cauchy', 'laplace', 'logistic', 'quadratic', 'polynomial', 'monotonic_nn', 'learnable', 'network'],
                        help='Noise schedule for CT and diffusion models')
    parser.add_argument('--noise_schedule_learnable', action='store_const', const=True, default=False,
                        help='Make noise schedule parameters learnable (default: False)')
    parser.add_argument('--noise_schedule_fixed', dest='noise_schedule_learnable', action='store_const', const=False,
                        help='Freeze noise schedule parameters (default: False)')
    parser.add_argument('--encoder_model_type', type=str, default='mlp',
                        choices=['mlp', 'mlp_normal', 'resnet', 'resnet_normal', 'identity', 'linear'],
                        help='Encoder model type. If None, determined automatically based on embed_dim.')
    parser.add_argument('--decoder_model_type', type=str, default='mlp',
                        choices=['mlp', 'resnet', 'identity'],
                        help='Decoder model type. If None, determined automatically based on embed_dim.')
    parser.add_argument('--decoder_type', type=str, default=None,
                        choices=['linear', 'softmax', 'none'],
                        help='Decoder output type (linear transformation, softmax, or none). If None, determined automatically based on embed_dim.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='artifacts/sequence_training')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--unconditional', action='store_true', help='Train for unconditional generation (x=None)')

    args = parser.parse_args()

    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = f"artifacts/sequence_{timestamp}/{args.model_type}"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load or generate synthetic sequence data
    if args.data_path is not None:
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)
        x_train = jnp.array(data['train']['x'])
        y_train = jnp.array(data['train']['y'])
        x_val = jnp.array(data['val']['x']) if 'val' in data else None
        y_val = jnp.array(data['val']['y']) if 'val' in data else None
    else:
        # Generate synthetic sequence data for testing
        print("No data_path provided, generating synthetic sequence data...")
        key = jr.PRNGKey(args.seed)
        key, x_key, y_key = jr.split(key, 3)
        
        # Generate synthetic sequences
        n_train = 1000
        n_val = 200
        x_train = jr.normal(x_key, (n_train, args.x_seq_len, args.embed_dim))
        y_train = jr.normal(y_key, (n_train, args.z_seq_len, args.embed_dim))
        x_val = jr.normal(jr.PRNGKey(args.seed + 1), (n_val, args.x_seq_len, args.embed_dim))
        y_val = jr.normal(jr.PRNGKey(args.seed + 2), (n_val, args.z_seq_len, args.embed_dim))

    # For conditional generation: inputs=x (conditional), targets=y (output sequence)
    train_x, train_y = x_train, y_train
    val_x, val_y = (x_val, y_val) if x_val is not None else (None, None)

    # Build config with sequence shapes
    # For unconditional generation, set input_shape to empty tuple since x is None
    input_shape = () if args.unconditional else (args.x_seq_len, args.embed_dim)
    config = build_config(
        model=args.model_type,
        input_shape=input_shape,   # empty for unconditional, otherwise (x_seq_len, embed_dim)
        output_shape=(args.z_seq_len, args.embed_dim),  # output is y sequence
        latent_shape=(args.z_seq_len, args.embed_dim),  # latent z is same shape as output
        crn_type=args.crn_type,
        network_type=args.network_type,
        hidden_dims=args.hidden_dims,
        recon_loss_type=args.recon_loss_type,
        reg_weight=args.reg_weight,
        recon_weight=args.recon_weight,
        vae_weight=args.vae_weight,
        noise_schedule=args.noise_schedule,
        noise_schedule_learnable=args.noise_schedule_learnable,
        use_snr_weight=args.use_snr_weight,
        encoder_type=args.encoder_model_type,
        decoder_type=args.decoder_type,
        decoder_model_type=args.decoder_model_type,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )

    trainer = SequenceTrainer(
        config=config,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        seed=args.seed,
        unconditional=args.unconditional
    )

    # Initialize
    bs = min(args.batch_size, train_y.shape[0])
    if args.unconditional:
        # For unconditional, pass None for x since CRN input_shape is empty
        x_sample = None
    else:
        x_sample = train_x[:bs]
    y_sample = train_y[:bs]
    z_sample = jr.normal(jr.PRNGKey(args.seed), (bs, args.z_seq_len, args.embed_dim))
    t_sample = jr.uniform(jr.PRNGKey(args.seed+1), (bs,), minval=0.0, maxval=1.0)
    trainer.initialize(x_sample, y_sample, z_sample, t_sample)

    # Train - for unconditional, pass None for x_data
    train_x_input = None if args.unconditional else train_x
    val_x_input = None if args.unconditional else (val_x if val_x is not None else None)
    
    # Set dropout_epochs: if None, use num_epochs (all epochs), otherwise use specified value
    dropout_epochs = args.dropout_epochs if args.dropout_epochs is not None else args.num_epochs
    
    validation_data = (val_x_input, val_y) if val_y is not None else None
    
    history = trainer.train(
        x_data=train_x_input,
        y_data=train_y,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=validation_data,
        dropout_epochs=dropout_epochs,
        verbose=args.verbose,
    )

    # Save minimal training history and params
    with open(Path(args.save_dir) / 'training_results.pkl', 'wb') as f:
        pickle.dump(history, f)
    trainer.save_params(str(Path(args.save_dir) / 'model_params.pkl'))

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
    
    # Generate trajectory plot with 40 trajectories
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
        print(f"Saved loss trends plot to {args.save_dir}/loss_trends.png")
        print(f"Saved trajectory plot to {args.save_dir}/latent_trajectories.png")


if __name__ == '__main__':
    main()

