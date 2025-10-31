#!/usr/bin/env python3
"""
Training script for conditional generation (x | y) on Two Moons.

This script trains the selected model with reversed mapping (inputs=y, targets=x)
and evaluates conditional generation by sampling with PRNGKey.
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

from src.flow_models.trainer_gen import GenerationTrainer
from src.flow_models.fm import VAEFlowConfig as FMConfig
from src.flow_models.df import VAEFlowConfig as DFConfig
from src.flow_models.ct import VAEFlowConfig as CTConfig


def load_two_moons_data(data_path: str = "data/two_moons_formatted.pkl"):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    x_train = jnp.array(data['train']['x'])
    y_train = jnp.array(data['train']['y'])
    x_val = jnp.array(data['val']['x'])
    y_val = jnp.array(data['val']['y'])
    # labels {-1,1} already
    return x_train, y_train, x_val, y_val


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
                 noise_schedule: str,
                 noise_schedule_learnable: bool = False,
                 use_snr_weight: bool = None):
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
        'use_snr_weight': use_snr_weight,
        'integration_method': 'midpoint' if model in ('ct', 'diffusion') else 'euler',
        'sigma': 0.02,
        'noise_schedule': noise_schedule,  # Legacy support
    })

    crn = FrozenDict({
        'model_type': crn_type,
        'network_type': network_type,
        'hidden_dims': tuple(hidden_dims),
        'time_embed_dim': 64,
        'time_embed_method': 'sinusoidal',
        'activation_fn': 'swish',
        'use_batch_norm': False,
        'dropout_rate': 0.1,
    })
    enc = FrozenDict({
        'model_type': 'identity',
        'encoder_type': 'deterministic',
        'input_shape': input_shape,
        'latent_shape': latent_shape,
        'hidden_dims': (8,),
        'activation': 'swish',
        'dropout_rate': 0.0,
    })
    dec = FrozenDict({
        'model_type': 'identity',
        'decoder_type': 'none',
        'latent_shape': latent_shape,
        'output_shape': output_shape,
        'hidden_dims': (16, 16),
        'activation': 'swish',
        'dropout_rate': 0.0,
    })

    if model == 'diffusion':
        # Add noise schedule config for diffusion model
        noise_schedule_config = FrozenDict({
            'schedule_type': noise_schedule,
            'learnable': noise_schedule_learnable,
        })
        return DFConfig(main=main, noise_schedule=noise_schedule_config, crn=crn, encoder=enc, decoder=dec)
    if model == 'ct':
        # Add noise schedule config for CT model
        noise_schedule_config = FrozenDict({
            'schedule_type': noise_schedule,
            'learnable': noise_schedule_learnable,
        })
        return CTConfig(main=main, noise_schedule=noise_schedule_config, crn=crn, encoder=enc, decoder=dec)
    # Flow matching doesn't use noise schedule
    return FMConfig(main=main, crn=crn, encoder=enc, decoder=dec)


def main():
    parser = argparse.ArgumentParser(description='Conditional generation training on Two Moons (x | y)')
    parser.add_argument('--model_type', type=str, default='flow_matching', choices=['flow_matching', 'diffusion', 'ct'])
    parser.add_argument('--data_path', type=str, default='data/two_moons_formatted.pkl')
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--crn_type', type=str, default='vanilla')
    parser.add_argument('--network_type', type=str, default='mlp')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 32, 32, 32, 32, 32])
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--recon_loss_type', type=str, default='mse', choices=['mse', 'cross_entropy', 'none'])
    parser.add_argument('--use_snr_weight', action='store_const', const=True, default=True,
                        help='Apply SNR weighting to reconstruction loss (default: True)')
    parser.add_argument('--no_snr_weight', dest='use_snr_weight', action='store_const', const=False,
                        help='Disable SNR weighting for reconstruction loss')
    parser.add_argument('--reg_weight', type=float, default=0.0)
    parser.add_argument('--noise_schedule', type=str, default='linear',
                        choices=['linear', 'cosine', 'sigmoid', 'exponential', 'cauchy', 'laplace', 'logistic', 'quadratic', 'polynomial', 'monotonic_nn', 'learnable', 'network'],
                        help='Noise schedule for CT and diffusion models')
    parser.add_argument('--noise_schedule_learnable', action='store_const', const=True, default=False,
                        help='Make noise schedule parameters learnable (default: False)')
    parser.add_argument('--noise_schedule_fixed', dest='noise_schedule_learnable', action='store_const', const=False,
                        help='Freeze noise schedule parameters (default: False)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--unconditional', action='store_true', help='Train for unconditional generation (x=None)')

    args = parser.parse_args()

    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = f"artifacts/two_moons_{timestamp}_gen/{args.model_type}"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    x_train, y_train, x_val, y_val = load_two_moons_data(args.data_path)

    # Reverse mapping for generation: inputs=y, targets=x
    train_x, train_y = y_train, x_train
    val_x, val_y = y_val, x_val

    # Build config with reversed shapes
    # For unconditional generation, set input_shape to empty tuple since x is None
    input_shape = () if args.unconditional else (args.output_dim,)
    config = build_config(
        model=args.model_type,
        input_shape=input_shape,   # empty for unconditional, otherwise label y (dim=2 here)
        output_shape=(args.input_dim,),  # output is x (2D coordinates)
        latent_shape=(args.latent_dim,),
        crn_type=args.crn_type,
        network_type=args.network_type,
        hidden_dims=args.hidden_dims,
        recon_loss_type=args.recon_loss_type,
        reg_weight=args.reg_weight,
        recon_weight=args.recon_weight,
        noise_schedule=args.noise_schedule,
        noise_schedule_learnable=args.noise_schedule_learnable,
        use_snr_weight=args.use_snr_weight,
    )

    trainer = GenerationTrainer(
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
    z_sample = jr.normal(jr.PRNGKey(args.seed), (bs, args.latent_dim))
    t_sample = jr.uniform(jr.PRNGKey(args.seed+1), (bs,), minval=0.0, maxval=1.0)
    trainer.initialize(x_sample, y_sample, z_sample, t_sample)

    # Train - for unconditional, pass None for x_data
    train_x_input = None if args.unconditional else train_x
    val_x_input = None if args.unconditional else val_x
    
    history = trainer.train(
        x_data=train_x_input,
        y_data=train_y,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=(val_x_input, val_y),
        dropout_epochs=0,
        verbose=args.verbose,
    )

    # Save minimal training history and params
    with open(Path(args.save_dir) / 'training_results.pkl', 'wb') as f:
        pickle.dump(history, f)
    trainer.save_params(str(Path(args.save_dir) / 'model_params.pkl'))

    # Generation
    num_gen = min(2000, val_y.shape[0])
    prng = jr.PRNGKey(args.seed + 123)
    
    if args.unconditional:
        # Unconditional generation
        x_gen = np.array(trainer.unconditional_generate(
            batch_shape=(num_gen,),
            num_steps=20,
            prng_key=prng
        ))
        x_real = np.array(val_y[:num_gen])
        y_labels = None
        cond_y = None
    else:
        # Conditional generation
        cond_y = val_x[:num_gen]  # remember, reversed: cond is labels
        x_gen = np.array(trainer.conditional_generate(cond_y, num_steps=20, prng_key=prng))
        x_real = np.array(val_y[:num_gen])
        y_labels = np.array(cond_y)

    # Plot
    trainer.save_generation_plot(x_real=x_real, y_labels=y_labels, x_gen=x_gen, output_dir=args.save_dir)
    trainer.save_loss_trends_plot(history, output_dir=args.save_dir)
    
    # Generate trajectory plot with 40 trajectories
    trajectory_prng = jr.PRNGKey(args.seed + 456)
    trainer.save_trajectory_plot(
        cond_y=cond_y,
        num_trajectories=40,
        num_steps=20,
        prng_key=trajectory_prng,
        output_dir=args.save_dir
    )

    if args.verbose:
        print(f"Saved generation assets to {args.save_dir}")
        print(f"Saved loss trends plot to {args.save_dir}/loss_trends.png")
        print(f"Saved trajectory plot to {args.save_dir}/latent_trajectories.png")


if __name__ == '__main__':
    main()


