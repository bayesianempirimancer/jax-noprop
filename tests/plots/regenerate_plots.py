#!/usr/bin/env python3
"""
Regenerate plots from saved model with updated plotting code.
"""

import pickle
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from src.flow_models.trainer_seq import SequenceTrainer
from src.flow_models.fm import VAEFlowConfig as FMConfig
from flax.core import FrozenDict
import jax.random as jr

def build_config(input_shape, output_shape, latent_shape, hidden_dims,
                 num_layers: int = 4, num_heads: int = 4, mlp_ratio: float = 4.0):
    main = FrozenDict({
        'input_shape': input_shape,
        'output_shape': output_shape,
        'latent_shape': latent_shape,
        'recon_loss_type': 'mse',
        'recon_weight': 1.0,
        'reg_weight': 0.0,
        'use_snr_weight': False,
        'integration_method': 'euler',
        'sigma': 0.02,
        'noise_schedule': 'exponential',
    })

    crn = FrozenDict({
        'model_type': 'vanilla',
        'network_type': 'transformer_seq2seq',
        'hidden_dims': tuple(hidden_dims),
        'time_embed_dim': 32,
        'time_embed_method': 'sinusoidal',
        'activation_fn': 'swish',
        'use_batch_norm': False,
        'dropout_rate': 0.0,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'qkv_bias': True,
    })
    
    encoder_model_type = 'identity'
    decoder_model_type = 'identity'
    decoder_output_type = 'none'
    
    enc = FrozenDict({
        'model_type': encoder_model_type,
        'encoder_type': 'deterministic',
        'input_shape': input_shape,
        'latent_shape': latent_shape,
        'hidden_dims': (16, 16),
        'activation': 'swish',
        'dropout_rate': 0.0,
    })
    dec = FrozenDict({
        'model_type': decoder_model_type,
        'decoder_type': decoder_output_type,
        'latent_shape': latent_shape,
        'output_shape': output_shape,
        'hidden_dims': (32, 16),
        'activation': 'swish',
        'dropout_rate': 0.0,
    })

    noise_schedule_config = FrozenDict({
        'schedule_type': 'exponential',
        'learnable': False,
    })
    
    return FMConfig(main=main, noise_schedule=noise_schedule_config, crn=crn, encoder=enc, decoder=dec)


def main():
    # Load saved model params
    save_dir = Path('artifacts/stock_sequences_200epochs')
    model_params_path = save_dir / 'model_params.pkl'
    history_path = save_dir / 'training_history.pkl'
    data_path = 'data/stock_sequences_projected.pkl'
    
    print("Loading saved model and data...")
    
    # Load model params
    with open(model_params_path, 'rb') as f:
        saved_params = pickle.load(f)
    
    # Load training history
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    x_val = jnp.array(data['val']['x'])
    y_val = jnp.array(data['val']['y'])
    
    print(f"Loaded validation data: x_val={x_val.shape}, y_val={y_val.shape}")
    
    # Get sequence dimensions
    x_seq_len = x_val.shape[1]
    y_seq_len = y_val.shape[1]
    embed_dim = x_val.shape[2]
    
    # Build config
    input_shape = (x_seq_len, embed_dim)
    output_shape = (y_seq_len, embed_dim)
    latent_shape = (y_seq_len, embed_dim)
    
    config = build_config(
        input_shape=input_shape,
        output_shape=output_shape,
        latent_shape=latent_shape,
        hidden_dims=[embed_dim],
        num_layers=4,
        num_heads=4,
        mlp_ratio=4.0,
    )
    
    # Create trainer
    trainer = SequenceTrainer(
        config=config,
        learning_rate=1e-3,
        optimizer_name='adam',
        seed=42,
        unconditional=False
    )
    
    # Restore params (we need to initialize first, then replace params)
    # Initialize with dummy data
    key = jr.PRNGKey(42)
    bs = min(32, y_val.shape[0])
    x_sample = x_val[:bs]
    y_sample = y_val[:bs]
    z_sample = jr.normal(jr.PRNGKey(42), (bs, y_seq_len, embed_dim))
    t_sample = jr.uniform(jr.PRNGKey(43), (bs,), minval=0.0, maxval=1.0)
    
    print("\nInitializing trainer to restore params...")
    trainer.initialize(x_sample, y_sample, z_sample, t_sample)
    trainer.params = saved_params  # Restore saved params
    
    # Generate samples
    print("\nGenerating samples for plotting...")
    num_gen = min(100, y_val.shape[0])
    cond_x = x_val[:num_gen]
    key, gen_key = jr.split(key)
    y_gen = np.array(trainer.conditional_generate(cond_x, num_steps=20))
    y_real = np.array(y_val[:num_gen])
    
    print(f"Generated sequences: {y_gen.shape}")
    
    # Regenerate plots
    print("\nRegenerating plots...")
    
    # 1. Loss trends plot
    trainer.save_loss_trends_plot(history, output_dir=str(save_dir))
    print("✓ Loss trends plot saved")
    
    # 2. Sequence comparison plots
    trainer.save_sequence_plot(
        y_real=y_real,
        x_labels=np.array(cond_x),
        y_gen=y_gen,
        output_dir=str(save_dir),
        data_path=data_path
    )
    print("✓ Sequence comparison plots saved")
    
    # 3. Price comparison plot (10:30 AM - 2:30 PM)
    trainer.save_price_comparison_plot(
        y_real=y_real,
        y_pred=y_gen,
        data_path=data_path,
        output_dir=str(save_dir),
        num_samples=8,
        start_time="10:30",
        end_time="14:30"
    )
    print("✓ Price comparison plot saved")
    
    print(f"\n✓ All plots regenerated and saved to {save_dir}")


if __name__ == '__main__':
    main()

