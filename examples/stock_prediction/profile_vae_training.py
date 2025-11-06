#!/usr/bin/env python3
"""
Profile VAE training to identify performance bottlenecks.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import jax.random as jr
import time
import pickle
from flax.core import FrozenDict

from src.models.vae.vae import VAE, VAEConfig
from src.models.vae.trainer import VAETrainer

# Enable JAX profiling
jax.config.update("jax_traceback_filtering", "off")

def main():
    print("Loading data...")
    with open('data/stock_sequences_full_day_2d.pkl', 'rb') as f:
        data = pickle.load(f)
    
    train_sequences = data['train']['sequences']
    y_seq_len = data.get('metadata', {}).get('y_seq_len', 12)
    
    # Extract sequences
    train_hour_sequences = []
    for seq in train_sequences[:100]:  # Just use 100 sequences for profiling
        for i in range(len(seq) - y_seq_len + 1):
            train_hour_sequences.append(seq[i:i+y_seq_len])
            break  # Just one per sequence
    
    train_data = jnp.array(train_hour_sequences)
    print(f"Data shape: {train_data.shape}")
    
    # Create config
    config = VAEConfig(
        main=FrozenDict({
            "input_shape": (12, 2),
            "latent_shape": (6,),
            "output_shape": (12, 2),
            "recon_loss_type": "mse",
            "recon_weight": 1.0,
            "kl_weight": 1.0,
        }),
        encoder=FrozenDict({
            "model_type": "mlp_normal",
            "encoder_type": "normal",
            "input_shape": (12, 2),
            "latent_shape": (6,),
            "hidden_dims": (64, 32),
            "activation": "swish",
            "dropout_rate": 0.1,
        }),
        decoder=FrozenDict({
            "model_type": "mlp",
            "decoder_type": "none",
            "latent_shape": (6,),
            "output_shape": (12, 2),
            "hidden_dims": (32, 64),
            "activation": "swish",
            "dropout_rate": 0.1,
        }),
    )
    
    trainer = VAETrainer(config=config, learning_rate=1e-3, seed=42)
    
    print("\n=== Profiling Initialization ===")
    start = time.time()
    trainer.initialize(train_data[:64])
    init_time = time.time() - start
    print(f"Initialization time: {init_time:.4f}s")
    
    print("\n=== Profiling Single Training Step ===")
    x_batch = train_data[:64]
    key = jr.PRNGKey(42)
    
    # Warm up
    print("Warming up (first call compiles)...")
    start = time.time()
    trainer.params, trainer.opt_state, loss, metrics = trainer.train_step(
        trainer.params, x_batch, trainer.opt_state, key, training=True
    )
    first_time = time.time() - start
    print(f"First step (compilation): {first_time:.4f}s")
    
    # Profile subsequent steps
    print("Profiling 10 steps...")
    times = []
    for i in range(10):
        trainer.rng, step_key = jr.split(trainer.rng)
        start = time.time()
        trainer.params, trainer.opt_state, loss, metrics = trainer.train_step(
            trainer.params, x_batch, trainer.opt_state, step_key, training=True
        )
        step_time = time.time() - start
        times.append(step_time)
        if i < 3:
            print(f"  Step {i+1}: {step_time:.4f}s")
    
    print(f"Average step time: {sum(times)/len(times):.4f}s")
    print(f"Min: {min(times):.4f}s, Max: {max(times):.4f}s")
    
    print("\n=== Profiling Loss Function ===")
    # Profile loss function directly
    start = time.time()
    loss, metrics = trainer.model.loss(trainer.params, x_batch, key, training=True)
    loss_time = time.time() - start
    print(f"Loss function time: {loss_time:.4f}s")
    
    print("\n=== Profiling Encode/Decode ===")
    start = time.time()
    mu, logvar = trainer.model.apply(trainer.params, x_batch, method='encode', training=True, rngs={'dropout': key})
    encode_time = time.time() - start
    print(f"Encode time: {encode_time:.4f}s")
    
    std = jnp.exp(0.5 * logvar)
    z = mu + std * jr.normal(jr.PRNGKey(43), mu.shape)
    
    start = time.time()
    x_recon = trainer.model.apply(trainer.params, z, method='decode', training=True, rngs={'dropout': key})
    decode_time = time.time() - start
    print(f"Decode time: {decode_time:.4f}s")
    
    print("\n=== Profiling Full Epoch ===")
    # Warm up epoch
    print("Warming up epoch (first epoch may compile)...")
    start = time.time()
    epoch_metrics = trainer.train_epoch(train_data, batch_size=64, use_dropout=True)
    first_epoch_time = time.time() - start
    print(f"First epoch time: {first_epoch_time:.4f}s")
    
    # Profile subsequent epochs
    print("Profiling 3 more epochs...")
    epoch_times = []
    for i in range(3):
        start = time.time()
        epoch_metrics = trainer.train_epoch(train_data, batch_size=64, use_dropout=True)
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        print(f"  Epoch {i+2}: {epoch_time:.4f}s")
    
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    num_batches = len(train_data) // 64
    print(f"\nAverage epoch time: {avg_epoch_time:.4f}s")
    print(f"Number of batches per epoch: {num_batches}")
    print(f"Average time per batch: {avg_epoch_time/num_batches:.4f}s")
    
    print("\n=== Profiling Complete ===")

if __name__ == "__main__":
    main()

