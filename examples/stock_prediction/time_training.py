#!/usr/bin/env python3
"""
Time different parts of VAE training to identify bottlenecks.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import time
import pickle
from flax.core import FrozenDict

from src.models.vae.vae import VAEConfig
from src.models.vae.trainer import VAETrainer

# Disable tqdm for cleaner output
import tqdm
tqdm.tqdm.__init__ = lambda self, *args, **kwargs: None

def main():
    print("Loading data...")
    start = time.time()
    with open('data/stock_sequences_full_day_2d.pkl', 'rb') as f:
        data = pickle.load(f)
    load_time = time.time() - start
    print(f"Data loading: {load_time:.4f}s")
    
    train_sequences = data['train']['sequences']
    y_seq_len = data.get('metadata', {}).get('y_seq_len', 12)
    
    print("\nExtracting sequences...")
    start = time.time()
    train_hour_sequences = []
    for seq in train_sequences[:1000]:  # Use 1000 sequences
        for i in range(len(seq) - y_seq_len + 1):
            train_hour_sequences.append(seq[i:i+y_seq_len])
            break  # Just one per sequence
    extract_time = time.time() - start
    print(f"Sequence extraction: {extract_time:.4f}s")
    print(f"  Extracted {len(train_hour_sequences)} sequences")
    
    start = time.time()
    train_data = jnp.array(train_hour_sequences)
    array_conv_time = time.time() - start
    print(f"Array conversion: {array_conv_time:.4f}s")
    print(f"  Data shape: {train_data.shape}")
    
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
    
    print("\n=== Initialization ===")
    start = time.time()
    trainer.initialize(train_data[:64])
    init_time = time.time() - start
    print(f"Initialization: {init_time:.4f}s")
    
    print("\n=== Training 10 Epochs ===")
    batch_size = 64
    num_batches = len(train_data) // batch_size
    print(f"Batches per epoch: {num_batches}")
    
    # Time first epoch separately (includes compilation)
    print("\nFirst epoch (includes compilation):")
    start = time.time()
    epoch_metrics = trainer.train_epoch(train_data, batch_size=batch_size, use_dropout=True)
    first_epoch_time = time.time() - start
    print(f"  Time: {first_epoch_time:.4f}s")
    print(f"  Per batch: {first_epoch_time/num_batches:.4f}s")
    
    # Time subsequent epochs
    print("\nSubsequent epochs:")
    epoch_times = []
    for i in range(9):
        start = time.time()
        epoch_metrics = trainer.train_epoch(train_data, batch_size=batch_size, use_dropout=True)
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        if i < 3:
            print(f"  Epoch {i+2}: {epoch_time:.4f}s")
    
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"\nAverage epoch time: {avg_epoch_time:.4f}s")
    print(f"Average per batch: {avg_epoch_time/num_batches:.4f}s")
    print(f"Total for 10 epochs: {first_epoch_time + sum(epoch_times):.4f}s")
    
    # Time full training loop with tqdm
    print("\n=== Full Training Loop (with tqdm) ===")
    trainer2 = VAETrainer(config=config, learning_rate=1e-3, seed=42)
    trainer2.initialize(train_data[:64])
    
    start = time.time()
    history = trainer2.train(
        x_data=train_data,
        num_epochs=10,
        batch_size=batch_size,
        validation_data=None,  # No validation to isolate training
        verbose=True
    )
    full_train_time = time.time() - start
    print(f"\nFull training loop time: {full_train_time:.4f}s")
    print(f"Per epoch: {full_train_time/10:.4f}s")

if __name__ == "__main__":
    main()

