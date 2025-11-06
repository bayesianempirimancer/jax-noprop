#!/usr/bin/env python3
"""
Test if recompilation occurs when switching between training and evaluation.
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

from src.models.vae.vae import VAEConfig
from src.models.vae.trainer import VAETrainer

# Enable JAX compilation logging (too verbose, disable for now)
# jax.config.update("jax_log_compiles", True)

def main():
    print("Loading data...")
    with open('data/stock_sequences_full_day_2d.pkl', 'rb') as f:
        data = pickle.load(f)
    
    train_sequences = data['train']['sequences']
    val_sequences = data['val']['sequences']
    y_seq_len = data.get('metadata', {}).get('y_seq_len', 12)
    
    # Extract a small sample
    train_hour_sequences = []
    for seq in train_sequences[:100]:
        for i in range(len(seq) - y_seq_len + 1):
            train_hour_sequences.append(seq[i:i+y_seq_len])
            break
    
    val_hour_sequences = []
    for seq in val_sequences[:100]:
        for i in range(len(seq) - y_seq_len + 1):
            val_hour_sequences.append(seq[i:i+y_seq_len])
            break
    
    train_data = jnp.array(train_hour_sequences)
    val_data = jnp.array(val_hour_sequences)
    
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
    trainer.initialize(train_data[:64])
    
    print("\n=== Testing recompilation ===")
    print("\n1. First training step (compilation)...")
    x_batch = train_data[:64]
    key = jr.PRNGKey(42)
    start = time.time()
    trainer.params, trainer.opt_state, loss, metrics = trainer.train_step(
        trainer.params, x_batch, trainer.opt_state, key, training=True
    )
    time1 = time.time() - start
    print(f"   Time: {time1:.4f}s")
    
    print("\n2. Second training step (should be fast, no recompilation)...")
    trainer.rng, key2 = jr.split(trainer.rng)
    start = time.time()
    trainer.params, trainer.opt_state, loss, metrics = trainer.train_step(
        trainer.params, x_batch, trainer.opt_state, key2, training=True
    )
    time2 = time.time() - start
    print(f"   Time: {time2:.4f}s")
    
    print("\n3. First evaluation (compilation of _eval_batch)...")
    start = time.time()
    val_metrics = trainer.evaluate(val_data, batch_size=64)
    time3 = time.time() - start
    print(f"   Time: {time3:.4f}s")
    
    print("\n4. Training step after evaluation (checking for recompilation)...")
    trainer.rng, key3 = jr.split(trainer.rng)
    start = time.time()
    trainer.params, trainer.opt_state, loss, metrics = trainer.train_step(
        trainer.params, x_batch, trainer.opt_state, key3, training=True
    )
    time4 = time.time() - start
    print(f"   Time: {time4:.4f}s")
    if time4 > time2 * 2:
        print(f"   WARNING: Possible recompilation! Time increased from {time2:.4f}s to {time4:.4f}s")
    else:
        print(f"   OK: No significant recompilation (time: {time2:.4f}s -> {time4:.4f}s)")
    
    print("\n5. Second evaluation (should be fast)...")
    start = time.time()
    val_metrics = trainer.evaluate(val_data, batch_size=64)
    time5 = time.time() - start
    print(f"   Time: {time5:.4f}s")
    
    print("\n=== Summary ===")
    print(f"Training step times: {time1:.4f}s (first), {time2:.4f}s (subsequent), {time4:.4f}s (after eval)")
    print(f"Evaluation times: {time3:.4f}s (first), {time5:.4f}s (subsequent)")

if __name__ == "__main__":
    main()

