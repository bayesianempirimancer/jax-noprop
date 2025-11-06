#!/usr/bin/env python3
"""
Test different JIT strategies to understand the performance implications.
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
from functools import partial
from flax.core import FrozenDict

from src.models.vae.vae import VAE, VAEConfig
from src.models.vae.trainer import VAETrainer

def test_strategy_1_no_jit_on_loss():
    """Strategy 1: No JIT on loss(), only on train_step"""
    print("\n=== Strategy 1: No JIT on loss() ===")
    
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
    
    # Load small dataset
    with open('data/stock_sequences_full_day_2d.pkl', 'rb') as f:
        data = pickle.load(f)
    train_sequences = data['train']['sequences'][:200]
    y_seq_len = 12
    train_hour_sequences = []
    for seq in train_sequences:
        for i in range(len(seq) - y_seq_len + 1):
            train_hour_sequences.append(seq[i:i+y_seq_len])
            break
    train_data = jnp.array(train_hour_sequences)
    
    trainer = VAETrainer(config=config, learning_rate=1e-3, seed=42)
    trainer.initialize(train_data[:64])
    
    # Time training steps
    x_batch = train_data[:64]
    key = jr.PRNGKey(42)
    
    # Warm up
    trainer.params, trainer.opt_state, loss, metrics = trainer.train_step(
        trainer.params, x_batch, trainer.opt_state, key, training=True
    )
    
    # Time 10 training steps
    times = []
    for i in range(10):
        trainer.rng, step_key = jr.split(trainer.rng)
        start = time.time()
        trainer.params, trainer.opt_state, loss, metrics = trainer.train_step(
            trainer.params, x_batch, trainer.opt_state, step_key, training=True
        )
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"  Average train_step time: {avg_time:.4f}s")
    
    # Time evaluation
    val_data = train_data[:128]
    start = time.time()
    val_metrics = trainer.evaluate(val_data, batch_size=64)
    eval_time = time.time() - start
    print(f"  Evaluation time: {eval_time:.4f}s")
    
    # Time training step after evaluation
    trainer.rng, step_key = jr.split(trainer.rng)
    start = time.time()
    trainer.params, trainer.opt_state, loss, metrics = trainer.train_step(
        trainer.params, x_batch, trainer.opt_state, step_key, training=True
    )
    post_eval_time = time.time() - start
    print(f"  Train step after eval: {post_eval_time:.4f}s")
    
    return avg_time, eval_time, post_eval_time

def test_strategy_2_jit_on_loss():
    """Strategy 2: JIT on loss() with training as static"""
    print("\n=== Strategy 2: JIT on loss() with training static ===")
    
    # Temporarily add JIT to loss
    original_loss = VAE.loss
    @partial(jax.jit, static_argnums=(0, 4))
    def jitted_loss(self, params, x, key, training=True):
        return original_loss(self, params, x, key, training)
    
    VAE.loss = jitted_loss
    
    try:
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
        
        # Load small dataset
        with open('data/stock_sequences_full_day_2d.pkl', 'rb') as f:
            data = pickle.load(f)
        train_sequences = data['train']['sequences'][:200]
        y_seq_len = 12
        train_hour_sequences = []
        for seq in train_sequences:
            for i in range(len(seq) - y_seq_len + 1):
                train_hour_sequences.append(seq[i:i+y_seq_len])
                break
        train_data = jnp.array(train_hour_sequences)
        
        trainer = VAETrainer(config=config, learning_rate=1e-3, seed=42)
        trainer.initialize(train_data[:64])
        
        # Time training steps
        x_batch = train_data[:64]
        key = jr.PRNGKey(42)
        
        # Warm up
        trainer.params, trainer.opt_state, loss, metrics = trainer.train_step(
            trainer.params, x_batch, trainer.opt_state, key, training=True
        )
        
        # Time 10 training steps
        times = []
        for i in range(10):
            trainer.rng, step_key = jr.split(trainer.rng)
            start = time.time()
            trainer.params, trainer.opt_state, loss, metrics = trainer.train_step(
                trainer.params, x_batch, trainer.opt_state, step_key, training=True
            )
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        print(f"  Average train_step time: {avg_time:.4f}s")
        
        # Time evaluation
        val_data = train_data[:128]
        start = time.time()
        val_metrics = trainer.evaluate(val_data, batch_size=64)
        eval_time = time.time() - start
        print(f"  Evaluation time: {eval_time:.4f}s")
        
        # Time training step after evaluation
        trainer.rng, step_key = jr.split(trainer.rng)
        start = time.time()
        trainer.params, trainer.opt_state, loss, metrics = trainer.train_step(
            trainer.params, x_batch, trainer.opt_state, step_key, training=True
        )
        post_eval_time = time.time() - start
        print(f"  Train step after eval: {post_eval_time:.4f}s")
        
        return avg_time, eval_time, post_eval_time
    finally:
        # Restore original
        VAE.loss = original_loss

def main():
    print("Testing different JIT strategies...")
    
    # Strategy 1: No JIT on loss
    s1_train, s1_eval, s1_post = test_strategy_1_no_jit_on_loss()
    
    # Strategy 2: JIT on loss with training static
    s2_train, s2_eval, s2_post = test_strategy_2_jit_on_loss()
    
    print("\n=== Comparison ===")
    print(f"Strategy 1 (no JIT on loss):")
    print(f"  Train step: {s1_train:.4f}s, Eval: {s1_eval:.4f}s, Post-eval train: {s1_post:.4f}s")
    print(f"Strategy 2 (JIT on loss):")
    print(f"  Train step: {s2_train:.4f}s, Eval: {s2_eval:.4f}s, Post-eval train: {s2_post:.4f}s")
    print(f"\nImprovement: {((s1_train - s2_train) / s1_train * 100):.1f}% faster train_step")
    print(f"Post-eval overhead: Strategy 1: {((s1_post - s1_train) / s1_train * 100):.1f}%, Strategy 2: {((s2_post - s2_train) / s2_train * 100):.1f}%")

if __name__ == "__main__":
    main()

