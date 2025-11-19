#!/usr/bin/env python3
"""
Test script for VAE_flow (fm.py) on two moons dataset using GenerationTrainer.

This script tests the standard fm model in conditional generation mode
with use_noise_schedule=False.
"""

import sys
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import jax.random as jr
from flax.core import FrozenDict
from dataclasses import replace

from src.flow_models.trainer_gen import GenerationTrainer
from examples.two_moons.config import Config as TwoMoonsConfig


def load_two_moons_data(data_path: str = "./data/two_moons.pkl"):
    """Load two moons dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please run: python examples/two_moons/generate_two_moons.py"
        )
    
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    x_train = dataset['train']['x']
    x_val = dataset['val']['x']
    y_train = dataset['train']['y']
    y_val = dataset['val']['y']
    
    print(f"Loaded dataset:")
    print(f"  Train samples: {x_train.shape[0]}")
    print(f"  Val samples: {x_val.shape[0]}")
    print(f"  Data shape: {x_train.shape}")
    
    return x_train, y_train, x_val, y_val


def main():
    """Main test function."""
    print("=" * 60)
    print("VAE_flow (fm.py) Test on Two Moons Dataset")
    print("Conditional Generation with use_noise_schedule=False")
    print("=" * 60)
    
    # Configuration
    data_path = "./data/two_moons.pkl"
    num_epochs = 50
    batch_size = 256
    learning_rate = 1e-3
    seed = 42
    output_dir = "./artifacts/two_moons_fm"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    np.random.seed(seed)
    
    # Step 1: Load data
    print("\n[Step 1] Loading two moons dataset...")
    x_train, y_train, x_val, y_val = load_two_moons_data(data_path)
    x_train_jax = jnp.array(x_train)
    y_train_jax = jnp.array(y_train)
    x_val_jax = jnp.array(x_val)
    y_val_jax = jnp.array(y_val)
    
    # Step 2: Create config with use_noise_schedule=False
    print("\n[Step 2] Creating config with use_noise_schedule=False...")
    base_config = TwoMoonsConfig()
    
    # Set no_noise_schedule to True (which means use_noise_schedule=False)
    main_config = base_config.main.copy({
        "no_noise_schedule": True,  # This sets use_noise_schedule=False
    })
    
    config = replace(
        base_config,
        main=main_config,
    )
    
    print(f"  Config created: no_noise_schedule={config.main.get('no_noise_schedule', False)}")
    
    # Step 3: Create trainer
    print("\n[Step 3] Creating GenerationTrainer...")
    trainer = GenerationTrainer(
        config=config,
        learning_rate=learning_rate,
        optimizer_name="adam",
        seed=seed,
        unconditional=False,  # Conditional generation
        warmup_steps=0,
        model_type="flow_matching"
    )
    print("  Trainer created!")
    
    # Step 4: Initialize model
    print("\n[Step 4] Initializing model...")
    x_sample = x_train_jax[0:1]  # Use first sample for initialization
    y_sample = y_train_jax[0:1]
    trainer.initialize(x_sample, y_sample)
    print("  Model initialized!")
    
    # Step 5: Train model
    print(f"\n[Step 5] Training model for {num_epochs} epochs...")
    history = trainer.train(
        x_data=x_train_jax,  # Conditional inputs (y labels)
        y_data=y_train_jax,   # Targets (x coordinates)
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(x_val_jax, y_val_jax),
        dropout_epochs=num_epochs
    )
    print("  Training complete!")
    
    # Step 6: Generate samples
    print("\n[Step 6] Generating samples...")
    num_gen = min(1000, x_val_jax.shape[0])
    key = jr.PRNGKey(seed + 123)
    
    # Conditional generation: generate x coordinates given y labels
    cond_y = x_val_jax[:num_gen]  # Conditional inputs (y labels)
    x_gen = trainer.conditional_generate(cond_y, num_steps=20, prng_key=key)
    x_gen_np = np.array(x_gen)
    print(f"  Generated {x_gen_np.shape[0]} samples")
    
    # Step 7: Save results and create plots
    print("\n[Step 7] Saving results and creating plots...")
    x_real = y_val_jax[:num_gen]  # Real x coordinates (targets)
    y_labels = x_val_jax[:num_gen]  # y labels (conditions)
    
    # Store in history for plotting
    history['x_gen'] = x_gen_np
    history['x_real'] = np.array(x_real)
    
    trainer.save_results(
        history=history,
        output_dir=output_dir,
        x_real=np.array(x_real),
        x_gen=x_gen_np,
        y_labels=np.array(y_labels)
    )
    
    print("\n" + "=" * 60)
    print("Test complete! All outputs saved to:")
    print(f"  {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()


