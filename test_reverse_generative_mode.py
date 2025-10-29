#!/usr/bin/env python3
"""
Test reverse generative mode for the diffusion model.
This script loads a trained reverse model and generates samples to visualize predicted coordinates.
"""

import sys
import os
sys.path.append('/home/jebeck/GitHub/jax-noprop/src')

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

# Import the model
from flow_models.df import VAE_flow
from flax.core import FrozenDict

def load_reverse_model(model_path):
    """Load the reverse model and config."""
    print(f"Loading model from {model_path}")
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'params' in data:
        params = data['params']
        config = data['config']
        print("Loaded both parameters and config")
    else:
        params = data
        print("Loaded parameters only (old format)")
        # Create a default config for reverse model
        config = {
            'main': FrozenDict({
                'input_shape': (2,),  # Labels (one-hot)
                'output_shape': (2,),  # Coordinates
                'latent_shape': (2,),
                'recon_weight': 0.0,
                'recon_loss_type': 'mse'
            }),
            'crn': FrozenDict({
                'model_type': 'mlp',
                'hidden_dims': (64, 64, 64),
                'activation': 'swish',
                'use_layer_norm': True,
                'dropout_rate': 0.0
            }),
            'encoder': FrozenDict({
                'model_type': 'identity',
                'hidden_dims': (64, 64),
                'activation': 'swish',
                'use_layer_norm': True,
                'dropout_rate': 0.0
            }),
            'decoder': FrozenDict({
                'model_type': 'none',
                'hidden_dims': (64, 64),
                'activation': 'swish',
                'use_layer_norm': True,
                'dropout_rate': 0.0,
                'output_transformation': 'none'
            })
        }
    
    # Create model
    model = VAE_flow(config)
    
    return model, params, config

def generate_samples(model, params, num_samples=1000, prng_key=None):
    """Generate samples using the reverse model."""
    if prng_key is None:
        prng_key = jr.PRNGKey(42)
    
    # Create input labels (one-hot encoded)
    # For two moons, we have 2 classes, so labels are [1,0] or [0,1]
    labels = jnp.array([[1, 0], [0, 1]])  # Two possible labels
    labels = jnp.tile(labels, (num_samples // 2, 1))  # Repeat to get num_samples
    
    # Generate predictions
    predictions = []
    for i in range(0, num_samples, 100):  # Process in batches of 100
        batch_labels = labels[i:i+100]
        batch_key = jr.fold_in(prng_key, i)
        
        # Use generative mode (sample z_0 from unit normal)
        batch_pred = model.predict(
            params, 
            batch_labels, 
            num_steps=20, 
            integration_method="midpoint",
            output_type="end_point",
            prng_key=batch_key
        )
        predictions.append(batch_pred)
    
    return jnp.concatenate(predictions, axis=0)

def plot_predictions(predictions, labels, save_path=None):
    """Plot the predicted coordinates colored by input label."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Separate predictions by label
    label_0_mask = labels[:, 0] == 1  # Label [1, 0]
    label_1_mask = labels[:, 1] == 1  # Label [0, 1]
    
    # Plot predictions colored by label
    ax.scatter(predictions[label_0_mask, 0], predictions[label_0_mask, 1], 
               c='red', alpha=0.6, s=20, label='Label [1,0]', marker='o')
    ax.scatter(predictions[label_1_mask, 0], predictions[label_1_mask, 1], 
               c='blue', alpha=0.6, s=20, label='Label [0,1]', marker='s')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Reverse Model: Predicted Coordinates from Labels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add some statistics
    stats_text = f'Total samples: {len(predictions)}\n'
    stats_text += f'Label [1,0]: {jnp.sum(label_0_mask)} samples\n'
    stats_text += f'Label [0,1]: {jnp.sum(label_1_mask)} samples\n'
    stats_text += f'X range: [{jnp.min(predictions[:, 0]):.2f}, {jnp.max(predictions[:, 0]):.2f}]\n'
    stats_text += f'Y range: [{jnp.min(predictions[:, 1]):.2f}, {jnp.max(predictions[:, 1]):.2f}]'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    """Main function to test reverse generative mode."""
    # Find the most recent reverse model
    artifacts_dir = Path("/home/jebeck/GitHub/jax-noprop/artifacts")
    reverse_models = list(artifacts_dir.glob("*reverse*identity*mlp*recon1*snr_weighted*"))
    
    if not reverse_models:
        print("No reverse models found in artifacts directory")
        return
    
    # Get the most recent one
    latest_model = max(reverse_models, key=lambda p: p.stat().st_mtime)
    model_path = latest_model / "model_params.pkl"
    
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return
    
    print(f"Using model: {latest_model.name}")
    
    # Load model
    model, params, config = load_reverse_model(model_path)
    
    # Generate samples
    print("Generating 1000 samples...")
    predictions = generate_samples(model, params, num_samples=1000)
    
    # Create labels for plotting
    labels = jnp.array([[1, 0], [0, 1]])
    labels = jnp.tile(labels, (500, 1))  # 500 of each label
    
    # Plot results
    save_path = artifacts_dir / "reverse_generative_mode_visualization.png"
    plot_predictions(predictions, labels, save_path)
    
    print(f"Generated {len(predictions)} samples")
    print(f"Predictions shape: {predictions.shape}")
    print(f"X range: [{jnp.min(predictions[:, 0]):.3f}, {jnp.max(predictions[:, 0]):.3f}]")
    print(f"Y range: [{jnp.min(predictions[:, 1]):.3f}, {jnp.max(predictions[:, 1]):.3f}]")

if __name__ == "__main__":
    main()