#!/usr/bin/env python3
"""
Compare original two moons data with reverse model predictions.
This script loads the original dataset and the reverse model predictions to compare them.
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

def load_original_data():
    """Load the original two moons dataset."""
    data_path = "/home/jebeck/GitHub/jax-noprop/data/two_moons_formatted.pkl"
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    x_train = jnp.array(data['train']['x'])
    y_train = jnp.array(data['train']['y'])
    x_val = jnp.array(data['val']['x'])
    y_val = jnp.array(data['val']['y'])
    
    return x_train, y_train, x_val, y_val

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

def generate_reverse_predictions(model, params, num_samples=1000, prng_key=None):
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
    
    return jnp.concatenate(predictions, axis=0), labels

def plot_comparison(original_x, original_y, predicted_x, predicted_labels, save_path=None):
    """Plot comparison between original data and reverse model predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Original two moons data
    ax1 = axes[0]
    
    # Separate original data by label
    # original_y is 2D array (10000, 2) with one-hot encoded labels
    label_0_mask = original_y[:, 0] == 1  # [1, 0] pattern
    label_1_mask = original_y[:, 1] == 1  # [0, 1] pattern
    
    ax1.scatter(original_x[label_0_mask, 0], original_x[label_0_mask, 1], 
                c='red', alpha=0.6, s=20, label='Class [1,0]', marker='o')
    ax1.scatter(original_x[label_1_mask, 0], original_x[label_1_mask, 1], 
                c='blue', alpha=0.6, s=20, label='Class [0,1]', marker='s')
    
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Original Two Moons Dataset')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics for original data
    stats_text1 = f'Total samples: {len(original_x)}\n'
    stats_text1 += f'Class [1,0]: {jnp.sum(label_0_mask)} samples\n'
    stats_text1 += f'Class [0,1]: {jnp.sum(label_1_mask)} samples\n'
    stats_text1 += f'X range: [{jnp.min(original_x[:, 0]):.2f}, {jnp.max(original_x[:, 0]):.2f}]\n'
    stats_text1 += f'Y range: [{jnp.min(original_x[:, 1]):.2f}, {jnp.max(original_x[:, 1]):.2f}]'
    
    ax1.text(0.02, 0.98, stats_text1, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Reverse model predictions
    ax2 = axes[1]
    
    # Separate predictions by label
    label_0_mask_pred = predicted_labels[:, 0] == 1  # Label [1, 0]
    label_1_mask_pred = predicted_labels[:, 1] == 1  # Label [0, 1]
    
    ax2.scatter(predicted_x[label_0_mask_pred, 0], predicted_x[label_0_mask_pred, 1], 
                c='red', alpha=0.6, s=20, label='Label [1,0]', marker='o')
    ax2.scatter(predicted_x[label_1_mask_pred, 0], predicted_x[label_1_mask_pred, 1], 
                c='blue', alpha=0.6, s=20, label='Label [0,1]', marker='s')
    
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('Reverse Model: Predicted Coordinates from Labels')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics for predictions
    stats_text2 = f'Total samples: {len(predicted_x)}\n'
    stats_text2 += f'Label [1,0]: {jnp.sum(label_0_mask_pred)} samples\n'
    stats_text2 += f'Label [0,1]: {jnp.sum(label_1_mask_pred)} samples\n'
    stats_text2 += f'X range: [{jnp.min(predicted_x[:, 0]):.2f}, {jnp.max(predicted_x[:, 0]):.2f}]\n'
    stats_text2 += f'Y range: [{jnp.min(predicted_x[:, 1]):.2f}, {jnp.max(predicted_x[:, 1]):.2f}]'
    
    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()

def main():
    """Main function to compare original data with reverse model predictions."""
    # Load original data
    print("Loading original two moons dataset...")
    x_train, y_train, x_val, y_val = load_original_data()
    
    # Combine train and val for comparison
    original_x = jnp.concatenate([x_train, x_val], axis=0)
    original_y = jnp.concatenate([y_train, y_val], axis=0)
    
    print(f"Original data shapes:")
    print(f"  x: {original_x.shape}")
    print(f"  y: {original_y.shape}")
    print(f"  y unique values: {jnp.unique(original_y)}")
    
    # Find the most recent reverse model
    artifacts_dir = Path("/home/jebeck/GitHub/jax-noprop/artifacts")
    reverse_models = list(artifacts_dir.glob("*reverse*identity*identity*recon0*"))
    
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
    
    # Generate predictions
    print("Generating 1000 samples...")
    predicted_x, predicted_labels = generate_reverse_predictions(model, params, num_samples=1000)
    
    # Plot comparison
    save_path = artifacts_dir / "reverse_vs_original_comparison.png"
    plot_comparison(original_x, original_y, predicted_x, predicted_labels, save_path)
    
    print(f"Generated {len(predicted_x)} samples")
    print(f"Predictions shape: {predicted_x.shape}")
    print(f"Original data shape: {original_x.shape}")

if __name__ == "__main__":
    main()
