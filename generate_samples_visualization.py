#!/usr/bin/env python3
"""
Generate samples using both regression and generative modes and create a 4-panel visualization.
Shows train/val data colored by deterministic predictions vs stochastic predictions.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append('src')

from flow_models.df import VAEFlowConfig, VAE_flow

def load_data_and_model():
    """Load the two moons dataset and trained model."""
    
    # Load dataset
    with open('data/two_moons_formatted.pkl', 'rb') as f:
        data = pickle.load(f)
    
    x_train, y_train = data['train']['x'], data['train']['y']
    x_val, y_val = data['val']['x'], data['val']['y']
    
    print(f"Loaded dataset:")
    print(f"  Train: x={x_train.shape}, y={y_train.shape}")
    print(f"  Val: x={x_val.shape}, y={y_val.shape}")
    
    # Load model
    model_path = "artifacts/test_generative_with_config_20251029_070323/model_params.pkl"
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    model_params = data['params']
    config = data['config']
    model = VAE_flow(config)
    
    print(f"Loaded model with config type: {type(config)}")
    
    return x_train, y_train, x_val, y_val, model, model_params

def generate_predictions(model, params, x_data, num_samples=5):
    """Generate both deterministic and stochastic predictions."""
    
    # Deterministic predictions (regression mode)
    y_det = model.predict(params, x_data, num_steps=20, prng_key=None)
    
    # Stochastic predictions (generative mode) - multiple samples
    key = jr.PRNGKey(42)
    y_stoch_samples = []
    
    for i in range(num_samples):
        key_i, key = jr.split(key)
        y_stoch_i = model.predict(params, x_data, num_steps=20, prng_key=key_i)
        y_stoch_samples.append(y_stoch_i)
    
    # Average the stochastic samples
    y_stoch = jnp.mean(jnp.stack(y_stoch_samples), axis=0)
    
    return y_det, y_stoch, y_stoch_samples

def create_visualization(x_train, y_train, x_val, y_val, y_train_det, y_train_stoch, y_val_det, y_val_stoch):
    """Create 4-panel visualization comparing deterministic vs stochastic predictions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Regression vs Generative Mode Predictions', fontsize=16, fontweight='bold')
    
    # Convert to numpy for plotting
    x_train_np = np.array(x_train)
    y_train_np = np.array(y_train)
    x_val_np = np.array(x_val)
    y_val_np = np.array(y_val)
    
    y_train_det_np = np.array(y_train_det)
    y_train_stoch_np = np.array(y_train_stoch)
    y_val_det_np = np.array(y_val_det)
    y_val_stoch_np = np.array(y_val_stoch)
    
    # Get true class labels (argmax of y)
    train_true_classes = np.argmax(y_train_np, axis=1)
    val_true_classes = np.argmax(y_val_np, axis=1)
    
    # Get predicted class labels
    train_det_classes = np.argmax(y_train_det_np, axis=1)
    train_stoch_classes = np.argmax(y_train_stoch_np, axis=1)
    val_det_classes = np.argmax(y_val_det_np, axis=1)
    val_stoch_classes = np.argmax(y_val_stoch_np, axis=1)
    
    # Panel 1: Train - Deterministic
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(x_train_np[:, 0], x_train_np[:, 1], c=train_det_classes, 
                          cmap='viridis', alpha=0.7, s=20)
    ax1.set_title('Train Data - Deterministic Predictions', fontweight='bold')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Predicted Class')
    
    # Panel 2: Train - Stochastic
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(x_train_np[:, 0], x_train_np[:, 1], c=train_stoch_classes, 
                          cmap='viridis', alpha=0.7, s=20)
    ax2.set_title('Train Data - Stochastic Predictions', fontweight='bold')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Predicted Class')
    
    # Panel 3: Val - Deterministic
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(x_val_np[:, 0], x_val_np[:, 1], c=val_det_classes, 
                          cmap='viridis', alpha=0.7, s=20)
    ax3.set_title('Validation Data - Deterministic Predictions', fontweight='bold')
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Predicted Class')
    
    # Panel 4: Val - Stochastic
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(x_val_np[:, 0], x_val_np[:, 1], c=val_stoch_classes, 
                          cmap='viridis', alpha=0.7, s=20)
    ax4.set_title('Validation Data - Stochastic Predictions', fontweight='bold')
    ax4.set_xlabel('X1')
    ax4.set_ylabel('X2')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=ax4, label='Predicted Class')
    
    plt.tight_layout()
    
    # Calculate and print accuracies
    train_det_acc = np.mean(train_det_classes == train_true_classes)
    train_stoch_acc = np.mean(train_stoch_classes == train_true_classes)
    val_det_acc = np.mean(val_det_classes == val_true_classes)
    val_stoch_acc = np.mean(val_stoch_classes == val_true_classes)
    
    print(f"\nAccuracy Comparison:")
    print(f"Train - Deterministic: {train_det_acc:.3f}")
    print(f"Train - Stochastic:    {train_stoch_acc:.3f}")
    print(f"Val - Deterministic:   {val_det_acc:.3f}")
    print(f"Val - Stochastic:      {val_stoch_acc:.3f}")
    
    return fig

def main():
    """Main function to generate samples and create visualization."""
    
    print("Generating Samples and Creating Visualization")
    print("=" * 50)
    
    # Load data and model
    x_train, y_train, x_val, y_val, model, params = load_data_and_model()
    
    # Generate predictions for training data
    print("\nGenerating predictions for training data...")
    y_train_det, y_train_stoch, y_train_stoch_samples = generate_predictions(model, params, x_train, num_samples=5)
    
    # Generate predictions for validation data
    print("Generating predictions for validation data...")
    y_val_det, y_val_stoch, y_val_stoch_samples = generate_predictions(model, params, x_val, num_samples=5)
    
    # Create visualization
    print("Creating visualization...")
    fig = create_visualization(x_train, y_train, x_val, y_val, 
                              y_train_det, y_train_stoch, y_val_det, y_val_stoch)
    
    # Save the figure
    output_path = "artifacts/samples_comparison_visualization.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Show some example predictions
    print(f"\nExample Predictions (first 5 training samples):")
    print("Input\t\t\tTrue\tDet\tStoch")
    print("-" * 50)
    for i in range(5):
        true_class = np.argmax(y_train[i])
        det_class = np.argmax(y_train_det[i])
        stoch_class = np.argmax(y_train_stoch[i])
        print(f"{x_train[i]}\t{true_class}\t{det_class}\t{stoch_class}")
    
    plt.show()

if __name__ == "__main__":
    main()
