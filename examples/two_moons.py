#!/usr/bin/env python3
"""
Two Moons NoProp-CT Example

This example demonstrates NoProp-CT on the classic two moons dataset:
- x: 2D coordinates from the two moons dataset
- z: One-hot encoded labels (which moon the point belongs to)
- Training: Learn to predict z from x using NoProp-CT
- Visualization: Show learning curves, MSE, and z(t) trajectories

The two moons dataset is perfect for visualizing:
1. How the model learns to separate the two classes
2. The continuous-time evolution of predictions z(t)
3. The effect of different noise schedules
"""
import time
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple
import os

# Import our NoProp implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.jax_noprop.noprop_ct import NoPropCT
from src.jax_noprop.noprop_fm import NoPropFM
from src.jax_noprop.models import SimpleMLP
from src.jax_noprop.noise_schedules import (
    LinearNoiseSchedule, 
    CosineNoiseSchedule, 
    SigmoidNoiseSchedule,
    LearnableNoiseSchedule,
    SimpleMonotonicNetwork
)


def generate_two_moons_data(n_samples: int = 1000, noise: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two moons dataset.
    
    Args:
        n_samples: Number of samples to generate
        noise: Amount of noise to add
        random_state: Random seed for reproducibility
        
    Returns:
        x: 2D coordinates [n_samples, 2]
        z: One-hot encoded labels [n_samples, 2]
    """
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Standardize the features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    # Convert to 2D binary labels: class 0 = [-1, 1], class 1 = [1, -1]
    z = np.zeros((n_samples, 2), dtype=np.float32)
    z[y == 0, 0] = -1.0  # Class 0: [-1, 1]
    z[y == 0, 1] = 1.0
    z[y == 1, 0] = 1.0   # Class 1: [1, -1]
    z[y == 1, 1] = -1.0
    
    return x.astype(np.float32), z.astype(np.float32)




def train_model(
    x_train: np.ndarray,
    z_train: np.ndarray,
    x_val: np.ndarray,
    z_val: np.ndarray,
    model_type: str = "ct",
    noise_schedule_name: str = "cosine",
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    batch_size: int = 64,
    random_seed: int = 42,
    sigma_t: float = 0.05,
    reg_weight: float = 0.0
) -> Dict[str, Any]:
    """Train NoProp model (CT or FM) on two moons dataset.
    
    Args:
        x_train: Training features [n_train, 2]
        z_train: Training targets [n_train, 1]
        x_val: Validation features [n_val, 2]
        z_val: Validation targets [n_val, 1]
        model_type: "ct" for NoProp-CT or "fm" for NoProp-FM
        noise_schedule_name: Name of noise schedule to use (for CT only)
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        random_seed: Random seed for reproducibility
        sigma_t: Noise standard deviation for z_t (for FM only)
        reg_weight: Regularization weight (for FM only)
        
    Returns:
        Dictionary containing training results
    """
    # Set random seed
    key = jr.PRNGKey(random_seed)
    
    # Create model using the new SimpleMLP with multiple hidden layers
    model = SimpleMLP(hidden_dims=(64, 64, 64))
    
    # Create the appropriate NoProp model
    if model_type == "ct":
        # Create noise schedule
        if noise_schedule_name == "linear":
            noise_schedule = LinearNoiseSchedule()
        elif noise_schedule_name == "cosine":
            noise_schedule = CosineNoiseSchedule()
        elif noise_schedule_name == "sigmoid":
            noise_schedule = SigmoidNoiseSchedule()
        elif noise_schedule_name == "learnable":
            noise_schedule = LearnableNoiseSchedule(
                hidden_dims=(32, 32),
                monotonic_network=SimpleMonotonicNetwork(hidden_dims=(32, 32))
            )
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule_name}")
        
        # Create NoProp-CT
        noprop_model = NoPropCT(target_dim=2, model=model, noise_schedule=noise_schedule)
        model_name = f"NoProp-CT ({noise_schedule_name})"
        
    elif model_type == "fm":
        # Create NoProp-FM
        noprop_model = NoPropFM(
            target_dim=2, 
            model=model,
            reg_weight=reg_weight,
            sigma_t=sigma_t
        )
        model_name = f"NoProp-FM (σ_t={sigma_t}, reg={reg_weight})"
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize parameters
    key, subkey = jr.split(key)
    dummy_x = jnp.ones((batch_size, 2))
    dummy_z = jnp.ones((batch_size, 2))
    dummy_t = jnp.ones((batch_size,))
    
    params = noprop_model.init(subkey, dummy_z, dummy_x, dummy_t)
    
    # Create optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_mses = []
    val_mses = []
    train_accuracies = []
    val_accuracies = []
    
    n_train = len(x_train)
    n_val = len(x_val)
    
    print(f"Training {model_name}...")
    print(f"Training samples: {n_train}, Validation samples: {n_val}")
    
    # Timing measurements
    epoch_times = []
    total_train_start = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        # Training
        epoch_train_losses = []
        epoch_train_mses = []
        epoch_train_accuracies = []
        
        # Shuffle training data
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, n_train)
        x_train_shuffled = x_train[perm]
        z_train_shuffled = z_train[perm]
        
        # Training batches
        for i in range(0, n_train, batch_size):
            batch_end = min(i + batch_size, n_train)
            x_batch = x_train_shuffled[i:batch_end]
            z_batch = z_train_shuffled[i:batch_end]
            
            # Ensure batch size is consistent
            if len(x_batch) < batch_size:
                # Pad with random samples if needed
                n_pad = batch_size - len(x_batch)
                key, subkey = jr.split(key)
                pad_indices = jr.choice(subkey, n_train, (n_pad,))
                x_batch = jnp.concatenate([x_batch, x_train[pad_indices]], axis=0)
                z_batch = jnp.concatenate([z_batch, z_train[pad_indices]], axis=0)
            
            # Training step
            key, subkey = jr.split(key)
            params, opt_state, loss, metrics = noprop_model.train_step(
                params, opt_state, x_batch, z_batch, subkey, optimizer
            )
            
            epoch_train_losses.append(loss)
            epoch_train_mses.append(metrics['mse'])
        
        # Compute actual validation metrics
        val_loss, val_metrics = noprop_model.compute_loss(params, x_val, z_val, jr.PRNGKey(42))
        epoch_val_losses = [val_loss]
        epoch_val_mses = [val_metrics['mse']]
        
        # Compute training accuracy for this epoch
        z_train_pred = noprop_model.predict(params, x_train, 2, 10, "euler")
        train_pred_classes = (z_train_pred[:, 0] > z_train_pred[:, 1]).astype(int)
        train_true_classes = (z_train[:, 0] > z_train[:, 1]).astype(int)
        epoch_train_accuracy = jnp.mean(train_pred_classes == train_true_classes)
        
        # Compute validation accuracy for this epoch
        z_val_pred = noprop_model.predict(params, x_val, 2, 10, "euler")
        val_pred_classes = (z_val_pred[:, 0] > z_val_pred[:, 1]).astype(int)
        val_true_classes = (z_val[:, 0] > z_val[:, 1]).astype(int)
        epoch_val_accuracy = jnp.mean(val_pred_classes == val_true_classes)
        
        epoch_val_accuracies = [epoch_val_accuracy]
        
        # Record metrics
        train_losses.append(np.mean(epoch_train_losses))
        val_losses.append(np.mean(epoch_val_losses))
        train_mses.append(np.mean(epoch_train_mses))
        val_mses.append(np.mean(epoch_val_mses))
        train_accuracies.append(epoch_train_accuracy)
        val_accuracies.append(epoch_val_accuracy)
        
        # Record epoch timing
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss = {train_losses[-1]:.4f}, "
                  f"Val Loss = {val_losses[-1]:.4f}, "
                  f"Train MSE = {train_mses[-1]:.4f}, "
                  f"Val MSE = {val_mses[-1]:.4f}, "
                  f"Train Acc = {train_accuracies[-1]:.4f}, "
                  f"Val Acc = {val_accuracies[-1]:.4f}, "
                  f"Time = {epoch_time:.3f}s")
    
    # Compute final accuracies and measure inference time
    print("Computing final accuracies and measuring inference time...")
    
    # Measure inference time for training set
    inference_start = time.time()
    z_train_pred = noprop_model.predict(params, x_train, 2, 10, "euler")
    train_inference_time = time.time() - inference_start
    train_pred_classes = (z_train_pred[:, 0] > z_train_pred[:, 1]).astype(int)  # z[0] > z[1]
    train_true_classes = (z_train[:, 0] > z_train[:, 1]).astype(int)  # z[0] > z[1]
    final_train_accuracy = jnp.mean(train_pred_classes == train_true_classes)
    
    # Measure inference time for validation set
    inference_start = time.time()
    z_val_pred = noprop_model.predict(params, x_val, 2, 10, "euler")
    val_inference_time = time.time() - inference_start
    val_pred_classes = (z_val_pred[:, 0] > z_val_pred[:, 1]).astype(int)  # z[0] > z[1]
    val_true_classes = (z_val[:, 0] > z_val[:, 1]).astype(int)  # z[0] > z[1]
    final_val_accuracy = jnp.mean(val_pred_classes == val_true_classes)
    
    # Calculate total training time
    total_train_time = time.time() - total_train_start
    
    print(f"Final Training Accuracy: {final_train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
    print(f"Total Training Time: {total_train_time:.3f}s")
    print(f"Average Epoch Time: {np.mean(epoch_times):.3f}s")
    print(f"Train Inference Time: {train_inference_time:.3f}s")
    print(f"Val Inference Time: {val_inference_time:.3f}s")
    
    # Return results with model-specific keys
    results = {
        'params': params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mses': train_mses,
        'val_mses': val_mses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'final_train_accuracy': final_train_accuracy,
        'final_val_accuracy': final_val_accuracy,
        'model_type': model_type,
        'model_name': model_name,
        'total_train_time': total_train_time,
        'epoch_times': epoch_times,
        'avg_epoch_time': np.mean(epoch_times),
        'train_inference_time': train_inference_time,
        'val_inference_time': val_inference_time
    }
    
    if model_type == "ct":
        results['noprop_ct'] = noprop_model
        results['noise_schedule_name'] = noise_schedule_name
    elif model_type == "fm":
        results['noprop_fm'] = noprop_model
        results['sigma_t'] = sigma_t
        results['reg_weight'] = reg_weight
    
    return results


def predict_trajectories(
    noprop_model,
    params: Dict[str, Any],
    x: np.ndarray,
    num_timesteps: int = 50,
    random_seed: int = 42
) -> np.ndarray:
    """Predict z(t) trajectories for given inputs using actual ODE integration.
    
    Args:
        noprop_model: Trained NoProp model (CT or FM)
        params: Model parameters
        x: Input features [n_samples, 2]
        num_timesteps: Number of timesteps to use for prediction
        random_seed: Random seed for reproducibility
        
    Returns:
        z_trajectories: Predicted trajectories [n_samples, num_timesteps + 1, 2]
    """
    # Use the new predict_trajectory method to get actual ODE integration trajectories
    z_trajectories = noprop_model.predict_trajectory(
        params, x, "euler", 2, num_timesteps
    )
    # Transpose from (num_steps+1, batch_size, output_dim) to (batch_size, num_steps+1, output_dim) for plotting
    z_trajectories = jnp.transpose(z_trajectories, (1, 0, 2))
    
    return z_trajectories  # [n_samples, num_timesteps + 1, 2]


def plot_results(
    x_train: np.ndarray,
    z_train: np.ndarray,
    x_val: np.ndarray,
    z_val: np.ndarray,
    results: Dict[str, Any],
    save_dir: str = "artifacts"
):
    """Plot training results and trajectories.
    
    Args:
        x_train: Training features
        z_train: Training targets
        x_val: Validation features
        z_val: Validation targets
        results: Training results dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract results
    train_losses = results['train_losses']
    val_losses = results['val_losses']
    train_mses = results['train_mses']
    val_mses = results['val_mses']
    train_accuracies = results['train_accuracies']
    val_accuracies = results['val_accuracies']
    model_name = results['model_name']
    model_type = results['model_type']
    
    # Get model-specific information
    if model_type == "ct":
        noise_schedule_name = results['noise_schedule_name']
        noprop_model = results['noprop_ct']
        plot_suffix = noise_schedule_name
    elif model_type == "fm":
        sigma_t = results['sigma_t']
        reg_weight = results['reg_weight']
        noprop_model = results['noprop_fm']
        plot_suffix = f"fm_sigma{sigma_t}_reg{reg_weight}"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 1. Plot learning curves
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    epochs = range(len(train_losses))
    
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
    ax1.plot(epochs, val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Progress - {model_name}')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_mses, label='Train MSE', color='blue')
    ax2.plot(epochs, val_mses, label='Val MSE', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title(f'MSE Progress - {model_name}')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    ax3.plot(epochs, val_accuracies, label='Val Accuracy', color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title(f'Accuracy Progress - {model_name}')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_curves_{plot_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot data and predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # True labels
    colors = ['red', 'blue']
    labels = ['Moon 0', 'Moon 1']
    
    # Plot true labels (binary: +1 for class 1, -1 for class 0)
    mask_class0 = z_train[:, 0] == -1  # Class 0 (Moon 0)
    mask_class1 = z_train[:, 0] == 1   # Class 1 (Moon 1)
    ax1.scatter(x_train[mask_class0, 0], x_train[mask_class0, 1], c=colors[0], label=labels[0], alpha=0.6, s=20)
    ax1.scatter(x_train[mask_class1, 0], x_train[mask_class1, 1], c=colors[1], label=labels[1], alpha=0.6, s=20)
    
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('True Two Moons Dataset')
    ax1.legend()
    ax1.grid(True)
    
    # Predictions
    params = results['params']
    
    # Get final predictions
    key = jr.PRNGKey(42)
    num_steps = 40  # Number of integration steps for prediction
    z_pred = noprop_model.predict(params, x_val, 2, num_steps, "euler")
    
    # Convert to class predictions using z[0] > z[1]
    pred_classes = (z_pred[:, 0] > z_pred[:, 1]).astype(int)  # z[0] > z[1]
    
    for i in range(2):
        mask = pred_classes == i
        ax2.scatter(x_val[mask, 0], x_val[mask, 1], c=colors[i], label=f'Predicted {labels[i]}', alpha=0.6, s=20)
    
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title(f'{model_name} Predictions')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/predictions_{plot_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot z(t) trajectories for a few samples
    print("Computing z(t) trajectories...")
    z_trajectories = predict_trajectories(noprop_model, params, x_val[:10], num_timesteps=20)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(10):
        ax = axes[i]
        
        # Plot trajectory - now using actual ODE integration (2D output)
        t_points = np.linspace(0, 1, z_trajectories.shape[1])
        ax.plot(t_points, z_trajectories[i, :, 0], 'b-', label='z[0](t)', linewidth=2)
        ax.plot(t_points, z_trajectories[i, :, 1], 'g-', label='z[1](t)', linewidth=2)
        
        # Mark start and end points
        ax.scatter([0], [z_trajectories[i, 0, 0]], c='red', s=100, marker='o', label='z[0](0)')
        ax.scatter([0], [z_trajectories[i, 0, 1]], c='orange', s=100, marker='o', label='z[1](0)')
        ax.scatter([1], [z_trajectories[i, -1, 0]], c='blue', s=100, marker='s', label='z[0](1)')
        ax.scatter([1], [z_trajectories[i, -1, 1]], c='purple', s=100, marker='s', label='z[1](1)')
        
        # True label
        true_class = int((z_val[i, 0] > z_val[i, 1]))  # z[0] > z[1]
        ax.set_title(f'Sample {i+1} (True: {labels[true_class]})')
        ax.set_xlabel('Time t')
        ax.set_ylabel('z(t)')
        ax.grid(True)
        ax.legend(fontsize=8)
    
    plt.suptitle(f'z(t) Trajectories - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/trajectories_{plot_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Plot 2D trajectory evolution in feature space
    print("Creating 2D trajectory evolution plot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Select a few interesting samples to show
    sample_indices = [0, 1, 2, 3, 4, 5]
    time_indices = [0, 4, 8, 12, 16, 20]  # Show evolution at different time points
    
    for i, sample_idx in enumerate(sample_indices):
        ax = axes[i]
        
        # Plot the full trajectory in 2D space
        trajectory_1d = z_trajectories[sample_idx, :, 0]  # [num_timesteps+1] - 1D output
        t_points = np.linspace(0, 1, len(trajectory_1d))
        
        # Plot trajectory line
        ax.plot(t_points, trajectory_1d, 'k-', alpha=0.3, linewidth=1)
        
        # Plot trajectory points with color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory_1d)))
        for j, (t, z) in enumerate(zip(t_points, trajectory_1d)):
            ax.scatter(t, z, c=[colors[j]], s=30, alpha=0.7)
        
        # Mark start and end points
        ax.scatter(t_points[0], trajectory_1d[0], c='red', s=200, marker='o', 
                  label='Start (t=0)', edgecolors='black', linewidth=2)
        ax.scatter(t_points[-1], trajectory_1d[-1], c='blue', s=200, marker='s', 
                  label='End (t=1)', edgecolors='black', linewidth=2)
        
        # Mark intermediate time points
        for t_idx in time_indices[1:-1]:  # Skip start and end
            if t_idx < len(trajectory_1d):
                ax.scatter(t_points[t_idx], trajectory_1d[t_idx], 
                          c='orange', s=100, marker='^', alpha=0.8)
        
        # True label
        true_class = int((z_val[sample_idx, 0] > z_val[sample_idx, 1]))  # z[0] > z[1]
        ax.set_title(f'Sample {sample_idx+1} (True: {labels[true_class]})', fontsize=12)
        ax.set_xlabel('Time t')
        ax.set_ylabel('z(t)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'2D Trajectory Evolution - {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/trajectory_evolution_{plot_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run the two moons example."""
    print("🌙 Two Moons NoProp Comparison Example")
    print("=" * 50)
    
    # Generate data
    print("Generating two moons dataset...")
    x, z = generate_two_moons_data(n_samples=1000, noise=0.1, random_state=42)
    
    # Split into train/val with random shuffling
    np.random.seed(42)
    n_samples = len(x)
    indices = np.random.permutation(n_samples)
    x_shuffled = x[indices]
    z_shuffled = z[indices]
    
    n_train = int(0.8 * n_samples)
    x_train, x_val = x_shuffled[:n_train], x_shuffled[n_train:]
    z_train, z_val = z_shuffled[:n_train], z_shuffled[n_train:]
    
    print(f"Dataset: {len(x)} samples")
    print(f"Training: {len(x_train)} samples")
    print(f"Validation: {len(x_val)} samples")
    
    # Train both models for runtime comparison
    models_to_train = [
        {"type": "ct", "noise_schedule": "linear", "name": "NoProp-CT (Linear)"},
        {"type": "fm", "sigma_t": 0.05, "reg_weight": 0.0, "name": "NoProp-FM (σ_t=0.05)"}
    ]
    
    all_results = []
    
    for model_config in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_config['name']}")
        print(f"{'='*60}")
        
        # Train model
        if model_config["type"] == "ct":
            results = train_model(
                x_train=x_train,
                z_train=z_train,
                x_val=x_val,
                z_val=z_val,
                model_type="ct",
                noise_schedule_name=model_config["noise_schedule"],
                learning_rate=1e-3,
                num_epochs=50,
                batch_size=32,
                random_seed=42
            )
        elif model_config["type"] == "fm":
            results = train_model(
                x_train=x_train,
                z_train=z_train,
                x_val=x_val,
                z_val=z_val,
                model_type="fm",
                learning_rate=1e-3,
                num_epochs=50,
                batch_size=32,
                random_seed=42,
                sigma_t=model_config["sigma_t"],
                reg_weight=model_config["reg_weight"]
            )
        
        # Plot results
        plot_results(x_train, z_train, x_val, z_val, results)
        
        # Print final metrics
        print(f"\nFinal Results for {model_config['name']}:")
        print(f"  Final Train Loss: {results['train_losses'][-1]:.4f}")
        print(f"  Final Val Loss: {results['val_losses'][-1]:.4f}")
        print(f"  Final Train MSE: {results['train_mses'][-1]:.4f}")
        print(f"  Final Val MSE: {results['val_mses'][-1]:.4f}")
        print(f"  Final Train Accuracy: {results['final_train_accuracy']:.4f}")
        print(f"  Final Val Accuracy: {results['final_val_accuracy']:.4f}")
        
        all_results.append(results)
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("📊 RUNTIME COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Train Acc':<10} {'Val Acc':<10} {'Avg Epoch':<10} {'Train Inf':<10} {'Val Inf':<10}")
    print("-" * 80)
    
    for results in all_results:
        model_name = results['model_name']
        train_acc = results['final_train_accuracy']
        val_acc = results['final_val_accuracy']
        avg_epoch_time = results['avg_epoch_time']
        train_inf_time = results['train_inference_time']
        val_inf_time = results['val_inference_time']
        
        print(f"{model_name:<30} {train_acc:<10.4f} {val_acc:<10.4f} {avg_epoch_time:<10.3f} {train_inf_time:<10.3f} {val_inf_time:<10.3f}")
    
    print(f"\n{'='*80}")
    print("📈 DETAILED TIMING BREAKDOWN")
    print(f"{'='*80}")
    
    for results in all_results:
        model_name = results['model_name']
        total_train_time = results['total_train_time']
        avg_epoch_time = results['avg_epoch_time']
        train_inf_time = results['train_inference_time']
        val_inf_time = results['val_inference_time']
        
        print(f"\n{model_name}:")
        print(f"  Total Training Time: {total_train_time:.3f}s")
        print(f"  Average Epoch Time:  {avg_epoch_time:.3f}s")
        print(f"  Train Inference:     {train_inf_time:.3f}s")
        print(f"  Val Inference:       {val_inf_time:.3f}s")
    
    print(f"\n{'='*60}")
    print("🎉 Two Moons Comparison Complete!")
    print("Check the 'artifacts' directory for visualization results.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
