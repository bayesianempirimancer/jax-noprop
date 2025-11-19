"""
Plotting utilities for generation tasks (conditional and unconditional).

This module provides functions for creating plots for generative models,
including generation comparisons, loss trends, and latent trajectories.
"""
import os
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax.numpy as jnp
import jax.random as jr


def create_generation_plot(
    x_real: np.ndarray, 
    y_labels: Optional[np.ndarray], 
    x_gen: np.ndarray, 
    output_dir: str,
    unconditional: bool = False,
    x_real_labels: Optional[np.ndarray] = None
):
    """
    Create generation comparison plot showing real vs generated samples.
    
    Args:
        x_real: Real samples [N, 2]
        y_labels: Conditional labels [N, ...] or None for unconditional (used for x_gen)
        x_gen: Generated samples [N, 2]
        output_dir: Directory to save the plot
        unconditional: Whether this is unconditional generation
        x_real_labels: Optional labels for x_real [N, ...]. If None, uses y_labels.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    if y_labels is not None and not unconditional:
        # Conditional generation: color by class labels
        # Use x_real_labels if provided, otherwise use y_labels for both
        labels_for_real = x_real_labels if x_real_labels is not None else y_labels
        labels_for_gen = y_labels
        
        # Convert one-hot to class indices for coloring
        def get_class_indices(labels):
            if len(labels.shape) == 2 and labels.shape[1] > 1:
            # One-hot encoded: use argmax to get class indices
                return np.argmax(labels, axis=1)
        else:
            # Integer labels: use directly
                return labels.flatten().astype(int)
        
        class_indices_real = get_class_indices(labels_for_real)
        class_indices_gen = get_class_indices(labels_for_gen)
        
        # Ensure lengths match
        n_real = len(x_real)
        n_gen = len(x_gen)
        n_labels_real = len(class_indices_real)
        n_labels_gen = len(class_indices_gen)
        
        if n_real != n_labels_real:
            print(f"Warning: Mismatch in lengths - x_real: {n_real}, x_real_labels: {n_labels_real}")
            min_len = min(n_real, n_labels_real)
            x_real = x_real[:min_len]
            class_indices_real = class_indices_real[:min_len]
        
        if n_gen != n_labels_gen:
            print(f"Warning: Mismatch in lengths - x_gen: {n_gen}, y_labels: {n_labels_gen}")
            min_len = min(n_gen, n_labels_gen)
            x_gen = x_gen[:min_len]
            class_indices_gen = class_indices_gen[:min_len]
        
        # Real - color by the labels that correspond to x_real
        ax[0].scatter(x_real[:, 0], x_real[:, 1], c=class_indices_real, s=6, cmap='coolwarm', alpha=0.8)
        ax[0].set_title('Real samples (x)')
        ax[0].set_aspect('equal', 'box')
        # Generated - color by the labels that were used for conditioning (should match x_gen order)
        ax[1].scatter(x_gen[:, 0], x_gen[:, 1], c=class_indices_gen, s=6, cmap='coolwarm', alpha=0.8)
        ax[1].set_title('Generated samples (x | y)')
    else:
        # Unconditional generation: single color
        # Real
        ax[0].scatter(x_real[:, 0], x_real[:, 1], c='blue', s=6, alpha=0.8)
        ax[0].set_title('Real samples (x)')
        ax[0].set_aspect('equal', 'box')
        # Generated
        ax[1].scatter(x_gen[:, 0], x_gen[:, 1], c='purple', s=6, alpha=0.8)
        ax[1].set_title('Generated samples (x) - Unconditional')
    
    ax[1].set_aspect('equal', 'box')
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_name = 'unconditional_generation.png' if unconditional else 'conditional_generation.png'
    fig.savefig(os.path.join(output_dir, plot_name), dpi=200, bbox_inches='tight')
    plt.close(fig)


def create_loss_trends_plot(history: Dict[str, Any], model_type: str, output_dir: str):
    """
    Plot loss terms over training epochs to diagnose training issues.
    Always uses 2x3 layout with 6 panels.
    
    Args:
        history: Training history dictionary containing loss values
        model_type: Type of model ('flow_matching', 'diffusion', 'ct')
        output_dir: Directory to save the plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Always use 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Loss Trends - {model_type.title()} Model', fontsize=16, fontweight='bold')
    
    epochs = range(len(history['train_losses']))
    
    # Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_losses'], label='Train Total', color='blue', linewidth=2)
    if history.get('val_losses') and len(history['val_losses']) > 0:
        ax.plot(epochs, history['val_losses'], label='Val Total', color='red', linewidth=2, linestyle='--')
    ax.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Flow Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_flow_losses'], label='Train Flow', color='green', linewidth=2)
    if history.get('val_flow_losses') and len(history['val_flow_losses']) > 0:
        ax.plot(epochs, history['val_flow_losses'], label='Val Flow', color='orange', linewidth=2, linestyle='--')
    ax.set_title('Flow Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reconstruction Loss
    ax = axes[0, 2]
    ax.plot(epochs, history['train_recon_losses'], label='Train Recon', color='purple', linewidth=2)
    if history.get('val_recon_losses') and len(history['val_recon_losses']) > 0:
        ax.plot(epochs, history['val_recon_losses'], label='Val Recon', color='brown', linewidth=2, linestyle='--')
    ax.set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Regularization Loss
    ax = axes[1, 0]
    ax.plot(epochs, history['train_reg_losses'], label='Train Reg', color='cyan', linewidth=2)
    if history.get('val_reg_losses') and len(history['val_reg_losses']) > 0:
        ax.plot(epochs, history['val_reg_losses'], label='Val Reg', color='magenta', linewidth=2, linestyle='--')
    ax.set_title('Regularization Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Chamfer Distance
    ax = axes[1, 1]
    train_chamfer = []
    val_chamfer = []
    val_chamfer_epochs = []
    
    # Compute train chamfer distance from final generation results
    if 'x_gen' in history and 'x_real' in history:
        from src.utils.metrics import chamfer_distance
        x_gen = np.array(history['x_gen'])
        x_real = np.array(history['x_real'])
        
        # Reshape to (num_samples, feature_dim) if needed
        if x_gen.ndim > 2:
            x_gen = x_gen.reshape(-1, x_gen.shape[-1])
        if x_real.ndim > 2:
            x_real = x_real.reshape(-1, x_real.shape[-1])
        
        # Convert to JAX arrays for chamfer_distance
        chamfer_dist = chamfer_distance(jnp.array(x_gen), jnp.array(x_real))
        if np.isfinite(chamfer_dist):
            train_chamfer = [chamfer_dist] * len(epochs)  # Same value for all epochs (final generation)
    
    # Get validation chamfer distances (computed every 10 epochs)
    if 'val_chamfer_distances' in history and len(history['val_chamfer_distances']) > 0:
        val_chamfer = history['val_chamfer_distances']
        # Chamfer distance is computed every 10 epochs, so map to actual epoch numbers
        val_chamfer_epochs = [i * 10 for i in range(len(val_chamfer))]
        # Make sure the last epoch is included
        if len(val_chamfer_epochs) > 0 and val_chamfer_epochs[-1] != epochs[-1]:
            val_chamfer_epochs[-1] = epochs[-1]
    
    if len(train_chamfer) > 0:
        ax.plot(epochs, train_chamfer, label='Train Chamfer Distance', color='darkgreen', linewidth=2)
    if len(val_chamfer) > 0:
        ax.plot(val_chamfer_epochs, val_chamfer, label='Val Chamfer Distance', color='darkred', linewidth=2, linestyle='--', marker='o', markersize=4)
    
    ax.set_title('Chamfer Distance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Chamfer Distance', color='darkgreen')
    ax.tick_params(axis='y', labelcolor='darkgreen')
    if len(train_chamfer) > 0 or len(val_chamfer) > 0:
        ax.legend()
    ax.grid(True, alpha=0.3)
    # Set y-axis to start from 0, but allow upper limit to be determined by data
    if len(train_chamfer) > 0 or len(val_chamfer) > 0:
        all_values = [v for v in train_chamfer if np.isfinite(v)] + [v for v in val_chamfer if np.isfinite(v)]
        if len(all_values) > 0:
            max_val = max(all_values)
            ax.set_ylim([0, max_val * 1.1])  # Add 10% padding at top
    
    # VAE Loss
    ax = axes[1, 2]
    if history.get('train_vae_losses') and len(history['train_vae_losses']) > 0:
        ax.plot(epochs, history['train_vae_losses'], label='Train VAE', color='teal', linewidth=2)
        if history.get('val_vae_losses') and len(history['val_vae_losses']) > 0:
            ax.plot(epochs, history['val_vae_losses'], label='Val VAE', color='coral', linewidth=2, linestyle='--')
        ax.set_title('VAE Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'VAE Loss\n(Not Available)', ha='center', va='center', fontsize=12, alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_trends.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def create_latent_trajectories_plot(
    model,
    params: Any,
    model_type: str,
    unconditional: bool,
    output_dir: str,
    cond_y: Optional[jnp.ndarray] = None,
    num_trajectories: int = 20,
    num_steps: int = 20,
    prng_key: Optional[jr.PRNGKey] = None,
    rng: Optional[jr.PRNGKey] = None
):
    """
    Generate and plot latent z trajectories during integration.
    
    Args:
        model: The model instance
        params: Model parameters
        model_type: Type of model ('flow_matching', 'diffusion', 'ct')
        unconditional: Whether this is unconditional generation
        output_dir: Directory to save the plot
        cond_y: Conditional inputs for conditional generation
        num_trajectories: Number of trajectories to plot
        num_steps: Number of integration steps
        prng_key: PRNG key for generation (optional)
        rng: Alternative PRNG key (optional, for backward compatibility)
    """
    if params is None:
        raise ValueError("Model parameters not provided.")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    n_samples = num_trajectories
    
    # Use rng if provided, otherwise prng_key, otherwise generate new
    if rng is not None:
        key = rng
    elif prng_key is not None:
        key = prng_key
    else:
        key = jr.PRNGKey(42)
    
    integration_method = "midpoint" if model_type == "ct" else "euler"
    
    if unconditional:
        # Use sample() for unconditional generation with batch
        traj = model.sample(
            params,
            key,
            batch_shape=(n_samples,),
            num_steps=num_steps,
            integration_method=integration_method,
            output_type="trajectory"
        )
        # traj shape: [num_steps, n_samples, output_dim]
        # Reshape to [n_samples, num_steps, output_dim] for plotting
        trajectories = np.array(traj).transpose(1, 0, 2)  # [n_samples, num_steps, output_dim]
    else:
        # Use predict() for conditional generation with batch
        if cond_y is None:
            raise ValueError("cond_y must be provided for conditional generation")
        cond_subset = cond_y[:n_samples]
        traj = model.predict(
            params,
            cond_subset,  # Full batch of conditions
            num_steps=num_steps,
            integration_method=integration_method,
            output_type="trajectory",
            prng_key=key  # predict will generate different z_0 for each sample in the batch
        )
        # traj shape: [num_steps, n_samples, output_dim]
        # Reshape to [n_samples, num_steps, output_dim] for plotting
        trajectories = np.array(traj).transpose(1, 0, 2)  # [n_samples, num_steps, output_dim]
    
    # Plot trajectories
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if unconditional:
        # Unconditional: all trajectories same color
        for i in range(n_samples):
            traj = trajectories[i]  # [num_steps, 2]
            ax.plot(traj[:, 0], traj[:, 1], color='purple', alpha=0.6, linewidth=1.5)
            # Mark end point
            ax.scatter(traj[-1, 0], traj[-1, 1], color='purple', s=50, marker='s', edgecolors='black', linewidths=1, zorder=5)
        
        ax.set_title(f'Latent z Trajectories During Integration - Unconditional ({n_samples} samples)', fontsize=14, fontweight='bold')
        legend_elements = [
            Line2D([0], [0], color='purple', linewidth=2, label='Unconditional'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='End', markeredgecolor='black')
        ]
    else:
        # Conditional: color by class
        cond_subset = cond_y[:n_samples]
        class_labels = np.array((cond_subset[:, 0] > 0).astype(int))  # 0 for class -1, 1 for class +1
        class_colors = {0: 'blue', 1: 'red'}  # Discrete colors for each class
        
        for i in range(n_samples):
            traj = trajectories[i]  # [num_steps, 2]
            color = class_colors[int(class_labels[i])]
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.6, linewidth=1.5)
            # Mark end point
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=50, marker='s', edgecolors='black', linewidths=1, zorder=5)
        
        ax.set_title(f'Latent z Trajectories During Integration ({n_samples} samples)', fontsize=14, fontweight='bold')
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, label='Class -1'),
            Line2D([0], [0], color='red', linewidth=2, label='Class +1'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='End', markeredgecolor='black')
        ]
    
    ax.set_xlabel('z[0]', fontsize=12)
    ax.set_ylabel('z[1]', fontsize=12)
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, loc='upper right')
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'latent_trajectories.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

