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
    unconditional: bool = False
):
    """
    Create generation comparison plot showing real vs generated samples.
    
    Args:
        x_real: Real samples [N, 2]
        y_labels: Conditional labels [N, ...] or None for unconditional
        x_gen: Generated samples [N, 2]
        output_dir: Directory to save the plot
        unconditional: Whether this is unconditional generation
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    if y_labels is not None and not unconditional:
        # Conditional generation: color by class labels
        # Convert one-hot to class indices for coloring
        if len(y_labels.shape) == 2 and y_labels.shape[1] > 1:
            # One-hot encoded: use argmax to get class indices
            class_indices = np.argmax(y_labels, axis=1)
        else:
            # Integer labels: use directly
            class_indices = y_labels.flatten().astype(int)
        
        # Real
        ax[0].scatter(x_real[:, 0], x_real[:, 1], c=class_indices, s=6, cmap='coolwarm', alpha=0.8)
        ax[0].set_title('Real samples (x)')
        ax[0].set_aspect('equal', 'box')
        # Generated
        ax[1].scatter(x_gen[:, 0], x_gen[:, 1], c=class_indices, s=6, cmap='coolwarm', alpha=0.8)
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
    
    # Chamfer Distance (if available, otherwise leave empty)
    ax = axes[1, 1]
    if history.get('val_chamfer_distances') and len(history['val_chamfer_distances']) > 0:
        chamfer_epochs = range(len(history['val_chamfer_distances']))
        ax.plot(chamfer_epochs, history['val_chamfer_distances'], label='Val Chamfer', color='darkorange', linewidth=2, linestyle='--')
        ax.set_title('Chamfer Distance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Chamfer Distance\n(Not Available)', ha='center', va='center', fontsize=12, alpha=0.5)
    
    # Leave last panel empty or add additional metric if needed
    ax = axes[1, 2]
    ax.axis('off')
    
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
    
    # Split keys for each trajectory
    prng_keys = jr.split(key, n_samples)
    trajectories = []
    
    integration_method = "midpoint" if model_type == "ct" else "euler"
    
    for i in range(n_samples):
        if unconditional:
            # Use sample() for unconditional generation
            traj = model.sample(
                params,
                prng_keys[i],
                batch_shape=(1,),
                num_steps=num_steps,
                integration_method=integration_method,
                output_type="trajectory"
            )
        else:
            # Use predict() for conditional generation
            if cond_y is None:
                raise ValueError("cond_y must be provided for conditional generation")
            cond_subset = cond_y[:n_samples]
            traj = model.predict(
                params,
                cond_subset[i:i+1],  # Single condition with batch dim
                num_steps=num_steps,
                integration_method=integration_method,
                output_type="trajectory",
                prng_key=prng_keys[i]
            )
        
        # Remove batch dimension: [num_steps, 1, output_dim] -> [num_steps, output_dim]
        if traj.ndim == 3:
            traj = traj[:, 0, :]
        trajectories.append(np.array(traj))
    
    trajectories = np.array(trajectories)  # [n_samples, num_steps, output_dim]
    
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

