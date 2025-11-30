"""
Trajectory plotting utilities for all model types.

This module provides comprehensive trajectory analysis plots that can be used
by any model trainer to visualize model behavior, trajectories, and dynamics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, Any, Optional
from pathlib import Path
import jax.numpy as jnp
import jax.random as jr


def create_simple_trajectory_plot(
    trajectories: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    model_name: str = "Model",
    num_samples: int = 5
) -> None:
    """
    Create a simple trajectory plot for quick visualization.
    
    Args:
        trajectories: Array of shape (num_samples, num_steps, output_dim) containing trajectories
        targets: Array of shape (num_samples, output_dim) containing target endpoints
        output_path: Path to save the plot
        model_name: Name of the model for the title
        num_samples: Number of sample trajectories to plot
    """
    num_steps, output_dim = trajectories.shape[1], trajectories.shape[2]
    time_points = np.linspace(0.0, 1.0, num_steps)
    
    # Create figure
    fig, axes = plt.subplots(1, min(output_dim, 3), figsize=(5 * min(output_dim, 3), 4))
    if output_dim == 1:
        axes = [axes]
    
    fig.suptitle(f'{model_name} - Simple Trajectory Plot', fontsize=14, fontweight='bold')
    
    # Plot trajectories for each output dimension
    for dim in range(min(output_dim, 3)):
        ax = axes[dim]
        
        # Plot trajectories for each sample
        for sample_idx in range(min(num_samples, trajectories.shape[0])):
            trajectory = trajectories[sample_idx, :, dim]
            target = targets[sample_idx, dim]
            
            # Plot the trajectory against time
            ax.plot(time_points, trajectory, alpha=0.7, linewidth=1.5)
            
            # Mark the target endpoint as a dot at t=1.0
            ax.scatter(1.0, target, color='red', s=50, alpha=0.8, zorder=5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Output Dimension {dim}')
        ax.set_title(f'Dimension {dim}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simple trajectory plot saved to: {output_path}")


def plot_latent_trajectories(
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
        # Use predict_latent() for conditional generation with batch (returns latent trajectories directly)
        if cond_y is None:
            raise ValueError("cond_y must be provided for conditional generation")
        cond_subset = cond_y[:n_samples]
        traj = model.predict_latent(
            params,
            cond_subset,  # Full batch of conditions
            num_steps=num_steps,
            integration_method=integration_method,
            output_type="trajectory",
            prng_key=key  # predict_latent will generate different z_0 for each sample in the batch
        )
        # traj shape: [num_steps, n_samples, latent_dim]
        # Reshape to [n_samples, num_steps, latent_dim] for plotting
        trajectories = np.array(traj).transpose(1, 0, 2)  # [n_samples, num_steps, latent_dim]
    
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
    
    print(f"✓ Saved latent trajectories plot to {os.path.join(output_dir, 'latent_trajectories.png')}")
