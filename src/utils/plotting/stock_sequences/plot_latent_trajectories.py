"""
Plot latent z trajectories during integration for sequence data.
"""
import os
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from typing import Optional, Tuple
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


def plot_latent_trajectories(
    model,
    params: dict,
    model_type: str,
    unconditional: bool,
    output_dir: str,
    cond_x: Optional[jnp.ndarray] = None,
    num_trajectories: int = 20,
    num_steps: int = 20,
    prng_key: Optional[jr.PRNGKey] = None,
    rng: Optional[jr.PRNGKey] = None
):
    """
    Generate and plot latent z trajectories during integration for sequence data.
    
    Args:
        model: Model instance (VAE_flow)
        params: Model parameters
        model_type: Type of model ('flow_matching', 'diffusion', 'ct')
        unconditional: Whether model is unconditional
        output_dir: Directory to save the plot
        cond_x: Conditional input for conditional generation
        num_trajectories: Number of trajectories to plot
        num_steps: Number of integration steps
        prng_key: Optional PRNG key for generation
        rng: Optional RNG state (will be split if prng_key is None)
    """
    if params is None:
        raise ValueError("Model not initialized. Call initialize() first.")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    n_samples = num_trajectories
    
    # Generate trajectories
    if prng_key is None:
        if rng is None:
            raise ValueError("Either prng_key or rng must be provided")
        rng, prng_key = jr.split(rng)
    
    # Split keys for each trajectory
    prng_keys = jr.split(prng_key, n_samples)
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
            if cond_x is None:
                raise ValueError("cond_x must be provided for conditional generation")
            cond_subset = cond_x[:n_samples]
            traj = model.predict(
                params,
                cond_subset[i:i+1],  # Single condition with batch dim
                num_steps=num_steps,
                integration_method=integration_method,
                output_type="trajectory"
            )
        
        # For sequences, trajectories are [num_steps, 1, seq_len, embed_dim]
        # We'll flatten to show sequence evolution: [num_steps, seq_len * embed_dim]
        if traj.ndim >= 4:
            # Flatten sequence dimensions: [num_steps, 1, seq_len, embed_dim] -> [num_steps, seq_len * embed_dim]
            traj = traj.reshape(traj.shape[0], -1)
        elif traj.ndim == 3:
            traj = traj[:, 0, :]  # Remove batch dim
        trajectories.append(np.array(traj))
    
    trajectories = np.array(trajectories)  # [n_samples, num_steps, flattened_dim]
    
    # Plot trajectories - show first 2 principal components or first 2 dims
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use PCA or just first 2 dims for visualization
    # For simplicity, just use first 2 dimensions of flattened space
    for i in range(n_samples):
        traj = trajectories[i]  # [num_steps, flattened_dim]
        if traj.shape[1] >= 2:
            ax.plot(traj[:, 0], traj[:, 1], color='purple', alpha=0.6, linewidth=1.5)
            # Mark end point
            ax.scatter(traj[-1, 0], traj[-1, 1], color='purple', s=50, marker='s', edgecolors='black', linewidths=1, zorder=5)
    
    ax.set_title(f'Latent z Trajectories During Integration - Sequences ({n_samples} samples)', fontsize=14, fontweight='bold')
    ax.set_xlabel('z[0]', fontsize=12)
    ax.set_ylabel('z[1]', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    legend_elements = [
        Line2D([0], [0], color='purple', linewidth=2, label='Trajectory'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='End', markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'latent_trajectories.png')
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved latent trajectory plot to {plot_path}")

