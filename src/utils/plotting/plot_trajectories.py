"""
Trajectory plotting utilities for all model types.

This module provides comprehensive trajectory analysis plots that can be used
by any model trainer to visualize model behavior, trajectories, and dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def create_trajectory_diagnostic_plot(
    trajectories: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    model_name: str = "Model",
    num_samples: int = 10
) -> None:
    """
    Create a diagnostic plot showing model trajectories for different data points.
    
    Args:
        trajectories: Array of shape (num_samples, num_steps, output_dim) containing trajectories
        targets: Array of shape (num_samples, output_dim) containing target endpoints
        output_path: Path to save the plot
        model_name: Name of the model for the title
        num_samples: Number of sample trajectories to plot
    """
    num_steps, output_dim = trajectories.shape[1], trajectories.shape[2]
    
    # Create time points for the x-axis (0.0 to 1.0)
    time_points = np.linspace(0.0, 1.0, num_steps)
    
    # Create figure with subplots for each output dimension
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'{model_name} - Trajectory Diagnostics', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot trajectories for each output dimension
    for dim in range(output_dim):
        ax = axes_flat[dim]
        
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
        ax.set_title(f'Dimension {dim} Trajectories')
        ax.grid(True, alpha=0.3)
        
        # Add legend for the first subplot only
        if dim == 0:
            ax.plot([], [], 'b-', alpha=0.7, linewidth=1.5, label='Model Trajectory')
            ax.scatter([], [], c='red', s=50, alpha=0.8, label='Target Endpoint')
            ax.legend()
    
    # Hide unused subplots
    for dim in range(output_dim, 9):
        axes_flat[dim].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory diagnostic plot saved to: {output_path}")


def create_model_output_trajectory_plot(
    model,
    params: Dict[str, Any],
    x_sample: np.ndarray,
    target_sample: np.ndarray,
    output_path: str,
    model_name: str = "Model",
    num_steps: int = 20,
    num_samples: int = 10
) -> None:
    """
    Create a plot showing model output as a function of time by manual integration.
    
    This function is specifically designed for models that have a predict method
    that can return trajectories and a _get_model_output method for raw neural network outputs.
    
    Args:
        model: The trained model instance (must have predict and _get_model_output methods)
        params: Model parameters
        x_sample: Input x values [num_samples, x_dim]
        target_sample: Target y values [num_samples, y_dim]
        output_path: Path to save the plot
        model_name: Name of the model for the title
        num_steps: Number of time steps for integration
        num_samples: Number of sample trajectories to plot
    """
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jr
    except ImportError:
        print("Warning: JAX not available. This function requires JAX for NoProp models.")
        return
    
    # Convert to JAX arrays
    x_jax = jnp.array(x_sample[:num_samples])
    target_jax = jnp.array(target_sample[:num_samples])
    
    # Time points for integration
    t_points = jnp.linspace(0.0, 1.0, num_steps + 1)
    
    # Get z(t) trajectories from the predict routine
    print("Getting z(t) trajectories from predict routine...")
    z_trajectories = model.predict(
        params, 
        x_jax, 
        num_steps=num_steps,
        output_type="trajectory",
        key=jr.PRNGKey(42)
    )  # Shape: [num_steps, num_samples, z_dim]
    
    print(f"z_trajectories shape: {z_trajectories.shape}")
    print(f"z_trajectories[0] (t=0): mean={jnp.mean(z_trajectories[0]):.6f}, std={jnp.std(z_trajectories[0]):.6f}")
    print(f"z_trajectories[-1] (t=1): mean={jnp.mean(z_trajectories[-1]):.6f}, std={jnp.std(z_trajectories[-1]):.6f}")
    print(f"z_trajectories range: [{jnp.min(z_trajectories):.6f}, {jnp.max(z_trajectories):.6f}]")
    
    # Store model outputs at each time step
    model_outputs = []
    
    # Get model output for each z(t) from the trajectories
    for i, t in enumerate(t_points):
        t_batch = jnp.full((num_samples,), t)
        z_t = z_trajectories[i]  # Get z(t) from the trajectory
        
        # Get CRN model output at current time (raw neural network output)
        # Use the same pattern as in compute_loss method
        model_output = model.apply(
            params, 
            z_t, 
            x_jax, 
            t_batch, 
            training=False, 
            method=model._get_model_output,
            rngs={'dropout': jr.PRNGKey(42)}  # Use a fixed key for deterministic output
        )
        
        # Debug: print statistics for first few time steps
        if i < 3:
            print(f"t={t:.3f}: z_t mean={jnp.mean(z_t):.4f}, std={jnp.std(z_t):.4f}")
            print(f"t={t:.3f}: model_output mean={jnp.mean(model_output):.4f}, std={jnp.std(model_output):.4f}")
            print(f"t={t:.3f}: target mean={jnp.mean(target_jax):.4f}, std={jnp.std(target_jax):.4f}")
            print(f"t={t:.3f}: model_output vs target MSE={jnp.mean((model_output - target_jax)**2):.6f}")
            print("---")
        
        # Store model output
        model_outputs.append(model_output)
    
    # Convert to numpy for plotting
    model_outputs = np.array(model_outputs)  # [num_steps, num_samples, output_dim]
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'{model_name} - CRN Output Trajectories', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    output_dim = model_outputs.shape[2]
    
    # Plot model outputs for each dimension
    for dim in range(output_dim):
        ax = axes_flat[dim]
        
        # Plot model outputs for each sample
        for sample_idx in range(num_samples):
            model_output_traj = model_outputs[:, sample_idx, dim]
            target_val = target_jax[sample_idx, dim]
            
            # Plot the model output trajectory
            ax.plot(t_points, model_output_traj, alpha=0.7, linewidth=1.5)
            
            # Mark the target value as a dot
            ax.scatter(1.0, target_val, color='red', s=50, alpha=0.8, zorder=5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'CRN Output Dimension {dim}')
        ax.set_title(f'Dimension {dim} CRN Outputs')
        ax.grid(True, alpha=0.3)
        
        # Add legend for the first subplot only
        if dim == 0:
            ax.plot([], [], 'b-', alpha=0.7, linewidth=1.5, label='CRN Output')
            ax.scatter([], [], c='red', s=50, alpha=0.8, label='Target Value')
            ax.legend()
    
    # Hide unused subplots
    for dim in range(output_dim, 9):
        axes_flat[dim].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model output trajectory plot saved to: {output_path}")


def create_dzdt_trajectory_plot(
    model,
    params: Dict[str, Any],
    x_sample: np.ndarray,
    target_sample: np.ndarray,
    output_path: str,
    model_name: str = "Model",
    num_steps: int = 20,
    num_samples: int = 10
) -> None:
    """
    Create a plot showing dz_dt (vector field) as a function of time using z(t) from predict trajectories.
    
    This function is specifically designed for models that have a predict method
    that can return trajectories and a dz_dt method for computing the vector field.
    
    Args:
        model: The trained model instance (must have predict and dz_dt methods)
        params: Model parameters
        x_sample: Input x values [num_samples, x_dim]
        target_sample: Target y values [num_samples, y_dim]
        output_path: Path to save the plot
        model_name: Name of the model for the title
        num_steps: Number of time steps for integration
        num_samples: Number of sample trajectories to plot
    """
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jr
    except ImportError:
        print("Warning: JAX not available. This function requires JAX for NoProp models.")
        return
    
    # Convert to JAX arrays
    x_jax = jnp.array(x_sample[:num_samples])
    target_jax = jnp.array(target_sample[:num_samples])
    
    # Time points for integration
    t_points = jnp.linspace(0.0, 1.0, num_steps + 1)
    
    # Get z(t) trajectories from the predict routine
    print("Getting z(t) trajectories from predict routine...")
    z_trajectories = model.predict(
        params, 
        x_jax, 
        num_steps=num_steps,
        output_type="trajectory",
        key=jr.PRNGKey(42)
    )  # Shape: [num_steps, num_samples, z_dim]
    
    print(f"z_trajectories shape: {z_trajectories.shape}")
    print(f"z_trajectories[0] (t=0): mean={jnp.mean(z_trajectories[0]):.6f}, std={jnp.std(z_trajectories[0]):.6f}")
    print(f"z_trajectories[-1] (t=1): mean={jnp.mean(z_trajectories[-1]):.6f}, std={jnp.std(z_trajectories[-1]):.6f}")
    
    # Store dz_dt values at each time step
    dzdt_trajectories = []
    
    # Get dz_dt for each z(t) from the trajectories
    for i, t in enumerate(t_points):
        t_batch = jnp.full((num_samples,), t)
        z_t = z_trajectories[i]  # Get z(t) from the trajectory
        
        # Get dz_dt at current time
        dz_dt = model.dz_dt(params, z_t, x_jax, t_batch)
        
        # Debug: print statistics for first few time steps
        if i < 3:
            print(f"t={t:.3f}: z_t mean={jnp.mean(z_t):.4f}, std={jnp.std(z_t):.4f}")
            print(f"t={t:.3f}: dz_dt mean={jnp.mean(dz_dt):.4f}, std={jnp.std(dz_dt):.4f}")
            print(f"t={t:.3f}: dz_dt range=[{jnp.min(dz_dt):.4f}, {jnp.max(dz_dt):.4f}]")
            print("---")
        
        # Store dz_dt
        dzdt_trajectories.append(dz_dt)
    
    # Convert to numpy for plotting
    dzdt_trajectories = np.array(dzdt_trajectories)  # [num_steps, num_samples, z_dim]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - dz/dt Vector Field Trajectories', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    z_dim = dzdt_trajectories.shape[2]
    
    # Plot dz_dt for each dimension
    for dim in range(min(z_dim, 4)):  # Plot up to 4 dimensions
        ax = axes_flat[dim]
        
        # Plot dz_dt trajectories for each sample
        for sample_idx in range(num_samples):
            dzdt_traj = dzdt_trajectories[:, sample_idx, dim]
            
            # Plot the dz_dt trajectory
            ax.plot(t_points, dzdt_traj, alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'dz/dt Dimension {dim}')
        ax.set_title(f'Dimension {dim} Vector Field')
        ax.grid(True, alpha=0.3)
        
        # Add legend for the first subplot only
        if dim == 0:
            ax.plot([], [], 'b-', alpha=0.7, linewidth=1.5, label='dz/dt')
            ax.legend()
    
    # Hide unused subplots
    for dim in range(z_dim, 4):
        axes_flat[dim].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"dz/dt trajectory plot saved to: {output_path}")


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
