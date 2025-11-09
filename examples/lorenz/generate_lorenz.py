#!/usr/bin/env python3
"""
Generate Lorenz system dataset for sequence modeling.

This script creates trajectories from the Lorenz system, a classic chaotic dynamical system.
The trajectories are split into sequences for conditional generation tasks.

NOTE: This script should be called from the project root directory:
    python examples/lorenz/generate_lorenz.py [args]

All paths (output_dir) are relative to the project root directory.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
from src.utils.ode_integration import integrate_ode


def lorenz_vector_field(params, z, x, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Lorenz system vector field for JAX ODE integration.
    
    Args:
        params: Model parameters (unused for Lorenz, but required by interface)
        z: Current state [batch_size, 3] where each row is [x, y, z]
        x: Input data (unused for Lorenz, but required by interface)
        t: Current time [batch_size] or scalar
        sigma: Lorenz parameter (default: 10.0)
        rho: Lorenz parameter (default: 28.0)
        beta: Lorenz parameter (default: 8.0/3.0)
        
    Returns:
        Derivative [batch_size, 3] where each row is [dx/dt, dy/dt, dz/dt]
    """
    x_coord = z[..., 0]
    y_coord = z[..., 1]
    z_coord = z[..., 2]
    
    dx_dt = sigma * (y_coord - x_coord)
    dy_dt = x_coord * (rho - z_coord) - y_coord
    dz_dt = x_coord * y_coord - beta * z_coord
    
    return jnp.stack([dx_dt, dy_dt, dz_dt], axis=-1)


def generate_lorenz_trajectory(
    initial_state: np.ndarray,
    t_span: tuple,
    n_points: int,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0/3.0,
    noise: float = 0.0
) -> np.ndarray:
    """
    Generate a single Lorenz trajectory using JAX ODE integration.
    
    Args:
        initial_state: Initial state [x0, y0, z0]
        t_span: Time span (t_start, t_end)
        n_points: Number of points in trajectory
        sigma: Lorenz parameter
        rho: Lorenz parameter
        beta: Lorenz parameter
        noise: Standard deviation of Gaussian noise to add (default: 0.0)
        
    Returns:
        Trajectory array [n_points, 3]
    """
    # Convert to JAX arrays
    z0 = jnp.array(initial_state[None, :])  # Shape: [1, 3] for batch dimension
    x = None  # No input for Lorenz system
    
    # Create vector field with parameters
    def vector_field(params, z, x, t):
        return lorenz_vector_field(params, z, x, t, sigma=sigma, rho=rho, beta=beta)
    
    # Use RK4 with fine time steps (n_points * 4 for accuracy)
    num_steps = (n_points - 1) * 4  # More steps for accuracy
    
    # Integrate ODE
    trajectory = integrate_ode(
        vector_field=vector_field,
        params={},  # No parameters needed for Lorenz
        z0=z0,
        x=x,
        time_span=t_span,
        num_steps=num_steps,
        method="rk4",
        output_type="trajectory"
    )
    
    # trajectory shape: [num_steps+1, 1, 3]
    trajectory = trajectory[:, 0, :]  # Remove batch dimension: [num_steps+1, 3]
    
    # Downsample to n_points
    indices = np.linspace(0, len(trajectory) - 1, n_points, dtype=int)
    trajectory = np.array(trajectory[indices])
    
    # Add noise if specified
    if noise > 0:
        trajectory += np.random.normal(0, noise, trajectory.shape)
    
    return trajectory


def generate_lorenz_dataset(
    n_trajectories: int = 1000,
    trajectory_length: int = 200,
    t_span: tuple = (0.0, 20.0),
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0/3.0,
    noise: float = 0.0,
    initial_state_range: tuple = ((-20, 20), (-20, 20), (0, 50)),
    seed: int = 42
) -> np.ndarray:
    """
    Generate multiple Lorenz trajectories.
    
    Args:
        n_trajectories: Number of trajectories to generate
        trajectory_length: Length of each trajectory
        t_span: Time span (t_start, t_end)
        sigma: Lorenz parameter
        rho: Lorenz parameter
        beta: Lorenz parameter
        noise: Standard deviation of Gaussian noise
        initial_state_range: Range for initial states ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        seed: Random seed
        
    Returns:
        Array of trajectories [n_trajectories, trajectory_length, 3]
    """
    np.random.seed(seed)
    
    trajectories = []
    for i in range(n_trajectories):
        # Sample random initial state
        x0 = np.random.uniform(initial_state_range[0][0], initial_state_range[0][1])
        y0 = np.random.uniform(initial_state_range[1][0], initial_state_range[1][1])
        z0 = np.random.uniform(initial_state_range[2][0], initial_state_range[2][1])
        initial_state = np.array([x0, y0, z0])
        
        trajectory = generate_lorenz_trajectory(
            initial_state,
            t_span,
            trajectory_length,
            sigma=sigma,
            rho=rho,
            beta=beta,
            noise=noise
        )
        trajectories.append(trajectory)
    
    return np.array(trajectories)


def split_sequences(
    trajectories: np.ndarray,
    input_seq_len: int = 20,
    output_seq_len: int = 20,
    stride: int = 1
) -> tuple:
    """
    Split trajectories into input-output sequence pairs.
    
    Args:
        trajectories: Array of trajectories [n_trajectories, traj_len, 3]
        input_seq_len: Length of input sequences
        output_seq_len: Length of output sequences
        stride: Stride for sliding window (default: 1)
        
    Returns:
        Tuple of (x_sequences, y_sequences) where:
        - x_sequences: Input sequences [n_sequences, input_seq_len, 3]
        - y_sequences: Output sequences [n_sequences, output_seq_len, 3]
    """
    x_sequences = []
    y_sequences = []
    
    for traj in trajectories:
        traj_len = traj.shape[0]
        total_len = input_seq_len + output_seq_len
        
        # Generate sequences using sliding window
        for start in range(0, traj_len - total_len + 1, stride):
            x_seq = traj[start:start + input_seq_len]
            y_seq = traj[start + input_seq_len:start + total_len]
            x_sequences.append(x_seq)
            y_sequences.append(y_seq)
    
    return np.array(x_sequences), np.array(y_sequences)


def visualize_trajectories(trajectories: np.ndarray, save_path: str = None, n_samples: int = 4):
    """
    Visualize sample Lorenz trajectories.
    
    Args:
        trajectories: Array of trajectories [n_trajectories, traj_len, 3]
        save_path: Path to save the plot (optional)
        n_samples: Number of trajectories to plot (default: 4)
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 3D line plot
    ax1 = fig.add_subplot(131, projection='3d')
    for i in range(min(n_samples, len(trajectories))):
        traj = trajectories[i]
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.7, linewidth=1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Lorenz Trajectories (3D)')
    
    # X-Y projection line plot
    ax2 = fig.add_subplot(132)
    for i in range(min(n_samples, len(trajectories))):
        traj = trajectories[i]
        ax2.plot(traj[:, 0], traj[:, 1], alpha=0.7, linewidth=1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('X-Y Projection')
    ax2.grid(True, alpha=0.3)
    
    # Time series line plot
    ax3 = fig.add_subplot(133)
    for i in range(min(n_samples, len(trajectories))):
        traj = trajectories[i]
        t = np.arange(len(traj))
        ax3.plot(t, traj[:, 0], label=f'X (traj {i+1})', alpha=0.7, linewidth=1)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('X Value')
    ax3.set_title('X Component Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate Lorenz system dataset for sequence modeling'
    )
    
    parser.add_argument('--n_trajectories', type=int, default=1000,
                       help='Number of trajectories to generate (default: 1000)')
    parser.add_argument('--trajectory_length', type=int, default=200,
                       help='Length of each trajectory (default: 200)')
    parser.add_argument('--input_seq_len', type=int, default=20,
                       help='Length of input sequences (default: 20)')
    parser.add_argument('--output_seq_len', type=int, default=20,
                       help='Length of output sequences (default: 20)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for sliding window (default: 1)')
    parser.add_argument('--sigma', type=float, default=10.0,
                       help='Lorenz parameter sigma (default: 10.0)')
    parser.add_argument('--rho', type=float, default=28.0,
                       help='Lorenz parameter rho (default: 28.0)')
    parser.add_argument('--beta', type=float, default=8.0/3.0,
                       help='Lorenz parameter beta (default: 8.0/3.0)')
    parser.add_argument('--noise', type=float, default=0.0,
                       help='Standard deviation of Gaussian noise (default: 0.0)')
    parser.add_argument('--t_span', type=float, nargs=2, default=[0.0, 20.0],
                       help='Time span [t_start, t_end] (default: [0.0, 20.0])')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory (default: ./data)')
    parser.add_argument('--filename', type=str, default='lorenz.pkl',
                       help='Output filename (default: lorenz.pkl)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data for training (default: 0.8)')
    parser.add_argument('--visualize', action='store_true',
                       help='Display visualization')
    parser.add_argument('--save_plot', action='store_true',
                       help='Save visualization plot')
    
    args = parser.parse_args()
    
    # Generate trajectories
    print(f"Generating {args.n_trajectories} Lorenz trajectories...")
    trajectories = generate_lorenz_dataset(
        n_trajectories=args.n_trajectories,
        trajectory_length=args.trajectory_length,
        t_span=tuple(args.t_span),
        sigma=args.sigma,
        rho=args.rho,
        beta=args.beta,
        noise=args.noise,
        seed=args.seed
    )
    print(f"Generated trajectories shape: {trajectories.shape}")
    
    # Split into sequences
    print(f"Splitting trajectories into sequences (input_len={args.input_seq_len}, output_len={args.output_seq_len})...")
    x_sequences, y_sequences = split_sequences(
        trajectories,
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        stride=args.stride
    )
    print(f"Generated sequences:")
    print(f"  X sequences: {x_sequences.shape}")
    print(f"  Y sequences: {y_sequences.shape}")
    
    # Shuffle and split into train/val
    n_total = len(x_sequences)
    
    # Calculate target to get exactly 1024 training sequences
    target_train = 1024
    target_total = int(target_train / args.train_ratio)  # Need 1280 total for 1024 train (80/20 split)
    
    # Limit to target_total if we have more
    if n_total > target_total:
        n_total = target_total
        x_sequences = x_sequences[:target_total]
        y_sequences = y_sequences[:target_total]
    
    indices = np.random.RandomState(args.seed).permutation(n_total)
    n_train = int(args.train_ratio * n_total)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    x_train = x_sequences[train_indices]
    y_train = y_sequences[train_indices]
    x_val = x_sequences[val_indices]
    y_val = y_sequences[val_indices]
    
    print(f"Train/Val split:")
    print(f"  Train: {len(x_train)} sequences")
    print(f"  Val: {len(x_val)} sequences")
    
    # Scale data to [0, 10] using min-max normalization per dimension
    print("Scaling data to [0, 10] range per dimension...")
    # Combine all data to compute global min/max per dimension
    all_x = np.concatenate([x_train, x_val], axis=0)
    all_y = np.concatenate([y_train, y_val], axis=0)
    all_data = np.concatenate([all_x, all_y], axis=0)  # Shape: [n_total*2, seq_len, 3]
    
    # Compute min and max for each dimension (axis 2, the 3D coordinates)
    min_vals = all_data.min(axis=(0, 1))  # Shape: (3,)
    max_vals = all_data.max(axis=(0, 1))  # Shape: (3,)
    
    print(f"  Min values per dimension: {min_vals}")
    print(f"  Max values per dimension: {max_vals}")
    
    # Normalize to [0, 1] then scale to [0, 10]
    def scale_sequences(sequences, min_vals, max_vals):
        """Scale sequences using min-max normalization per dimension."""
        # Normalize: (value - min) / (max - min)
        # Avoid division by zero
        ranges = max_vals - min_vals
        ranges = np.where(ranges == 0, 1.0, ranges)  # Replace zeros with 1.0
        
        # Normalize to [0, 1] then scale to [0, 10]
        normalized = (sequences - min_vals) / ranges
        scaled = normalized * 10.0
        return scaled
    
    x_train = scale_sequences(x_train, min_vals, max_vals)
    y_train = scale_sequences(y_train, min_vals, max_vals)
    x_val = scale_sequences(x_val, min_vals, max_vals)
    y_val = scale_sequences(y_val, min_vals, max_vals)
    
    print(f"  Scaled data range: [{x_train.min():.4f}, {x_train.max():.4f}]")
    
    # Save data
    output_path = Path(args.output_dir) / args.filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'train': {
            'x': x_train,
            'y': y_train
        },
        'val': {
            'x': x_val,
            'y': y_val
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved data to {output_path}")
    
    # Visualization
    if args.visualize or args.save_plot:
        plot_path = None
        if args.save_plot:
            plot_path = Path(args.output_dir) / 'lorenz_visualization.png'
        visualize_trajectories(trajectories, save_path=plot_path, n_samples=4)


if __name__ == '__main__':
    main()

