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


def create_patched_trajectories(
    trajectories: np.ndarray,
    window_length: int = 4
) -> np.ndarray:
    """
    Create patched trajectories by flattening non-overlapping time windows.
    
    For each trajectory of shape (traj_len, features), creates a patched version
    of shape (traj_len // window_length, window_length * features) by:
    1. Taking non-overlapping windows of length window_length
    2. Flattening each window along the feature dimension
    
    Args:
        trajectories: Array of trajectories [n_trajectories, traj_len, features]
        window_length: Length of each time window to flatten (default: 4)
        
    Returns:
        Array of patched trajectories [n_trajectories, traj_len // window_length, window_length * features]
    """
    n_trajectories, traj_len, features = trajectories.shape
    
    # Check that traj_len is divisible by window_length
    if traj_len % window_length != 0:
        raise ValueError(
            f"traj_len ({traj_len}) must be divisible by window_length ({window_length})"
        )
    
    num_patches = traj_len // window_length
    patched_trajectories = []
    
    for traj in trajectories:
        # Reshape to (num_patches, window_length, features)
        windows = traj.reshape(num_patches, window_length, features)
        # Flatten each window: (num_patches, window_length * features)
        patched = windows.reshape(num_patches, window_length * features)
        patched_trajectories.append(patched)
    
    return np.array(patched_trajectories)


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
    
    parser.add_argument('--n_trajectories', type=int, default=320,
                       help='Number of trajectories to generate (default: 320)')
    parser.add_argument('--trajectory_length', type=int, default=1024,
                       help='Length of each trajectory (default: 1024)')
    parser.add_argument('--input_seq_len', type=int, default=6,
                       help='Length of input sequences (default: 6)')
    parser.add_argument('--output_seq_len', type=int, default=6,
                       help='Length of output sequences (default: 6)')
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
    parser.add_argument('--t_span', type=float, nargs=2, default=[0.0, 100.0],
                       help='Time span [t_start, t_end] (default: [0.0, 100.0])')
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
    print(f"Generating {args.n_trajectories} Lorenz trajectories of length {args.trajectory_length}...")
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
    
    # Center data by subtracting the mean
    print("Centering data (subtracting mean)...")
    # Compute mean for each dimension across all trajectories and time steps
    mean_vals = trajectories.mean(axis=(0, 1))  # Shape: (3,)
    
    print(f"  Mean values per dimension: {mean_vals}")
    print(f"  Original data range: [{trajectories.min():.4f}, {trajectories.max():.4f}]")
    
    # Center by subtracting mean
    trajectories = trajectories - mean_vals
    
    print(f"  Centered data range: [{trajectories.min():.4f}, {trajectories.max():.4f}]")
    print(f"  Centered data mean: {trajectories.mean(axis=(0, 1))}")
    
    # Create regression dataset: predict next N time points from previous N
    print(f"\nCreating regression dataset (predict next {args.output_seq_len} from previous {args.input_seq_len})...")
    x_sequences, y_sequences = split_sequences(
        trajectories,
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        stride=args.stride
    )
    
    print(f"  x_sequences shape: {x_sequences.shape}  # (num_batches, {args.input_seq_len}, 3)")
    print(f"  y_sequences shape: {y_sequences.shape}  # (num_batches, {args.output_seq_len}, 3)")
    print(f"  Total number of samples: {len(x_sequences)}")
    
    # Split into train/val if train_ratio < 1.0
    if args.train_ratio < 1.0:
        n_train = int(len(x_sequences) * args.train_ratio)
        x_train = x_sequences[:n_train]
        y_train = y_sequences[:n_train]
        x_val = x_sequences[n_train:]
        y_val = y_sequences[n_train:]
        print(f"  Train samples: {len(x_train)}")
        print(f"  Val samples: {len(x_val)}")
    else:
        x_train = x_sequences
        y_train = y_sequences
        x_val = None
        y_val = None
    
    # Save regression dataset
    print("\nSaving regression dataset...")
    output_path = Path(args.output_dir) / args.filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'train': {
            'x': x_train,  # (num_batches, 6, 3)
            'y': y_train   # (num_batches, 6, 3)
        },
        'val': {
            'x': x_val,    # (num_val_batches, 6, 3) or None
            'y': y_val     # (num_val_batches, 6, 3) or None
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved data to {output_path}")
    print(f"  Train x shape: {x_train.shape}")
    print(f"  Train y shape: {y_train.shape}")
    if x_val is not None:
        print(f"  Val x shape: {x_val.shape}")
        print(f"  Val y shape: {y_val.shape}")
    
    # Create patched trajectories: 320 trajectories of shape (256, 12)
    # Each window of length 4 from (1024, 3) gets flattened to (12,)
    # 1024 / 4 = 256 windows per trajectory
    print("\nCreating patched trajectories...")
    print(f"  Window length: 4 (flattening 4 timesteps)")
    print(f"  Original: (1024, 3) -> Patched: (256, 12)")
    
    patched_trajectories = create_patched_trajectories(trajectories, window_length=4)
    print(f"  Patched trajectories shape: {patched_trajectories.shape}")
    
    # Save lorenz_patches.pkl: 320 trajectories of shape (256, 12)
    print("\nSaving lorenz_patches.pkl (320 trajectories of shape (256, 12))...")
    patches_output_path = Path(args.output_dir) / 'lorenz_patches.pkl'
    
    patches_data = {
        'train': {
            'x': None,
            'y': patched_trajectories  # (320, 256, 12)
        },
        'val': {
            'x': None,
            'y': None  # No split for this dataset
        }
    }
    
    with open(patches_output_path, 'wb') as f:
        pickle.dump(patches_data, f)
    print(f"Saved patched data to {patches_output_path}")
    print(f"  Shape: {patched_trajectories.shape}")
    
    # Visualization
    if args.visualize or args.save_plot:
        plot_path = None
        if args.save_plot:
            plot_path = Path(args.output_dir) / 'lorenz_visualization.png'
        visualize_trajectories(trajectories, save_path=plot_path, n_samples=4)


if __name__ == '__main__':
    main()

