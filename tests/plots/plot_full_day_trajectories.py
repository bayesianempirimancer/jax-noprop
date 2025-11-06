#!/usr/bin/env python3
"""
Plot full-day trajectories for price and volume (post-processing).

Creates spaghetti plots showing:
- Price trajectories (feature 0) for 40 days
- Volume trajectories (feature 1) for 40 days
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_full_day_trajectories(data_path: str = 'data/stock_sequences_full_day_2d.pkl',
                                num_days: int = 40,
                                output_dir: str = 'artifacts'):
    """
    Plot full-day trajectories for price and volume.
    
    Args:
        data_path: Path to processed data pickle file
        num_days: Number of days to plot (default: 40)
        output_dir: Directory to save plots
    """
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get full-day sequences
    train_sequences = data['train']['sequences']
    val_sequences = data['val']['sequences']
    
    # Combine train and val sequences
    all_sequences = train_sequences + val_sequences
    
    print(f"Loaded {len(all_sequences)} full-day sequences")
    
    # Select num_days sequences (or all if fewer)
    num_days = min(num_days, len(all_sequences))
    selected_sequences = all_sequences[:num_days]
    
    print(f"Plotting {num_days} full-day trajectories...")
    
    # Extract price and volume trajectories
    price_trajectories = []
    volume_trajectories = []
    
    for seq in selected_sequences:
        if len(seq.shape) == 2 and seq.shape[1] >= 2:
            price_trajectories.append(seq[:, 0])  # Feature 0: price
            volume_trajectories.append(seq[:, 1])  # Feature 1: volume
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot price trajectories
    print("Creating price spaghetti plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, price_traj in enumerate(price_trajectories):
        ax.plot(price_traj, alpha=0.6, linewidth=0.8)
    
    ax.set_xlabel('Time (5-minute intervals)', fontsize=12)
    ax.set_ylabel('Price (normalized, standardized)', fontsize=12)
    ax.set_title(f'Full-Day Price Trajectories (Post-Processing)\n{num_days} days', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    price_output = output_path / 'full_day_price_trajectories.png'
    plt.savefig(price_output, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved price plot to {price_output}")
    plt.close()
    
    # Plot volume trajectories
    print("Creating volume spaghetti plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, volume_traj in enumerate(volume_trajectories):
        ax.plot(volume_traj, alpha=0.6, linewidth=0.8)
    
    ax.set_xlabel('Time (5-minute intervals)', fontsize=12)
    ax.set_ylabel('Volume (normalized, standardized)', fontsize=12)
    ax.set_title(f'Full-Day Volume Trajectories (Post-Processing)\n{num_days} days', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    volume_output = output_path / 'full_day_volume_trajectories.png'
    plt.savefig(volume_output, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved volume plot to {volume_output}")
    plt.close()
    
    # Print statistics
    print(f"\nStatistics:")
    all_prices = np.concatenate([traj for traj in price_trajectories])
    all_volumes = np.concatenate([traj for traj in volume_trajectories])
    
    print(f"  Price (feature 0):")
    print(f"    Mean: {all_prices.mean():.6f}")
    print(f"    Std: {all_prices.std():.6f}")
    print(f"    Min: {all_prices.min():.6f}")
    print(f"    Max: {all_prices.max():.6f}")
    
    print(f"  Volume (feature 1):")
    print(f"    Mean: {all_volumes.mean():.6f}")
    print(f"    Std: {all_volumes.std():.6f}")
    print(f"    Min: {all_volumes.min():.6f}")
    print(f"    Max: {all_volumes.max():.6f}")
    
    print(f"\n✓ Plots saved to {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot full-day price and volume trajectories')
    parser.add_argument('--data_path', type=str, default='data/stock_sequences_full_day_2d.pkl',
                       help='Path to processed data pickle file')
    parser.add_argument('--num_days', type=int, default=40,
                       help='Number of days to plot (default: 40)')
    parser.add_argument('--output_dir', type=str, default='artifacts',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    plot_full_day_trajectories(
        data_path=args.data_path,
        num_days=args.num_days,
        output_dir=args.output_dir
    )


