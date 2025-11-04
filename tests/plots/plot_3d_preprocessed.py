#!/usr/bin/env python3
"""
Plot preprocessed stock data with 2 dimensions (price, volume)
plus spaghetti plot of 20D embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from preprocess_stock_data import preprocess_stock_data


def compute_ask_bid_spread(raw_ask: np.ndarray, raw_bid: np.ndarray, 
                           previous_close: float) -> np.ndarray:
    """
    Compute ask-bid spread from raw data, then log-normalize.
    
    Args:
        raw_ask: Raw ask prices
        raw_bid: Raw bid prices
        previous_close: Previous day's closing price
    
    Returns:
        Log-normalized spread: log((ask - bid) / previous_close)
    """
    spread = raw_ask - raw_bid
    # Normalize by previous close and log transform
    spread_normalized = np.log((spread + 1e-8) / (previous_close + 1e-8))
    return spread_normalized


def plot_2d_preprocessed_with_embedding(
    raw_data_path: str,
    projected_data_path: str,
    output_path: str = None,
    num_days: int = 5
):
    """
    Plot 2 preprocessed features (price, volume) plus 20D embedding spaghetti plot.
    
    Args:
        raw_data_path: Path to preprocessed stock data file (2D: price, volume)
        projected_data_path: Path to projected data file (20D embeddings)
        output_path: Path to save the plot
        num_days: Number of days to plot
    """
    # We need to load the raw data BEFORE preprocessing to get actual bid/ask values
    # But the data file might already be preprocessed. Let's check prepare_stock_data_multiple.py
    # to see if we can get raw data, or if we need to reconstruct it.
    
    # For now, let's load the preprocessed data and reconstruct raw values
    print(f"Loading preprocessed data from {raw_data_path}...")
    with open(raw_data_path, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    # Load projected data
    print(f"Loading projected data from {projected_data_path}...")
    with open(projected_data_path, 'rb') as f:
        proj_data = pickle.load(f)
    
    # Get preprocessed sequences (2D: price, volume)
    if 'train' in preprocessed_data:
        y_preprocessed_2d = preprocessed_data['train']['y']  # [n_samples, seq_len, 2]
    else:
        y_preprocessed_2d = preprocessed_data.get('y', None)
    
    # Get projected sequences (20D embeddings)
    if 'train' in proj_data:
        y_projected = proj_data['train']['y']  # [n_samples, seq_len, 20]
    else:
        y_projected = proj_data.get('y', None)
    
    if y_preprocessed_2d is None or y_projected is None:
        print("ERROR: Missing data")
        return
    
    n_samples = min(num_days, len(y_preprocessed_2d), len(y_projected))
    
    # Use preprocessed data directly (already 2D: price, volume)
    preprocessed_y_2d = y_preprocessed_2d[:n_samples]  # [n_samples, seq_len, 2]
    
    # Create 3-panel plot (Price, Volume, 20D Embedding)
    fig = plt.figure(figsize=(18, 6))
    
    seq_len = preprocessed_y_2d.shape[1]
    time_steps = np.arange(seq_len)
    
    # Panel 1: Price
    ax1 = plt.subplot(1, 3, 1)
    for day_idx in range(n_samples):
        ax1.plot(time_steps, preprocessed_y_2d[day_idx, :, 0], 
                linewidth=1.5, alpha=0.7, label=f'Day {day_idx+1}')
    ax1.set_title('Price (Standardized)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time Step (5-min intervals)', fontsize=10)
    ax1.set_ylabel('Standardized Value', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')
    
    # Panel 2: Volume
    ax2 = plt.subplot(1, 3, 2)
    for day_idx in range(n_samples):
        ax2.plot(time_steps, preprocessed_y_2d[day_idx, :, 1], 
                linewidth=1.5, alpha=0.7, label=f'Day {day_idx+1}')
    ax2.set_title('Volume (Standardized)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Step (5-min intervals)', fontsize=10)
    ax2.set_ylabel('Standardized Value', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best')
    
    # Panel 3: 20D Embedding Spaghetti Plot
    ax3 = plt.subplot(1, 3, 3)
    # Plot all 20 dimensions for one day (or multiple days)
    day_to_plot = 0  # Plot first day
    seq_len_proj = y_projected.shape[1]
    embed_dim = y_projected.shape[2]
    time_steps_proj = np.arange(seq_len_proj)
    
    # Plot all 20 dimensions as spaghetti plot
    for dim_idx in range(embed_dim):
        ax3.plot(time_steps_proj, y_projected[day_to_plot, :, dim_idx], 
                linewidth=1.0, alpha=0.5, color=f'C{dim_idx % 10}')
    
    ax3.set_title(f'20D Embedding (Day {day_to_plot+1}) - All Dimensions', 
                 fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Step (5-min intervals)', fontsize=10)
    ax3.set_ylabel('Embedding Value', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {y_projected[day_to_plot].mean():.3f}\n'
    stats_text += f'Std: {y_projected[day_to_plot].std():.3f}\n'
    stats_text += f'Range: [{y_projected[day_to_plot].min():.2f}, {y_projected[day_to_plot].max():.2f}]\n'
    stats_text += f'Dimensions: {embed_dim}'
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved plot to {output_path}")
    else:
        plt.savefig('artifacts/2d_preprocessed_with_embedding.png', dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved plot to artifacts/2d_preprocessed_with_embedding.png")
    
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot 3D preprocessed data with 20D embedding')
    parser.add_argument('--raw_data_path', type=str, 
                       default='data/stock_sequences_2d_test.pkl',
                       help='Path to preprocessed stock data file (2D: price, volume)')
    parser.add_argument('--projected_data_path', type=str,
                       default=None,
                       help='Path to projected data file (20D embeddings)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save the plot')
    parser.add_argument('--num_days', type=int, default=5,
                       help='Number of days to plot')
    
    args = parser.parse_args()
    
    # If no projected data path provided, try to generate it
    if args.projected_data_path is None:
        # Try to find a projected data file
        import os
        if os.path.exists('data/stock_sequences_projected_2d.pkl'):
            args.projected_data_path = 'data/stock_sequences_projected_2d.pkl'
        else:
            print("ERROR: No projected data file found. Please provide --projected_data_path")
            exit(1)
    
    plot_2d_preprocessed_with_embedding(
        args.raw_data_path,
        args.projected_data_path,
        args.output_path,
        args.num_days
    )

