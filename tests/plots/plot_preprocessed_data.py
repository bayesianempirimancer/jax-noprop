#!/usr/bin/env python3
"""
Plot preprocessed stock data for a few different days.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from preprocess_stock_data import preprocess_stock_data


def plot_preprocessed_features(data_path: str, output_path: str = None, num_days: int = 5):
    """
    Plot preprocessed features for a few different days.
    
    Args:
        data_path: Path to pickle file with raw stock data
        output_path: Path to save the plot (optional)
        num_days: Number of days to plot
    """
    # Load raw data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get raw sequences
    if 'train' in data:
        x_sequences = data['train']['x']
        y_sequences = data['train']['y']
        previous_closes = data.get('metadata', {}).get('previous_closes_train', None)
        previous_avg_volumes = data.get('metadata', {}).get('previous_avg_volumes_train', None)
    else:
        x_sequences = data.get('x', [])
        y_sequences = data.get('y', None)
        previous_closes = data.get('metadata', {}).get('previous_closes', None)
        previous_avg_volumes = data.get('metadata', {}).get('previous_avg_volumes', None)
    
    if y_sequences is None:
        print("ERROR: No y sequences found in data")
        return
    
    # Convert to numpy arrays if needed
    if isinstance(x_sequences, list):
        x_list = x_sequences
    else:
        # Convert array to list
        x_list = [x_sequences[i] for i in range(x_sequences.shape[0])]
    
    n_samples = min(num_days, len(y_sequences))
    
    # For now, let's use the raw data directly if it's already log-normalized
    # Otherwise we need to check what format the data is in
    
    # Check if data is already preprocessed or raw
    # If it's from prepare_stock_data_multiple.py, it might already be log-normalized
    # Let's check the statistics
    print(f"\nData statistics:")
    print(f"  Number of samples: {len(y_sequences)}")
    print(f"  y shape: {y_sequences.shape}")
    print(f"  Sample y[0, 0, :] = {y_sequences[0, 0, :]}")
    print(f"  y mean: {y_sequences.mean(axis=(0, 1))}")
    print(f"  y std: {y_sequences.std(axis=(0, 1))}")
    
    # Check if we have previous_closes
    if previous_closes is None:
        print("WARNING: No previous_closes found. Creating synthetic data for demonstration.")
        # Create synthetic data for demonstration
        previous_closes = 100.0 + 20.0 * np.random.randn(len(y_sequences))
        previous_closes = np.maximum(previous_closes, 50.0)
        previous_avg_volumes = 1000000.0 + 500000.0 * np.random.randn(len(y_sequences))
        previous_avg_volumes = np.maximum(previous_avg_volumes, 100000.0)
        
        # Create synthetic raw sequences
        x_raw = []
        y_raw = np.zeros_like(y_sequences)
        
        for i in range(len(y_sequences)):
            prev_close = previous_closes[i]
            prev_avg_vol = previous_avg_volumes[i]
            
            # Create synthetic y sequences
            y_raw[i, :, 0] = prev_close * (1.0 + 0.02 * np.random.randn(y_sequences.shape[1]))  # Price
            y_raw[i, :, 0] = np.maximum(y_raw[i, :, 0], prev_close * 0.95)
            y_raw[i, :, 1] = prev_avg_vol * (1.0 + 0.2 * np.random.randn(y_sequences.shape[1]))  # Volume
            y_raw[i, :, 1] = np.maximum(y_raw[i, :, 1], 1000.0)
            y_raw[i, :, 2] = y_raw[i, :, 0] * 0.999  # Bid
            y_raw[i, :, 3] = y_raw[i, :, 0] * 1.001  # Ask
            
            # Create synthetic x sequences
            if i < len(x_list):
                seq_len = len(x_list[i]) if isinstance(x_list, list) else x_list.shape[1]
                x_seq = np.zeros((seq_len, 4))
                x_seq[:, 0] = prev_close * (1.0 + 0.02 * np.random.randn(seq_len))
                x_seq[:, 0] = np.maximum(x_seq[:, 0], prev_close * 0.95)
                x_seq[:, 1] = prev_avg_vol * (1.0 + 0.2 * np.random.randn(seq_len))
                x_seq[:, 1] = np.maximum(x_seq[:, 1], 1000.0)
                x_seq[:, 2] = x_seq[:, 0] * 0.999
                x_seq[:, 3] = x_seq[:, 0] * 1.001
                x_raw.append(x_seq)
            else:
                x_raw.append(np.zeros((10, 4)))
        
        x_sequences = x_raw
        y_sequences = y_raw
    else:
        # Assume data is already log-normalized from prepare_stock_data_multiple.py
        # We need to convert it back to raw for preprocessing demo
        # Actually, let's just use the data as-is and apply preprocessing
        # But first let's check if it's already preprocessed
        if abs(y_sequences[0, :, 0].mean()) < 0.1 and y_sequences[0, :, 0].std() < 2.0:
            # Likely already preprocessed (standardized)
            print("Data appears to be already preprocessed. Using as-is for plotting.")
            preprocessed_y = y_sequences
        else:
            # Need to apply preprocessing
            print("Applying preprocessing...")
            _, preprocessed_y, _ = preprocess_stock_data(
                x_list[:n_samples], 
                y_sequences[:n_samples], 
                previous_closes[:n_samples], 
                previous_avg_volumes[:n_samples] if previous_avg_volumes is not None else np.ones(n_samples) * 1000000.0
            )
            n_samples = len(preprocessed_y)
    
    # Get preprocessed data (use first n_samples)
    if 'preprocessed_y' not in locals():
        preprocessed_y = y_sequences[:n_samples]
    
    # Create plot
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    feature_names = ['Price (standardized)', 'Volume (standardized)', 'Bid (standardized)', 'Ask (standardized)']
    
    for day_idx in range(n_samples):
        for feat_idx in range(4):
            ax = axes[day_idx, feat_idx]
            
            # Plot the sequence
            seq_len = preprocessed_y.shape[1]
            time_steps = np.arange(seq_len)
            
            ax.plot(time_steps, preprocessed_y[day_idx, :, feat_idx], 
                   linewidth=2, alpha=0.7, color='blue')
            ax.set_title(f'Day {day_idx+1} - {feature_names[feat_idx]}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time Step (5-min intervals)', fontsize=9)
            ax.set_ylabel('Standardized Value', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = preprocessed_y[day_idx, :, feat_idx].mean()
            std_val = preprocessed_y[day_idx, :, feat_idx].std()
            ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved plot to {output_path}")
    else:
        plt.savefig('preprocessed_data_plot.png', dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved plot to preprocessed_data_plot.png")
    
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot preprocessed stock data')
    parser.add_argument('--data_path', type=str, default='data/stock_sequences_test_new_preprocessing.pkl',
                       help='Path to raw stock data file')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save the plot (default: preprocessed_data_plot.png)')
    parser.add_argument('--num_days', type=int, default=5,
                       help='Number of days to plot')
    
    args = parser.parse_args()
    
    plot_preprocessed_features(args.data_path, args.output_path, args.num_days)

