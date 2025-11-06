"""
Plot raw prediction vs ground truth trajectories over time.
"""
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def plot_trajectory_comparison(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    num_samples: int = 20
):
    """
    Plot raw prediction vs ground truth trajectories over time.
    
    Shows time series plots for a random selection of sequences, comparing
    predicted and ground truth trajectories for each dimension.
    
    Args:
        y_real: Real sequences [batch, seq_len, 2] in standardized 2D space
        y_pred: Predicted sequences [batch, seq_len, 2] in standardized 2D space
        output_dir: Directory to save the plot
        num_samples: Number of random sequences to plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    batch_size, seq_len, feature_dim = y_real.shape
    num_samples = min(num_samples, batch_size)
    
    # Randomly select samples
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(batch_size, num_samples, replace=False)
    
    # Calculate a more balanced grid layout
    # Total panels = feature_dim * num_samples
    total_panels = feature_dim * num_samples
    # Calculate optimal grid: aim for roughly square layout
    n_cols = int(np.ceil(np.sqrt(total_panels)))
    n_rows = int(np.ceil(total_panels / n_cols))
    
    # Create figure with subplots
    panel_width = 3.5  # Width per panel
    panel_height = 3.5  # Height per panel
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_width*n_cols, panel_height*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle('Trajectory Comparison: Predictions vs Ground Truth (Model Space)', 
                 fontsize=16, fontweight='bold')
    
    dim_names = ['Price (standardized)', 'Volume (standardized)']
    
    plot_idx = 0
    for dim_idx in range(feature_dim):
        for sample_idx in sample_indices:
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            plot_idx += 1
            
            # Get sequences for this sample
            real_seq = y_real[sample_idx, :, dim_idx]
            pred_seq = y_pred[sample_idx, :, dim_idx]
            
            # Plot trajectories
            time_steps = np.arange(seq_len)
            ax.plot(time_steps, real_seq, label='Ground Truth', color='blue', 
                   linewidth=2, marker='o', markersize=4, alpha=0.7)
            ax.plot(time_steps, pred_seq, label='Prediction', color='red', 
                   linewidth=2, marker='s', markersize=4, linestyle='--', alpha=0.7)
            
            # Set y-axis limits to match both sequences to avoid visual bias
            y_min = min(real_seq.min(), pred_seq.min())
            y_max = max(real_seq.max(), pred_seq.max())
            y_range = y_max - y_min
            if y_range > 0:
                # Add 10% padding
                y_pad = y_range * 0.1
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
            else:
                # If both sequences are constant, add small padding around the value
                ax.set_ylim(y_min - 0.1, y_max + 0.1)
            
            # Compute sample-specific metrics
            mse_sample = np.mean((real_seq - pred_seq) ** 2)
            mae_sample = np.mean(np.abs(real_seq - pred_seq))
            
            # Set title and labels
            if col == 0:
                ax.set_ylabel(dim_names[dim_idx], fontsize=11)
            ax.set_title(f'Sample {sample_idx} ({dim_names[dim_idx]})\nMSE: {mse_sample:.4f}, MAE: {mae_sample:.4f}', 
                       fontsize=10, fontweight='bold')
            
            if row == n_rows - 1:
                ax.set_xlabel('Time Step', fontsize=10)
            
            ax.grid(True, alpha=0.3)
            if plot_idx == 1:  # Show legend on first plot
                ax.legend(fontsize=8)
            
            # Hide unused subplots
            if plot_idx >= total_panels:
                ax.axis('off')
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'trajectory_comparison_model_space.png')
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved trajectory comparison plot to {plot_path}")

