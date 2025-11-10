"""
Plot generated sequences vs real sequences for Lorenz system data.
Shows x, y, z coordinates in a 6x6 grid for 12 samples.
"""
import numpy as np
from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt


def plot_lorenz_sequence_comparison(
    y_real: np.ndarray,
    y_gen: np.ndarray,
    output_dir: str,
    num_samples: int = 12
):
    """
    Plot generated sequences vs real sequences for Lorenz system.
    
    Creates a 6x6 grid showing true and predicted x, y, z coordinates for 12 samples.
    Layout: 6 rows x 6 columns = 36 subplots
    - Each row shows 2 samples (side by side)
    - For each sample: 3 columns showing x, y, z coordinates
    
    Args:
        y_real: Real sequences [batch, seq_len, 3] where last dim is (x, y, z)
        y_gen: Generated sequences [batch, seq_len, 3] where last dim is (x, y, z)
        output_dir: Directory to save the plot
        num_samples: Number of samples to plot (default: 12)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    batch_size = min(num_samples, y_real.shape[0], y_gen.shape[0])
    seq_len = y_real.shape[1]
    
    if y_real.shape[2] != 3 or y_gen.shape[2] != 3:
        raise ValueError(f"Expected 3D coordinates (x, y, z), got shapes: y_real={y_real.shape}, y_gen={y_gen.shape}")
    
    # Create 6x6 subplot grid
    fig, axes = plt.subplots(6, 6, figsize=(18, 18))
    fig.suptitle('Lorenz System: True vs Predicted Sequences (x, y, z coordinates)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    coord_names = ['x', 'y', 'z']
    coord_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Plot 12 samples: 6 rows, each row has 2 samples side by side
    # Each sample shows 3 coordinates (x, y, z) in 3 columns
    for sample_idx in range(batch_size):
        row = sample_idx // 2  # Which row (0-5)
        col_offset = (sample_idx % 2) * 3  # Which set of 3 columns (0 or 3)
        
        # Extract sequences for this sample
        real_seq = y_real[sample_idx]  # [seq_len, 3]
        gen_seq = y_gen[sample_idx]    # [seq_len, 3]
        
        # Plot each coordinate (x, y, z) in its column
        for coord_idx, (coord_name, color) in enumerate(zip(coord_names, coord_colors)):
            col = col_offset + coord_idx
            ax = axes[row, col]
            
            # Plot true and predicted sequences
            time_steps = np.arange(seq_len)
            ax.plot(time_steps, real_seq[:, coord_idx], 
                   label='True', color=color, linewidth=2, alpha=0.7)
            ax.plot(time_steps, gen_seq[:, coord_idx], 
                   label='Predicted', color=color, linewidth=2, 
                   linestyle='--', alpha=0.7)
            
            # Formatting
            ax.set_title(f'Sample {sample_idx+1}, {coord_name.upper()}', 
                        fontsize=9, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=8)
            ax.set_ylabel(f'{coord_name.upper()} Value', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best')
            
            # Only show y-axis labels on leftmost column
            if col % 3 != 0:
                ax.set_ylabel('')
            
            # Only show x-axis labels on bottom row
            if row < 5:
                ax.set_xlabel('')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plot_path = Path(output_dir) / 'lorenz_sequence_comparison.png'
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved Lorenz sequence comparison plot to {plot_path}")


