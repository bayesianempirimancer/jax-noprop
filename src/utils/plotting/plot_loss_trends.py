"""
Plot loss trends over training epochs.
"""
import os
import numpy as np
from typing import Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt


def plot_loss_trends(history: Dict[str, Any], model_type: str, output_dir: str):
    """
    Plot loss terms over training epochs to diagnose training issues.
    
    Args:
        history: Training history dictionary containing loss values
        model_type: Type of model ('flow_matching', 'diffusion', 'ct')
        output_dir: Directory to save the plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if we have sequence metrics to determine subplot layout
    has_seq_metrics = history.get('val_seq_metrics') and len(history['val_seq_metrics']) > 0
    if has_seq_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Loss Trends - {model_type.title()} Model (Sequence Data)', fontsize=16, fontweight='bold')
    
    epochs = range(len(history['train_losses']))
    
    # Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_losses'], label='Train Total', color='blue', linewidth=2)
    if history.get('val_losses') and len(history['val_losses']) > 0:
        ax.plot(epochs, history['val_losses'], label='Val Total', color='red', linewidth=2, linestyle='--')
    ax.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Flow Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_flow_losses'], label='Train Flow', color='green', linewidth=2)
    if history.get('val_flow_losses') and len(history['val_flow_losses']) > 0:
        ax.plot(epochs, history['val_flow_losses'], label='Val Flow', color='orange', linewidth=2, linestyle='--')
    ax.set_title('Flow Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Percent Variance Explained (if available) - use the unused third panel in top row
    if has_seq_metrics:
        ax = axes[0, 2]
        seq_epochs = range(len(history['val_seq_metrics']))
        pve_vals = [m.get('percent_variance_explained', float('nan')) for m in history['val_seq_metrics'] 
                   if 'percent_variance_explained' in m and np.isfinite(m.get('percent_variance_explained', float('nan')))]
        
        if pve_vals:
            ax.plot(seq_epochs[:len(pve_vals)], pve_vals, label='% Variance Explained', color='green', linewidth=2, linestyle='-')
        
        ax.set_title('% Variance Explained', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('% Variance Explained', color='green')
        ax.tick_params(axis='y', labelcolor='green')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Reconstruction Loss
    ax = axes[1, 0]
    ax.plot(epochs, history['train_recon_losses'], label='Train Recon', color='purple', linewidth=2)
    if history.get('val_recon_losses') and len(history['val_recon_losses']) > 0:
        ax.plot(epochs, history['val_recon_losses'], label='Val Recon', color='brown', linewidth=2, linestyle='--')
    ax.set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Regularization Loss
    ax = axes[1, 1]
    ax.plot(epochs, history['train_reg_losses'], label='Train Reg', color='cyan', linewidth=2)
    if history.get('val_reg_losses') and len(history['val_reg_losses']) > 0:
        ax.plot(epochs, history['val_reg_losses'], label='Val Reg', color='magenta', linewidth=2, linestyle='--')
    ax.set_title('Regularization Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MSE (if available) - use the bottom row third panel
    if has_seq_metrics:
        ax = axes[1, 2]
        seq_epochs = range(len(history['val_seq_metrics']))
        mse_vals = [m['mse'] for m in history['val_seq_metrics'] if 'mse' in m]
        
        if mse_vals:
            ax.plot(seq_epochs[:len(mse_vals)], mse_vals, label='MSE', color='darkorange', linewidth=2, linestyle='--')
        
        ax.set_title('Sequence Metrics (MSE)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE', color='darkorange')
        ax.tick_params(axis='y', labelcolor='darkorange')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'loss_trends.png')
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved loss trends plot to {plot_path}")

