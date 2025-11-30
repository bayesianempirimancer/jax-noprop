"""
Unified loss trends plotting for flow models.

This module provides a single function to create comprehensive loss trends plots
with 8 panels covering all major loss components and metrics for both generation
and regression tasks.
"""

import os
import numpy as np
from typing import Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt


def create_loss_trends_plot(history: Dict[str, Any], model_type: str, output_dir: str):
    """
    Create a unified loss trends plot with 8 panels.
    
    Panels (2x4 layout):
    1. Total Loss
    2. Flow Loss
    3. Reconstruction Loss
    4. VAE Loss
    5. Regularization Loss
    6. Percent Variance Explained (R²)
    7. Mean Squared Error (MSE)
    8. Chamfer Distance
    
    Args:
        history: Training history dictionary containing loss values and metrics
        model_type: Type of model ('flow_matching', 'diffusion', 'ct')
        output_dir: Directory to save the plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create 2x4 layout for 8 panels
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Loss Trends - {model_type.title()} Model', fontsize=16, fontweight='bold')
    
    epochs = range(len(history['train_losses']))
    
    # Panel 1: Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_losses'], label='Train', color='blue', linewidth=2)
    if history.get('val_losses') and len(history['val_losses']) > 0:
        ax.plot(epochs, history['val_losses'], label='Val', color='red', linewidth=2, linestyle='--')
    ax.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Flow Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_flow_losses'], label='Train', color='green', linewidth=2)
    if history.get('val_flow_losses') and len(history['val_flow_losses']) > 0:
        ax.plot(epochs, history['val_flow_losses'], label='Val', color='orange', linewidth=2, linestyle='--')
    ax.set_title('Flow Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Reconstruction Loss
    ax = axes[0, 2]
    ax.plot(epochs, history['train_recon_losses'], label='Train', color='purple', linewidth=2)
    if history.get('val_recon_losses') and len(history['val_recon_losses']) > 0:
        ax.plot(epochs, history['val_recon_losses'], label='Val', color='brown', linewidth=2, linestyle='--')
    ax.set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: VAE Loss
    ax = axes[0, 3]
    if history.get('train_vae_losses') and len(history['train_vae_losses']) > 0:
        ax.plot(epochs, history['train_vae_losses'], label='Train', color='teal', linewidth=2)
        if history.get('val_vae_losses') and len(history['val_vae_losses']) > 0:
            ax.plot(epochs, history['val_vae_losses'], label='Val', color='coral', linewidth=2, linestyle='--')
        ax.set_title('VAE Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'VAE Loss\n(Not Available)', ha='center', va='center', fontsize=12, alpha=0.5)
    
    # Panel 5: Regularization Loss
    ax = axes[1, 0]
    ax.plot(epochs, history['train_reg_losses'], label='Train', color='cyan', linewidth=2)
    if history.get('val_reg_losses') and len(history['val_reg_losses']) > 0:
        ax.plot(epochs, history['val_reg_losses'], label='Val', color='magenta', linewidth=2, linestyle='--')
    ax.set_title('Regularization Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 6: Percent Variance Explained (R²)
    ax = axes[1, 1]
    train_pve = []
    val_pve = []
    
    # Check if this is a generation task (has x_gen/x_real) - PVE is not applicable
    is_generation_task = 'x_gen' in history and 'x_real' in history
    
    if not is_generation_task:
        # Use per-epoch PVE values if available
        if 'train_pve' in history and len(history['train_pve']) > 0:
            train_pve = history['train_pve']
        # Fallback: compute from final predictions if available
        elif 'train_pred' in history and 'train_y' in history:
            train_pred = np.array(history['train_pred'])
            train_y = np.array(history['train_y'])
            ss_res = np.sum((train_y - train_pred) ** 2)
            ss_tot = np.sum((train_y - np.mean(train_y, axis=0, keepdims=True)) ** 2)
            if ss_tot > 0:
                r2_train = 1 - (ss_res / ss_tot)
                train_pve = [r2_train * 100] * len(epochs)
        
        if 'val_pve' in history and len(history['val_pve']) > 0:
            val_pve = history['val_pve']
        elif 'val_pred' in history and 'val_y' in history and len(history.get('val_pred', [])) > 0:
            val_pred = np.array(history['val_pred'])
            val_y = np.array(history['val_y'])
            ss_res = np.sum((val_y - val_pred) ** 2)
            ss_tot = np.sum((val_y - np.mean(val_y, axis=0, keepdims=True)) ** 2)
            if ss_tot > 0:
                r2_val = 1 - (ss_res / ss_tot)
                val_pve = [r2_val * 100] * len(epochs)
    
    if len(train_pve) > 0 or len(val_pve) > 0:
        if len(train_pve) > 0:
            ax.plot(epochs[:len(train_pve)], train_pve, label='Train', color='darkgreen', linewidth=2)
        if len(val_pve) > 0:
            ax.plot(epochs[:len(val_pve)], val_pve, label='Val', color='darkred', linewidth=2, linestyle='--')
        ax.set_title('% Variance Explained (R²)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('% Variance Explained', color='darkgreen')
        ax.tick_params(axis='y', labelcolor='darkgreen')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Show N/A for generation tasks where PVE is not computed
        ax.axis('off')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, alpha=0.5)
        ax.set_title('% Variance Explained (R²)', fontsize=12, fontweight='bold')
    
    # Panel 7: Mean Squared Error (MSE)
    ax = axes[1, 2]
    train_mse = []
    val_mse = []
    
    # Check if this is a generation task (has x_gen/x_real) - MSE is not applicable
    is_generation_task = 'x_gen' in history and 'x_real' in history
    
    if not is_generation_task:
        # Use per-epoch MSE values if available
        if 'train_mse' in history and len(history['train_mse']) > 0:
            train_mse = history['train_mse']
        # Fallback: compute from final predictions if available
        elif 'train_pred' in history and 'train_y' in history:
            train_pred = np.array(history['train_pred'])
            train_y = np.array(history['train_y'])
            mse_val = np.mean((train_y - train_pred) ** 2)
            train_mse = [mse_val] * len(epochs)
        
        if 'val_mse' in history and len(history['val_mse']) > 0:
            val_mse = history['val_mse']
        elif 'val_pred' in history and 'val_y' in history and len(history.get('val_pred', [])) > 0:
            val_pred = np.array(history['val_pred'])
            val_y = np.array(history['val_y'])
            mse_val = np.mean((val_y - val_pred) ** 2)
            val_mse = [mse_val] * len(epochs)
    
    if len(train_mse) > 0 or len(val_mse) > 0:
        if len(train_mse) > 0:
            ax.plot(epochs[:len(train_mse)], train_mse, label='Train', color='darkblue', linewidth=2)
        if len(val_mse) > 0:
            ax.plot(epochs[:len(val_mse)], val_mse, label='Val', color='darkorange', linewidth=2, linestyle='--')
        ax.set_title('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE', color='darkblue')
        ax.tick_params(axis='y', labelcolor='darkblue')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Show N/A for generation tasks where MSE is not computed
        ax.axis('off')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, alpha=0.5)
        ax.set_title('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
    
    # Panel 8: Chamfer Distance
    ax = axes[1, 3]
    train_chamfer = []
    val_chamfer = []
    val_chamfer_epochs = []
    
    # Compute train chamfer distance from final generation results
    if 'x_gen' in history and 'x_real' in history:
        import jax.numpy as jnp
        from src.utils.metrics import chamfer_distance
        x_gen = np.array(history['x_gen'])
        x_real = np.array(history['x_real'])
        
        # Reshape to (num_samples, feature_dim) if needed
        if x_gen.ndim > 2:
            x_gen = x_gen.reshape(-1, x_gen.shape[-1])
        if x_real.ndim > 2:
            x_real = x_real.reshape(-1, x_real.shape[-1])
        
        chamfer_dist = chamfer_distance(jnp.array(x_gen), jnp.array(x_real))
        if np.isfinite(chamfer_dist):
            train_chamfer = [chamfer_dist] * len(epochs)
    
    # Get validation chamfer distances (computed every 10 epochs)
    if 'val_chamfer_distances' in history and len(history['val_chamfer_distances']) > 0:
        val_chamfer = history['val_chamfer_distances']
        val_chamfer_epochs = [i * 10 for i in range(len(val_chamfer))]
        if len(val_chamfer_epochs) > 0 and val_chamfer_epochs[-1] != epochs[-1]:
            val_chamfer_epochs[-1] = epochs[-1]
    
    if len(train_chamfer) > 0 or len(val_chamfer) > 0:
        if len(train_chamfer) > 0:
            ax.plot(epochs, train_chamfer, label='Train', color='darkgreen', linewidth=2)
        if len(val_chamfer) > 0:
            ax.plot(val_chamfer_epochs, val_chamfer, label='Val', color='darkred', linewidth=2, 
                    linestyle='--', marker='o', markersize=4)
        ax.set_title('Chamfer Distance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Chamfer Distance', color='darkgreen')
        ax.tick_params(axis='y', labelcolor='darkgreen')
        ax.legend()
        # Set y-axis to start from 0
        all_values = [v for v in train_chamfer if np.isfinite(v)] + [v for v in val_chamfer if np.isfinite(v)]
        if len(all_values) > 0:
            max_val = max(all_values)
            ax.set_ylim([0, max_val * 1.1])
        ax.grid(True, alpha=0.3)
    else:
        # Show N/A for regression tasks where chamfer distance is not computed
        ax.axis('off')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, alpha=0.5)
        ax.set_title('Chamfer Distance', fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'loss_trends.png')
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved loss trends plot to {plot_path}")
