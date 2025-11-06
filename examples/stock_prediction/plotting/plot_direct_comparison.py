"""
Direct comparison plot of predictions vs ground truth in model input/output space.
"""
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def plot_direct_comparison(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    num_samples: int = 100
):
    """
    Direct comparison plot of predictions vs ground truth in model input/output space (2D).
    
    This plots predictions vs ground truth directly without any transformations,
    showing the model's performance in the standardized 2D space (price, volume).
    
    Args:
        y_real: Real sequences [batch, seq_len, 2] in standardized 2D space
        y_pred: Predicted sequences [batch, seq_len, 2] in standardized 2D space
        output_dir: Directory to save the plot
        num_samples: Number of samples to plot (will use first num_samples)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Use first num_samples
    num_samples = min(num_samples, y_real.shape[0])
    y_real_subset = y_real[:num_samples]
    y_pred_subset = y_pred[:num_samples]
    
    # Flatten sequences for comparison: [num_samples * seq_len, 2]
    y_real_flat = y_real_subset.reshape(-1, y_real_subset.shape[-1])  # [N, 2]
    y_pred_flat = y_pred_subset.reshape(-1, y_pred_subset.shape[-1])  # [N, 2]
    
    # Compute metrics directly in this space
    mse_price = np.mean((y_real_flat[:, 0] - y_pred_flat[:, 0]) ** 2)
    mse_volume = np.mean((y_real_flat[:, 1] - y_pred_flat[:, 1]) ** 2)
    mse_total = np.mean((y_real_flat - y_pred_flat) ** 2)
    
    # Compute R² for each dimension
    def compute_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot > 1e-10:
            return 1.0 - (ss_res / ss_tot)
        return float('nan')
    
    r2_price = compute_r2(y_real_flat[:, 0], y_pred_flat[:, 0])
    r2_volume = compute_r2(y_real_flat[:, 1], y_pred_flat[:, 1])
    
    # Overall R² (on flattened data)
    ss_res_total = np.sum((y_real_flat - y_pred_flat) ** 2)
    y_real_mean = np.mean(y_real_flat, axis=0, keepdims=True)
    ss_tot_total = np.sum((y_real_flat - y_real_mean) ** 2)
    if ss_tot_total > 1e-10:
        r2_total = 1.0 - (ss_res_total / ss_tot_total)
        pve_total = r2_total * 100.0
    else:
        r2_total = float('nan')
        pve_total = float('nan')
    
    # Create figure with scatter plots for each dimension
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Direct Comparison: Predictions vs Ground Truth (Model Input/Output Space)', 
                 fontsize=14, fontweight='bold')
    
    # Compute PVE for price and volume separately
    pve_price = r2_price * 100.0 if np.isfinite(r2_price) else float('nan')
    pve_volume = r2_volume * 100.0 if np.isfinite(r2_volume) else float('nan')
    
    # Price dimension (dim 0)
    ax = axes[0]
    ax.scatter(y_real_flat[:, 0], y_pred_flat[:, 0], alpha=0.5, s=10)
    # Add diagonal line
    min_val = min(y_real_flat[:, 0].min(), y_pred_flat[:, 0].min())
    max_val = max(y_real_flat[:, 0].max(), y_pred_flat[:, 0].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    # Set equal aspect ratio and matching limits to avoid visual bias
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel('Ground Truth (Price, standardized)', fontsize=11)
    ax.set_ylabel('Prediction (Price, standardized)', fontsize=11)
    title_str = f'Price\nMSE: {mse_price:.6f}, R²: {r2_price:.4f}'
    if np.isfinite(pve_price):
        title_str += f'\n% Variance Explained: {pve_price:.2f}%'
    ax.set_title(title_str, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Volume dimension (dim 1)
    ax = axes[1]
    ax.scatter(y_real_flat[:, 1], y_pred_flat[:, 1], alpha=0.5, s=10, color='green')
    # Add diagonal line
    min_val = min(y_real_flat[:, 1].min(), y_pred_flat[:, 1].min())
    max_val = max(y_real_flat[:, 1].max(), y_pred_flat[:, 1].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    # Set equal aspect ratio and matching limits to avoid visual bias
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel('Ground Truth (Volume, standardized)', fontsize=11)
    ax.set_ylabel('Prediction (Volume, standardized)', fontsize=11)
    title_str = f'Volume\nMSE: {mse_volume:.6f}, R²: {r2_volume:.4f}'
    if np.isfinite(pve_volume):
        title_str += f'\n% Variance Explained: {pve_volume:.2f}%'
    ax.set_title(title_str, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Combined scatter (all dimensions flattened)
    ax = axes[2]
    y_real_all = y_real_flat.flatten()
    y_pred_all = y_pred_flat.flatten()
    ax.scatter(y_real_all, y_pred_all, alpha=0.5, s=10, color='purple')
    # Add diagonal line
    min_val = min(y_real_all.min(), y_pred_all.min())
    max_val = max(y_real_all.max(), y_pred_all.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    # Set equal aspect ratio and matching limits to avoid visual bias
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel('Ground Truth (All dimensions, standardized)', fontsize=11)
    ax.set_ylabel('Prediction (All dimensions, standardized)', fontsize=11)
    title_str = f'Combined\nMSE: {mse_total:.6f}'
    if np.isfinite(r2_total):
        title_str += f', R²: {r2_total:.4f}\n% Variance Explained: {pve_total:.2f}%'
    ax.set_title(title_str, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'direct_comparison_model_space.png')
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved direct comparison plot to {plot_path}")
    print(f"  Metrics in model space:")
    pve_price = r2_price * 100.0 if np.isfinite(r2_price) else float('nan')
    pve_volume = r2_volume * 100.0 if np.isfinite(r2_volume) else float('nan')
    if np.isfinite(pve_price):
        print(f"    Price: MSE={mse_price:.6f}, R²={r2_price:.4f}, % Variance Explained={pve_price:.2f}%")
    else:
        print(f"    Price: MSE={mse_price:.6f}, R²={r2_price:.4f}, % Variance Explained=N/A")
    if np.isfinite(pve_volume):
        print(f"    Volume: MSE={mse_volume:.6f}, R²={r2_volume:.4f}, % Variance Explained={pve_volume:.2f}%")
    else:
        print(f"    Volume: MSE={mse_volume:.6f}, R²={r2_volume:.4f}, % Variance Explained=N/A")
    print(f"    Combined: MSE={mse_total:.6f}, R²={r2_total:.4f}, % Variance Explained={pve_total:.2f}%")

