"""Unified plotting utilities for all VAE training progress visualization.

This module provides a single function that can handle standard VAE, VBVAE, and VQVAE
training progress plots, showing N/A for unavailable metrics.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def create_vae_loss_trends_plot(history: Dict[str, Any], output_dir: str, save_name: str = "vae_training_progress.png") -> None:
    """Create unified training progress plot for VAE, VBVAE, or VQVAE.
    
    Automatically detects the VAE type based on available keys in history and creates
    appropriate plots. Shows "N/A" for unavailable metrics.
    
    Args:
        history: Training history dictionary. Should contain:
            - train_losses, val_losses (required)
            - train_recon_losses, val_recon_losses (optional)
            - train_kl_losses, val_kl_losses (optional, for standard VAE)
            - train_gmm_losses, val_gmm_losses (optional, for VBVAE)
            - train_vq_losses, val_vq_losses (optional, for VQVAE)
            - active_clusters (optional, for VBVAE)
            - normalized_pi (optional, for VBVAE)
            - train_pve_by_dim, val_pve_by_dim (optional, for VQVAE)
        output_dir: Directory to save the plot
        save_name: Name of the output file (default: "vae_training_progress.png")
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if 'train_losses' not in history or len(history['train_losses']) == 0:
        print("Warning: No training losses found in history")
        return
    
    epochs = range(len(history['train_losses']))
    
    # Detect VAE type based on available keys
    has_kl = 'train_kl_losses' in history and len(history.get('train_kl_losses', [])) > 0
    has_gmm = 'train_gmm_losses' in history and len(history.get('train_gmm_losses', [])) > 0
    has_vq = 'train_vq_losses' in history and len(history.get('train_vq_losses', [])) > 0
    has_active_clusters = 'active_clusters' in history and len(history.get('active_clusters', [])) > 0
    has_normalized_pi = 'normalized_pi' in history and len(history.get('normalized_pi', [])) > 0
    has_pve_by_dim = 'train_pve_by_dim' in history and len(history.get('train_pve_by_dim', [])) > 0
    
    # Determine VAE type
    if has_gmm or has_active_clusters or has_normalized_pi:
        vae_type = "VBVAE"
    elif has_vq or has_pve_by_dim:
        vae_type = "VQVAE"
    else:
        vae_type = "VAE"
    
    # Calculate validation epochs (only where validation was actually run)
    val_epochs = []
    if 'val_losses' in history and len(history['val_losses']) > 0:
        # Find epochs where validation was run (every 10 epochs + last epoch)
        val_epochs = [i for i in range(len(history['train_losses'])) 
                     if i % 10 == 0 or i == len(history['train_losses']) - 1]
        val_epochs = val_epochs[:len(history['val_losses'])]
    
    # Create 3x3 layout to accommodate all possible metrics
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'{vae_type} Training Progress', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Panel 1: Total Loss
    ax = axes_flat[0]
    ax.plot(epochs, history['train_losses'], label='Train', color='blue', linewidth=2)
    if 'val_losses' in history and len(history['val_losses']) > 0 and len(val_epochs) == len(history['val_losses']):
        ax.plot(val_epochs, history['val_losses'], label='Val', color='red', linewidth=2, linestyle='--', marker='o', markersize=4)
    ax.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Reconstruction Loss
    ax = axes_flat[1]
    if 'train_recon_losses' in history and len(history['train_recon_losses']) > 0:
        ax.plot(epochs, history['train_recon_losses'], label='Train', color='blue', linewidth=2)
        if 'val_recon_losses' in history and len(history['val_recon_losses']) > 0 and len(val_epochs) == len(history['val_recon_losses']):
            ax.plot(val_epochs, history['val_recon_losses'], label='Val', color='red', linewidth=2, linestyle='--', marker='o', markersize=4)
        ax.set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, alpha=0.5)
        ax.set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
    
    # Panel 3: KL Divergence Loss (Standard VAE)
    ax = axes_flat[2]
    if has_kl:
        ax.plot(epochs, history['train_kl_losses'], label='Train', color='green', linewidth=2)
        if 'val_kl_losses' in history and len(history['val_kl_losses']) > 0 and len(val_epochs) == len(history['val_kl_losses']):
            ax.plot(val_epochs, history['val_kl_losses'], label='Val', color='orange', linewidth=2, linestyle='--', marker='o', markersize=4)
        ax.set_title('KL Divergence Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, alpha=0.5)
        ax.set_title('KL Divergence Loss', fontsize=12, fontweight='bold')
    
    # Panel 4: GMM Loss (VBVAE)
    ax = axes_flat[3]
    if has_gmm:
        ax.plot(epochs, history['train_gmm_losses'], label='Train', color='purple', linewidth=2)
        if 'val_gmm_losses' in history and len(history['val_gmm_losses']) > 0 and len(val_epochs) == len(history['val_gmm_losses']):
            ax.plot(val_epochs, history['val_gmm_losses'], label='Val', color='magenta', linewidth=2, linestyle='--', marker='o', markersize=4)
        ax.set_title('GMM Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, alpha=0.5)
        ax.set_title('GMM Loss', fontsize=12, fontweight='bold')
    
    # Panel 5: VQ Loss (VQVAE)
    ax = axes_flat[4]
    if has_vq:
        ax.plot(epochs, history['train_vq_losses'], label='Train', color='cyan', linewidth=2)
        if 'val_vq_losses' in history and len(history['val_vq_losses']) > 0 and len(val_epochs) == len(history['val_vq_losses']):
            ax.plot(val_epochs, history['val_vq_losses'], label='Val', color='teal', linewidth=2, linestyle='--', marker='o', markersize=4)
        ax.set_title('Vector Quantization Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, alpha=0.5)
        ax.set_title('Vector Quantization Loss', fontsize=12, fontweight='bold')
    
    # Panel 6: Active Clusters (VBVAE)
    ax = axes_flat[5]
    if has_active_clusters:
        ax.plot(epochs, history['active_clusters'], label='Active Clusters', color='green', linewidth=2)
        ax.set_title('Active Clusters Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Number of Active Clusters')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, alpha=0.5)
        ax.set_title('Active Clusters', fontsize=12, fontweight='bold')
    
    # Panel 7: Normalized Mixing Weights (VBVAE)
    ax = axes_flat[6]
    if has_normalized_pi:
        num_clusters = len(history['normalized_pi'][0]) if len(history['normalized_pi']) > 0 else 0
        if num_clusters > 0:
            final_pi = np.array(history['normalized_pi'][-1])
            top_cluster_indices = np.argsort(final_pi)[-min(10, num_clusters):][::-1]  # Top 10 clusters
            
            pi_over_time = np.array(history['normalized_pi'])  # [num_epochs, num_clusters]
            colors = plt.cm.tab10(np.linspace(0, 1, len(top_cluster_indices)))
            
            for idx, k in enumerate(top_cluster_indices):
                ax.plot(epochs, pi_over_time[:, k], label=f'Cluster {k}', 
                       color=colors[idx], alpha=0.7, linewidth=1.5)
            
            ax.set_title('Top Cluster Mixing Weights', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Normalized Mixing Weight')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=1)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, alpha=0.5)
            ax.set_title('Mixing Weights', fontsize=12, fontweight='bold')
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, alpha=0.5)
        ax.set_title('Mixing Weights', fontsize=12, fontweight='bold')
    
    # Panel 8: Percent Variance Explained by Dimension (VQVAE)
    ax = axes_flat[7]
    if has_pve_by_dim:
        first_pve = history['train_pve_by_dim'][0]
        if len(first_pve) > 0:
            num_dims = len(first_pve)
            
            # Extract PVE for each dimension across epochs
            for dim in range(min(num_dims, 5)):  # Limit to 5 dimensions for readability
                train_pve_dim = []
                for epoch_pve in history['train_pve_by_dim']:
                    if len(epoch_pve) > dim and np.isfinite(epoch_pve[dim]):
                        train_pve_dim.append(epoch_pve[dim])
                    else:
                        train_pve_dim.append(np.nan)
                
                ax.plot(epochs[:len(train_pve_dim)], train_pve_dim, 
                       label=f'Dim {dim} (Train)', alpha=0.7, linestyle='-')
            
            # Plot validation PVE if available
            if 'val_pve_by_dim' in history and len(history['val_pve_by_dim']) > 0 and len(val_epochs) > 0:
                for dim in range(min(num_dims, 5)):
                    val_pve_dim = []
                    for epoch_pve in history['val_pve_by_dim']:
                        if len(epoch_pve) > dim and np.isfinite(epoch_pve[dim]):
                            val_pve_dim.append(epoch_pve[dim])
                        else:
                            val_pve_dim.append(np.nan)
                    
                    if len(val_pve_dim) == len(val_epochs):
                        ax.plot(val_epochs, val_pve_dim, 
                               label=f'Dim {dim} (Val)', marker='o', alpha=0.7, linestyle='--')
        
        ax.set_title('Percent Variance Explained', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('% Variance Explained')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Set ylim appropriately
        all_pve_values = []
        for epoch_pve in history.get('train_pve_by_dim', []):
            all_pve_values.extend([p for p in epoch_pve if np.isfinite(p)])
        for epoch_pve in history.get('val_pve_by_dim', []):
            all_pve_values.extend([p for p in epoch_pve if np.isfinite(p)])
        if all_pve_values:
            pve_min = min(all_pve_values)
            pve_max = max(all_pve_values)
            y_range = pve_max - pve_min
            ax.set_ylim(bottom=pve_min - 0.1 * y_range, top=pve_max + 0.1 * y_range)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, alpha=0.5)
        ax.set_title('Percent Variance Explained', fontsize=12, fontweight='bold')
    
    # Panel 9: Loss Components Comparison (Standard VAE)
    ax = axes_flat[8]
    has_recon = 'train_recon_losses' in history and len(history['train_recon_losses']) > 0
    has_kl_comp = has_kl
    has_gmm_comp = has_gmm
    has_vq_comp = has_vq
    
    if has_recon and (has_kl_comp or has_gmm_comp or has_vq_comp):
        ax.plot(epochs, history['train_recon_losses'], label='Recon', color='blue', linewidth=2)
        if has_kl_comp:
            ax.plot(epochs, history['train_kl_losses'], label='KL', color='green', linewidth=2)
        if has_gmm_comp:
            ax.plot(epochs, history['train_gmm_losses'], label='GMM', color='purple', linewidth=2)
        if has_vq_comp:
            ax.plot(epochs, history['train_vq_losses'], label='VQ', color='cyan', linewidth=2)
        ax.set_title('Loss Components Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, alpha=0.5)
        ax.set_title('Loss Components', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = Path(output_dir) / save_name
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {vae_type} training progress plot to {save_path}")


def create_cluster_means_over_time_plot(
    history: Dict[str, list],
    save_path: Optional[str] = None,
    top_n_clusters: int = 10
) -> None:
    """Plot cluster means (first two dimensions) over time for most commonly used clusters.
    
    This function is specific to VBVAE and will show N/A if cluster means are not available.
    
    Args:
        history: Training history dictionary with 'cluster_means' and 'normalized_pi' keys
        save_path: Optional path to save the plot
        top_n_clusters: Number of top clusters to plot
    """
    if 'cluster_means' not in history or len(history['cluster_means']) == 0:
        print("Warning: No cluster means tracked in history")
        return
    
    if 'normalized_pi' not in history or len(history['normalized_pi']) == 0:
        print("Warning: No mixing weights tracked in history")
        return
    
    # Get final mixing weights to determine top clusters
    final_pi = np.array(history['normalized_pi'][-1])
    top_cluster_indices = np.argsort(final_pi)[-top_n_clusters:][::-1]  # Top N clusters
    
    # Get cluster means over time
    cluster_means_over_time = np.array(history['cluster_means'])  # [num_epochs, num_clusters, 2]
    epochs = range(len(cluster_means_over_time))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Cluster Means Over Time (Top Clusters)', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_cluster_indices)))
    
    # Plot 1: Cluster means trajectory (x1, x2) over epochs
    ax = axes[0]
    for idx, k in enumerate(top_cluster_indices):
        means_k = cluster_means_over_time[:, k, :]  # [num_epochs, 2]
        ax.plot(means_k[:, 0], means_k[:, 1], 
               color=colors[idx], alpha=0.6, linewidth=2, label=f'Cluster {k}')
        # Mark start and end points
        ax.scatter(means_k[0, 0], means_k[0, 1], 
                  color=colors[idx], marker='o', s=100, alpha=0.8, zorder=5)
        ax.scatter(means_k[-1, 0], means_k[-1, 1], 
                  color=colors[idx], marker='s', s=100, alpha=0.8, zorder=5)
    
    ax.set_xlabel('Latent Dimension 0 (x1)')
    ax.set_ylabel('Latent Dimension 1 (x2)')
    ax.set_title(f'Cluster Mean Trajectories (Top {len(top_cluster_indices)} Clusters)\nCircles=Start, Squares=End')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot 2: Cluster means over epochs (separate plots for x1 and x2)
    ax = axes[1]
    for idx, k in enumerate(top_cluster_indices):
        means_k = cluster_means_over_time[:, k, :]  # [num_epochs, 2]
        ax.plot(epochs, means_k[:, 0], 
               color=colors[idx], linestyle='-', alpha=0.7, linewidth=1.5, 
               label=f'Cluster {k} (x1)')
        ax.plot(epochs, means_k[:, 1], 
               color=colors[idx], linestyle='--', alpha=0.7, linewidth=1.5, 
               label=f'Cluster {k} (x2)')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cluster Mean Value')
    ax.set_title(f'Cluster Means Over Time\nSolid=x1, Dashed=x2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Cluster means plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

