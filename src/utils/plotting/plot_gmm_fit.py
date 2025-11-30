"""Plotting utilities for GMM visualization."""

import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt


def plot_gmm_clusters(
    cluster_means: np.ndarray,
    data_points: np.ndarray,
    assignments: Optional[np.ndarray] = None,
    output_path: str = "gmm_clusters.png",
    title: str = "GMM Clusters"
) -> None:
    """Plot GMM cluster means and data point assignments.
    
    Args:
        cluster_means: Cluster means array of shape [num_clusters, dim]
        data_points: Data points array of shape [num_samples, dim]
        assignments: Optional cluster assignments array of shape [num_samples].
                    If None, will be computed using nearest cluster mean.
        output_path: Path to save the plot
        title: Plot title
    """
    cluster_means = np.array(cluster_means)
    data_points = np.array(data_points)
    
    # Check dimensions
    if cluster_means.shape[1] != 2 or data_points.shape[1] != 2:
        print(f"Skipping GMM plot: only 2D data supported (cluster_means: {cluster_means.shape}, data_points: {data_points.shape})")
        return
    
    num_clusters = cluster_means.shape[0]
    
    # Compute assignments if not provided (nearest cluster mean)
    if assignments is None:
        # Compute distances from each data point to each cluster mean
        distances = np.sum((data_points[:, np.newaxis, :] - cluster_means[np.newaxis, :, :]) ** 2, axis=2)
        assignments = np.argmin(distances, axis=1)
    
    assignments = np.array(assignments)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot data points colored by cluster assignment
    colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))
    for k in range(num_clusters):
        mask = assignments == k
        if np.any(mask):
            ax.scatter(
                data_points[mask, 0],
                data_points[mask, 1],
                alpha=0.5,
                s=20,
                c=[colors[k]],
                label=f'Cluster {k}' if k < 10 else None,
                edgecolors='black',
                linewidths=0.5
            )
    
    # Plot cluster means
    ax.scatter(
        cluster_means[:, 0],
        cluster_means[:, 1],
        s=400,
        c='red',
        marker='X',
        linewidths=3,
        edgecolors='black',
        label='Cluster Means',
        zorder=10
    )
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved GMM cluster plot to {save_path}")


def create_gmm_fit_plot(
    model,
    params: Dict[str, Any],
    output_dir: str,
    y_data: Optional[jnp.ndarray] = None,
    max_samples: int = 2000,
    rng: Optional[Any] = None
) -> None:
    """Create a visualization of the GMM fit in latent space.
    
    This function visualizes how the GMM clusters fit the data by:
    1. Encoding data points to latent space
    2. Assigning data points to clusters based on latent representations
    3. Plotting data points colored by cluster assignment
    4. Plotting cluster means
    
    Args:
        model: Model instance with encode method and flow_planner.gmm
        params: Model parameters dictionary
        output_dir: Directory to save the plot
        y_data: Training data to encode and visualize (optional)
        max_samples: Maximum number of samples to visualize
        rng: Random key for sampling (optional)
    """
    import jax.random as jr
    
    try:
        # Get GMM parameters
        gmm_params = params['params']['flow_planner']['gmm']
        cluster_means = np.array(gmm_params['mu_n'])  # [num_clusters, latent_dim]
        num_clusters = cluster_means.shape[0]
        latent_dim = cluster_means.shape[1]
        
        # Only plot if latent_dim is 2D
        if latent_dim != 2:
            print(f"Skipping GMM fit plot: latent_dim={latent_dim} (only 2D supported)")
            return
        
        if y_data is None:
            # Just plot cluster means if no data provided
            plot_gmm_clusters(
                cluster_means=cluster_means,
                data_points=np.array([]).reshape(0, 2),
                output_path=Path(output_dir) / "gmm_fit.png",
                title="GMM Cluster Means"
            )
            return
        
        # Sample a subset for visualization
        n_viz = min(max_samples, y_data.shape[0])
        if rng is None:
            key = jr.PRNGKey(42)
        else:
            key = rng
            
        if n_viz < y_data.shape[0]:
            key, sample_key = jr.split(key)
            indices = jr.choice(sample_key, y_data.shape[0], shape=(n_viz,), replace=False)
            y_viz = y_data[indices]
        else:
            y_viz = y_data
        
        # Encode to latent space
        key, encode_key = jr.split(key)
        mu_z_target, _ = model.apply(
            params, y_viz, method='encode', training=False, rngs={'dropout': encode_key}
        )
        z_target = np.array(mu_z_target)  # [n_viz, latent_dim]
        
        # Get cluster assignments using GMM
        z_target_jax = jnp.array(z_target)
        
        # Try to use model's flow_planner.gmm if available
        try:
            log_p_tilde = model.flow_planner.gmm.apply(
                {'params': gmm_params},
                z_target_jax,
                training=False,
                method='log_p_tilde'
            )
        except (AttributeError, TypeError):
            # Fallback: create temporary GMM instance
            from src.vae.vb_gmm import create_gmm_vbem, GMMVBEMConfig
            from flax.core import freeze
            gmm_config = model.config.flow_planner.get('gmm', {})
            gmm_vbem_config = GMMVBEMConfig(
                num_clusters=gmm_config.get('num_clusters', 8),
                latent_dim=latent_dim
            )
            gmm_temp = create_gmm_vbem(gmm_vbem_config)
            log_p_tilde = gmm_temp.apply(
                freeze({'params': gmm_params}),
                z_target_jax,
                training=False,
                method='log_p_tilde'
            )
        
        assignments = np.argmax(np.array(log_p_tilde), axis=1)
        
        # Plot using the simplified function
        plot_gmm_clusters(
            cluster_means=cluster_means,
            data_points=z_target,
            assignments=assignments,
            output_path=Path(output_dir) / "gmm_fit.png",
            title="GMM Fit in Latent Space"
        )
        
    except Exception as e:
        import traceback
        print(f"Warning: Error creating GMM fit plot: {e}")
        traceback.print_exc()
