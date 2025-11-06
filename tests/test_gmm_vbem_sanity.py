"""Sanity check test: Single Gaussian with severe over-clustering."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flax.core import freeze, unfreeze
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from src.models.vae.vb_gmm import GMMVBEM


def generate_uniform_data(
    n_samples: int = 2000,
    latent_dim: int = 2,
    low: float = -2.0,
    high: float = 2.0,
    key: jr.PRNGKey = None
) -> tuple:
    """Generate data from a uniform distribution on [low, high] for each dimension."""
    if key is None:
        key = jr.PRNGKey(42)
    
    # Generate data from uniform distribution
    key, sample_key = jr.split(key)
    z_e = jr.uniform(sample_key, (n_samples, latent_dim), minval=low, maxval=high)
    
    # Mean and variance of uniform distribution (2D: same for each dimension)
    mean = jnp.array([(low + high) / 2.0] * latent_dim)  # [latent_dim]
    variance = jnp.array([(high - low) ** 2 / 12.0] * latent_dim)  # [latent_dim] Variance of uniform distribution
    
    return z_e, mean, variance


def test_single_gaussian_overclustering():
    """Test GMMVBEM with single Gaussian and severe over-clustering."""
    # Parameters
    num_clusters = 200  # Severe over-clustering
    latent_dim = 2
    n_samples = 2000
    batch_size = 32
    num_epochs = 20
    
    # Generate data
    key = jr.PRNGKey(1234)
    key, data_key = jr.split(key)
    z_e_data, true_mean, true_variance = generate_uniform_data(
        n_samples=n_samples,
        latent_dim=latent_dim,
        low=-2.0,
        high=2.0,
        key=data_key
    )
    
    print(f"Generated {n_samples} samples from uniform distribution on [-2, 2]")
    print(f"True mean: {true_mean}")
    print(f"True variance: {true_variance}")
    print(f"Using {num_clusters} clusters (severe over-clustering)\n")
    
    # Initialize GMMVBEM
    key, init_key = jr.split(key)
    gmm_vbem = GMMVBEM(
        num_clusters=num_clusters,
        latent_dim=latent_dim,
        prior_mu=0.0,
        prior_alpha=0.5,
        prior_beta=0.5 / num_clusters,
        prior_alpha_mix=0.5
    )
    
    # Initialize parameters
    z_e_sample = z_e_data[:10]
    dummy_params = gmm_vbem.init(init_key, z_e_sample)
    gmm_params = unfreeze(dummy_params['params'])
    
    print("Initialized GMM parameters")
    print(f"  mu_n shape: {gmm_params['mu_n'].shape}")
    print(f"  alpha_n shape: {gmm_params['alpha_n'].shape}")
    print(f"  beta_n shape: {gmm_params['beta_n'].shape}")
    print(f"  alpha_mix shape: {gmm_params['alpha_mix'].shape}\n")
    
    # Track parameter evolution
    alpha_mix_history = []
    
    # Training loop
    N_eff = float(n_samples)
    
    for epoch in range(num_epochs):
        # Shuffle data
        key, shuffle_key = jr.split(key)
        perm = jr.permutation(shuffle_key, n_samples)
        z_e_shuffled = z_e_data[perm]
        
        # Process in batches
        num_batches = n_samples // batch_size
        z_e_batches = z_e_shuffled[:num_batches * batch_size].reshape(num_batches, batch_size, latent_dim)
        
        for batch_idx in range(num_batches):
            z_e_batch = z_e_batches[batch_idx]
            
            # Get cluster probabilities and logZ
            gmm_params_frozen = freeze({'params': gmm_params})
            _, cluster_probs, logZ = gmm_vbem.apply(gmm_params_frozen, z_e_batch)
            
            # Update GMM parameters via VBEM (with fill_unused)
            gmm_params = gmm_vbem.update(
                params=gmm_params,
                z_e=z_e_batch,
                cluster_probs=cluster_probs,
                logZ=logZ,
                N_eff=N_eff,
                use_fill_unused=True  # Use fill_unused
            )
        
        # Store history every epoch
        alpha_mix_history.append(np.array(gmm_params['alpha_mix']))
        
        # Compute expected statistics
        gmm_params_frozen = freeze({'params': gmm_params})
        expectations = gmm_vbem.apply(
            gmm_params_frozen,
            method='nat_to_stats'
        )
        
        E_pi = np.array(expectations['E_pi'])
        
        # Count active clusters (alpha_mix > 5.5 means more than 5 data points)
        active_clusters = np.sum(gmm_params['alpha_mix'] > 5.5)
        
        print(f"Epoch {epoch}:")
        print(f"  Active clusters (alpha_mix > 5.5): {active_clusters}")
        print(f"  Max mixing weight: {np.max(E_pi):.6f}")
        print(f"  Min mixing weight: {np.min(E_pi):.6f}")
        print(f"  Mean mixing weight: {np.mean(E_pi):.6f}\n")
    
    # Final results
    gmm_params_frozen = freeze({'params': gmm_params})
    final_expectations = gmm_vbem.apply(gmm_params_frozen, method='nat_to_stats')
    
    E_mu_final = np.array(final_expectations['E_mu'])
    alpha_n_final = np.array(gmm_params['alpha_n'])
    beta_n_final = np.array(gmm_params['beta_n'])
    E_var_final = beta_n_final / (alpha_n_final - 1.0)
    E_pi_final = np.array(final_expectations['E_pi'])
    alpha_mix_final = np.array(gmm_params['alpha_mix'])
    
    # Count active clusters
    active_mask = alpha_mix_final > 5.5
    active_cluster_indices = np.where(active_mask)[0]
    num_active = len(active_cluster_indices)
    
    print("=" * 60)
    print("Final Results:")
    print("=" * 60)
    print(f"Total clusters: {num_clusters}")
    print(f"Active clusters (alpha_mix > 5.5): {num_active}")
    print(f"True mean: {true_mean}")
    print(f"True variance: {true_variance}")
    print(f"\nLearned statistics (from active clusters):")
    if num_active > 0:
        active_means = E_mu_final[active_cluster_indices]
        active_vars = E_var_final[active_cluster_indices]
        active_weights = E_pi_final[active_cluster_indices]
        
        # Weighted mean and variance
        weighted_mean = np.sum(active_weights[:, None] * active_means, axis=0)
        weighted_var = np.sum(active_weights[:, None] * active_vars, axis=0)
        
        print(f"  Weighted mean: {weighted_mean}")
        print(f"  Weighted variance: {weighted_var}")
        print(f"  Mean error: {np.linalg.norm(weighted_mean - true_mean):.4f}")
        print(f"  Variance error: {np.abs(weighted_var - true_variance).mean():.4f}")
    else:
        print("  No active clusters found!")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Data distribution
    ax = axes[0, 0]
    ax.scatter(z_e_data[:, 0], z_e_data[:, 1], alpha=0.3, s=10, c='blue', label='Data')
    # Handle true_mean as either scalar or array
    if isinstance(true_mean, (int, float)):
        true_mean_plot = [true_mean, true_mean]
    else:
        true_mean_plot = true_mean
    ax.scatter(true_mean_plot[0], true_mean_plot[1], c='red', marker='x', s=200, linewidths=3, label='True mean')
    ax.set_title('Data Distribution')
    ax.set_xlabel('Dimension 0')
    ax.set_ylabel('Dimension 1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot 2: Learned clusters - color data points by their most likely assignment
    ax = axes[0, 1]
    if num_active > 0:
        # Get cluster assignments for all data points
        gmm_params_frozen = freeze({'params': gmm_params})
        _, cluster_probs_all, _ = gmm_vbem.apply(gmm_params_frozen, z_e_data)
        cluster_probs_all = np.array(cluster_probs_all)
        assignments = np.argmax(cluster_probs_all, axis=-1)  # [n_samples]
        
        # Create color map for active clusters only
        cmap = plt.cm.tab20  # Use tab20 colormap for better color distinction
        active_cluster_colors = {idx: cmap(i % 20) for i, idx in enumerate(active_cluster_indices)}
        
        # Color points by their assigned cluster (only if assigned to active cluster)
        point_colors = []
        for assignment in assignments:
            if assignment in active_cluster_indices:
                point_colors.append(active_cluster_colors[assignment])
            else:
                point_colors.append('lightgray')  # Gray for inactive clusters
        
        ax.scatter(z_e_data[:, 0], z_e_data[:, 1], alpha=0.5, s=15, c=point_colors)
        
        # Plot cluster means
        for idx, k in enumerate(active_cluster_indices):
            mean = E_mu_final[k]
            color = active_cluster_colors[k]
            ax.scatter(mean[0], mean[1], c=[color], marker='x', s=100, linewidths=3, 
                      edgecolors='black', zorder=10)
        
        ax.set_title(f'Data Colored by Cluster Assignment ({num_active} active clusters)')
    else:
        ax.scatter(z_e_data[:, 0], z_e_data[:, 1], alpha=0.3, s=10, c='gray', label='Data (no active clusters)')
        ax.set_title('Data (No Active Clusters)')
    ax.set_xlabel('Dimension 0')
    ax.set_ylabel('Dimension 1')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(axes[0, 0].get_xlim())
    ax.set_ylim(axes[0, 0].get_ylim())
    
    # Plot 3: Mixing weights evolution (top 20 clusters by final weight)
    ax = axes[1, 0]
    epochs_plot = list(range(len(alpha_mix_history)))
    if num_active > 0:
        # Get top clusters by final mixing weight
        top_indices = np.argsort(E_pi_final)[-20:][::-1]
        for k in top_indices:
            pi_history = [alpha_mix[k] / (alpha_mix.sum() + 1e-8) for alpha_mix in alpha_mix_history]
            ax.plot(epochs_plot, pi_history, marker='o', label=f'Cluster {k}', markersize=3, linewidth=1)
    ax.set_title('Top 20 Mixing Weights Evolution')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mixing Weight')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Number of active clusters over time
    ax = axes[1, 1]
    active_counts = [np.sum(alpha_mix > 5.5) for alpha_mix in alpha_mix_history]
    ax.plot(epochs_plot, active_counts, marker='o', linewidth=2, markersize=6)
    ax.set_title('Number of Active Clusters Over Time')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Active Clusters (alpha_mix > 5.5)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Expected (1 cluster)')
    ax.legend()
    
    plt.tight_layout()
    output_path = Path('test_gmm_vbem_sanity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    test_single_gaussian_overclustering()

