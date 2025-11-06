"""Test script for GMMVBEM component."""

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


def generate_gmm_data(
    n_samples: int = 1000,
    num_clusters: int = 3,
    latent_dim: int = 2,
    key: jr.PRNGKey = None
) -> tuple:
    """Generate synthetic GMM data."""
    if key is None:
        key = jr.PRNGKey(42)
    
    key, k1, k2, k3 = jr.split(key, 4)
    
    # True means
    true_means = jr.normal(k1, (num_clusters, latent_dim)) * 1.0
    
    # True variances (diagonal)
    true_vars = jnp.ones((num_clusters, latent_dim)) * 0.2 + jr.uniform(k2, (num_clusters, latent_dim)) * 0.3
    
    # True mixing weights
    true_mix_weights = jnp.ones(num_clusters) / num_clusters
    
    # Generate data
    assignments = jr.categorical(k3, logits=jnp.log(true_mix_weights + 1e-8), shape=(n_samples,))
    
    # Sample from assigned clusters
    keys = jr.split(k1, n_samples)
    samples = []
    for i, (key_i, cluster_idx) in enumerate(zip(keys, assignments)):
        mean = true_means[cluster_idx]
        var = true_vars[cluster_idx]
        sample = mean + jr.normal(key_i, (latent_dim,)) * jnp.sqrt(var)
        samples.append(sample)
    
    z_e = jnp.array(samples)
    
    # Shuffle data
    key, shuffle_key = jr.split(key)
    perm = jr.permutation(shuffle_key, n_samples)
    z_e = z_e[perm]
    assignments = assignments[perm]
    
    return z_e, assignments, true_means, true_vars, true_mix_weights


def test_gmm_vbem():
    """Test GMMVBEM update function."""
    # Parameters
    true_num_clusters = 3
    num_clusters = 9  # Over-clustering
    latent_dim = 2
    n_samples = 800
    batch_size = 32
    num_epochs = 50
    
    # Generate data
    key = jr.PRNGKey(1234)
    key, data_key = jr.split(key)
    z_e_data, true_assignments, true_means, true_vars, true_mix_weights = generate_gmm_data(
        n_samples=n_samples,
        num_clusters=true_num_clusters,
        latent_dim=latent_dim,
        key=data_key
    )
    
    print(f"Generated {n_samples} samples from {true_num_clusters} true clusters")
    print(f"Using {num_clusters} clusters")
    print(f"True means:\n{true_means}")
    print(f"True variances:\n{true_vars}")
    print(f"True mixing weights: {true_mix_weights}\n")
    
    # Initialize GMMVBEM
    key, init_key = jr.split(key)
    gmm_vbem = GMMVBEM(
        num_clusters=num_clusters,
        latent_dim=latent_dim,
        prior_mu=0.0,
        prior_alpha=1.5,
        prior_beta=0.1 / num_clusters,
        prior_alpha_mix=1.0
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
    mu_history = []
    alpha_mix_history = []
    
    # Training loop
    N_eff = 800.0
    
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
            
            # Update GMM parameters via VBEM
            gmm_params = gmm_vbem.update(
                params=gmm_params,
                z_e=z_e_batch,
                cluster_probs=cluster_probs,
                logZ=logZ,
                N_eff=N_eff
                # lr defaults to 0.2
            )
        
        # Store history every epoch
        mu_history.append(np.array(gmm_params['mu_n']))
        alpha_mix_history.append(np.array(gmm_params['alpha_mix']))
        
        # Compute expected statistics
        expectations = gmm_vbem.apply(
            freeze({'params': gmm_params}),
            method='nat_to_stats'
        )
        
        E_mu = np.array(expectations['E_mu'])
        alpha_n = np.array(gmm_params['alpha_n'])
        beta_n = np.array(gmm_params['beta_n'])
        E_var = beta_n / (alpha_n - 1.0)
        E_pi = np.array(expectations['E_pi'])
        
        print(f"Epoch {epoch}:")
        print(f"  Learned means:\n{E_mu}")
        print(f"  Learned variances:\n{E_var}")
        print(f"  Learned mixing weights: {E_pi}\n")
    
    # Final results
    gmm_params_frozen = freeze({'params': gmm_params})
    final_expectations = gmm_vbem.apply(gmm_params_frozen, method='nat_stats')
    
    E_mu_final = np.array(final_expectations['E_mu'])
    alpha_n_final = np.array(gmm_params['alpha_n'])
    beta_n_final = np.array(gmm_params['beta_n'])
    E_var_final = beta_n_final / (alpha_n_final - 1.0)
    E_pi_final = np.array(final_expectations['E_pi'])
    alpha_mix_final = np.array(gmm_params['alpha_mix'])
    
    # Filter clusters with more than 5 assigned data points (alpha_mix > 5.5) for upper right plot
    active_mask = alpha_mix_final > 5.5
    active_cluster_indices = np.where(active_mask)[0]
    
    print("=" * 60)
    print("Final Results:")
    print("=" * 60)
    
    # Match learned clusters to true clusters using Hungarian algorithm
    from scipy.optimize import linear_sum_assignment
    
    distances = np.zeros((num_clusters, true_num_clusters))
    for i in range(num_clusters):
        for j in range(true_num_clusters):
            distances[i, j] = np.linalg.norm(E_mu_final[i] - true_means[j])
    
    row_ind, col_ind = linear_sum_assignment(distances)
    E_mu_matched = E_mu_final[row_ind[:true_num_clusters]]
    E_var_matched = E_var_final[row_ind[:true_num_clusters]]
    E_pi_matched = E_pi_final[row_ind[:true_num_clusters]]
    
    # Compute errors
    mean_error = np.mean(np.abs(E_mu_matched - true_means))
    mean_rel_error = np.mean(np.abs(E_mu_matched - true_means) / (np.abs(true_means) + 1e-8)) * 100
    var_error = np.mean(np.abs(E_var_matched - true_vars))
    var_rel_error = np.mean(np.abs(E_var_matched - true_vars) / (true_vars + 1e-8)) * 100
    pi_error = np.mean(np.abs(E_pi_matched - true_mix_weights))
    
    print(f"True means:\n{true_means}")
    print(f"Learned means:\n{E_mu_matched}")
    print(f"Mean error: {mean_error:.4f} ({mean_rel_error:.1f}%)\n")
    
    print(f"True variances:\n{true_vars}")
    print(f"Learned variances:\n{E_var_matched}")
    print(f"Variance error: {var_error:.4f} ({var_rel_error:.1f}%)\n")
    
    print(f"True mixing weights: {true_mix_weights}")
    print(f"Learned mixing weights: {E_pi_matched}")
    print(f"Mixing weight error: {pi_error:.4f}\n")
    
    # Assessment
    print("Accuracy Assessment:")
    print(f"  Means: {'✓' if mean_rel_error < 25 else '~' if mean_rel_error < 50 else '✗'}")
    print(f"  Variances: {'✓' if var_rel_error < 100 else '~' if var_rel_error < 200 else '✗'}")
    print(f"  Mixing weights: {'✓' if pi_error < 0.15 else '~' if pi_error < 0.30 else '✗'}\n")
    
    # Plot results
    def plot_ellipse(ax, mean, var, color, alpha=0.3, linewidth=2):
        """Plot 2-sigma ellipse for a Gaussian with diagonal covariance."""
        from matplotlib.patches import Ellipse
        var = np.asarray(var)
        if var.ndim > 0:
            var = np.maximum(var[:2], 0.01)  # Clip negative variances to small positive value
            width, height = 4 * np.sqrt(var)  # 2-sigma ellipse for 2D
        else:
            var = max(var, 0.01)
            width = height = 4 * np.sqrt(var)
        ellipse = Ellipse(mean, width, height, angle=0, 
                         facecolor=color, edgecolor=color, alpha=alpha, linewidth=linewidth)
        ax.add_patch(ellipse)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    # Plot 1: True clusters
    ax = axes[0, 0]
    for k in range(true_num_clusters):
        mask = true_assignments == k
        ax.scatter(z_e_data[mask, 0], z_e_data[mask, 1], 
                  c=colors[k], alpha=0.3, s=20)
        plot_ellipse(ax, true_means[k], true_vars[k], colors[k], alpha=0.2, linewidth=2)
    ax.scatter(true_means[:, 0], true_means[:, 1], 
              c='black', marker='x', s=200, linewidths=3, label='True means')
    ax.set_title('Data with True Clusters')
    ax.set_xlabel('Dimension 0')
    ax.set_ylabel('Dimension 1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot 2: Learned clusters (only active clusters)
    ax = axes[0, 1]
    for k in range(true_num_clusters):
        mask = true_assignments == k
        ax.scatter(z_e_data[mask, 0], z_e_data[mask, 1], 
                  c=colors[k], alpha=0.3, s=20)
    for idx, k in enumerate(active_cluster_indices):
        plot_ellipse(ax, E_mu_final[k], E_var_final[k], colors[idx % len(colors)], 
                    alpha=0.2, linewidth=2)
    if len(active_cluster_indices) > 0:
        ax.scatter(E_mu_final[active_cluster_indices, 0], E_mu_final[active_cluster_indices, 1], 
                  c='black', marker='x', s=200, linewidths=3, label='Learned means')
    ax.set_title('Data with Learned Clusters')
    ax.set_xlabel('Dimension 0')
    ax.set_ylabel('Dimension 1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(axes[0, 0].get_xlim())
    ax.set_ylim(axes[0, 0].get_ylim())
    
    # Plot 3: Mean evolution (all clusters)
    ax = axes[1, 0]
    epochs_plot = list(range(len(mu_history)))
    for k in range(num_clusters):
        mu_0_history = [mu[k, 0] for mu in mu_history]
        mu_1_history = [mu[k, 1] for mu in mu_history]
        ax.plot(epochs_plot, mu_0_history, marker='o', label=f'C{k}, dim 0', markersize=4)
        ax.plot(epochs_plot, mu_1_history, marker='s', linestyle='--', label=f'C{k}, dim 1', markersize=4)
    ax.set_title('Cluster Mean Evolution')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Value')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Mixing weights evolution (all clusters)
    ax = axes[1, 1]
    for k in range(num_clusters):
        pi_history = [alpha_mix[k] / (alpha_mix.sum() + 1e-8) for alpha_mix in alpha_mix_history]
        ax.plot(epochs_plot, pi_history, marker='o', label=f'Cluster {k}', markersize=4)
    for k in range(true_num_clusters):
        ax.axhline(true_mix_weights[k], color=colors[k], linestyle=':', alpha=0.5)
    ax.set_title('Mixing Weight Evolution')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mixing Weight')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path('test_gmm_vbem_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    test_gmm_vbem()

