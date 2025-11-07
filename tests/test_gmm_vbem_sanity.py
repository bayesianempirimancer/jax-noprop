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
import time

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
    num_clusters = 100  # Severe over-clustering
    latent_dim = 2
    n_samples = 2000
    batch_size = 256
    num_epochs = 100
    
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
        prior_mu=0.0,  # Scalar, will be converted to array [latent_dim]
        prior_alpha=0.5,
        prior_beta=0.5 / num_clusters,
        prior_alpha_mix=1.0,  # Standard prior
        beta_mix=0.1,  # Low mixing temperature
        tie_precisions=False  # Allow clusters to have different precisions
    )
    
    # Initialize parameters
    z_e_sample = z_e_data[:10]
    dummy_params = gmm_vbem.init(init_key, z_e_sample)
    gmm_params = unfreeze(dummy_params['params'])
    
    # Initialize cluster means from random data points
    key, init_subset_key = jr.split(key)
    # Must have at least num_clusters samples
    n_init_samples = max(num_clusters, min(num_clusters * 2, n_samples))
    init_indices = jr.permutation(init_subset_key, n_samples)[:n_init_samples]
    z_e_init = z_e_data[init_indices]
    gmm_params = gmm_vbem.initialize_cluster_means(
        params=gmm_params,
        z_e=z_e_init,
        key=init_subset_key
    )
    
    print("Initialized GMM parameters")
    print(f"  mu_n shape: {gmm_params['mu_n'].shape}")
    print(f"  alpha_n shape: {gmm_params['alpha_n'].shape}")
    print(f"  beta_n shape: {gmm_params['beta_n'].shape}")
    print(f"  alpha_mix shape: {gmm_params['alpha_mix'].shape}\n")
    
    # Track parameter evolution and loss
    alpha_mix_history = []
    loss_history = []
    kl_history = []
    neg_log_likelihood_history = []
    active_clusters_history = []
    epoch_times = []
    
    # Training loop
    N_eff = 10000.0  # Effective number of data points
    
    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        # Shuffle data for each epoch
        key, shuffle_key = jr.split(key)
        perm = jr.permutation(shuffle_key, n_samples)
        z_e_shuffled = z_e_data[perm]
        
        # Process in batches
        num_batches = n_samples // batch_size
        z_e_batches = z_e_shuffled[:num_batches * batch_size].reshape(num_batches, batch_size, latent_dim)
        
        for batch_idx in range(num_batches):
            z_e_batch = z_e_batches[batch_idx]
            
            # Update GMM parameters via VBEM
            # Call update via apply since it's now @nn.compact
            gmm_params_frozen = freeze({'params': gmm_params})
            gmm_params = gmm_vbem.apply(
                gmm_params_frozen,
                z_e_batch,
                N_eff=N_eff,
                lr=0.2,
                training=True,
                method='update'
            )
        
        # Store history every epoch
        alpha_mix_history.append(np.array(gmm_params['alpha_mix']))
        
        # Compute loss for this epoch and track components
        gmm_params_frozen_loss = freeze({'params': gmm_params})
        # Use full dataset to compute loss
        loss_value = gmm_vbem.apply(gmm_params_frozen_loss, z_e_data, training=False, method='loss')
        loss_history.append(float(loss_value))
        
        # Compute KL divergence separately and break it down
        kl_value = gmm_vbem.apply(gmm_params_frozen_loss, training=False, method='kl_prior')
        kl_history.append(float(kl_value))
        
        # Debug: compute KL components separately for first few epochs
        if epoch < 3:
            # Get parameters
            mu_n = np.array(gmm_params['mu_n'])  # [100, 2]
            alpha_n = np.array(gmm_params['alpha_n'])  # [100, 1]
            beta_n = np.array(gmm_params['beta_n'])  # [100, 2]
            alpha_mix = np.array(gmm_params['alpha_mix'])  # [100]
            
            # Compute Normal-Gamma KL manually to see breakdown
            from src.utils.kl_divergence import normal_gamma_kl, dirichlet_kl
            kappa_n = 2.0 * alpha_n  # [100, 1]
            kappa_prior = 2.0 * 0.5  # scalar = 1.0
            prior_beta_scaled = 0.5 / num_clusters**(2/latent_dim)
            
            kl_ng = normal_gamma_kl(
                kappa_p=kappa_n,
                mu_p=mu_n,
                alpha_p=alpha_n,
                beta_p=beta_n,
                kappa_q=kappa_prior,
                mu_q=0.0,
                alpha_q=0.5,
                beta_q=prior_beta_scaled
            )  # [100, 2]
            kl_ng_total = float(np.sum(kl_ng))
            
            kl_dir = dirichlet_kl(alpha_mix, 1.0)  # [1]
            kl_dir_total = float(np.sum(kl_dir))
            
            print(f"    KL breakdown: Normal-Gamma={kl_ng_total:.2f}, Dirichlet={kl_dir_total:.2f}, Total={kl_value:.2f}")
            print(f"    Per cluster-dim KL (mean): {np.mean(kl_ng):.2f}, (max): {np.max(kl_ng):.2f}, (min): {np.min(kl_ng):.2f}")
        
        # Compute negative log-likelihood separately
        log_p_tilde = gmm_vbem.apply(gmm_params_frozen_loss, z_e_data, training=False, method='log_p_tilde')
        from src.utils.math_utils import logsumexp
        logZ = logsumexp(log_p_tilde, axis=-1)
        neg_log_likelihood = -float(jnp.mean(logZ))
        neg_log_likelihood_history.append(neg_log_likelihood)
        
        # Compute expected statistics
        gmm_params_frozen = freeze({'params': gmm_params})
        expectations = gmm_vbem.apply(
            gmm_params_frozen,
            training=False,
            method='nat_to_stats'
        )
        
        E_pi = np.array(expectations['E_pi'])
        
        # Count active clusters via actual assignments
        gmm_params_frozen_epoch = freeze({'params': gmm_params})
        log_p_tilde = gmm_vbem.apply(gmm_params_frozen_epoch, z_e_data, training=False, method='log_p_tilde')
        log_p_tilde = np.array(log_p_tilde)  # [n_samples, num_clusters]
        cluster_assignments = np.argmax(log_p_tilde, axis=-1)  # [n_samples]
        unique_clusters = np.unique(cluster_assignments)
        active_clusters = len(unique_clusters)
        active_clusters_history.append(active_clusters)
        
        epoch_end_time = time.perf_counter()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Calculate average times
        avg_epoch_time = np.mean(epoch_times) if epoch_times else epoch_time
        num_batches = n_samples // batch_size
        avg_step_time = avg_epoch_time / num_batches if num_batches > 0 else 0.0
        
        print(f"Epoch {epoch}:")
        print(f"  Active clusters (via assignments): {active_clusters}")
        print(f"  Max mixing weight: {np.max(E_pi):.6f}")
        print(f"  Min mixing weight: {np.min(E_pi):.6f}")
        print(f"  Mean mixing weight: {np.mean(E_pi):.6f}")
        print(f"  Epoch time: {epoch_time:.3f}s (avg={avg_epoch_time:.3f}s), step_time={avg_step_time*1000:.2f}ms")
        if epoch < 10 or epoch % 10 == 0:  # Print loss components for first 10 epochs and every 10th
            print(f"  Loss: {loss_history[-1]:.4f}, NLL: {neg_log_likelihood_history[-1]:.4f}, KL: {kl_history[-1]:.4f}\n")
        else:
            print()
    
    # Final results
    gmm_params_frozen = freeze({'params': gmm_params})
    final_expectations = gmm_vbem.apply(gmm_params_frozen, training=False, method='nat_to_stats')
    
    E_mu_final = np.array(final_expectations['E_mu'])
    alpha_n_final = np.array(gmm_params['alpha_n'])
    beta_n_final = np.array(gmm_params['beta_n'])
    
    # Compute variance: if tie_precisions is True, use summed alpha and beta
    if gmm_vbem.tie_precisions:
        # Sum across clusters, then compute variance (same for all clusters)
        alpha_sum = np.sum(alpha_n_final, axis=0, keepdims=True)  # [1, latent_dim]
        beta_sum = np.sum(beta_n_final, axis=0, keepdims=True)  # [1, latent_dim]
        E_var_tied = beta_sum / (alpha_sum - 1.0)  # [1, latent_dim]
        # Broadcast to all clusters
        E_var_final = np.broadcast_to(E_var_tied, (gmm_vbem.num_clusters, gmm_vbem.latent_dim))
    else:
        E_var_final = beta_n_final / (alpha_n_final - 1.0)
    
    E_pi_final = np.array(final_expectations['E_pi'])
    alpha_mix_final = np.array(gmm_params['alpha_mix'])
    
    # Count active clusters via actual assignments
    log_p_tilde = gmm_vbem.apply(gmm_params_frozen, z_e_data, training=False, method='log_p_tilde')
    log_p_tilde = np.array(log_p_tilde)  # [n_samples, num_clusters]
    cluster_assignments = np.argmax(log_p_tilde, axis=-1)  # [n_samples]
    unique_clusters = np.unique(cluster_assignments)
    active_cluster_indices = unique_clusters
    num_active = len(active_cluster_indices)
    
    # Print timing summary
    if epoch_times:
        total_time = np.sum(epoch_times)
        avg_epoch_time = np.mean(epoch_times)
        num_batches = n_samples // batch_size
        avg_step_time = avg_epoch_time / num_batches if num_batches > 0 else 0.0
        print("=" * 60)
        print("Training Timing Summary:")
        print("=" * 60)
        print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Average time per epoch: {avg_epoch_time:.3f}s")
        print(f"Average time per step (batch): {avg_step_time*1000:.2f}ms")
        print()
    
    print("=" * 60)
    print("Final Results:")
    print("=" * 60)
    print(f"Total clusters: {num_clusters}")
    print(f"Active clusters (via assignments): {num_active}")
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
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
        # Get cluster assignments for all data points using minibatches (same batch_size as training)
        gmm_params_frozen = freeze({'params': gmm_params})
        assignments_list = []
        try:
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                z_e_batch = z_e_data[i:batch_end]
                _, log_p_tilde_batch = gmm_vbem.apply(gmm_params_frozen, z_e_batch, training=False, method='quantize')
                # Compute cluster_probs from log_p_tilde using numerically stable softmax
                from src.utils.math_utils import stable_softmax
                cluster_probs_batch = stable_softmax(log_p_tilde_batch, axis=-1)  # [batch, num_clusters]
                assignments_batch = np.argmax(np.array(cluster_probs_batch), axis=-1)  # [batch]
                assignments_list.append(assignments_batch)
            # Concatenate all assignments (handles variable batch sizes)
            assignments = np.concatenate([a.flatten() for a in assignments_list])  # [n_samples]
        except Exception as e:
            print(f"Warning: Could not compute assignments for plotting: {e}")
            # Fallback: just plot the cluster means without coloring points
            assignments = None
        
        # Create color map for active clusters only
        cmap = plt.cm.tab20  # Use tab20 colormap for better color distinction
        active_cluster_colors = {idx: cmap(i % 20) for i, idx in enumerate(active_cluster_indices)}
        
        # Color points by their assigned cluster (only if assigned to active cluster)
        if assignments is not None:
            point_colors = []
            active_set = set(active_cluster_indices.tolist())  # Convert to set for faster lookup
            for assignment in assignments:
                assignment_int = int(assignment)  # Convert numpy scalar to Python int
                if assignment_int in active_set:
                    point_colors.append(active_cluster_colors[assignment_int])
                else:
                    point_colors.append('lightgray')  # Gray for inactive clusters
            
            ax.scatter(z_e_data[:, 0], z_e_data[:, 1], alpha=0.5, s=15, c=point_colors)
        else:
            # Fallback: just plot all points in gray
            ax.scatter(z_e_data[:, 0], z_e_data[:, 1], alpha=0.3, s=15, c='lightgray', label='Data')
        
        # Plot all cluster means as red dots with ellipses showing 2*std
        # Debug: print some statistics about cluster means
        print(f"\nCluster means statistics (for plotting):")
        print(f"  Total clusters: {num_clusters}")
        print(f"  Mean of all cluster means: {np.mean(E_mu_final, axis=0)}")
        print(f"  Std of all cluster means: {np.std(E_mu_final, axis=0)}")
        print(f"  Range: [{np.min(E_mu_final, axis=0)}, {np.max(E_mu_final, axis=0)}]")
        print(f"  Number near (±1, ±1): {np.sum((np.abs(np.abs(E_mu_final[:, 0]) - 1.0) < 0.2) & (np.abs(np.abs(E_mu_final[:, 1]) - 1.0) < 0.2))}")
        
        # Import Ellipse for plotting
        from matplotlib.patches import Ellipse
        
        # Plot ellipses and means for each cluster
        # Debug: check if variances differ along axes
        if num_clusters > 0:
            sample_var = E_var_final[0]
            print(f"  Sample cluster variance: {sample_var}")
            print(f"  Sample std: {np.sqrt(np.maximum(sample_var, 0.01))}")
            print(f"  Variances differ along axes? {not np.allclose(sample_var[0], sample_var[1])}")
        
        for k in range(num_clusters):
            mean = E_mu_final[k]
            var = E_var_final[k]
            
            # Compute 2*std for ellipse (2*std in each direction = 4*std total width/height)
            std = np.sqrt(np.maximum(var, 0.01))  # Clip to avoid negative variances
            width = 4 * std[0]  # 2*std in x direction
            height = 4 * std[1]  # 2*std in y direction
            
            # Plot ellipse (2-sigma ellipse)
            ellipse = Ellipse(mean, width, height, angle=0, 
                            facecolor='red', edgecolor='darkred', alpha=0.2, 
                            linewidth=1.5, zorder=5)
            ax.add_patch(ellipse)
            
            # Plot cluster mean as red dot
            ax.scatter(mean[0], mean[1], c='red', marker='o', s=50, 
                      edgecolors='darkred', linewidths=1, zorder=10, alpha=0.7)
        
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
    
    # Plot 3: Mixing weights evolution (all clusters)
    ax = axes[1, 0]
    epochs_plot = list(range(len(alpha_mix_history)))
    if num_active > 0:
        # Plot all clusters
        for k in range(num_clusters):
            pi_history = [alpha_mix[k] / (alpha_mix.sum() + 1e-8) for alpha_mix in alpha_mix_history]
            ax.plot(epochs_plot, pi_history, marker='o', markersize=2, linewidth=0.5, alpha=0.6)
    ax.set_title('All Mixing Weights Evolution')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mixing Weight')
    ax.grid(True, alpha=0.3)
    
    # Plot 3 (upper right): Cluster centers weighted by frequency of use
    ax = axes[0, 2]
    # Get final mixing weights (mean of Dirichlet)
    E_pi_plot = E_pi_final
    # Scale circle sizes by mixing weights (normalize to reasonable range)
    circle_sizes = E_pi_plot * 1000  # Scale factor for visibility
    circle_sizes = np.maximum(circle_sizes, 10)  # Minimum size for visibility
    
    # Plot all cluster means with circles scaled by mixing weights
    for k in range(num_clusters):
        if not (np.any(np.isnan(E_mu_final[k])) or np.any(np.isnan(E_pi_plot[k]))):
            ax.scatter(E_mu_final[k, 0], E_mu_final[k, 1], 
                      s=circle_sizes[k], alpha=0.6, edgecolors='black', linewidths=1.5,
                      c='red', label=f'C{k} (π={E_pi_plot[k]:.4f})' if E_pi_plot[k] > 0.01 and k < 10 else None)
    
    # Also plot data points for context
    ax.scatter(z_e_data[:, 0], z_e_data[:, 1], c='gray', alpha=0.1, s=5)
    ax.set_title('Cluster Centers (Circle Size ∝ Mixing Weight)')
    ax.set_xlabel('Dimension 0')
    ax.set_ylabel('Dimension 1')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(axes[0, 0].get_xlim())
    ax.set_ylim(axes[0, 0].get_ylim())
    
    # Plot 4: Number of active clusters over time
    ax = axes[1, 1]
    epochs_plot_active = list(range(len(active_clusters_history)))
    ax.plot(epochs_plot_active, active_clusters_history, marker='o', linewidth=2, markersize=6)
    ax.set_title('Number of Active Clusters Over Time')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Active Clusters (via assignments)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Expected (1 cluster)')
    ax.legend()
    
    # Plot 5 (bottom right): Loss evolution
    ax = axes[1, 2]
    ax.plot(epochs_plot, loss_history, marker='o', linewidth=2, markersize=4, color='purple')
    ax.set_title('Loss Evolution')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save to artifacts directory
    output_dir = Path('artifacts/test_gmm_vbem')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'sanity_test_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved sanity test plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    test_single_gaussian_overclustering()

