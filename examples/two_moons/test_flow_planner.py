#!/usr/bin/env python3
"""
Test script for GMMFlowPlanner on two moons dataset.

This script demonstrates the full pipeline:
1. Load two moons dataset
2. Fit GMM to the dataset
3. Subsample points as x_target
4. Conditionally sample x_0 from GMM given x_target
5. Apply Sinkhorn refinement to reorder x_0
6. Visualize results
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import jax.random as jr
from flax.core import freeze

from src.flow_models.flow_planner import GMMFlowPlanner
from src.vae.vb_gmm import GMMVBEMConfig


def load_two_moons_data(data_path: str = "./data/two_moons.pkl"):
    """Load two moons dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please run: python examples/two_moons/generate_two_moons.py"
        )
    
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Combine train and val for fitting GMM
    x_train = dataset['train']['x']
    x_val = dataset['val']['x']
    x_all = np.concatenate([x_train, x_val], axis=0)
    
    print(f"Loaded dataset:")
    print(f"  Train samples: {x_train.shape[0]}")
    print(f"  Val samples: {x_val.shape[0]}")
    print(f"  Total samples: {x_all.shape[0]}")
    print(f"  Data shape: {x_all.shape}")
    
    return x_all, x_train, x_val


def visualize_gmm_fit(
    x_data: np.ndarray,
    planner: GMMFlowPlanner,
    params: dict,
    save_path: str = None
):
    """Visualize GMM fit to data."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot data points
    ax.scatter(x_data[:, 0], x_data[:, 1], alpha=0.3, s=10, c='gray', label='Data')
    
    # Get GMM cluster means and visualize them
    gmm_params = params['params']['gmm']
    cluster_means = np.array(gmm_params['mu_n'])  # [num_clusters, latent_dim]
    
    # Plot cluster means
    ax.scatter(
        cluster_means[:, 0], 
        cluster_means[:, 1], 
        s=200, 
        c='red', 
        marker='x', 
        linewidths=3,
        label='GMM Cluster Means'
    )
    
    # Get cluster assignments for a sample of points to show cluster boundaries
    # Sample a subset for visualization
    n_viz = min(1000, x_data.shape[0])
    x_viz = jnp.array(x_data[:n_viz])
    
    # Get log probabilities - need to create GMM instance
    from src.vae.vb_gmm import create_gmm_vbem
    gmm = create_gmm_vbem(planner.gmm_config)
    log_p_tilde = gmm.apply(
        freeze({'params': gmm_params}),
        x_viz,
        training=False,
        method='log_p_tilde'
    )
    assignments = np.argmax(log_p_tilde, axis=1)
    
    # Plot with colors based on assignments
    colors = plt.cm.tab20(np.linspace(0, 1, planner.gmm_config.num_clusters))
    for k in range(planner.gmm_config.num_clusters):
        mask = assignments == k
        if np.any(mask):
            ax.scatter(
                x_viz[mask, 0], 
                x_viz[mask, 1], 
                alpha=0.5, 
                s=20, 
                c=[colors[k]], 
                label=f'Cluster {k}' if k < 10 else None
            )
    
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_title('GMM Fit to Two Moons Dataset', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved GMM fit visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_conditional_samples(
    x_target: np.ndarray,
    x_0: np.ndarray,
    save_path: str = None
):
    """Visualize conditional samples with lines connecting to source x_target."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot x_target points
    ax.scatter(
        x_target[:, 0], 
        x_target[:, 1], 
        s=100, 
        c='blue', 
        marker='o', 
        alpha=0.7,
        label='x_target',
        edgecolors='darkblue',
        linewidths=1.5
    )
    
    # Plot x_0 points
    ax.scatter(
        x_0[:, 0], 
        x_0[:, 1], 
        s=100, 
        c='orange', 
        marker='s', 
        alpha=0.7,
        label='x_0 (conditional samples)',
        edgecolors='darkorange',
        linewidths=1.5
    )
    
    # Draw lines connecting x_target to x_0
    for i in range(len(x_target)):
        ax.plot(
            [x_target[i, 0], x_0[i, 0]], 
            [x_target[i, 1], x_0[i, 1]], 
            'k-', 
            alpha=0.3, 
            linewidth=0.5
        )
    
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_title('Conditional Samples: x_0 from GMM given x_target', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved conditional samples visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_sinkhorn_refinement(
    x_target: np.ndarray,
    x_0_original: np.ndarray,
    x_0_refined: np.ndarray,
    save_path: str = None
):
    """Visualize Sinkhorn refinement with lines connecting to x_target."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot x_target points
    ax.scatter(
        x_target[:, 0], 
        x_target[:, 1], 
        s=100, 
        c='blue', 
        marker='o', 
        alpha=0.7,
        label='x_target',
        edgecolors='darkblue',
        linewidths=1.5
    )
    
    # Plot x_0_refined points
    ax.scatter(
        x_0_refined[:, 0], 
        x_0_refined[:, 1], 
        s=100, 
        c='green', 
        marker='^', 
        alpha=0.7,
        label='x_0 (after Sinkhorn refinement)',
        edgecolors='darkgreen',
        linewidths=1.5
    )
    
    # Draw lines connecting x_target to x_0_refined
    for i in range(len(x_target)):
        ax.plot(
            [x_target[i, 0], x_0_refined[i, 0]], 
            [x_target[i, 1], x_0_refined[i, 1]], 
            'g-', 
            alpha=0.4, 
            linewidth=1.0
        )
    
    ax.set_xlabel('X1', fontsize=12)
    ax.set_ylabel('X2', fontsize=12)
    ax.set_title('After Sinkhorn Refinement: Optimal Transport Alignment', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Sinkhorn refinement visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    """Main test function."""
    print("=" * 60)
    print("GMMFlowPlanner Test on Two Moons Dataset")
    print("=" * 60)
    
    # Configuration
    data_path = "./data/two_moons.pkl"
    num_clusters = 8
    num_target_samples = 80  # Increased by factor of 4 (was 20)
    num_epochs = 20
    seed = 42
    output_dir = "./artifacts/two_moons_flow_planner"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    key = jr.PRNGKey(seed)
    np.random.seed(seed)
    
    # Step 1: Load two moons dataset
    print("\n[Step 1] Loading two moons dataset...")
    x_all, x_train, x_val = load_two_moons_data(data_path)
    x_all_jax = jnp.array(x_all)
    latent_dim = x_all_jax.shape[1]
    
    # Step 2: Create and initialize GMMFlowPlanner
    print("\n[Step 2] Creating GMMFlowPlanner...")
    gmm_config = GMMVBEMConfig(
        num_clusters=num_clusters,
        latent_dim=latent_dim,
        prior_mu=0.0,
        prior_alpha=1.0,
        prior_beta=1.0,
        prior_alpha_mix=0.5,
        beta_mix=0.1,
        tie_precisions=False
    )
    
    planner = GMMFlowPlanner(
        ndims=1,
        learnable=False,
        alpha_min=0.05,
        alpha_max=0.95,
        sigma_min=0.05,
        sigma_max=0.95,
        gmm_config=gmm_config,
        top_k=3
    )
    
    # Initialize model
    print("  Initializing model parameters...")
    key, init_key = jr.split(key)
    dummy_x = jnp.zeros((1, latent_dim))
    params = planner.init(init_key, dummy_x, init_key)
    
    # Step 3: Test gmm_update on a minibatch
    print("\n[Step 3] Testing gmm_update on a minibatch...")
    # Get a small batch for testing
    test_batch_size = 32
    key, batch_key = jr.split(key)
    batch_indices = jr.choice(batch_key, x_all_jax.shape[0], shape=(test_batch_size,), replace=False)
    x_batch = x_all_jax[batch_indices]
    
    # Get initial GMM parameters
    gmm_params_before_update = params['params']['gmm']
    mu_n_before = np.array(gmm_params_before_update['mu_n'])
    alpha_n_before = np.array(gmm_params_before_update['alpha_n'])
    print(f"  Before update:")
    print(f"    mu_n - mean: {mu_n_before.mean():.4f}, std: {mu_n_before.std():.4f}")
    print(f"    alpha_n - mean: {alpha_n_before.mean():.4f}, std: {alpha_n_before.std():.4f}")
    
    # Perform a single minibatch update
    params = planner.gmm_update(
        params=params,
        x_batch=x_batch,
        N_eff=float(x_all_jax.shape[0]),
        lr=0.2,
        training=True
    )
    
    # Get updated GMM parameters
    gmm_params_after_update = params['params']['gmm']
    mu_n_after = np.array(gmm_params_after_update['mu_n'])
    alpha_n_after = np.array(gmm_params_after_update['alpha_n'])
    print(f"  After update:")
    print(f"    mu_n - mean: {mu_n_after.mean():.4f}, std: {mu_n_after.std():.4f}")
    print(f"    alpha_n - mean: {alpha_n_after.mean():.4f}, std: {alpha_n_after.std():.4f}")
    
    # Verify parameters changed
    mu_diff = np.abs(mu_n_after - mu_n_before).mean()
    alpha_diff = np.abs(alpha_n_after - alpha_n_before).mean()
    print(f"  Parameter changes:")
    print(f"    mu_n change: {mu_diff:.6f}")
    print(f"    alpha_n change: {alpha_diff:.6f}")
    
    if mu_diff > 1e-6 or alpha_diff > 1e-6:
        print("  ✓ gmm_update successfully updated parameters!")
    else:
        print("  ⚠ Warning: Parameters did not change significantly")
    
    # Step 4: Fit GMM to data (with initialization)
    print(f"\n[Step 4] Fitting GMM to data ({num_epochs} epochs)...")
    print("  Initializing cluster means from data and fitting...")
    
    # Check cluster means before fitting
    gmm_params_before = params['params']['gmm']
    mu_n_before = np.array(gmm_params_before['mu_n'])
    print(f"    Cluster means before fit - shape: {mu_n_before.shape}, mean: {mu_n_before.mean():.4f}, std: {mu_n_before.std():.4f}")
    
    # Fit GMM using the flow planner's fit_gmm method
    params = planner.fit_gmm(
        params=params,
        x_data=x_all_jax,
        initialize=True,  # Initialize cluster means from data
        num_epochs=num_epochs,
        batch_size=256,
        N_eff=float(x_all_jax.shape[0]),
        lr=0.2,
        seed=seed
    )
    
    # Check cluster means after fitting
    gmm_params_after = params['params']['gmm']
    mu_n_after = np.array(gmm_params_after['mu_n'])
    print(f"    Cluster means after fit - mean: {mu_n_after.mean():.4f}, std: {mu_n_after.std():.4f}")
    print(f"    Data statistics - mean: {x_all.mean():.4f}, std: {x_all.std():.4f}")
    print("  GMM fitting complete!")
    
    # Visualize GMM fit
    print("\n[Visualization 1] Creating GMM fit visualization...")
    visualize_gmm_fit(
        x_data=x_all,
        planner=planner,
        params=params,
        save_path=os.path.join(output_dir, "gmm_fit.png")
    )
    
    # Step 5: Subsample points as x_target
    print(f"\n[Step 5] Subsampling {num_target_samples} points as x_target...")
    key, sample_key = jr.split(key)
    indices = jr.choice(sample_key, x_all_jax.shape[0], shape=(num_target_samples,), replace=False)
    x_target = x_all_jax[indices]
    x_target_np = np.array(x_target)
    print(f"  Selected {num_target_samples} target points")
    
    # Step 6: Conditionally sample x_0 from GMM given x_target
    print("\n[Step 6] Conditionally sampling x_0 from GMM...")
    key, sample_key = jr.split(key)
    x_0 = planner.apply(
        params,
        x_target,
        sample_key,
        method='sample_x_0',
        training=False
    )
    x_0_np = np.array(x_0)
    print(f"  Generated {x_0.shape[0]} conditional samples")
    
    # Visualize conditional samples
    print("\n[Visualization 2] Creating conditional samples visualization...")
    visualize_conditional_samples(
        x_target=x_target_np,
        x_0=x_0_np,
        save_path=os.path.join(output_dir, "conditional_samples.png")
    )
    
    # Step 7: Apply Sinkhorn refinement
    print("\n[Step 7] Applying Sinkhorn refinement...")
    key, refine_key = jr.split(key)
    x_0_refined = planner.apply(
        params,
        x_0,
        x_target,
        refine_key,
        method='apply_sinkhorn_refinement',
        training=False,
        epsilon=0.1,
        num_iterations=100
    )
    x_0_refined_np = np.array(x_0_refined)
    print("  Sinkhorn refinement complete!")
    
    # Visualize Sinkhorn refinement
    print("\n[Visualization 3] Creating Sinkhorn refinement visualization...")
    visualize_sinkhorn_refinement(
        x_target=x_target_np,
        x_0_original=x_0_np,
        x_0_refined=x_0_refined_np,
        save_path=os.path.join(output_dir, "sinkhorn_refinement.png")
    )
    
    # Step 8: Gaussian-based conditional sampling (for comparison)
    print("\n[Step 8] Conditionally sampling x_0 from Gaussian (normal method)...")
    # Create a separate planner with sample_method='normal' for comparison
    planner_gaussian = GMMFlowPlanner(
        ndims=1,
        learnable=False,
        alpha_min=0.05,
        alpha_max=0.95,
        sigma_min=0.05,
        sigma_max=0.95,
        gmm_config=gmm_config,
        top_k=3,
        sample_method="normal"  # Use normal distribution instead of GMM
    )
    key, init_key = jr.split(key)
    params_gaussian = planner_gaussian.init(init_key, x_target, init_key)
    
    key, sample_key = jr.split(key)
    x_0_gaussian = planner_gaussian.apply(
        params_gaussian,
        x_target,
        sample_key,
        method='sample_x_0',
        training=False
    )
    x_0_gaussian_np = np.array(x_0_gaussian)
    print(f"  Generated {x_0_gaussian.shape[0]} Gaussian samples")
    
    # Visualize Gaussian conditional samples
    print("\n[Visualization 4] Creating Gaussian conditional samples visualization...")
    visualize_conditional_samples(
        x_target=x_target_np,
        x_0=x_0_gaussian_np,
        save_path=os.path.join(output_dir, "conditional_samples_gaussian.png")
    )
    
    # Step 9: Apply Sinkhorn refinement to Gaussian samples
    print("\n[Step 9] Applying Sinkhorn refinement to Gaussian samples...")
    key, refine_key = jr.split(key)
    x_0_gaussian_refined = planner_gaussian.apply(
        params_gaussian,
        x_0_gaussian,
        x_target,
        refine_key,
        method='apply_sinkhorn_refinement',
        training=False,
        epsilon=0.1,
        num_iterations=100
    )
    x_0_gaussian_refined_np = np.array(x_0_gaussian_refined)
    print("  Sinkhorn refinement complete!")
    
    # Visualize Sinkhorn refinement for Gaussian
    print("\n[Visualization 5] Creating Gaussian Sinkhorn refinement visualization...")
    visualize_sinkhorn_refinement(
        x_target=x_target_np,
        x_0_original=x_0_gaussian_np,
        x_0_refined=x_0_gaussian_refined_np,
        save_path=os.path.join(output_dir, "sinkhorn_refinement_gaussian.png")
    )
    
    print("\n" + "=" * 60)
    print("Test complete! All visualizations saved to:")
    print(f"  {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

