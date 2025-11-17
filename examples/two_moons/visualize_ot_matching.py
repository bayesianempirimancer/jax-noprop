#!/usr/bin/env python3
"""
Visualize and benchmark different Optimal Transport matching algorithms.

This script compares how different OT algorithms match Gaussian noise to two moons data,
showing the quality of pairings and execution times in JAX.

Usage:
    python examples/two_moons/visualize_ot_matching.py [--n_points 500] [--num_runs 100]
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import time
import argparse
import os
from typing import Tuple, Dict, Optional

# Optional dependency - will skip if not available
try:
    from ott.geometry import pointcloud
    from ott.solvers import linear
    OTT_AVAILABLE = True
except ImportError:
    OTT_AVAILABLE = False
    print("Warning: ott-jax not available. Skipping ott_linear and ott_sinkhorn methods.")

def generate_two_moons(n_points: int, noise: float = 0.1, scale: float = 8.0, seed: int = 42) -> jnp.ndarray:
    """Generate two moons dataset."""
    x_data, _ = make_moons(n_samples=n_points, noise=noise, random_state=seed)
    x_data = x_data * scale
    x_data = x_data - x_data.mean(axis=0, keepdims=True)  # Center
    return jnp.array(x_data)


def match_random(z_0: jnp.ndarray, z_target: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Random pairing - baseline (no OT)."""
    perm = jr.permutation(key, z_0.shape[0])
    z_0_shuffled = z_0[perm]
    indices = perm
    return z_0_shuffled, indices


def match_sliced(z_0: jnp.ndarray, z_target: jnp.ndarray, key: jax.random.PRNGKey, num_slices: int = 10) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sliced OT via sorting along random projections."""
    n_points, dim = z_0.shape
    
    # Initialize permutation as identity
    perm = jnp.arange(n_points)
    
    for i in range(num_slices):
        key, subkey = jr.split(key)
        # Random projection direction
        direction = jr.normal(subkey, (dim,))
        direction = direction / jnp.linalg.norm(direction)
        
        # Project both point clouds
        proj_z0 = z_0 @ direction
        proj_zt = z_target @ direction
        
        # Sort and match
        sort_z0 = jnp.argsort(proj_z0)
        sort_zt = jnp.argsort(proj_zt)
        
        # Create new permutation
        inv_sort_zt = jnp.argsort(sort_zt)
        perm = perm[sort_z0][inv_sort_zt]
    
    z_0_matched = z_0[perm]
    return z_0_matched, perm


def match_minibatch(z_0: jnp.ndarray, z_target: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Minibatch OT using Hungarian algorithm (scipy)."""
    # Compute cost matrix (pairwise squared distances)
    cost_matrix = jnp.sum((np.array(z_0)[:, None] - np.array(z_target)[None, :])**2, axis=-1)
    
    # Solve linear assignment problem
    row_ind, col_ind = optax.assignment.hungarian_algorithm(cost_matrix)
    
    z_0_matched = z_0[row_ind]
    return z_0_matched, jnp.array(col_ind)


def match_ott_linear(z_0: jnp.ndarray, z_target: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """OTT-JAX linear solver (exact OT)."""
    if not OTT_AVAILABLE:
        raise ImportError("ott-jax not available")
    
    # Create uniform weights
    n = z_0.shape[0]
    a = jnp.ones(n) / n
    b = jnp.ones(n) / n
    
    # Create geometry and solve with Sinkhorn
    geom = pointcloud.PointCloud(z_0, z_target)
    out = linear.solve(geom, a, b)
    
    # Extract transport matrix and find hard assignment
    transport_matrix = out.matrix
    
    # Convert soft assignment to hard assignment (argmax for each source point)
    indices = jnp.argmax(transport_matrix, axis=1)
    z_0_matched = z_0  # Keep original order but use indices for pairing
    
    return z_0_matched, indices


def match_ott_sinkhorn(z_0: jnp.ndarray, z_target: jnp.ndarray, epsilon: float = 0.01) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """OTT-JAX Sinkhorn solver (regularized OT)."""
    if not OTT_AVAILABLE:
        raise ImportError("ott-jax not available")
    
    # Create uniform weights
    n = z_0.shape[0]
    a = jnp.ones(n) / n
    b = jnp.ones(n) / n
    
    # Create geometry with entropic regularization
    geom = pointcloud.PointCloud(z_0, z_target, epsilon=epsilon)
    out = linear.solve(geom, a, b)
    
    # Extract transport matrix and find hard assignment
    transport_matrix = out.matrix
    indices = jnp.argmax(transport_matrix, axis=1)
    z_0_matched = z_0
    
    return z_0_matched, indices


def benchmark_function(func, *args, num_runs: int = 100, warmup: int = 10, **kwargs) -> Dict[str, float]:
    """Benchmark a function with proper JAX warmup."""
    # Warmup runs for JIT compilation
    for _ in range(warmup):
        result = func(*args, **kwargs)
        if isinstance(result, tuple) and any(isinstance(r, jnp.ndarray) for r in result):
            jax.block_until_ready(result[0])
    
    # Actual timing runs
    times = []
    for _ in range(num_runs):
        jitted_func = jax.jit(func)
        jitted_func(*args, **kwargs)
        start = time.perf_counter()
        result = jitted_func(*args, **kwargs)
        if isinstance(result, tuple) and any(isinstance(r, jnp.ndarray) for r in result):
            jax.block_until_ready(result[0])
        elif isinstance(result, jnp.ndarray):
            jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    
    return {
        'mean': np.mean(times) * 1000,  # Convert to ms
        'std': np.std(times) * 1000,
        'median': np.median(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000
    }


def compute_matching_quality(z_0_matched: jnp.ndarray, z_target: jnp.ndarray, indices: jnp.ndarray) -> Dict[str, float]:
    """Compute quality metrics for a matching."""
    # Compute distances between matched pairs
    if z_0_matched.shape == z_target.shape:
        # Direct matching
        distances = jnp.sqrt(jnp.sum((z_0_matched - z_target[indices])**2, axis=1))
    else:
        distances = jnp.sqrt(jnp.sum((z_0_matched - z_target)**2, axis=1))
    
    total_cost = jnp.sum(distances**2)
    mean_distance = jnp.mean(distances)
    
    return {
        'total_cost': float(total_cost),
        'mean_distance': float(mean_distance),
        'max_distance': float(jnp.max(distances)),
        'min_distance': float(jnp.min(distances))
    }


def visualize_matching(ax, z_0: jnp.ndarray, z_target: jnp.ndarray, 
                      z_0_matched: jnp.ndarray, indices: jnp.ndarray,
                      method_name: str, timing: float, quality: Dict[str, float],
                      max_lines: int = 200):
    """Visualize a single matching method."""
    # Subsample lines if too many points
    n_points = z_0.shape[0]
    if n_points > max_lines:
        line_indices = np.linspace(0, n_points - 1, max_lines, dtype=int)
    else:
        line_indices = np.arange(n_points)
    
    # Plot matching lines first (so they appear behind points)
    for i in line_indices:
        j = int(indices[i])
        ax.plot([z_0_matched[i, 0], z_target[j, 0]], 
                [z_0_matched[i, 1], z_target[j, 1]], 
                'k-', alpha=0.15, linewidth=0.5)
    
    # Plot points
    ax.scatter(z_0[:, 0], z_0[:, 1], c='blue', alpha=0.6, s=30, label='Source (Gaussian)', edgecolors='darkblue', linewidth=0.5)
    ax.scatter(z_target[:, 0], z_target[:, 1], c='red', alpha=0.6, s=30, label='Target (Two Moons)', edgecolors='darkred', linewidth=0.5)
    
    # Title with timing and quality
    title = f'{method_name}\n'
    title += f'Time: {timing:.2f} ms | Mean Dist: {quality["mean_distance"]:.2f}'
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=7)


def main():
    parser = argparse.ArgumentParser(description='Visualize OT matching algorithms')
    parser.add_argument('--n_points', type=int, default=500, help='Number of points in each cloud')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise level for two moons')
    parser.add_argument('--scale', type=float, default=8.0, help='Scale factor for two moons')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of timing runs')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--max_lines', type=int, default=200, help='Maximum matching lines to draw')
    parser.add_argument('--output_dir', type=str, default='artifacts/ot_matching', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Sinkhorn regularization')
    parser.add_argument('--num_slices', type=int, default=10, help='Number of slices for sliced OT')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("OT Matching Algorithm Comparison")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Number of points: {args.n_points}")
    print(f"  Two moons noise: {args.noise}")
    print(f"  Two moons scale: {args.scale}")
    print(f"  Timing runs: {args.num_runs} (warmup: {args.warmup})")
    print(f"  Random seed: {args.seed}")
    print(f"  OTT-JAX available: {OTT_AVAILABLE}")
    print("=" * 70)
    
    # Generate point clouds
    key = jr.PRNGKey(args.seed)
    key, subkey1, subkey2 = jr.split(key, 3)
    
    z_0 = jr.normal(subkey1, (args.n_points, 2))  # Gaussian noise
    z_target = generate_two_moons(args.n_points, noise=args.noise, scale=args.scale, seed=args.seed)
    
    print(f"\nPoint clouds generated:")
    print(f"  Source (Gaussian): {z_0.shape}, mean={jnp.mean(z_0):.3f}, std={jnp.std(z_0):.3f}")
    print(f"  Target (Two Moons): {z_target.shape}, mean={jnp.mean(z_target):.3f}, std={jnp.std(z_target):.3f}")
    
    # Define methods to test
    methods = [
        ('Random (baseline)', lambda: match_random(z_0, z_target, subkey2)),
        ('Sliced OT', lambda: match_sliced(z_0, z_target, subkey2, num_slices=args.num_slices)),
        ('Minibatch OT (Hungarian)', lambda: match_minibatch(z_0, z_target)),
    ]
    
    if OTT_AVAILABLE:
        methods.extend([
            ('OTT Linear (exact)', lambda: match_ott_linear(z_0, z_target)),
            ('OTT Sinkhorn', lambda: match_ott_sinkhorn(z_0, z_target, epsilon=args.epsilon)),
        ])
    
    # Run benchmarks and collect results
    results = []
    print(f"\nRunning benchmarks...")
    print("-" * 70)
    
    for method_name, method_func in methods:
        print(f"Testing {method_name}...", end=' ', flush=True)
        
        # Benchmark timing
        timing_stats = benchmark_function(method_func, num_runs=args.num_runs, warmup=args.warmup)
        
        # Get matching results
        z_0_matched, indices = method_func()
        
        # Compute quality
        quality_stats = compute_matching_quality(z_0_matched, z_target, indices)
        
        results.append({
            'name': method_name,
            'z_0_matched': z_0_matched,
            'indices': indices,
            'timing': timing_stats,
            'quality': quality_stats
        })
        
        print(f"✓ ({timing_stats['mean']:.2f} ms)")
    
    # Create visualization
    n_methods = len(results)
    fig = plt.figure(figsize=(15, 10))
    
    # Determine grid layout
    if n_methods <= 3:
        nrows, ncols = 1, n_methods
    else:
        nrows = 2
        ncols = (n_methods + 1) // 2
    
    for idx, result in enumerate(results):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        visualize_matching(
            ax, z_0, z_target,
            result['z_0_matched'], result['indices'],
            result['name'], result['timing']['mean'], result['quality'],
            max_lines=args.max_lines
        )
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(args.output_dir, 'matching_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Algorithm':<30} | {'Time (ms)':<10} | {'Total Cost':<12} | {'Mean Dist':<10}")
    print("-" * 70)
    
    baseline_time = results[0]['timing']['mean']
    
    for result in results:
        timing = result['timing']['mean']
        quality = result['quality']
        speedup = baseline_time / timing
        
        print(f"{result['name']:<30} | {timing:>8.2f} " +
              f"({speedup:>4.2f}x) | {quality['total_cost']:>10.1f} | {quality['mean_distance']:>8.2f}")
    
    # Save results to file
    results_path = os.path.join(args.output_dir, 'timing_results.txt')
    with open(results_path, 'w') as f:
        f.write("OT Matching Algorithm Performance\n")
        f.write("=" * 70 + "\n")
        f.write(f"Configuration:\n")
        f.write(f"  Number of points: {args.n_points}\n")
        f.write(f"  Latent dimension: 2\n")
        f.write(f"  Timing runs: {args.num_runs}\n\n")
        
        f.write(f"{'Algorithm':<30} | {'Time (ms)':<10} | {'Total Cost':<12} | {'Mean Dist':<10}\n")
        f.write("-" * 70 + "\n")
        
        for result in results:
            timing = result['timing']['mean']
            quality = result['quality']
            speedup = baseline_time / timing
            
            f.write(f"{result['name']:<30} | {timing:>8.2f} " +
                   f"({speedup:>4.2f}x) | {quality['total_cost']:>10.1f} | {quality['mean_distance']:>8.2f}\n")
    
    print(f"\n✓ Results saved to: {results_path}")
    print("\n" + "=" * 70)
    print("✓ Done! Check the output directory for results.")
    print("=" * 70)
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
