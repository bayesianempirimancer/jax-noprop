"""
Demo script exploring different position encoding methods for point clouds with transformers.

This script demonstrates various positional encoding methods for point cloud data,
showing how each method encodes spatial information differently.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.point_cloud_positional_encoding import (
    sinusoidal_3d_positional_encoding,
    fourier_features_3d,
    relative_positional_encoding_point_cloud,
    distance_based_positional_encoding,
    context_aware_positional_encoding,
    get_point_cloud_positional_encoding,
)


def generate_sample_point_cloud(n_points: int = 100, seed: int = 42) -> jnp.ndarray:
    """Generate a sample 3D point cloud."""
    rng = np.random.RandomState(seed)
    # Generate points in a sphere
    coords = rng.randn(n_points, 3)
    coords = coords / np.linalg.norm(coords, axis=1, keepdims=True) * rng.uniform(0.5, 1.0, (n_points, 1))
    return jnp.array(coords)


def visualize_encoding_statistics(coords: jnp.ndarray, method_name: str, encoding: jnp.ndarray):
    """Visualize statistics of the positional encoding."""
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")
    print(f"Input coords shape: {coords.shape}")
    print(f"Encoding shape: {encoding.shape}")
    print(f"Encoding mean: {jnp.mean(encoding):.4f}")
    print(f"Encoding std: {jnp.std(encoding):.4f}")
    print(f"Encoding min: {jnp.min(encoding):.4f}")
    print(f"Encoding max: {jnp.max(encoding):.4f}")
    print(f"Encoding range: {jnp.max(encoding) - jnp.min(encoding):.4f}")


def demo_sinusoidal_3d(coords: jnp.ndarray):
    """Demonstrate 3D sinusoidal positional encoding."""
    print("\n" + "="*60)
    print("1. 3D Sinusoidal Positional Encoding")
    print("="*60)
    
    encoding = sinusoidal_3d_positional_encoding(coords, d_model=64, normalize=True)
    visualize_encoding_statistics(coords, "Sinusoidal 3D", encoding)
    
    return encoding


def demo_fourier_features(coords: jnp.ndarray):
    """Demonstrate Fourier features encoding."""
    print("\n" + "="*60)
    print("2. Fourier Features (Multi-scale)")
    print("="*60)
    
    encoding = fourier_features_3d(coords, num_frequencies=8, include_original=True, normalize=True)
    visualize_encoding_statistics(coords, "Fourier Features", encoding)
    
    print(f"\nFeature dimension breakdown:")
    print(f"  - Fourier features per coord: 2 * 8 = 16")
    print(f"  - Total Fourier: 3 * 16 = 48")
    print(f"  - Original coords: 3")
    print(f"  - Total: {encoding.shape[-1]}")
    
    return encoding


def demo_relative_encoding(coords: jnp.ndarray):
    """Demonstrate relative positional encoding."""
    print("\n" + "="*60)
    print("3. Relative Positional Encoding")
    print("="*60)
    
    encoding = relative_positional_encoding_point_cloud(coords, d_model=64, normalize=True)
    visualize_encoding_statistics(coords, "Relative Positional", encoding)
    
    print(f"\nNote: This creates a {encoding.shape[0]}x{encoding.shape[1]} matrix")
    print(f"      where encoding[i, j] encodes the relative position from point i to point j")
    
    return encoding


def demo_distance_based(coords: jnp.ndarray):
    """Demonstrate distance-based encoding."""
    print("\n" + "="*60)
    print("4. Distance-based Positional Encoding")
    print("="*60)
    
    # Try with k nearest neighbors
    encoding_knn = distance_based_positional_encoding(
        coords, d_model=64, k_neighbors=5, normalize=True
    )
    visualize_encoding_statistics(coords, "Distance-based (k=5)", encoding_knn)
    
    # Try with reference points (grid)
    n_ref = 8
    ref_points = jnp.array([
        [x, y, z] 
        for x in [-1, 0, 1] 
        for y in [-1, 0, 1] 
        for z in [-1, 0, 1]
    ][:n_ref])  # Take first n_ref points
    ref_points = ref_points * 0.5  # Scale to reasonable range
    
    encoding_ref = distance_based_positional_encoding(
        coords, d_model=64, reference_points=ref_points, normalize=True
    )
    visualize_encoding_statistics(coords, "Distance-based (reference points)", encoding_ref)
    
    return encoding_knn, encoding_ref


def demo_context_aware(coords: jnp.ndarray):
    """Demonstrate context-aware multi-scale encoding."""
    print("\n" + "="*60)
    print("5. Context-Aware Multi-scale Positional Encoding")
    print("="*60)
    
    encoding = context_aware_positional_encoding(coords, d_model=64, num_scales=3, normalize=True)
    visualize_encoding_statistics(coords, "Context-Aware", encoding)
    
    print(f"\nMulti-scale breakdown:")
    print(f"  - Number of scales: 3")
    print(f"  - Dimensions per scale: ~{encoding.shape[-1] // 3}")
    print(f"  - Captures both local and global spatial relationships")
    
    return encoding


def demo_unified_api(coords: jnp.ndarray):
    """Demonstrate the unified API."""
    print("\n" + "="*60)
    print("6. Unified API Demo")
    print("="*60)
    
    methods = ["sinusoidal_3d", "fourier_features", "distance_based", "context_aware"]
    
    for method in methods:
        try:
            if method == "fourier_features":
                encoding = get_point_cloud_positional_encoding(
                    coords, d_model=None, method=method, num_frequencies=6
                )
            else:
                encoding = get_point_cloud_positional_encoding(
                    coords, d_model=64, method=method
                )
            print(f"\n{method}: shape {encoding.shape}")
        except Exception as e:
            print(f"\n{method}: Error - {e}")


def compare_encodings(coords: jnp.ndarray):
    """Compare different encoding methods on the same point cloud."""
    print("\n" + "="*60)
    print("Comparison of Encoding Methods")
    print("="*60)
    
    methods = {
        "Sinusoidal 3D": lambda c: sinusoidal_3d_positional_encoding(c, d_model=64),
        "Fourier Features": lambda c: fourier_features_3d(c, num_frequencies=6),
        "Distance-based (k=5)": lambda c: distance_based_positional_encoding(c, d_model=64, k_neighbors=5),
        "Context-Aware": lambda c: context_aware_positional_encoding(c, d_model=64, num_scales=3),
    }
    
    results = {}
    for name, func in methods.items():
        try:
            encoding = func(coords)
            results[name] = {
                "shape": encoding.shape,
                "mean": float(jnp.mean(encoding)),
                "std": float(jnp.std(encoding)),
                "min": float(jnp.min(encoding)),
                "max": float(jnp.max(encoding)),
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    
    print("\nSummary:")
    print("-" * 60)
    for name, stats in results.items():
        if "error" in stats:
            print(f"{name:25s}: ERROR - {stats['error']}")
        else:
            print(f"{name:25s}: shape {str(stats['shape']):20s} | "
                  f"mean={stats['mean']:7.4f} | std={stats['std']:7.4f} | "
                  f"range=[{stats['min']:7.4f}, {stats['max']:7.4f}]")


def main():
    """Run all demos."""
    print("="*60)
    print("Point Cloud Positional Encoding Methods Demo")
    print("="*60)
    
    # Generate sample point cloud
    coords = generate_sample_point_cloud(n_points=50, seed=42)
    print(f"\nGenerated point cloud with {coords.shape[0]} points")
    print(f"Coordinate range: [{jnp.min(coords):.3f}, {jnp.max(coords):.3f}]")
    
    # Run individual demos
    demo_sinusoidal_3d(coords)
    demo_fourier_features(coords)
    demo_relative_encoding(coords)
    demo_distance_based(coords)
    demo_context_aware(coords)
    demo_unified_api(coords)
    
    # Compare all methods
    compare_encodings(coords)
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Sinusoidal 3D: Simple, direct encoding of coordinates")
    print("2. Fourier Features: Multi-scale frequency representation (good for NeRF-style tasks)")
    print("3. Relative: Captures pairwise relationships (memory intensive for large point clouds)")
    print("4. Distance-based: Encodes distances to neighbors/references (flexible)")
    print("5. Context-Aware: Multi-scale encoding capturing local and global context")
    print("\nChoose based on:")
    print("  - Computational budget (relative encoding is O(n²))")
    print("  - Task requirements (local vs global relationships)")
    print("  - Point cloud size (some methods scale better)")


if __name__ == "__main__":
    main()

