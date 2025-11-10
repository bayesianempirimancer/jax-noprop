"""
Positional encoding functions for point cloud data in transformer architectures.

This module provides various positional encoding methods specifically designed for
point cloud data, where points are unordered sets of 3D coordinates. Unlike sequence
positional encodings, point cloud encodings need to handle spatial relationships
rather than sequential order.

Common methods implemented:
1. 3D Sinusoidal Encoding - Apply sinusoidal functions to 3D coordinates
2. Fourier Features - Multi-scale Fourier features for 3D coordinates
3. Relative Positional Encoding - Encode relative positions/distances between points
4. MLP-based Positional Encoding - Learnable position embeddings via MLP
5. Distance-based Encoding - Encode distances to neighbors or reference points
6. Context-Aware Positional Encoding - Multi-scale relative position encoding
"""

import jax.numpy as jnp
import jax
from typing import Optional, Tuple
from functools import partial


def sinusoidal_3d_positional_encoding(
    coords: jnp.ndarray,
    d_model: int,
    base: float = 10000.0,
    normalize: bool = True
) -> jnp.ndarray:
    """Apply sinusoidal positional encoding to 3D coordinates.
    
    This method applies sinusoidal functions to each coordinate dimension (x, y, z)
    similar to the standard transformer positional encoding, but adapted for 3D space.
    
    Args:
        coords: Point coordinates [batch, num_points, 3] or [num_points, 3]
        d_model: Output embedding dimension
        base: Base for frequency calculation (default: 10000.0)
        normalize: If True, normalize coordinates to [0, 1] range first
        
    Returns:
        Positional encodings [batch, num_points, d_model] or [num_points, d_model]
        
    Example:
        >>> coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        >>> pe = sinusoidal_3d_positional_encoding(coords, d_model=64)
        >>> print(pe.shape)  # (2, 64)
    """
    # Handle batch dimension
    has_batch = coords.ndim == 3
    if not has_batch:
        coords = coords[None, ...]  # [1, num_points, 3]
    
    batch_size, num_points, coord_dim = coords.shape
    assert coord_dim == 3, f"Expected 3D coordinates, got {coord_dim}D"
    
    # Normalize coordinates to [0, 1] if requested
    if normalize:
        coords_min = jnp.min(coords, axis=(0, 1), keepdims=True)  # [1, 1, 3]
        coords_max = jnp.max(coords, axis=(0, 1), keepdims=True)  # [1, 1, 3]
        coords_range = coords_max - coords_min
        coords_range = jnp.where(coords_range < 1e-8, 1.0, coords_range)  # Avoid division by zero
        coords = (coords - coords_min) / coords_range
    
    # Allocate encoding dimension per coordinate
    d_per_coord = d_model // 3
    remainder = d_model % 3
    
    # Create frequency terms for each coordinate dimension
    div_term = jnp.exp(jnp.arange(0, d_per_coord, 2) * -(jnp.log(base) / d_per_coord))
    
    # Apply sinusoidal encoding to each coordinate dimension
    encodings = []
    for coord_idx in range(3):
        coord_vals = coords[:, :, coord_idx:coord_idx+1]  # [batch, num_points, 1]
        
        # Compute encoding for this coordinate
        pe = jnp.zeros((batch_size, num_points, d_per_coord))
        
        # Handle even indices (sin)
        num_even = (d_per_coord + 1) // 2
        num_odd = d_per_coord // 2
        div_term_even = div_term[:num_even]
        div_term_odd = div_term[:num_odd]
        
        pe = pe.at[:, :, 0::2].set(jnp.sin(coord_vals * div_term_even))
        if num_odd > 0:
            pe = pe.at[:, :, 1::2].set(jnp.cos(coord_vals * div_term_odd))
        
        encodings.append(pe)
    
    # Concatenate encodings from all three coordinates
    pe_combined = jnp.concatenate(encodings, axis=-1)  # [batch, num_points, 3*d_per_coord]
    
    # Handle remainder dimension if d_model is not divisible by 3
    if remainder > 0:
        # Use average of all coordinates for remainder
        coord_avg = jnp.mean(coords, axis=-1, keepdims=True)  # [batch, num_points, 1]
        pe_remainder = jnp.zeros((batch_size, num_points, remainder))
        if remainder >= 2:
            pe_remainder = pe_remainder.at[:, :, 0::2].set(jnp.sin(coord_avg * div_term[:remainder//2]))
            pe_remainder = pe_remainder.at[:, :, 1::2].set(jnp.cos(coord_avg * div_term[:remainder//2]))
        else:
            pe_remainder = pe_remainder.at[:, :, 0].set(jnp.sin(coord_avg.squeeze(-1)))
        pe_combined = jnp.concatenate([pe_combined, pe_remainder], axis=-1)
    
    if not has_batch:
        pe_combined = pe_combined[0]  # Remove batch dimension
    
    return pe_combined


def fourier_features_2d(
    coords: jnp.ndarray,
    num_frequencies: int = 10,
    include_original: bool = True,
    normalize: bool = True
) -> jnp.ndarray:
    """Generate Fourier features for 2D coordinates (multi-scale positional encoding).
    
    This method generates multi-scale Fourier features by applying sin/cos to
    coordinates scaled by different frequencies. Commonly used in NeRF and 2D vision.
    
    Args:
        coords: Point coordinates [batch, num_points, 2] or [num_points, 2]
        num_frequencies: Number of frequency bands to use (default: 10)
        include_original: If True, include original coordinates in output
        normalize: If True, normalize coordinates to [0, 1] range first
        
    Returns:
        Fourier features [batch, num_points, 2*2*num_frequencies + (2 if include_original)] 
        or [num_points, ...]
        
    Example:
        >>> coords = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> features = fourier_features_2d(coords, num_frequencies=4)
        >>> print(features.shape)  # (2, 18) = 2*2*4 + 2
    """
    # Handle batch dimension
    has_batch = coords.ndim == 3
    if not has_batch:
        coords = coords[None, ...]  # [1, num_points, 2]
    
    batch_size, num_points, coord_dim = coords.shape
    assert coord_dim == 2, f"Expected 2D coordinates, got {coord_dim}D"
    
    # Normalize coordinates to [0, 1] if requested
    if normalize:
        coords_min = jnp.min(coords, axis=(0, 1), keepdims=True)  # [1, 1, 2]
        coords_max = jnp.max(coords, axis=(0, 1), keepdims=True)  # [1, 1, 2]
        coords_range = coords_max - coords_min
        coords_range = jnp.where(coords_range < 1e-8, 1.0, coords_range)
        coords = (coords - coords_min) / coords_range
    
    # Generate frequency bands: 2^0, 2^1, ..., 2^(num_frequencies-1)
    frequencies = 2.0 ** jnp.arange(num_frequencies)  # [num_frequencies]
    
    # Apply sin and cos to each coordinate at each frequency
    features_list = []
    
    for coord_idx in range(2):
        coord_vals = coords[:, :, coord_idx:coord_idx+1]  # [batch, num_points, 1]
        
        # Scale by frequencies: [batch, num_points, num_frequencies]
        scaled = coord_vals * frequencies[None, None, :]
        
        # Apply sin and cos
        sin_features = jnp.sin(scaled)  # [batch, num_points, num_frequencies]
        cos_features = jnp.cos(scaled)  # [batch, num_points, num_frequencies]
        
        # Interleave sin and cos: [batch, num_points, 2*num_frequencies]
        interleaved = jnp.stack([sin_features, cos_features], axis=-1)
        interleaved = interleaved.reshape(batch_size, num_points, 2 * num_frequencies)
        
        features_list.append(interleaved)
    
    # Concatenate features from all coordinates
    fourier_features = jnp.concatenate(features_list, axis=-1)  # [batch, num_points, 2*2*num_frequencies]
    
    # Optionally include original coordinates
    if include_original:
        fourier_features = jnp.concatenate([fourier_features, coords], axis=-1)
    
    if not has_batch:
        fourier_features = fourier_features[0]  # Remove batch dimension
    
    return fourier_features


def fourier_features_3d(
    coords: jnp.ndarray,
    num_frequencies: int = 10,
    include_original: bool = True,
    normalize: bool = True
) -> jnp.ndarray:
    """Generate Fourier features for 3D coordinates (multi-scale positional encoding).
    
    This method generates multi-scale Fourier features by applying sin/cos to
    coordinates scaled by different frequencies. Commonly used in NeRF and 3D vision.
    
    Args:
        coords: Point coordinates [batch, num_points, 3] or [num_points, 3]
        num_frequencies: Number of frequency bands to use (default: 10)
        include_original: If True, include original coordinates in output
        normalize: If True, normalize coordinates to [0, 1] range first
        
    Returns:
        Fourier features [batch, num_points, 3*2*num_frequencies + (3 if include_original)] 
        or [num_points, ...]
        
    Example:
        >>> coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        >>> features = fourier_features_3d(coords, num_frequencies=4)
        >>> print(features.shape)  # (2, 27) = 3*2*4 + 3
    """
    # Handle batch dimension
    has_batch = coords.ndim == 3
    if not has_batch:
        coords = coords[None, ...]  # [1, num_points, 3]
    
    batch_size, num_points, coord_dim = coords.shape
    assert coord_dim == 3, f"Expected 3D coordinates, got {coord_dim}D"
    
    # Normalize coordinates to [0, 1] if requested
    if normalize:
        coords_min = jnp.min(coords, axis=(0, 1), keepdims=True)  # [1, 1, 3]
        coords_max = jnp.max(coords, axis=(0, 1), keepdims=True)  # [1, 1, 3]
        coords_range = coords_max - coords_min
        coords_range = jnp.where(coords_range < 1e-8, 1.0, coords_range)
        coords = (coords - coords_min) / coords_range
    
    # Generate frequency bands: 2^0, 2^1, ..., 2^(num_frequencies-1)
    frequencies = 2.0 ** jnp.arange(num_frequencies)  # [num_frequencies]
    
    # Apply sin and cos to each coordinate at each frequency
    features_list = []
    
    for coord_idx in range(3):
        coord_vals = coords[:, :, coord_idx:coord_idx+1]  # [batch, num_points, 1]
        
        # Scale by frequencies: [batch, num_points, num_frequencies]
        scaled = coord_vals * frequencies[None, None, :]
        
        # Apply sin and cos
        sin_features = jnp.sin(scaled)  # [batch, num_points, num_frequencies]
        cos_features = jnp.cos(scaled)  # [batch, num_points, num_frequencies]
        
        # Interleave sin and cos: [batch, num_points, 2*num_frequencies]
        interleaved = jnp.stack([sin_features, cos_features], axis=-1)
        interleaved = interleaved.reshape(batch_size, num_points, 2 * num_frequencies)
        
        features_list.append(interleaved)
    
    # Concatenate features from all coordinates
    fourier_features = jnp.concatenate(features_list, axis=-1)  # [batch, num_points, 3*2*num_frequencies]
    
    # Optionally include original coordinates
    if include_original:
        fourier_features = jnp.concatenate([fourier_features, coords], axis=-1)
    
    if not has_batch:
        fourier_features = fourier_features[0]  # Remove batch dimension
    
    return fourier_features


def relative_positional_encoding_point_cloud(
    coords: jnp.ndarray,
    d_model: int,
    max_distance: Optional[float] = None,
    normalize: bool = True
) -> jnp.ndarray:
    """Compute relative positional encoding for point clouds.
    
    This method encodes the relative positions/distances between all pairs of points,
    similar to Point Transformer architecture. The encoding captures spatial
    relationships between points.
    
    Args:
        coords: Point coordinates [batch, num_points, 3] or [num_points, 3]
        d_model: Output embedding dimension per pair
        max_distance: Maximum distance to consider (for normalization). If None, uses max pairwise distance
        normalize: If True, normalize relative positions by max_distance
        
    Returns:
        Relative positional encodings [batch, num_points, num_points, d_model] 
        or [num_points, num_points, d_model]
        
    Example:
        >>> coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> pe = relative_positional_encoding_point_cloud(coords, d_model=64)
        >>> print(pe.shape)  # (3, 3, 64)
    """
    # Handle batch dimension
    has_batch = coords.ndim == 3
    if not has_batch:
        coords = coords[None, ...]  # [1, num_points, 3]
    
    batch_size, num_points, coord_dim = coords.shape
    assert coord_dim == 3, f"Expected 3D coordinates, got {coord_dim}D"
    
    # Compute relative positions: [batch, num_points, num_points, 3]
    # rel_pos[i, j] = coords[j] - coords[i]
    coords_expanded_i = coords[:, :, None, :]  # [batch, num_points, 1, 3]
    coords_expanded_j = coords[:, None, :, :]  # [batch, 1, num_points, 3]
    rel_pos = coords_expanded_j - coords_expanded_i  # [batch, num_points, num_points, 3]
    
    # Compute distances: [batch, num_points, num_points]
    distances = jnp.linalg.norm(rel_pos, axis=-1)  # [batch, num_points, num_points]
    
    # Normalize if requested
    if normalize:
        if max_distance is None:
            max_distance = jnp.max(distances)
        max_distance = jnp.maximum(max_distance, 1e-8)  # Avoid division by zero
        rel_pos = rel_pos / max_distance
        distances = distances / max_distance
    
    # Encode relative positions using sinusoidal encoding
    # We'll encode the 3D relative position vector
    d_per_coord = d_model // 3
    remainder = d_model % 3
    
    base = 10000.0
    div_term = jnp.exp(jnp.arange(0, d_per_coord, 2) * -(jnp.log(base) / d_per_coord))
    
    encodings = []
    for coord_idx in range(3):
        coord_vals = rel_pos[:, :, :, coord_idx:coord_idx+1]  # [batch, num_points, num_points, 1]
        
        pe = jnp.zeros((batch_size, num_points, num_points, d_per_coord))
        
        # Handle even/odd indices properly
        num_even = (d_per_coord + 1) // 2
        num_odd = d_per_coord // 2
        div_term_even = div_term[:num_even]
        div_term_odd = div_term[:num_odd]
        
        pe = pe.at[:, :, :, 0::2].set(jnp.sin(coord_vals * div_term_even))
        if num_odd > 0:
            pe = pe.at[:, :, :, 1::2].set(jnp.cos(coord_vals * div_term_odd))
        
        encodings.append(pe)
    
    pe_combined = jnp.concatenate(encodings, axis=-1)  # [batch, num_points, num_points, 3*d_per_coord]
    
    # Handle remainder
    if remainder > 0:
        # Use distance for remainder dimensions
        dist_expanded = distances[:, :, :, None]  # [batch, num_points, num_points, 1]
        pe_remainder = jnp.zeros((batch_size, num_points, num_points, remainder))
        if remainder >= 2:
            pe_remainder = pe_remainder.at[:, :, :, 0::2].set(jnp.sin(dist_expanded * div_term[:remainder//2]))
            pe_remainder = pe_remainder.at[:, :, :, 1::2].set(jnp.cos(dist_expanded * div_term[:remainder//2]))
        else:
            pe_remainder = pe_remainder.at[:, :, :, 0].set(jnp.sin(dist_expanded.squeeze(-1)))
        pe_combined = jnp.concatenate([pe_combined, pe_remainder], axis=-1)
    
    if not has_batch:
        pe_combined = pe_combined[0]  # Remove batch dimension
    
    return pe_combined


def distance_based_positional_encoding(
    coords: jnp.ndarray,
    d_model: int,
    k_neighbors: Optional[int] = None,
    reference_points: Optional[jnp.ndarray] = None,
    normalize: bool = True
) -> jnp.ndarray:
    """Encode positions based on distances to neighbors or reference points.
    
    This method encodes each point's position by its distances to k nearest neighbors
    or to a set of reference points (e.g., cluster centers, grid points).
    
    Args:
        coords: Point coordinates [batch, num_points, 3] or [num_points, 3]
        d_model: Output embedding dimension
        k_neighbors: Number of nearest neighbors to consider. If None, uses all points
        reference_points: Reference points to compute distances to [num_ref, 3]. 
                         If None and k_neighbors is None, uses all points
        normalize: If True, normalize distances
        
    Returns:
        Distance-based encodings [batch, num_points, d_model] or [num_points, d_model]
        
    Example:
        >>> coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> pe = distance_based_positional_encoding(coords, d_model=64, k_neighbors=2)
        >>> print(pe.shape)  # (3, 64)
    """
    # Handle batch dimension
    has_batch = coords.ndim == 3
    if not has_batch:
        coords = coords[None, ...]  # [1, num_points, 3]
    
    batch_size, num_points, coord_dim = coords.shape
    assert coord_dim == 3, f"Expected 3D coordinates, got {coord_dim}D"
    
    # Determine reference points
    if reference_points is not None:
        # Use provided reference points
        if reference_points.ndim == 2:
            ref_points = reference_points[None, :, :]  # [1, num_ref, 3]
            ref_points = jnp.broadcast_to(ref_points, (batch_size, *ref_points.shape[1:]))
        else:
            ref_points = reference_points  # [batch, num_ref, 3]
        num_ref = ref_points.shape[1]
    elif k_neighbors is not None:
        # Use k nearest neighbors from the same point cloud
        ref_points = coords  # [batch, num_points, 3]
        num_ref = num_points
    else:
        # Use all points as reference
        ref_points = coords  # [batch, num_points, 3]
        num_ref = num_points
    
    # Compute distances from each point to each reference point
    # [batch, num_points, num_ref]
    coords_expanded = coords[:, :, None, :]  # [batch, num_points, 1, 3]
    ref_expanded = ref_points[:, None, :, :]  # [batch, 1, num_ref, 3]
    distances = jnp.linalg.norm(coords_expanded - ref_expanded, axis=-1)  # [batch, num_points, num_ref]
    
    # Select k nearest neighbors if specified
    if k_neighbors is not None and k_neighbors < num_ref:
        # Get k nearest neighbors for each point
        _, top_k_indices = jax.lax.top_k(-distances, k=k_neighbors)  # Negative for ascending order
        batch_indices = jnp.arange(batch_size)[:, None, None]  # [batch, 1, 1]
        point_indices = jnp.arange(num_points)[None, :, None]  # [1, num_points, 1]
        distances = distances[batch_indices, point_indices, top_k_indices]  # [batch, num_points, k]
        num_ref = k_neighbors
    
    # Normalize distances
    if normalize:
        max_dist = jnp.max(distances)
        max_dist = jnp.maximum(max_dist, 1e-8)
        distances = distances / max_dist
    
    # Encode distances using sinusoidal encoding
    base = 10000.0
    d_per_dist = d_model // num_ref
    remainder = d_model % num_ref
    
    div_term = jnp.exp(jnp.arange(0, d_per_dist, 2) * -(jnp.log(base) / d_per_dist))
    
    encodings = []
    for ref_idx in range(num_ref):
        dist_vals = distances[:, :, ref_idx:ref_idx+1]  # [batch, num_points, 1]
        
        pe = jnp.zeros((batch_size, num_points, d_per_dist))
        
        # Handle even/odd indices properly
        num_even = (d_per_dist + 1) // 2
        num_odd = d_per_dist // 2
        div_term_even = div_term[:num_even]
        div_term_odd = div_term[:num_odd]
        
        pe = pe.at[:, :, 0::2].set(jnp.sin(dist_vals * div_term_even))
        if num_odd > 0:
            pe = pe.at[:, :, 1::2].set(jnp.cos(dist_vals * div_term_odd))
        
        encodings.append(pe)
    
    pe_combined = jnp.concatenate(encodings, axis=-1)  # [batch, num_points, num_ref*d_per_dist]
    
    # Handle remainder
    if remainder > 0:
        # Use average distance for remainder
        avg_dist = jnp.mean(distances, axis=-1, keepdims=True)  # [batch, num_points, 1]
        pe_remainder = jnp.zeros((batch_size, num_points, remainder))
        if remainder >= 2:
            pe_remainder = pe_remainder.at[:, :, 0::2].set(jnp.sin(avg_dist * div_term[:remainder//2]))
            pe_remainder = pe_remainder.at[:, :, 1::2].set(jnp.cos(avg_dist * div_term[:remainder//2]))
        else:
            pe_remainder = pe_remainder.at[:, :, 0].set(jnp.sin(avg_dist.squeeze(-1)))
        pe_combined = jnp.concatenate([pe_combined, pe_remainder], axis=-1)
    
    if not has_batch:
        pe_combined = pe_combined[0]  # Remove batch dimension
    
    return pe_combined


def context_aware_positional_encoding(
    coords: jnp.ndarray,
    d_model: int,
    num_scales: int = 3,
    normalize: bool = True
) -> jnp.ndarray:
    """Context-aware positional encoding with multiple scales.
    
    This method encodes positions at multiple scales to capture both local
    and global spatial relationships. Inspired by context-aware position encoding
    methods that address short- and long-range contexts.
    
    Args:
        coords: Point coordinates [batch, num_points, 3] or [num_points, 3]
        d_model: Output embedding dimension
        num_scales: Number of scale levels to use (default: 3)
        normalize: If True, normalize coordinates to [0, 1] range first
        
    Returns:
        Multi-scale positional encodings [batch, num_points, d_model] or [num_points, d_model]
        
    Example:
        >>> coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> pe = context_aware_positional_encoding(coords, d_model=64, num_scales=3)
        >>> print(pe.shape)  # (3, 64)
    """
    # Handle batch dimension
    has_batch = coords.ndim == 3
    if not has_batch:
        coords = coords[None, ...]  # [1, num_points, 3]
    
    batch_size, num_points, coord_dim = coords.shape
    assert coord_dim == 3, f"Expected 3D coordinates, got {coord_dim}D"
    
    # Normalize coordinates
    if normalize:
        coords_min = jnp.min(coords, axis=(0, 1), keepdims=True)
        coords_max = jnp.max(coords, axis=(0, 1), keepdims=True)
        coords_range = coords_max - coords_min
        coords_range = jnp.where(coords_range < 1e-8, 1.0, coords_range)
        coords = (coords - coords_min) / coords_range
    
    # Compute scale factors: 1.0, 0.5, 0.25, ... (or use different progression)
    scales = 2.0 ** (-jnp.arange(num_scales))  # [num_scales]
    
    # Allocate dimension per scale
    d_per_scale = d_model // num_scales
    remainder = d_model % num_scales
    
    base = 10000.0
    div_term = jnp.exp(jnp.arange(0, d_per_scale, 2) * -(jnp.log(base) / d_per_scale))
    
    encodings = []
    
    for scale_idx, scale in enumerate(scales):
        # Scale coordinates
        scaled_coords = coords * scale  # [batch, num_points, 3]
        
        # Compute relative positions at this scale
        coords_expanded_i = scaled_coords[:, :, None, :]  # [batch, num_points, 1, 3]
        coords_expanded_j = scaled_coords[:, None, :, :]  # [batch, 1, num_points, 3]
        rel_pos = coords_expanded_j - coords_expanded_i  # [batch, num_points, num_points, 3]
        
        # Aggregate relative positions (e.g., mean or max pooling)
        # Using mean pooling to get a single vector per point
        rel_pos_agg = jnp.mean(rel_pos, axis=2)  # [batch, num_points, 3]
        
        # Encode aggregated relative position using sinusoidal encoding
        # Flatten the 3D relative position vector and encode it
        rel_pos_flat = rel_pos_agg.reshape(batch_size, num_points, 3)  # [batch, num_points, 3]
        
        # Use a simpler approach: encode each coordinate separately
        d_per_coord_scale = d_per_scale // 3
        remainder_scale = d_per_scale % 3
        
        scale_encodings = []
        for coord_idx in range(3):
            coord_vals = rel_pos_agg[:, :, coord_idx:coord_idx+1]  # [batch, num_points, 1]
            
            pe = jnp.zeros((batch_size, num_points, d_per_coord_scale))
            
            # Handle even/odd indices properly
            num_even = (d_per_coord_scale + 1) // 2
            num_odd = d_per_coord_scale // 2
            div_term_even = div_term[:num_even] if num_even <= len(div_term) else div_term
            div_term_odd = div_term[:num_odd] if num_odd <= len(div_term) else div_term
            
            if num_even > 0:
                pe = pe.at[:, :, 0::2].set(jnp.sin(coord_vals * div_term_even[:num_even]))
            if num_odd > 0:
                pe = pe.at[:, :, 1::2].set(jnp.cos(coord_vals * div_term_odd[:num_odd]))
            
            scale_encodings.append(pe)
        
        scale_pe = jnp.concatenate(scale_encodings, axis=-1)  # [batch, num_points, 3*d_per_coord_scale]
        
        # Handle remainder for this scale
        if remainder_scale > 0:
            # Use distance to center for remainder
            center = jnp.mean(scaled_coords, axis=1, keepdims=True)
            dist_to_center = jnp.linalg.norm(scaled_coords - center, axis=-1, keepdims=True)
            pe_remainder = jnp.zeros((batch_size, num_points, remainder_scale))
            if remainder_scale >= 2:
                num_even_rem = (remainder_scale + 1) // 2
                num_odd_rem = remainder_scale // 2
                pe_remainder = pe_remainder.at[:, :, 0::2].set(jnp.sin(dist_to_center * div_term[:num_even_rem]))
                if num_odd_rem > 0:
                    pe_remainder = pe_remainder.at[:, :, 1::2].set(jnp.cos(dist_to_center * div_term[:num_odd_rem]))
            else:
                pe_remainder = pe_remainder.at[:, :, 0].set(jnp.sin(dist_to_center.squeeze(-1)))
            scale_pe = jnp.concatenate([scale_pe, pe_remainder], axis=-1)
        
        # Pad or truncate to exact d_per_scale
        if scale_pe.shape[-1] < d_per_scale:
            padding = jnp.zeros((batch_size, num_points, d_per_scale - scale_pe.shape[-1]))
            scale_pe = jnp.concatenate([scale_pe, padding], axis=-1)
        elif scale_pe.shape[-1] > d_per_scale:
            scale_pe = scale_pe[:, :, :d_per_scale]
        
        encodings.append(scale_pe)
    
    # Concatenate encodings from all scales
    pe_combined = jnp.concatenate(encodings, axis=-1)  # [batch, num_points, num_scales*d_per_scale]
    
    # Handle remainder
    if remainder > 0:
        # Use global center for remainder
        center = jnp.mean(coords, axis=1, keepdims=True)  # [batch, 1, 3]
        rel_to_center = coords - center  # [batch, num_points, 3]
        dist_to_center = jnp.linalg.norm(rel_to_center, axis=-1, keepdims=True)  # [batch, num_points, 1]
        
        pe_remainder = jnp.zeros((batch_size, num_points, remainder))
        if remainder >= 2:
            pe_remainder = pe_remainder.at[:, :, 0::2].set(jnp.sin(dist_to_center * div_term[:remainder//2]))
            pe_remainder = pe_remainder.at[:, :, 1::2].set(jnp.cos(dist_to_center * div_term[:remainder//2]))
        else:
            pe_remainder = pe_remainder.at[:, :, 0].set(jnp.sin(dist_to_center.squeeze(-1)))
        pe_combined = jnp.concatenate([pe_combined, pe_remainder], axis=-1)
    
    if not has_batch:
        pe_combined = pe_combined[0]  # Remove batch dimension
    
    return pe_combined


def get_point_cloud_positional_encoding(
    coords: jnp.ndarray,
    d_model: int,
    method: str = "sinusoidal_3d",
    **kwargs
) -> jnp.ndarray:
    """Get positional encoding for point clouds using specified method.
    
    Args:
        coords: Point coordinates [batch, num_points, 3] or [num_points, 3]
        d_model: Output embedding dimension
        method: Method to use. Options:
            - "sinusoidal_3d": 3D sinusoidal encoding
            - "fourier_features": Multi-scale Fourier features
            - "relative": Relative positional encoding (returns [..., num_points, num_points, d_model])
            - "distance_based": Distance-based encoding
            - "context_aware": Multi-scale context-aware encoding
        **kwargs: Additional arguments passed to the specific encoding function
        
    Returns:
        Positional encodings with shape depending on method
        
    Example:
        >>> coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        >>> pe = get_point_cloud_positional_encoding(coords, d_model=64, method="fourier_features")
        >>> print(pe.shape)
    """
    if method == "sinusoidal_3d":
        return sinusoidal_3d_positional_encoding(coords, d_model, **kwargs)
    elif method == "fourier_features":
        # For Fourier features, d_model might not be directly applicable
        # We'll compute features and project if needed
        num_frequencies = kwargs.pop("num_frequencies", 10)
        include_original = kwargs.pop("include_original", True)
        # Determine dimension from coords
        if coords.ndim == 3:
            coord_dim = coords.shape[-1]
        else:
            coord_dim = coords.shape[-1]
        
        if coord_dim == 2:
            features = fourier_features_2d(coords, num_frequencies=num_frequencies, 
                                          include_original=include_original, **kwargs)
        elif coord_dim == 3:
            features = fourier_features_3d(coords, num_frequencies=num_frequencies, 
                                          include_original=include_original, **kwargs)
        else:
            raise ValueError(f"fourier_features only supports 2D or 3D coordinates, got {coord_dim}D")
        # If d_model is specified and different from feature dim, we'd need projection
        # For now, return features as-is
        return features
    elif method == "relative":
        return relative_positional_encoding_point_cloud(coords, d_model, **kwargs)
    elif method == "distance_based":
        return distance_based_positional_encoding(coords, d_model, **kwargs)
    elif method == "context_aware":
        return context_aware_positional_encoding(coords, d_model, **kwargs)
    else:
        raise ValueError(
            f"Unknown point cloud positional encoding method: {method}. "
            f"Options: 'sinusoidal_3d', 'fourier_features', 'relative', "
            f"'distance_based', 'context_aware'"
        )

