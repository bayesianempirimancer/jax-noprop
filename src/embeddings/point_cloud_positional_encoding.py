"""
Positional encoding functions for point cloud data in transformer architectures.

This module provides various positional encoding methods specifically designed for
point cloud data, where points are unordered sets of 3D coordinates.
"""

import jax.numpy as jnp
import jax
import flax.linen as nn
from typing import Optional, Tuple, Union

# Constants
_EPSILON = 1e-8
_DEFAULT_BASE = 10000.0


# Helper functions
def _ensure_batch_dim(coords: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:
    """Ensure coordinates have batch dimension. Returns (coords, had_batch)."""
    has_batch = coords.ndim == 3
    if not has_batch:
        coords = coords[None, ...]
    return coords, has_batch


def _normalize_coords(coords: jnp.ndarray) -> jnp.ndarray:
    """Normalize coordinates to [0, 1] range."""
    coords_min = jnp.min(coords, axis=(0, 1), keepdims=True)
    coords_max = jnp.max(coords, axis=(0, 1), keepdims=True)
    coords_range = coords_max - coords_min
    coords_range = jnp.where(coords_range < _EPSILON, 1.0, coords_range)
    return (coords - coords_min) / coords_range


def _sinusoidal_encode(
    values: jnp.ndarray,
    d_model: int,
    base: float = _DEFAULT_BASE
) -> jnp.ndarray:
    """Apply sinusoidal encoding to values.
    
    Args:
        values: Values to encode, shape [..., 1] or [...,]
        d_model: Output dimension
        base: Base for frequency calculation
        
    Returns:
        Encoded values, shape [..., d_model]
    """
    # Preserve original shape prefix (everything except last dim)
    if values.ndim > 0 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    
    # Get shape prefix for output
    shape_prefix = values.shape if values.ndim > 0 else ()
    
    # Create frequency terms
    half_dim = max(d_model // 2, 1)
    if half_dim > 1:
        div_term = jnp.exp(jnp.arange(0, half_dim, 2) * -(jnp.log(base) / (half_dim - 1)))
    else:
        div_term = jnp.array([1.0])
    
    # Initialize output
    pe = jnp.zeros(shape_prefix + (d_model,))
    
    num_even = (d_model + 1) // 2
    num_odd = d_model // 2
    
    # Expand values for broadcasting
    if values.ndim == 0:
        values = values[None]
    values_expanded = values[..., None]  # [..., 1]
    
    # Apply sin/cos
    pe = pe.at[..., 0::2].set(jnp.sin(values_expanded * div_term[:num_even]))
    if num_odd > 0:
        pe = pe.at[..., 1::2].set(jnp.cos(values_expanded * div_term[:num_odd]))
    
    return pe


# Class-based implementations
class Sinusoidal3DPositionalEncoding(nn.Module):
    """3D sinusoidal positional encoding for point clouds."""
    
    d_model: int
    base: float = _DEFAULT_BASE
    normalize: bool = True
    
    def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Apply sinusoidal positional encoding to 3D coordinates.
        
        Args:
            coords: Point coordinates [batch, num_points, 3] or [num_points, 3]
            
        Returns:
            Positional encodings [batch, num_points, d_model] or [num_points, d_model]
        """
        coords, had_batch = _ensure_batch_dim(coords)
        batch_size, num_points, coord_dim = coords.shape
        
        if coord_dim != 3:
            raise ValueError(f"Expected 3D coordinates, got {coord_dim}D")
        
        if self.normalize:
            coords = _normalize_coords(coords)
        
        # Encode each coordinate dimension
        d_per_coord = self.d_model // 3
        remainder = self.d_model % 3
        
        encodings = []
        for coord_idx in range(3):
            coord_vals = coords[:, :, coord_idx:coord_idx+1]
            pe = _sinusoidal_encode(coord_vals, d_per_coord, self.base)
            encodings.append(pe)
        
        pe_combined = jnp.concatenate(encodings, axis=-1)
        
        # Handle remainder using average coordinate
        if remainder > 0:
            coord_avg = jnp.mean(coords, axis=-1, keepdims=True)
            pe_remainder = _sinusoidal_encode(coord_avg, remainder, self.base)
            pe_combined = jnp.concatenate([pe_combined, pe_remainder], axis=-1)
        
        return pe_combined[0] if not had_batch else pe_combined


class FourierFeaturesPositionalEncoding(nn.Module):
    """Multi-scale Fourier features for point cloud coordinates."""
    
    num_frequencies: int = 10
    include_original: bool = True
    normalize: bool = True
    
    def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Generate Fourier features for coordinates.
        
        Supports both 2D and 3D coordinates. Commonly used in NeRF and vision tasks.
        
        Args:
            coords: Point coordinates [batch, num_points, dim] or [num_points, dim]
            
        Returns:
            Fourier features [batch, num_points, dim*2*num_frequencies + (dim if include_original)]
            or [num_points, ...]
        """
        coords, had_batch = _ensure_batch_dim(coords)
        batch_size, num_points, coord_dim = coords.shape
        
        if coord_dim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D coordinates, got {coord_dim}D")
        
        if self.normalize:
            coords = _normalize_coords(coords)
        
        # Generate frequency bands: 2^0, 2^1, ..., 2^(num_frequencies-1)
        frequencies = 2.0 ** jnp.arange(self.num_frequencies)
        
        # Apply sin and cos to each coordinate at each frequency
        features_list = []
        for coord_idx in range(coord_dim):
            coord_vals = coords[:, :, coord_idx:coord_idx+1]  # [batch, num_points, 1]
            scaled = coord_vals * frequencies[None, None, :]  # [batch, num_points, num_frequencies]
            
            # Interleave sin and cos
            sin_cos = jnp.stack([jnp.sin(scaled), jnp.cos(scaled)], axis=-1)
            features_list.append(sin_cos.reshape(batch_size, num_points, 2 * self.num_frequencies))
        
        fourier_features = jnp.concatenate(features_list, axis=-1)
        
        if self.include_original:
            fourier_features = jnp.concatenate([fourier_features, coords], axis=-1)
        
        return fourier_features[0] if not had_batch else fourier_features


class RelativePositionalEncodingPointCloud(nn.Module):
    """Relative positional encoding for point clouds."""
    
    d_model: int
    max_distance: Optional[float] = None
    normalize: bool = True
    
    def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Compute relative positional encoding for point clouds.
        
        Encodes relative positions/distances between all pairs of points,
        similar to Point Transformer architecture.
        
        Args:
            coords: Point coordinates [batch, num_points, 3] or [num_points, 3]
            
        Returns:
            Relative positional encodings [batch, num_points, num_points, d_model]
            or [num_points, num_points, d_model]
        """
        coords, had_batch = _ensure_batch_dim(coords)
        batch_size, num_points, coord_dim = coords.shape
        
        if coord_dim != 3:
            raise ValueError(f"Expected 3D coordinates, got {coord_dim}D")
        
        # Compute relative positions: [batch, num_points, num_points, 3]
        rel_pos = coords[:, :, None, :] - coords[:, None, :, :]
        distances = jnp.linalg.norm(rel_pos, axis=-1)
        
        if self.normalize:
            max_distance = self.max_distance if self.max_distance is not None else jnp.max(distances)
            max_distance = jnp.maximum(max_distance, _EPSILON)
            rel_pos = rel_pos / max_distance
        
        # Encode relative positions
        d_per_coord = self.d_model // 3
        remainder = self.d_model % 3
        
        encodings = []
        for coord_idx in range(3):
            coord_vals = rel_pos[:, :, :, coord_idx:coord_idx+1]
            pe = _sinusoidal_encode(coord_vals, d_per_coord)
            encodings.append(pe)
        
        pe_combined = jnp.concatenate(encodings, axis=-1)
        
        # Handle remainder using distances
        if remainder > 0:
            dist_expanded = distances[:, :, :, None]
            pe_remainder = _sinusoidal_encode(dist_expanded, remainder)
            pe_combined = jnp.concatenate([pe_combined, pe_remainder], axis=-1)
        
        return pe_combined[0] if not had_batch else pe_combined


class DistanceBasedPositionalEncoding(nn.Module):
    """Distance-based positional encoding for point clouds."""
    
    d_model: int
    k_neighbors: Optional[int] = None
    reference_points: Optional[jnp.ndarray] = None
    normalize: bool = True
    
    def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Encode positions based on distances to neighbors or reference points.
        
        Args:
            coords: Point coordinates [batch, num_points, 3] or [num_points, 3]
            
        Returns:
            Distance-based encodings [batch, num_points, d_model] or [num_points, d_model]
        """
        coords, had_batch = _ensure_batch_dim(coords)
        batch_size, num_points, coord_dim = coords.shape
        
        if coord_dim != 3:
            raise ValueError(f"Expected 3D coordinates, got {coord_dim}D")
        
        # Determine reference points
        if self.reference_points is not None:
            if self.reference_points.ndim == 2:
                ref_points = self.reference_points[None, :, :]
                ref_points = jnp.broadcast_to(ref_points, (batch_size, *ref_points.shape[1:]))
            else:
                ref_points = self.reference_points
            num_ref = ref_points.shape[1]
        else:
            ref_points = coords
            num_ref = num_points
        
        # Compute distances: [batch, num_points, num_ref]
        coords_expanded = coords[:, :, None, :]
        ref_expanded = ref_points[:, None, :, :]
        distances = jnp.linalg.norm(coords_expanded - ref_expanded, axis=-1)
        
        # Select k nearest neighbors if specified
        if self.k_neighbors is not None and self.k_neighbors < num_ref:
            _, top_k_indices = jax.lax.top_k(-distances, k=self.k_neighbors)
            batch_indices = jnp.arange(batch_size)[:, None, None]
            point_indices = jnp.arange(num_points)[None, :, None]
            distances = distances[batch_indices, point_indices, top_k_indices]
            num_ref = self.k_neighbors
        
        if self.normalize:
            max_dist = jnp.maximum(jnp.max(distances), _EPSILON)
            distances = distances / max_dist
        
        # Encode distances
        d_per_dist = self.d_model // num_ref
        remainder = self.d_model % num_ref
        
        encodings = []
        for ref_idx in range(num_ref):
            dist_vals = distances[:, :, ref_idx:ref_idx+1]
            pe = _sinusoidal_encode(dist_vals, d_per_dist)
            encodings.append(pe)
        
        pe_combined = jnp.concatenate(encodings, axis=-1)
        
        # Handle remainder using average distance
        if remainder > 0:
            avg_dist = jnp.mean(distances, axis=-1, keepdims=True)
            pe_remainder = _sinusoidal_encode(avg_dist, remainder)
            pe_combined = jnp.concatenate([pe_combined, pe_remainder], axis=-1)
        
        return pe_combined[0] if not had_batch else pe_combined


class ContextAwarePositionalEncoding(nn.Module):
    """Context-aware positional encoding with multiple scales."""
    
    d_model: int
    num_scales: int = 3
    normalize: bool = True
    
    def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Context-aware positional encoding with multiple scales.
        
        Encodes positions at multiple scales to capture both local and global
        spatial relationships.
        
        Args:
            coords: Point coordinates [batch, num_points, 3] or [num_points, 3]
            
        Returns:
            Multi-scale positional encodings [batch, num_points, d_model] or [num_points, d_model]
        """
        coords, had_batch = _ensure_batch_dim(coords)
        batch_size, num_points, coord_dim = coords.shape
        
        if coord_dim != 3:
            raise ValueError(f"Expected 3D coordinates, got {coord_dim}D")
        
        if self.normalize:
            coords = _normalize_coords(coords)
        
        scales = 2.0 ** (-jnp.arange(self.num_scales))
        d_per_scale = self.d_model // self.num_scales
        remainder = self.d_model % self.num_scales
        
        encodings = []
        for scale in scales:
            scaled_coords = coords * scale
            
            # Compute and aggregate relative positions
            rel_pos = scaled_coords[:, :, None, :] - scaled_coords[:, None, :, :]
            rel_pos_agg = jnp.mean(rel_pos, axis=2)  # [batch, num_points, 3]
            
            # Encode aggregated relative position
            d_per_coord_scale = d_per_scale // 3
            remainder_scale = d_per_scale % 3
            
            scale_encodings = []
            for coord_idx in range(3):
                coord_vals = rel_pos_agg[:, :, coord_idx:coord_idx+1]
                pe = _sinusoidal_encode(coord_vals, d_per_coord_scale)
                scale_encodings.append(pe)
            
            scale_pe = jnp.concatenate(scale_encodings, axis=-1)
            
            # Handle remainder for this scale
            if remainder_scale > 0:
                center = jnp.mean(scaled_coords, axis=1, keepdims=True)
                dist_to_center = jnp.linalg.norm(scaled_coords - center, axis=-1, keepdims=True)
                pe_remainder = _sinusoidal_encode(dist_to_center, remainder_scale)
                scale_pe = jnp.concatenate([scale_pe, pe_remainder], axis=-1)
            
            # Ensure exact dimension
            if scale_pe.shape[-1] < d_per_scale:
                padding = jnp.zeros((batch_size, num_points, d_per_scale - scale_pe.shape[-1]))
                scale_pe = jnp.concatenate([scale_pe, padding], axis=-1)
            elif scale_pe.shape[-1] > d_per_scale:
                scale_pe = scale_pe[:, :, :d_per_scale]
            
            encodings.append(scale_pe)
        
        pe_combined = jnp.concatenate(encodings, axis=-1)
        
        # Handle remainder using distance to global center
        if remainder > 0:
            center = jnp.mean(coords, axis=1, keepdims=True)
            dist_to_center = jnp.linalg.norm(coords - center, axis=-1, keepdims=True)
            pe_remainder = _sinusoidal_encode(dist_to_center, remainder)
            pe_combined = jnp.concatenate([pe_combined, pe_remainder], axis=-1)
        
        return pe_combined[0] if not had_batch else pe_combined


# Factory function
def create_point_cloud_positional_encoding(
    method: str,
    d_model: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """Create a point cloud positional encoding instance.
    
    Args:
        method: Method to use:
            - "sinusoidal_3d": 3D sinusoidal encoding
            - "fourier_features": Multi-scale Fourier features
            - "relative": Relative positional encoding
            - "distance_based": Distance-based encoding
            - "context_aware": Multi-scale context-aware encoding
        d_model: Output embedding dimension (required for most methods, ignored for fourier_features)
        **kwargs: Additional arguments passed to the specific encoding class
        
    Returns:
        Positional encoding module instance
    """
    if method == "sinusoidal_3d":
        if d_model is None:
            raise ValueError("d_model is required for sinusoidal_3d method")
        return Sinusoidal3DPositionalEncoding(d_model=d_model, **kwargs)
    elif method == "fourier_features":
        return FourierFeaturesPositionalEncoding(**kwargs)
    elif method == "relative":
        if d_model is None:
            raise ValueError("d_model is required for relative method")
        return RelativePositionalEncodingPointCloud(d_model=d_model, **kwargs)
    elif method == "distance_based":
        if d_model is None:
            raise ValueError("d_model is required for distance_based method")
        return DistanceBasedPositionalEncoding(d_model=d_model, **kwargs)
    elif method == "context_aware":
        if d_model is None:
            raise ValueError("d_model is required for context_aware method")
        return ContextAwarePositionalEncoding(d_model=d_model, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Options: 'sinusoidal_3d', 'fourier_features', 'relative', "
            f"'distance_based', 'context_aware'"
        )


# Backward compatibility wrapper functions
def sinusoidal_3d_positional_encoding(
    coords: jnp.ndarray,
    d_model: int,
    base: float = _DEFAULT_BASE,
    normalize: bool = True
) -> jnp.ndarray:
    """Apply sinusoidal positional encoding to 3D coordinates.
    
    This is a wrapper function for backward compatibility.
    See Sinusoidal3DPositionalEncoding class for details.
    """
    encoding = Sinusoidal3DPositionalEncoding(d_model=d_model, base=base, normalize=normalize)
    return encoding(coords)


def fourier_features(
    coords: jnp.ndarray,
    num_frequencies: int = 10,
    include_original: bool = True,
    normalize: bool = True
) -> jnp.ndarray:
    """Generate Fourier features for coordinates.
    
    This is a wrapper function for backward compatibility.
    See FourierFeaturesPositionalEncoding class for details.
    """
    encoding = FourierFeaturesPositionalEncoding(
        num_frequencies=num_frequencies,
        include_original=include_original,
        normalize=normalize
    )
    return encoding(coords)


def fourier_features_2d(
    coords: jnp.ndarray,
    num_frequencies: int = 10,
    include_original: bool = True,
    normalize: bool = True
) -> jnp.ndarray:
    """Generate Fourier features for 2D coordinates. See fourier_features for details."""
    return fourier_features(coords, num_frequencies, include_original, normalize)


def fourier_features_3d(
    coords: jnp.ndarray,
    num_frequencies: int = 10,
    include_original: bool = True,
    normalize: bool = True
) -> jnp.ndarray:
    """Generate Fourier features for 3D coordinates. See fourier_features for details."""
    return fourier_features(coords, num_frequencies, include_original, normalize)


def relative_positional_encoding_point_cloud(
    coords: jnp.ndarray,
    d_model: int,
    max_distance: Optional[float] = None,
    normalize: bool = True
) -> jnp.ndarray:
    """Compute relative positional encoding for point clouds.
    
    This is a wrapper function for backward compatibility.
    See RelativePositionalEncodingPointCloud class for details.
    """
    encoding = RelativePositionalEncodingPointCloud(
        d_model=d_model,
        max_distance=max_distance,
        normalize=normalize
    )
    return encoding(coords)


def distance_based_positional_encoding(
    coords: jnp.ndarray,
    d_model: int,
    k_neighbors: Optional[int] = None,
    reference_points: Optional[jnp.ndarray] = None,
    normalize: bool = True
) -> jnp.ndarray:
    """Encode positions based on distances to neighbors or reference points.
    
    This is a wrapper function for backward compatibility.
    See DistanceBasedPositionalEncoding class for details.
    """
    encoding = DistanceBasedPositionalEncoding(
        d_model=d_model,
        k_neighbors=k_neighbors,
        reference_points=reference_points,
        normalize=normalize
    )
    return encoding(coords)


def context_aware_positional_encoding(
    coords: jnp.ndarray,
    d_model: int,
    num_scales: int = 3,
    normalize: bool = True
) -> jnp.ndarray:
    """Context-aware positional encoding with multiple scales.
    
    This is a wrapper function for backward compatibility.
    See ContextAwarePositionalEncoding class for details.
    """
    encoding = ContextAwarePositionalEncoding(
        d_model=d_model,
        num_scales=num_scales,
        normalize=normalize
    )
    return encoding(coords)


def get_point_cloud_positional_encoding(
    coords: jnp.ndarray,
    d_model: int,
    method: str = "sinusoidal_3d",
    **kwargs
) -> jnp.ndarray:
    """Get positional encoding for point clouds using specified method.
    
    This is a convenience function that allows switching between
    different positional encoding methods. Now uses class-based implementations.
    
    Args:
        coords: Point coordinates [batch, num_points, dim] or [num_points, dim]
        d_model: Output embedding dimension (may be ignored for fourier_features)
        method: Method to use:
            - "sinusoidal_3d": 3D sinusoidal encoding
            - "fourier_features": Multi-scale Fourier features
            - "relative": Relative positional encoding
            - "distance_based": Distance-based encoding
            - "context_aware": Multi-scale context-aware encoding
        **kwargs: Additional arguments passed to the specific encoding function
        
    Returns:
        Positional encodings with shape depending on method
    """
    encoding = create_point_cloud_positional_encoding(method=method, d_model=d_model, **kwargs)
    return encoding(coords)
