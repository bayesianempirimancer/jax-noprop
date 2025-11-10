"""
Embedding utilities for the jax_noprop package.

This module provides various embedding functions and classes for:
- Time embeddings (sinusoidal, linear, fourier, gaussian)
- Positional encodings (sinusoidal, relative, rotary)
- Noise schedules (linear, cosine, sigmoid, learnable)
"""

# Time embedding functions
from .embeddings import (
    sinusoidal_time_embedding,
    linear_time_embedding,
    fourier_time_embedding,
    gaussian_time_embedding,
    get_time_embedding,
)

# Positional encoding functions
from .positional_encoding import (
    positional_encoding,
    relative_positional_encoding,
    rotary_positional_encoding,
    get_positional_encoding,
)

# Point cloud positional encoding functions
from .point_cloud_positional_encoding import (
    sinusoidal_3d_positional_encoding,
    fourier_features_2d,
    fourier_features_3d,
    relative_positional_encoding_point_cloud,
    distance_based_positional_encoding,
    context_aware_positional_encoding,
    get_point_cloud_positional_encoding,
)

# Noise schedule classes and functions
from .noise_schedules import (
    NoiseSchedule,
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    SigmoidNoiseSchedule,
    PositiveDense,
    SimpleMonotonicNetwork,
    LearnableNoiseSchedule,
    create_noise_schedule,
)

__all__ = [
    # Time embeddings
    "sinusoidal_time_embedding",
    "linear_time_embedding", 
    "fourier_time_embedding",
    "gaussian_time_embedding",
    "get_time_embedding",
    
    # Positional encodings
    "positional_encoding",
    "relative_positional_encoding",
    "rotary_positional_encoding", 
    "get_positional_encoding",
    
    # Point cloud positional encodings
    "sinusoidal_3d_positional_encoding",
    "fourier_features_2d",
    "fourier_features_3d",
    "relative_positional_encoding_point_cloud",
    "distance_based_positional_encoding",
    "context_aware_positional_encoding",
    "get_point_cloud_positional_encoding",
    
    # Noise schedules
    "NoiseSchedule",
    "LinearNoiseSchedule",
    "CosineNoiseSchedule",
    "SigmoidNoiseSchedule",
    "PositiveDense",
    "SimpleMonotonicNetwork",
    "LearnableNoiseSchedule",
    "create_noise_schedule",
]
