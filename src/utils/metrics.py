"""
Metrics utilities for evaluating model performance.

This module contains standalone metric functions that can be used
across different trainers and evaluation scripts.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Any


def chamfer_distance(
    generated_samples: jnp.ndarray, 
    real_samples: jnp.ndarray
) -> float:
    """
    Compute Chamfer Distance between generated and real point clouds.
    
    Chamfer Distance measures the average distance from each generated point to its
    nearest neighbor in the real data, and from each real point to its nearest neighbor
    in the generated data.
    
    Args:
        generated_samples: Generated samples [num_gen, feature_dim]
        real_samples: Real samples [num_real, feature_dim]
        
    Returns:
        Chamfer Distance (scalar), or float('inf') if generation failed (NaN/Inf present)
    """
    # Check for NaN or Inf in generated samples - indicates generation failure
    gen_has_invalid = jnp.any(~jnp.isfinite(generated_samples))
    real_has_invalid = jnp.any(~jnp.isfinite(real_samples))
    
    if gen_has_invalid or real_has_invalid:
        # Return inf to indicate failure (we want to minimize, so inf is worst case)
        return float('inf')
    
    # Compute pairwise squared distances: [num_gen, num_real]
    # ||g_i - r_j||^2 = ||g_i||^2 - 2*g_i*r_j + ||r_j||^2
    gen_norm_sq = jnp.sum(generated_samples ** 2, axis=1, keepdims=True)  # [num_gen, 1]
    real_norm_sq = jnp.sum(real_samples ** 2, axis=1)  # [num_real,]
    dot_product = jnp.dot(generated_samples, real_samples.T)  # [num_gen, num_real]
    pairwise_dist_sq = gen_norm_sq - 2 * dot_product + real_norm_sq  # [num_gen, num_real]
    
    # Check for negative values due to numerical errors and clip
    pairwise_dist_sq = jnp.maximum(pairwise_dist_sq, 0.0)
    
    # Distance from each generated point to nearest real point
    min_dist_gen_to_real = jnp.sqrt(jnp.min(pairwise_dist_sq, axis=1))  # [num_gen,]
    chamfer_gen_to_real = jnp.mean(min_dist_gen_to_real)
    
    # Distance from each real point to nearest generated point
    min_dist_real_to_gen = jnp.sqrt(jnp.min(pairwise_dist_sq, axis=0))  # [num_real,]
    chamfer_real_to_gen = jnp.mean(min_dist_real_to_gen)
    
    # Bidirectional Chamfer Distance (average of both directions)
    chamfer_distance = (chamfer_gen_to_real + chamfer_real_to_gen) / 2.0
    
    # Final check for NaN/Inf (shouldn't happen now, but safety check)
    if not jnp.isfinite(chamfer_distance):
        return float('inf')
    
    return float(chamfer_distance)


def classification_accuracy(
    predictions: jnp.ndarray,
    targets: jnp.ndarray
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted values [num_samples, output_dim] or [num_samples]
        targets: Target values [num_samples, output_dim] (one-hot encoded) or [num_samples]
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    # Convert to numpy for easier computation
    pred_np = np.array(predictions)
    target_np = np.array(targets)
    
    # Handle different input shapes
    if pred_np.ndim == 1:
        pred_np = pred_np.reshape(-1, 1)
    if target_np.ndim == 1:
        target_np = target_np.reshape(-1, 1)
    
    # For classification, compare predicted vs true classes
    if target_np.shape[1] > 1:  # One-hot encoded
        # Labels are now {0, 1}, argmax works directly
        pred_classes = np.argmax(pred_np, axis=1)
        true_classes = np.argmax(target_np, axis=1)
    else:  # Binary classification
        # Predictions: if pred > 0, class 1, else class 0
        pred_classes = (pred_np > 0).astype(int).flatten()
        # Labels are now {0, 1}, use directly
        true_classes = target_np.astype(int).flatten()
    
    accuracy = np.mean(pred_classes == true_classes)
    return float(accuracy)


def mean_squared_error(
    predictions: jnp.ndarray,
    targets: jnp.ndarray
) -> float:
    """
    Compute Mean Squared Error (MSE).
    
    Args:
        predictions: Predicted values [num_samples, ...]
        targets: Target values [num_samples, ...]
        
    Returns:
        MSE as a float
    """
    mse = jnp.mean((predictions - targets) ** 2)
    return float(mse)


def mean_absolute_error(
    predictions: jnp.ndarray,
    targets: jnp.ndarray
) -> float:
    """
    Compute Mean Absolute Error (MAE).
    
    Args:
        predictions: Predicted values [num_samples, ...]
        targets: Target values [num_samples, ...]
        
    Returns:
        MAE as a float
    """
    mae = jnp.mean(jnp.abs(predictions - targets))
    return float(mae)


def r2_score(
    predictions: jnp.ndarray,
    targets: jnp.ndarray
) -> float:
    """
    Compute R² (coefficient of determination) score.
    
    Args:
        predictions: Predicted values [num_samples, ...]
        targets: Target values [num_samples, ...]
        
    Returns:
        R² score as a float (can be negative if model is worse than baseline)
    """
    ss_res = jnp.sum((targets - predictions) ** 2)
    ss_tot = jnp.sum((targets - jnp.mean(targets)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return float(r2)


def cosine_similarity(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    axis: int = 1
) -> float:
    """
    Compute average cosine similarity between predictions and targets.
    
    Args:
        predictions: Predicted values [num_samples, ...]
        targets: Target values [num_samples, ...]
        axis: Axis along which to compute similarity (default: 1)
        
    Returns:
        Average cosine similarity as a float between -1 and 1
    """
    # Flatten if needed
    if predictions.ndim > 2:
        predictions = predictions.reshape(predictions.shape[0], -1)
    if targets.ndim > 2:
        targets = targets.reshape(targets.shape[0], -1)
    
    # Normalize vectors
    pred_norm = predictions / (jnp.linalg.norm(predictions, axis=axis, keepdims=True) + 1e-8)
    target_norm = targets / (jnp.linalg.norm(targets, axis=axis, keepdims=True) + 1e-8)
    
    # Compute cosine similarity (average across samples)
    cosine_sim = jnp.mean(jnp.sum(pred_norm * target_norm, axis=axis))
    return float(cosine_sim)


def sequence_metrics(
    generated_sequences: jnp.ndarray,
    real_sequences: jnp.ndarray,
    price_dim: int = 0
) -> Dict[str, float]:
    """
    Compute comprehensive metrics between generated and real sequences.
    
    Args:
        generated_sequences: Generated sequences [num_gen, seq_len, embed_dim]
        real_sequences: Real sequences [num_real, seq_len, embed_dim]
        price_dim: Dimension index for price (used for R² calculation)
        
    Returns:
        Dictionary of metrics including MSE, MAE, cosine similarity, and R²
    """
    # Check for NaN or Inf in generated sequences - indicates generation failure
    gen_has_invalid = jnp.any(~jnp.isfinite(generated_sequences))
    real_has_invalid = jnp.any(~jnp.isfinite(real_sequences))
    
    if gen_has_invalid or real_has_invalid:
        # Return inf to indicate failure
        return {
            'mse': float('inf'),
            'mae': float('inf'),
            'cosine_sim': -1.0,
            'r2': float('-inf'),
            'percent_variance_explained': float('-inf')
        }
    
    # Flatten sequences for comparison
    gen_flat = generated_sequences.reshape(generated_sequences.shape[0], -1)
    real_flat = real_sequences.reshape(real_sequences.shape[0], -1)
    
    # Compute MSE (Mean Squared Error)
    mse = mean_squared_error(gen_flat, real_flat)
    
    # Compute MAE (Mean Absolute Error)
    mae = mean_absolute_error(gen_flat, real_flat)
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(gen_flat, real_flat)
    
    # Compute R² (coefficient of determination) on all dimensions (not just price_dim)
    # Flatten all dimensions for overall R² computation
    real_flat_all = real_sequences.reshape(-1)  # [batch * seq_len * embed_dim]
    gen_flat_all = generated_sequences.reshape(-1)  # [batch * seq_len * embed_dim]
    
    # Compute R² on all dimensions
    ss_res = jnp.sum((real_flat_all - gen_flat_all) ** 2)
    real_mean = jnp.mean(real_flat_all)
    ss_tot = jnp.sum((real_flat_all - real_mean) ** 2)
    
    # Avoid division by zero
    if ss_tot > 1e-10:
        r2 = 1.0 - (ss_res / ss_tot)
        percent_variance_explained = r2 * 100.0
    else:
        # If variance is zero, R² is undefined (all values are the same)
        r2 = float('nan')
        percent_variance_explained = float('nan')
    
    return {
        'mse': mse,
        'mae': mae,
        'cosine_sim': cosine_sim,
        'r2': float(r2),
        'percent_variance_explained': float(percent_variance_explained)
    }

