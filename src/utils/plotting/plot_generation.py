"""
Plotting utilities for generation tasks (conditional and unconditional).

This module provides functions for creating plots for generative models,
including generation comparisons, loss trends, and latent trajectories.
"""
import os
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax.numpy as jnp
import jax.random as jr


def create_generation_plot(
    x_real: np.ndarray, 
    y_labels: Optional[np.ndarray], 
    x_gen: np.ndarray, 
    output_dir: str,
    unconditional: bool = False,
    x_real_labels: Optional[np.ndarray] = None
):
    """
    Create generation comparison plot showing real vs generated samples.
    
    Args:
        x_real: Real samples [N, 2]
        y_labels: Conditional labels [N, ...] or None for unconditional (used for x_gen)
        x_gen: Generated samples [N, 2]
        output_dir: Directory to save the plot
        unconditional: Whether this is unconditional generation
        x_real_labels: Optional labels for x_real [N, ...]. If None, uses y_labels.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    if y_labels is not None and not unconditional:
        # Conditional generation: color by class labels
        # Use x_real_labels if provided, otherwise use y_labels for both
        labels_for_real = x_real_labels if x_real_labels is not None else y_labels
        labels_for_gen = y_labels
        
        # Convert one-hot to class indices for coloring
        def get_class_indices(labels):
            if len(labels.shape) == 2 and labels.shape[1] > 1:
                # One-hot encoded: use argmax to get class indices
                return np.argmax(labels, axis=1)
            else:
                # Integer labels: use directly
                return labels.flatten().astype(int)
        
        class_indices_real = get_class_indices(labels_for_real)
        class_indices_gen = get_class_indices(labels_for_gen)
        
        # Ensure lengths match
        n_real = len(x_real)
        n_gen = len(x_gen)
        n_labels_real = len(class_indices_real)
        n_labels_gen = len(class_indices_gen)
        
        if n_real != n_labels_real:
            print(f"Warning: Mismatch in lengths - x_real: {n_real}, x_real_labels: {n_labels_real}")
            min_len = min(n_real, n_labels_real)
            x_real = x_real[:min_len]
            class_indices_real = class_indices_real[:min_len]
        
        if n_gen != n_labels_gen:
            print(f"Warning: Mismatch in lengths - x_gen: {n_gen}, y_labels: {n_labels_gen}")
            min_len = min(n_gen, n_labels_gen)
            x_gen = x_gen[:min_len]
            class_indices_gen = class_indices_gen[:min_len]
        
        # Real - color by the labels that correspond to x_real
        ax[0].scatter(x_real[:, 0], x_real[:, 1], c=class_indices_real, s=6, cmap='coolwarm', alpha=0.8)
        ax[0].set_title('Real samples (x)')
        ax[0].set_aspect('equal', 'box')
        # Generated - color by the labels that were used for conditioning (should match x_gen order)
        ax[1].scatter(x_gen[:, 0], x_gen[:, 1], c=class_indices_gen, s=6, cmap='coolwarm', alpha=0.8)
        ax[1].set_title('Generated samples (x | y)')
    else:
        # Unconditional generation: single color
        # Real
        ax[0].scatter(x_real[:, 0], x_real[:, 1], c='blue', s=6, alpha=0.8)
        ax[0].set_title('Real samples (x)')
        ax[0].set_aspect('equal', 'box')
        # Generated
        ax[1].scatter(x_gen[:, 0], x_gen[:, 1], c='purple', s=6, alpha=0.8)
        ax[1].set_title('Generated samples (x) - Unconditional')
    
    ax[1].set_aspect('equal', 'box')
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_name = 'unconditional_generation.png' if unconditional else 'conditional_generation.png'
    fig.savefig(os.path.join(output_dir, plot_name), dpi=200, bbox_inches='tight')
    plt.close(fig)

