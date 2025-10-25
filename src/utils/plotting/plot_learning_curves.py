"""
Learning curve plotting utilities for all model types.

This module provides comprehensive learning analysis plots that can be used
by any model trainer (NoProp, ET, etc.) to visualize training progress,
model performance, and prediction quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def create_enhanced_learning_plot(
    results: Dict[str, Any],
    train_pred: np.ndarray,
    val_pred: np.ndarray,
    test_pred: Optional[np.ndarray] = None,
    train_y: np.ndarray = None,
    val_y: np.ndarray = None,
    test_y: Optional[np.ndarray] = None,
    output_path: str = None,
    model_name: str = "Model",
    skip_epochs: int = 4
) -> None:
    """
    Create a comprehensive learning analysis plot for any model type.
    
    This is a general-purpose function that works with any model's training results.
    It creates a 2x2 subplot layout showing:
    - Training/validation loss curves
    - MSE comparison across datasets
    - Prediction vs target scatter plots
    - Residuals analysis
    
    Args:
        results: Training results dictionary containing losses and metrics
        train_pred: Training predictions [num_train_samples, output_dim]
        val_pred: Validation predictions [num_val_samples, output_dim]
        test_pred: Test predictions [num_test_samples, output_dim] (optional)
        train_y: Training targets [num_train_samples, output_dim] (optional)
        val_y: Validation targets [num_val_samples, output_dim] (optional)
        test_y: Test targets [num_test_samples, output_dim] (optional)
        output_path: Path to save the plot
        model_name: Name of the model for the title
        skip_epochs: Number of epochs to skip from the beginning (default: 4)
    """
    # Extract training history
    train_losses = results.get('train_losses', [])
    val_losses = results.get('val_losses', [])
    
    # Handle optional test data
    if test_pred is None:
        test_pred = val_pred  # Use validation data if test not available
        test_y = val_y
        test_label = "Validation"
    else:
        test_label = "Test"
    
    # Use MSE from results if available (computed during training), otherwise compute from predictions
    if 'final_train_mse' in results and 'final_val_mse' in results:
        train_mse = results['final_train_mse']
        val_mse = results['final_val_mse']
        # Test MSE might not be available in older results
        if 'final_test_mse' in results and results['final_test_mse'] is not None:
            test_mse = results['final_test_mse']
            print(f"Using MSE from training results: train={train_mse:.6f}, val={val_mse:.6f}, test={test_mse:.6f}")
        else:
            test_mse = np.mean((test_pred - test_y) ** 2)
            print(f"Using MSE from training results for train/val: train={train_mse:.6f}, val={val_mse:.6f}, computing {test_label.lower()}={test_mse:.6f}")
    else:
        # Compute MSE from predictions if not available in results
        if train_y is not None and val_y is not None:
            train_mse = np.mean((train_pred - train_y) ** 2)
            val_mse = np.mean((val_pred - val_y) ** 2)
            test_mse = np.mean((test_pred - test_y) ** 2)
            print(f"Computing MSE from predictions: train={train_mse:.6f}, val={val_mse:.6f}, {test_label.lower()}={test_mse:.6f}")
        else:
            # Fallback: use loss values if targets not available
            train_mse = train_losses[-1] if train_losses else 0.0
            val_mse = val_losses[-1] if val_losses else 0.0
            test_mse = val_mse  # Use validation as proxy
            print(f"Using loss values as MSE proxy: train={train_mse:.6f}, val={val_mse:.6f}, {test_label.lower()}={test_mse:.6f}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Learning Analysis', fontsize=16, fontweight='bold')
    
    # Panel 1: Training and validation losses (skip first N epochs)
    ax1 = axes[0, 0]
    
    # Skip first N epochs if requested
    if skip_epochs > 0 and len(train_losses) > skip_epochs:
        epochs = range(skip_epochs + 1, len(train_losses) + 1)
        train_losses_plot = train_losses[skip_epochs:]
        val_losses_plot = val_losses[skip_epochs:]
        title_suffix = f" (after epoch {skip_epochs})"
    else:
        epochs = range(1, len(train_losses) + 1)
        train_losses_plot = train_losses
        val_losses_plot = val_losses
        title_suffix = ""
    
    ax1.plot(epochs, train_losses_plot, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses_plot, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training and Validation Losses{title_suffix}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: MSE comparison
    ax2 = axes[0, 1]
    mse_values = [train_mse, val_mse, test_mse]
    mse_labels = ['Train', 'Validation', test_label]
    colors = ['blue', 'red', 'green']
    bars = ax2.bar(mse_labels, mse_values, color=colors, alpha=0.7)
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('MSE Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, mse_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 3: Prediction vs Target scatter (if targets available)
    ax3 = axes[1, 0]
    if train_y is not None and val_y is not None:
        # Flatten arrays for scatter plot
        train_pred_flat = train_pred.flatten()
        train_y_flat = train_y.flatten()
        val_pred_flat = val_pred.flatten()
        val_y_flat = val_y.flatten()
        test_pred_flat = test_pred.flatten()
        test_y_flat = test_y.flatten()
        
        # Create scatter plots for all datasets with different colors
        ax3.scatter(train_y_flat, train_pred_flat, alpha=0.6, s=15, color='blue', label='Train')
        ax3.scatter(val_y_flat, val_pred_flat, alpha=0.6, s=15, color='red', label='Validation')
        ax3.scatter(test_y_flat, test_pred_flat, alpha=0.6, s=15, color='green', label=test_label)
        
        # Add perfect prediction line
        all_true = np.concatenate([train_y_flat, val_y_flat, test_y_flat])
        all_pred = np.concatenate([train_pred_flat, val_pred_flat, test_pred_flat])
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
        
        ax3.set_xlabel('True Values')
        ax3.set_ylabel('Predicted Values')
        ax3.set_title('Predicted vs True Values (All Datasets)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        # Fallback: show prediction distributions if targets not available
        ax3.hist(train_pred.flatten(), bins=50, alpha=0.7, color='blue', label='Train Predictions')
        ax3.hist(val_pred.flatten(), bins=50, alpha=0.7, color='red', label='Validation Predictions')
        ax3.hist(test_pred.flatten(), bins=50, alpha=0.7, color='green', label=f'{test_label} Predictions')
        ax3.set_xlabel('Prediction Values')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Prediction Value Distributions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Panel 4: Residuals analysis (if targets available) or loss history
    ax4 = axes[1, 1]
    if train_y is not None and val_y is not None:
        # Calculate residuals for all datasets
        train_residuals = train_pred_flat - train_y_flat
        val_residuals = val_pred_flat - val_y_flat
        test_residuals = test_pred_flat - test_y_flat
        
        # Create scatter plots for all datasets with different colors
        ax4.scatter(train_y_flat, train_residuals, alpha=0.6, s=15, color='blue', label='Train')
        ax4.scatter(val_y_flat, val_residuals, alpha=0.6, s=15, color='red', label='Validation')
        ax4.scatter(test_y_flat, test_residuals, alpha=0.6, s=15, color='green', label=test_label)
        
        ax4.axhline(y=0, color='k', linestyle='--', linewidth=2)
        ax4.set_xlabel('True Values')
        ax4.set_ylabel('Residuals (Predicted - True)')
        ax4.set_title('Residuals Analysis (All Datasets)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Fallback: show loss curves
        ax4.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
        ax4.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Loss Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Final Train Loss: {train_losses[-1]:.4f}\n'
    stats_text += f'Final Val Loss: {val_losses[-1]:.4f}\n'
    stats_text += f'Best Val Loss: {min(val_losses):.4f}\n'
    stats_text += f'Train MSE: {train_mse:.4f}\n'
    stats_text += f'Val MSE: {val_mse:.4f}\n'
    stats_text += f'{test_label} MSE: {test_mse:.4f}'
    
    # Add text box with statistics
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Learning analysis plot saved to: {output_path}")
    plt.close()


def create_simple_learning_plot(
    train_losses: list,
    val_losses: list,
    output_path: str,
    model_name: str = "Model",
    skip_epochs: int = 0
) -> None:
    """
    Create a simple learning curve plot showing only training and validation losses.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        output_path: Path to save the plot
        model_name: Name of the model for the title
        skip_epochs: Number of epochs to skip from the beginning
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Skip first N epochs if requested
    if skip_epochs > 0 and len(train_losses) > skip_epochs:
        epochs = range(skip_epochs + 1, len(train_losses) + 1)
        train_losses_plot = train_losses[skip_epochs:]
        val_losses_plot = val_losses[skip_epochs:]
        title_suffix = f" (after epoch {skip_epochs})"
    else:
        epochs = range(1, len(train_losses) + 1)
        train_losses_plot = train_losses
        val_losses_plot = val_losses
        title_suffix = ""
    
    ax.plot(epochs, train_losses_plot, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses_plot, 'r-', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_name} - Learning Curves{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simple learning curve plot saved to: {output_path}")


def create_mse_comparison_plot(
    train_mse: float,
    val_mse: float,
    test_mse: float = None,
    output_path: str = None,
    model_name: str = "Model"
) -> None:
    """
    Create a bar chart comparing MSE across train/val/test datasets.
    
    Args:
        train_mse: Training MSE
        val_mse: Validation MSE
        test_mse: Test MSE (optional)
        output_path: Path to save the plot
        model_name: Name of the model for the title
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    if test_mse is not None:
        mse_values = [train_mse, val_mse, test_mse]
        mse_labels = ['Train', 'Validation', 'Test']
        colors = ['blue', 'red', 'green']
    else:
        mse_values = [train_mse, val_mse]
        mse_labels = ['Train', 'Validation']
        colors = ['blue', 'red']
    
    bars = ax.bar(mse_labels, mse_values, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(f'{model_name} - MSE Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, mse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"MSE comparison plot saved to: {output_path}")
    plt.close()
