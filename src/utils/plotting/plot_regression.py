"""
Plotting utilities for regression/classification tasks.

This module provides functions for creating diagnostic plots for regression models,
including training progress, data visualization, predictions, and trajectories.
"""
import os
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt


def plot_class_labels(results: Dict[str, Any], output_dir: str):
    """Create data visualization plot with class labels."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Data Visualization - Two Moons Dataset', fontsize=16)
    
    # Training data
    train_x = results['train_x']
    train_y = results['train_y']
    
    if train_x.shape[1] >= 2:  # At least 2D data
        # Use model predictions for coloring if available, otherwise use ground truth
        if 'train_pred' in results:
            train_pred = np.array(results['train_pred'])
            color_values = train_pred[:, 0] if train_pred.shape[1] > 0 else train_pred
            color_label = 'Prediction'
        else:
            color_values = train_y[:, 0] if train_y.shape[1] > 0 else train_y
            color_label = 'Class'
        
        # Use smaller points and higher alpha for better density visualization
        scatter = axes[0].scatter(train_x[:, 0], train_x[:, 1], 
                                c=color_values, 
                                cmap='viridis', alpha=0.7, s=8)
        axes[0].set_title(f'Training Data ({train_x.shape[0]} samples)')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label=color_label)
    
    # Validation data if available
    if 'val_x' in results and 'val_y' in results:
        val_x = np.array(results['val_x'])
        val_y = np.array(results['val_y'])
        
        if val_x.shape[1] >= 2:  # At least 2D data
            # Use model predictions for coloring if available, otherwise use ground truth
            if 'val_pred' in results:
                val_pred = np.array(results['val_pred'])
                color_values = val_pred[:, 0] if val_pred.shape[1] > 0 else val_pred
                color_label = 'Prediction'
            else:
                color_values = val_y[:, 0] if val_y.shape[1] > 0 else val_y
                color_label = 'Class'
            
            scatter = axes[1].scatter(val_x[:, 0], val_x[:, 1], 
                                    c=color_values, 
                                    cmap='viridis', alpha=0.7, s=8)
            axes[1].set_title(f'Validation Data ({val_x.shape[0]} samples)')
            axes[1].set_xlabel('Feature 1')
            axes[1].set_ylabel('Feature 2')
            axes[1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1], label=color_label)
    else:
        # If no validation data, show training data density
        axes[1].hist2d(train_x[:, 0], train_x[:, 1], bins=50, cmap='viridis', alpha=0.8)
        axes[1].set_title(f'Training Data Density ({train_x.shape[0]} samples)')
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "data_visualization.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Data visualization plot saved to {plot_file}")


def create_all_regression_plots(results: Dict[str, Any], model, params, output_dir: str, model_type: Optional[str] = None):
    """Create all diagnostic plots for regression tasks.
    
    Args:
        results: Training history dictionary
        model: Model instance
        params: Model parameters
        output_dir: Directory to save plots
        model_type: Optional model type string ('flow_matching', 'diffusion', 'ct'). 
                   If not provided, will try to infer from model.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Loss Trends Plot (unified 8-panel layout)
        if 'train_losses' in results and len(results['train_losses']) > 0:
            # Get model_type from parameter, model attribute, or infer from class name
            if model_type is None:
                model_type = getattr(model, 'model_type', None)
            if model_type is None:
                # Try to infer from model class name
                model_class_name = model.__class__.__name__.lower()
                if 'diffusion' in model_class_name:
                    model_type = 'diffusion'
                elif 'ct' in model_class_name or 'continuous' in model_class_name:
                    model_type = 'ct'
                else:
                    model_type = 'flow_matching'
            from src.utils.plotting.plot_loss_trends import create_loss_trends_plot
            create_loss_trends_plot(results, model_type, output_dir)
        
        # 2. Data visualization
        if 'train_x' in results and 'train_y' in results:
            plot_class_labels(results, output_dir)
        
        # 3. Trajectory plot
        if 'train_x' in results and 'train_y' in results:
            from src.utils.plotting.plot_trajectories import create_simple_trajectory_plot
            
            train_x = np.array(results['train_x'])
            train_y = np.array(results['train_y'])
            
            # Trajectory plot (5 samples)
            n_samples = min(5, len(train_x))
            x_sample = train_x[:n_samples]
            y_sample = train_y[:n_samples]
            
            trajectories = []
            for i in range(n_samples):
                x_single = x_sample[i:i+1]  # Keep batch dimension
                traj = model.predict(params, x_single, num_steps=20, output_type="trajectory")
                if traj.ndim == 3 and traj.shape[1] == 1:
                    traj = traj[:, 0, :]  # Remove batch dimension
                trajectories.append(traj)
            
            trajectories = np.array(trajectories)  # Shape: [n_samples, n_steps, output_dim]
            plot_file = os.path.join(output_dir, "trajectories.png")
            create_simple_trajectory_plot(
                trajectories=trajectories,
                targets=y_sample,
                output_path=plot_file,
                model_name="Model",
                num_samples=n_samples
            )
            
    except Exception as e:
        import traceback
        print(f"Warning: Error creating plots: {e}")
        traceback.print_exc()

