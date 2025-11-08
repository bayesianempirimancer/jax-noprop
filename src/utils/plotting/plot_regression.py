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


def create_training_progress_plot(results: Dict[str, Any], output_dir: str):
    """Create 4-panel training progress plot: Total loss, Reconstruction loss, Flow loss, Residuals vs Targets."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    epochs = range(len(results['train_losses']))
    
    # Panel 1: Total loss
    axes[0, 0].plot(epochs, results['train_losses'], label='Train', color='blue', linewidth=2)
    if 'val_losses' in results and len(results['val_losses']) > 0:
        val_epochs = range(len(results['val_losses']))
        axes[0, 0].plot(val_epochs, results['val_losses'], label='Validation', color='red', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel 2: Reconstruction loss
    if 'train_recon_losses' in results and len(results['train_recon_losses']) > 0:
        axes[0, 1].plot(epochs, results['train_recon_losses'], label='Train', color='blue', linewidth=2)
        if 'val_recon_losses' in results and len(results['val_recon_losses']) > 0:
            axes[0, 1].plot(val_epochs, results['val_recon_losses'], label='Validation', color='red', linewidth=2)
    axes[0, 1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel 3: Flow loss
    if 'train_flow_losses' in results and len(results['train_flow_losses']) > 0:
        axes[1, 0].plot(epochs, results['train_flow_losses'], label='Train', color='blue', linewidth=2)
        if 'val_flow_losses' in results and len(results['val_flow_losses']) > 0:
            val_epochs = range(len(results['val_flow_losses']))
            axes[1, 0].plot(val_epochs, results['val_flow_losses'], label='Validation', color='red', linewidth=2)
    axes[1, 0].set_title('Flow Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel 4: Residuals vs Targets
    if 'train_pred' in results and 'train_y' in results:
        train_pred = np.array(results['train_pred'])
        train_y = np.array(results['train_y'])
        val_pred = np.array(results.get('val_pred', []))
        val_y = np.array(results.get('val_y', []))

        # Flatten
        train_pred_flat = train_pred.reshape(-1) if train_pred.ndim > 2 else train_pred.flatten()
        train_y_flat = train_y.reshape(-1) if train_y.ndim > 2 else train_y.flatten()

        # Sample indices
        n_sample = min(1000, len(train_y_flat))
        indices = np.random.choice(len(train_y_flat), n_sample, replace=False)

        residuals = train_pred_flat - train_y_flat
        axes[1, 1].scatter(train_y_flat[indices], residuals[indices], alpha=0.6, s=15, color='blue', label='Train')

        if len(val_pred) > 0 and len(val_y) > 0:
            val_pred_flat = val_pred.reshape(-1) if val_pred.ndim > 2 else val_pred.flatten()
            val_y_flat = val_y.reshape(-1) if val_y.ndim > 2 else val_y.flatten()
            n_val_sample = min(500, len(val_y_flat))
            val_indices = np.random.choice(len(val_y_flat), n_val_sample, replace=False)
            val_residuals = val_pred_flat - val_y_flat
            axes[1, 1].scatter(val_y_flat[val_indices], val_residuals[val_indices], alpha=0.6, s=15, color='red', label='Val')

        axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('True Values')
        axes[1, 1].set_ylabel('Residuals (Predicted - True)')
        axes[1, 1].set_title('Residuals vs Targets', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "training_progress.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved to {plot_file}")


def create_data_visualization_plot(results: Dict[str, Any], output_dir: str):
    """Create data visualization plot with more data points."""
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


def create_predictions_plot(results: Dict[str, Any], output_dir: str):
    """Create predictions vs targets plot."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if 'train_pred' not in results or 'train_y' not in results:
        return
    
    train_pred = results['train_pred']
    train_y = results['train_y']
    
    # Convert to numpy if needed
    if hasattr(train_pred, 'shape'):
        pred_np = np.array(train_pred)
    else:
        pred_np = train_pred
    
    if hasattr(train_y, 'shape'):
        y_np = np.array(train_y)
    else:
        y_np = train_y
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Predictions vs Targets', fontsize=16)
    
    # Flatten for plotting if needed
    if pred_np.ndim > 2:
        pred_flat = pred_np.reshape(-1)
    else:
        pred_flat = pred_np.flatten()
    
    if y_np.ndim > 2:
        y_flat = y_np.reshape(-1)
    else:
        y_flat = y_np.flatten()
    
    # Scatter plot
    axes[0].scatter(y_flat, pred_flat, alpha=0.6)
    axes[0].plot([y_flat.min(), y_flat.max()], [y_flat.min(), y_flat.max()], 'r--', lw=2)
    axes[0].set_xlabel('Target')
    axes[0].set_ylabel('Prediction')
    axes[0].set_title('Predictions vs Targets')
    axes[0].grid(True)
    
    # Residuals
    residuals = pred_flat - y_flat
    axes[1].scatter(y_flat, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Target')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals')
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "predictions.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Predictions plot saved to {plot_file}")


def create_trajectory_plot(results: Dict[str, Any], model, params, output_dir: str):
    """Create trajectory plot using the trajectory plotting utilities."""
    try:
        from src.utils.plotting.plot_trajectories import create_simple_trajectory_plot
        
        # Get sample data for trajectory generation
        if 'train_x' not in results or 'train_y' not in results:
            return
        
        # Sample a few points for trajectory visualization
        train_x = np.array(results['train_x'])
        train_y = np.array(results['train_y'])
        n_samples = min(5, len(train_x))
        
        # Get trajectories using the model's predict method
        x_sample = train_x[:n_samples]
        y_sample = train_y[:n_samples]
        
        # Generate trajectories using the model
        trajectories = []
        for i in range(n_samples):
            x_single = x_sample[i:i+1]  # Keep batch dimension
            y_single = y_sample[i:i+1]
            traj = model.predict(
                params, 
                x_single, 
                num_steps=20, 
                output_type="trajectory"
            )
            if traj.ndim == 3 and traj.shape[1] == 1:
                traj = traj[:, 0, :]  # Remove batch dimension
            trajectories.append(traj)
        
        trajectories = np.array(trajectories)  # Shape: [n_samples, n_steps, output_dim]
        
        # Create trajectory plot
        plot_file = os.path.join(output_dir, "trajectories.png")
        create_simple_trajectory_plot(
            trajectories=trajectories,
            targets=y_sample,
            output_path=plot_file,
            model_name="Model",
            num_samples=n_samples
        )
        
    except Exception:
        import traceback
        traceback.print_exc()


def create_trajectory_diagnostics_plot(results: Dict[str, Any], model, params, output_dir: str):
    """Create trajectory diagnostics plot using the trajectory plotting utilities."""
    try:
        from src.utils.plotting.plot_trajectories import create_trajectory_diagnostic_plot
        
        # Get sample data for trajectory generation
        if 'train_x' not in results or 'train_y' not in results:
            return
        
        # Sample more points for diagnostics
        train_x = np.array(results['train_x'])
        train_y = np.array(results['train_y'])
        n_samples = min(10, len(train_x))
        
        # Get trajectories using the model's predict method
        x_sample = train_x[:n_samples]
        y_sample = train_y[:n_samples]
        
        trajectories = []
        for i in range(n_samples):
            x_single = x_sample[i:i+1]  # Keep batch dimension
            y_single = y_sample[i:i+1]
            traj = model.predict(
                params, 
                x_single, 
                num_steps=20, 
                output_type="trajectory"
            )
            if traj.ndim == 3 and traj.shape[1] == 1:
                traj = traj[:, 0, :]  # Remove batch dimension
            trajectories.append(traj)
        
        trajectories = np.array(trajectories)  # Shape: [n_samples, n_steps, output_dim]
        
        # Create trajectory diagnostics plot
        plot_file = os.path.join(output_dir, "trajectory_diagnostics.png")
        create_trajectory_diagnostic_plot(
            trajectories=trajectories,
            targets=y_sample,
            output_path=plot_file,
            model_name="Model",
            num_samples=n_samples
        )
        
    except Exception:
        import traceback
        traceback.print_exc()


def create_all_regression_plots(results: Dict[str, Any], model, params, output_dir: str):
    """Create all diagnostic plots for regression tasks."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Training Progress Plot
        if 'train_losses' in results and len(results['train_losses']) > 0:
            create_training_progress_plot(results, output_dir)
        
        # 2. Data visualization
        if 'train_x' in results and 'train_y' in results:
            create_data_visualization_plot(results, output_dir)
        
        # 3. Trajectory plot
        if 'train_pred' in results and 'train_y' in results:
            create_trajectory_plot(results, model, params, output_dir)
        
        # 4. Trajectory diagnostics plot
        if 'train_pred' in results and 'train_y' in results:
            create_trajectory_diagnostics_plot(results, model, params, output_dir)
            
    except Exception as e:
        import traceback
        print(f"Warning: Error creating plots: {e}")
        traceback.print_exc()

