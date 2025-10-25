"""
Example usage of the plotting functions.

This shows how any model trainer can use the plotting functions
to create comprehensive learning analysis plots and trajectory visualizations.
"""

import numpy as np
from scripts.plotting.plot_learning_curves import (
    create_enhanced_learning_plot,
    create_simple_learning_plot,
    create_mse_comparison_plot
)
from scripts.plotting.plot_trajectories import (
    create_trajectory_diagnostic_plot,
    create_simple_trajectory_plot
)


def example_usage():
    """Example of how to use the plotting functions."""
    
    # Example training results
    results = {
        'train_losses': [0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.16, 0.15, 0.14, 0.13],
        'val_losses': [0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25, 0.24, 0.23],
        'final_train_mse': 0.13,
        'final_val_mse': 0.23,
        'final_test_mse': 0.25
    }
    
    # Example predictions and targets
    train_pred = np.random.randn(100, 5) * 0.1 + np.array([1, 2, 3, 4, 5])
    val_pred = np.random.randn(20, 5) * 0.1 + np.array([1, 2, 3, 4, 5])
    test_pred = np.random.randn(20, 5) * 0.1 + np.array([1, 2, 3, 4, 5])
    
    train_y = np.random.randn(100, 5) * 0.1 + np.array([1, 2, 3, 4, 5])
    val_y = np.random.randn(20, 5) * 0.1 + np.array([1, 2, 3, 4, 5])
    test_y = np.random.randn(20, 5) * 0.1 + np.array([1, 2, 3, 4, 5])
    
    # Create enhanced learning plot
    create_enhanced_learning_plot(
        results=results,
        train_pred=train_pred,
        val_pred=val_pred,
        test_pred=test_pred,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        output_path="example_enhanced_learning.png",
        model_name="Example Model"
    )
    
    # Create simple learning plot
    create_simple_learning_plot(
        train_losses=results['train_losses'],
        val_losses=results['val_losses'],
        output_path="example_simple_learning.png",
        model_name="Example Model",
        skip_epochs=2
    )
    
    # Create MSE comparison plot
    create_mse_comparison_plot(
        train_mse=results['final_train_mse'],
        val_mse=results['final_val_mse'],
        test_mse=results['final_test_mse'],
        output_path="example_mse_comparison.png",
        model_name="Example Model"
    )
    
    # Example trajectory data
    num_samples, num_steps, output_dim = 10, 20, 3
    trajectories = np.random.randn(num_samples, num_steps, output_dim) * 0.1
    targets = np.random.randn(num_samples, output_dim) * 0.1
    
    # Create trajectory diagnostic plot
    create_trajectory_diagnostic_plot(
        trajectories=trajectories,
        targets=targets,
        output_path="example_trajectory_diagnostics.png",
        model_name="Example Model",
        num_samples=5
    )
    
    # Create simple trajectory plot
    create_simple_trajectory_plot(
        trajectories=trajectories,
        targets=targets,
        output_path="example_simple_trajectory.png",
        model_name="Example Model",
        num_samples=3
    )


if __name__ == "__main__":
    example_usage()
