"""
Streamlined trainer for VAE_flow (NoProp-CT) implementation.

This trainer leverages the built-in methods from VAE_flow and provides
plotting and saving functionality consistent with the original trainer.
"""
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from typing import Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm
import sys
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import traceback
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.flow_models_wip.fm_wip import VAE_flow, VAEFlowConfig

# Optional plotting imports (with try-except for graceful degradation)
try:
    from src.utils.plotting.plot_learning_curves import create_enhanced_learning_plot
    from src.utils.plotting.plot_trajectories import (
        create_trajectory_diagnostic_plot,
        create_simple_trajectory_plot
    )
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class VAEFlowTrainer:
    """Streamlined trainer for VAE_flow model using built-in methods."""
    
    def __init__(
        self,
        config: VAEFlowConfig,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        seed: int = 42
    ):
        """
        Initialize the trainer.
        
        Args:
            config: VAE flow configuration
            learning_rate: Learning rate for optimizer
            optimizer_name: Name of optimizer ('adam', 'sgd', etc.)
            seed: Random seed for reproducibility
        """
        self.config = config
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.seed = seed
        
        # Initialize model
        self.model = VAE_flow(config=config)
        
        # Initialize optimizer
        if optimizer_name.lower() == "adam":
            self.optimizer = optax.adam(learning_rate)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optax.sgd(learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Initialize state
        self.params = None
        self.opt_state = None
        self.rng = jr.PRNGKey(seed)
        
    def initialize(self, x_sample: jnp.ndarray, y_sample: jnp.ndarray, z_sample: jnp.ndarray, t_sample: jnp.ndarray):
        """
        Initialize model parameters and optimizer state.
        
        Args:
            x_sample: Sample input data [batch_size, input_dim]
            y_sample: Sample target data [batch_size, output_dim]
            z_sample: Sample latent data [batch_size, latent_dim]
            t_sample: Sample time data [batch_size]
        """
        self.rng, init_rng = jr.split(self.rng)
        # Use the model's __call__ method for initialization
        self.params = self.model.init(init_rng, x_sample, y_sample, init_rng)
        self.opt_state = self.optimizer.init(self.params)
        print(f"Model initialized with {sum(x.size for x in jax.tree_leaves(self.params))} parameters")
        
    def train_step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, use_dropout: bool = True) -> Dict[str, float]:
        """
        Single training step using VAE_flow's built-in train_step method.
        
        Args:
            x_batch: Input batch [batch_size, input_dim]
            y_batch: Target batch [batch_size, output_dim]
            use_dropout: Whether to use dropout during training
            
        Returns:
            Dictionary of training metrics
        """
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        # Use VAE_flow's built-in train_step method
        self.rng, train_rng = jr.split(self.rng)
        if use_dropout:
            self.params, self.opt_state, loss, metrics = self.model.train_step_with_dropout(
                self.params, x_batch, y_batch, self.opt_state, self.optimizer, train_rng
            )
        else:
            self.params, self.opt_state, loss, metrics = self.model.train_step_without_dropout(
                self.params, x_batch, y_batch, self.opt_state, self.optimizer, train_rng
            )
        
        return metrics
    
    def train_epoch(self, x_data: jnp.ndarray, y_data: jnp.ndarray, batch_size: int = 256, use_dropout: bool = True) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            x_data: Training input data [num_samples, input_dim]
            y_data: Training target data [num_samples, output_dim]
            batch_size: Batch size for training
            use_dropout: Whether to use dropout during training
            
        Returns:
            Dictionary of epoch metrics
        """
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        num_samples = x_data.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        epoch_metrics = {
            'total_loss': 0.0,
            'flow_loss': 0.0,
            'recon_loss': 0.0,
            'step': 0
        }
        
        # Shuffle data
        self.rng, shuffle_rng = jr.split(self.rng)
        perm = jr.permutation(shuffle_rng, num_samples)
        x_data = x_data[perm]
        y_data = y_data[perm]
        
        # Train on batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            x_batch = x_data[start_idx:end_idx]
            y_batch = y_data[start_idx:end_idx]
            
            # Training step
            metrics = self.train_step(x_batch, y_batch, use_dropout=use_dropout)
            
            # Accumulate metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
            epoch_metrics['step'] += 1
        
        # Average metrics
        for key in epoch_metrics:
            if key != 'step':
                epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train(
        self,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        num_epochs: int = 100,
        batch_size: int = 256,
        validation_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        dropout_epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            x_data: Training input data [num_samples, input_dim]
            y_data: Training target data [num_samples, output_dim]
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Optional validation data (x_val, y_val)
            dropout_epochs: Number of epochs with dropout (if None, uses all epochs)
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training history
        """
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        # Set dropout epochs
        if dropout_epochs is None:
            dropout_epochs = num_epochs
        
        # Initialize history
        history = {
            'train_losses': [],
            'train_flow_losses': [],
            'train_recon_losses': [],
            'val_losses': [],
            'val_flow_losses': [],
            'val_recon_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'train_pred': [],
            'val_pred': [],
            'train_x': [],
            'train_y': [],
            'val_x': [],
            'val_y': []
        }
        
        if verbose:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Dropout epochs: {dropout_epochs}")
            print(f"Training data shape: x={x_data.shape}, y={y_data.shape}")
            if validation_data is not None:
                print(f"Validation data shape: x={validation_data[0].shape}, y={validation_data[1].shape}")
        
        # Training loop
        for epoch in tqdm(range(num_epochs), desc="Training", disable=not verbose):
            # Determine if we should use dropout
            use_dropout = epoch < dropout_epochs
            
            # Train epoch
            train_metrics = self.train_epoch(x_data, y_data, batch_size, use_dropout)
            
            # Store training metrics
            history['train_losses'].append(train_metrics['total_loss'])
            history['train_flow_losses'].append(train_metrics['flow_loss'])
            history['train_recon_losses'].append(train_metrics['recon_loss'])
            
            # Validation
            if validation_data is not None:
                val_metrics = self.evaluate(validation_data[0], validation_data[1], batch_size)
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_flow_losses'].append(val_metrics['flow_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
                history['val_accuracies'].append(val_metrics.get('accuracy', 0.0))
            
            # Compute accuracy for training data (sample)
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                train_acc = self.compute_accuracy(x_data[:100], y_data[:100])
                history['train_accuracies'].append(train_acc)
                
                if validation_data is not None:
                    val_acc = self.compute_accuracy(validation_data[0][:100], validation_data[1][:100])
                    history['val_accuracies'].append(val_acc)
        
        # Store final predictions and data (use more samples for better visualization)
        num_viz_samples = min(2000, x_data.shape[0])  # Use up to 2000 samples for visualization
        history['train_pred'] = self.predict(x_data[:num_viz_samples])
        history['train_x'] = np.array(x_data[:num_viz_samples])
        history['train_y'] = np.array(y_data[:num_viz_samples])
        
        if validation_data is not None:
            val_num_viz_samples = min(1000, validation_data[0].shape[0])  # Use up to 1000 validation samples
            history['val_pred'] = self.predict(validation_data[0][:val_num_viz_samples])
            history['val_x'] = np.array(validation_data[0][:val_num_viz_samples])
            history['val_y'] = np.array(validation_data[1][:val_num_viz_samples])
        
        if verbose:
            print("Training completed!")
        
        return history
    
    def evaluate(self, x_data: jnp.ndarray, y_data: jnp.ndarray, batch_size: int = 256) -> Dict[str, float]:
        """
        Evaluate the model on given data.
        
        Args:
            x_data: Input data [num_samples, input_dim]
            y_data: Target data [num_samples, output_dim]
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        num_samples = x_data.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        total_loss = 0.0
        flow_loss = 0.0
        recon_loss = 0.0
        
        # Evaluate on batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            x_batch = x_data[start_idx:end_idx]
            y_batch = y_data[start_idx:end_idx]
            
            # Use model's loss method for evaluation
            self.rng, eval_rng = jr.split(self.rng)
            loss, metrics = self.model.loss(self.params, x_batch, y_batch, eval_rng, training=False)
            
            total_loss += loss
            flow_loss += metrics.get('flow_loss', 0.0)
            recon_loss += metrics.get('recon_loss', 0.0)
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_flow_loss = flow_loss / num_batches
        avg_recon_loss = recon_loss / num_batches
        
        return {
            'total_loss': float(avg_loss),
            'flow_loss': float(avg_flow_loss),
            'recon_loss': float(avg_recon_loss)
        }
    
    def predict(self, x_data: jnp.ndarray, num_steps: int = 20, integration_method: str = "euler", output_type: str = "end_point") -> jnp.ndarray:
        """
        Make predictions using the model's predict method.
        
        Args:
            x_data: Input data [num_samples, input_dim]
            num_steps: Number of integration steps
            integration_method: Integration method for ODE solving
            output_type: Type of output ('end_point', 'trajectory')
            
        Returns:
            Predictions [num_samples, output_dim] or [num_samples, num_steps, output_dim]
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        return self.model.predict(self.params, x_data, num_steps, integration_method, output_type)
    
    def compute_accuracy(self, x_data: jnp.ndarray, y_data: jnp.ndarray) -> float:
        """
        Compute classification accuracy.
        
        Args:
            x_data: Input data [num_samples, input_dim]
            y_data: Target data [num_samples, output_dim] (one-hot encoded)
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        predictions = self.predict(x_data)
        
        # Convert to numpy for easier computation
        pred_np = np.array(predictions)
        y_np = np.array(y_data)
        
        # For classification, compare predicted vs true classes
        if y_np.shape[1] > 1:  # One-hot encoded
            pred_classes = np.argmax(pred_np, axis=1)
            true_classes = np.argmax(y_np, axis=1)
        else:  # Binary classification
            pred_classes = (pred_np > 0.5).astype(int).flatten()
            true_classes = y_np.astype(int).flatten()
        
        accuracy = np.mean(pred_classes == true_classes)
        return float(accuracy)
    
    def save_params(self, filepath: str):
        """Save model parameters to file."""
        if self.params is None:
            raise ValueError("Model not initialized. No parameters to save.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"Parameters saved to {filepath}")
    
    def load_params(self, filepath: str):
        """Load model parameters from file."""
        with open(filepath, 'rb') as f:
            self.params = pickle.load(f)
        print(f"Parameters loaded from {filepath}")
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save training results and create plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        results_file = os.path.join(output_dir, "training_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {results_file}")
        
        # Save parameters
        params_file = os.path.join(output_dir, "model_params.pkl")
        self.save_params(params_file)
        
        # Create plots if plotting is available
        if PLOTTING_AVAILABLE:
            self._create_plots(results, output_dir)
        else:
            print("Plotting not available. Install plotting dependencies to generate plots.")
    
    def _create_plots(self, results: Dict[str, Any], output_dir: str):
        """Create diagnostic plots."""
        try:
            # Learning curves
            if 'train_losses' in results and len(results['train_losses']) > 0:
                self._create_learning_curves_plot(results, output_dir)
            
            # Data visualization
            if 'train_x' in results and 'train_y' in results:
                self._create_data_visualization_plot(results, output_dir)
            
            # Predictions vs targets
            if 'train_pred' in results and 'train_y' in results:
                self._create_predictions_plot(results, output_dir)
                
        except Exception as e:
            print(f"Warning: Error creating plots: {e}")
            traceback.print_exc()
    
    def _create_learning_curves_plot(self, results: Dict[str, Any], output_dir: str):
        """Create learning curves plot."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Progress', fontsize=16)
        
        epochs = range(len(results['train_losses']))
        
        # Total loss
        axes[0, 0].plot(epochs, results['train_losses'], label='Train', color='blue')
        if 'val_losses' in results and len(results['val_losses']) > 0:
            val_epochs = range(len(results['val_losses']))
            axes[0, 0].plot(val_epochs, results['val_losses'], label='Validation', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Flow loss
        axes[0, 1].plot(epochs, results['train_flow_losses'], label='Train', color='blue')
        if 'val_flow_losses' in results and len(results['val_flow_losses']) > 0:
            axes[0, 1].plot(val_epochs, results['val_flow_losses'], label='Validation', color='red')
        axes[0, 1].set_title('Flow Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Reconstruction loss
        axes[1, 0].plot(epochs, results['train_recon_losses'], label='Train', color='blue')
        if 'val_recon_losses' in results and len(results['val_recon_losses']) > 0:
            axes[1, 0].plot(val_epochs, results['val_recon_losses'], label='Validation', color='red')
        axes[1, 0].set_title('Reconstruction Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Accuracy
        if 'train_accuracies' in results and len(results['train_accuracies']) > 0:
            acc_epochs = range(len(results['train_accuracies']))
            axes[1, 1].plot(acc_epochs, results['train_accuracies'], label='Train', color='blue')
            if 'val_accuracies' in results and len(results['val_accuracies']) > 0:
                val_acc_epochs = range(len(results['val_accuracies']))
                axes[1, 1].plot(val_acc_epochs, results['val_accuracies'], label='Validation', color='red')
            axes[1, 1].set_title('Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, "learning_curves.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning curves plot saved to {plot_file}")
    
    def _create_data_visualization_plot(self, results: Dict[str, Any], output_dir: str):
        """Create data visualization plot with more data points."""
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
    
    def _create_predictions_plot(self, results: Dict[str, Any], output_dir: str):
        """Create predictions vs targets plot."""
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


def main():
    """Example usage of the VAEFlowTrainer."""
    # This would be used for testing or as a standalone script
    pass


if __name__ == "__main__":
    main()
