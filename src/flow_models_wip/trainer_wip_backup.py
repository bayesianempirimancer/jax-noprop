"""
Trainer for VAE_flow (NoProp-CT) implementation.

This trainer handles the training loop for the VAE_flow model with flow matching.
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
    """Trainer for VAE_flow model."""
    
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
            config: VAE_flow configuration
            learning_rate: Learning rate for optimizer
            optimizer_name: Name of optimizer ("adam", "adamw", "sgd")
            seed: Random seed
        """
        self.config = config
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.seed = seed
        
        # Initialize model (shapes are now computed from config)
        self.model = VAE_flow(config=config)
        
        # Initialize optimizer
        if optimizer_name.lower() == "adam":
            self.optimizer = optax.adam(learning_rate)
        elif optimizer_name.lower() == "adamw":
            self.optimizer = optax.adamw(learning_rate)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optax.sgd(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Initialize random key
        self.rng = jr.PRNGKey(seed)
        
        # Training state
        self.params = None
        self.opt_state = None
        self.step = 0
        
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
        # Use standard Flax initialization - @nn.compact methods will be initialized automatically
        # The __call__ method now takes (x, y, key, training)
        self.params = self.model.init(init_rng, x_sample, y_sample, init_rng)
        self.opt_state = self.optimizer.init(self.params)
        print(f"Model initialized with {sum(x.size for x in jax.tree_leaves(self.params))} parameters")
        
    def train_step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, use_dropout: bool = True) -> Dict[str, float]:
        """
        Single training step using VAE_flow's optimized JIT-compiled train_step method.
        
        Args:
            x_batch: Input batch [batch_size, input_dim]
            y_batch: Target batch [batch_size, output_dim]
            use_dropout: Whether to use dropout during training
            
        Returns:
            Dictionary of training metrics
        """
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        # Use VAE_flow's JIT-compiled train_step method (now works with BaseModel inheritance)
        self.rng, train_rng = jr.split(self.rng)
        if use_dropout:
            self.params, self.opt_state, loss, metrics = self.model.train_step_with_dropout(
                self.params, x_batch, y_batch, self.opt_state, self.optimizer, train_rng
            )
        else:
            self.params, self.opt_state, loss, metrics = self.model.train_step_without_dropout(
                self.params, x_batch, y_batch, self.opt_state, self.optimizer, train_rng
            )
        
        # Update step counter
        self.step += 1
        
        # Convert metrics to float for logging
        metrics_float = {k: float(v) for k, v in metrics.items()}
        metrics_float['step'] = self.step
        
        return metrics_float
    
    def train_epoch(self, x_data: jnp.ndarray, y_data: jnp.ndarray, batch_size: int = 256, use_dropout: bool = True) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            x_data: Input data [num_samples, input_dim]
            y_data: Target data [num_samples, output_dim]
            batch_size: Batch size for training
            use_dropout: Whether to use dropout during training
            
        Returns:
            Dictionary of average metrics for the epoch
        """
        num_samples = x_data.shape[0]
        num_batches = num_samples // batch_size  # Use integer division for efficiency
        
        epoch_losses = []
        epoch_flow_losses = []
        epoch_recon_losses = []
        
        # Shuffle data for this epoch
        self.rng, shuffle_key = jr.split(self.rng)
        indices = jr.permutation(shuffle_key, num_samples)
        x_shuffled = x_data[indices]
        y_shuffled = y_data[indices]
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            x_batch = x_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Single training step
            metrics = self.train_step(x_batch, y_batch, use_dropout)
            
            epoch_losses.append(metrics['total_loss'])
            epoch_flow_losses.append(metrics['flow_loss'])
            epoch_recon_losses.append(metrics['recon_loss'])
        
        # Average metrics across batches
        avg_metrics = {
            'total_loss': jnp.mean(jnp.array(epoch_losses)),
            'flow_loss': jnp.mean(jnp.array(epoch_flow_losses)),
            'recon_loss': jnp.mean(jnp.array(epoch_recon_losses)),
            'step': self.step
        }
        
        return avg_metrics
    
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
        
        history = {
            'train_losses': [],
            'train_flow_losses': [],
            'train_recon_losses': [],
            'val_losses': [],
            'val_flow_losses': [],
            'val_recon_losses': [],
            'train_accuracies': [],
            'val_accuracies': []
        }
        
        if verbose:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Dropout epochs: {dropout_epochs}")
            print(f"Training data shape: x={x_data.shape}, y={y_data.shape}")
            if validation_data is not None:
                print(f"Validation data shape: x={validation_data[0].shape}, y={validation_data[1].shape}")
        
        for epoch in tqdm(range(num_epochs), desc="Training", disable=not verbose):
            # Determine if we should use dropout for this epoch
            use_dropout = epoch < dropout_epochs
            
            # Training
            train_metrics = self.train_epoch(x_data, y_data, batch_size, use_dropout)
            
            # Validation using the same JIT-compiled loss function
            if validation_data is not None:
                val_metrics = self.evaluate(validation_data[0], validation_data[1], batch_size)
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_flow_losses'].append(val_metrics['flow_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
            
            # Record training metrics
            history['train_losses'].append(train_metrics['total_loss'])
            history['train_flow_losses'].append(train_metrics['flow_loss'])
            history['train_recon_losses'].append(train_metrics['recon_loss'])
            
            # Skip expensive accuracy computation during training
            # Accuracy will be computed only at the end
            history['train_accuracies'].append(0.0)  # Placeholder
            history['val_accuracies'].append(0.0)    # Placeholder
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_metrics['total_loss']:.4f} "
                      f"(Flow: {train_metrics['flow_loss']:.4f}, "
                      f"Recon: {train_metrics['recon_loss']:.4f})")
                if validation_data is not None:
                    print(f"  Val Loss: {val_metrics['total_loss']:.4f} "
                          f"(Flow: {val_metrics['flow_loss']:.4f}, "
                          f"Recon: {val_metrics['recon_loss']:.4f})")
                print()
        
        # Add final predictions and data to history for plotting
        print("\nComputing final predictions...")
        history['train_pred'] = self._compute_predictions(x_data)
        history['val_pred'] = self._compute_predictions(validation_data[0]) if validation_data is not None else None
        history['train_x'] = x_data
        history['train_y'] = y_data
        history['val_x'] = validation_data[0] if validation_data is not None else None
        history['val_y'] = validation_data[1] if validation_data is not None else None
        
        # Compute final accuracies (expensive operation, done only once at the end)
        print("Evaluating classification performance...")
        final_train_acc = self._compute_accuracy(x_data, y_data)
        final_val_acc = self._compute_accuracy(validation_data[0], validation_data[1]) if validation_data is not None else 0.0
        
        # Update the last accuracy values in history
        if history['train_accuracies']:
            history['train_accuracies'][-1] = final_train_acc
        if history['val_accuracies']:
            history['val_accuracies'][-1] = final_val_acc
        
        return history
    
    def evaluate(self, x_data: jnp.ndarray, y_data: jnp.ndarray, batch_size: int = 256, training: bool = False) -> Dict[str, float]:
        """
        Evaluate the model on given data using the same JIT-compiled loss function as training.
        
        Args:
            x_data: Input data [num_samples, input_dim]
            y_data: Target data [num_samples, output_dim]
            batch_size: Batch size for evaluation (unused, kept for compatibility)
            training: Whether to use training mode (affects dropout, etc.)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        # Use the same JIT-compiled loss function as training
        # Generate a single random key for the entire evaluation
        self.rng, eval_rng = jr.split(self.rng)
        
        # Stop gradient computation for evaluation (speed optimization)
        params_no_grad = jax.lax.stop_gradient(self.params)
        
        # Compute loss on the entire dataset at once (JIT-compiled) with specified training mode
        loss, metrics = self.model.loss(params_no_grad, x_data, y_data, eval_rng, training=training)
        
        # Convert to float for logging (outside JIT compilation)
        return {k: float(v) for k, v in metrics.items()}
    
    def save_params(self, filepath: str):
        """Save model parameters to file."""
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"Parameters saved to {filepath}")
    
    def load_params(self, filepath: str):
        """Load model parameters from file."""
        with open(filepath, 'rb') as f:
            self.params = pickle.load(f)
        print(f"Parameters loaded from {filepath}")
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save training results and generate plots."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training results
        with open(output_path / "training_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Generate plots
        print("Generating plots...")
        
        # Use plotting functions if available
        try:
            
            # 1. Learning curve plot
            print("Generating learning curve plot...")
            try:
                learning_plot_path = output_path / "learning_analysis.png"
                create_enhanced_learning_plot(
                    results=results,
                    train_pred=results['train_pred'],
                    val_pred=results['val_pred'],
                    test_pred=results.get('test_pred'),
                    train_y=results['train_y'],
                    val_y=results['val_y'],
                    test_y=results.get('test_y'),
                    output_path=str(learning_plot_path),
                    model_name="VAE Flow Model",
                    skip_epochs=0  # Don't skip any epochs for VAE
                )
                print(f"Learning curve plot saved to: {learning_plot_path}")
            except Exception as e:
                print(f"Warning: Could not generate learning curve plot: {e}")
            
            # 2. Data visualization plot (original data + class-colored points)
            print("Generating data visualization plot...")
            try:
                self._create_data_visualization_plot(results, output_path)
            except Exception as e:
                print(f"Warning: Could not generate data visualization plot: {e}")
            
            # 3. Trajectory plots
            print("Generating trajectory plots...")
            try:
                self._create_trajectory_plots(results, output_path)
            except Exception as e:
                print(f"Warning: Could not generate trajectory plots: {e}")
            
            # 4. Latent space flow plots
            print("Generating latent space flow plots...")
            try:
                self._create_latent_space_flow_plot(results, output_path)
            except Exception as e:
                print(f"Warning: Could not generate latent space flow plots: {e}")
                
        except ImportError as e:
            print(f"Warning: Could not import plotting utilities: {e}")
            print("Skipping plot generation...")
        
        print(f"Results saved to: {output_path}")
    
    def _create_vae_diagnostic_plots(self, results: Dict[str, Any], output_path):
        """Create VAE-specific diagnostic plots."""
        
        # 1. Loss components plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('VAE Flow Model - Training Analysis', fontsize=16, fontweight='bold')
        
        # Plot training losses
        epochs = range(1, len(results['train_loss']) + 1)
        
        # Total loss
        axes[0, 0].plot(epochs, results['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, results['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Flow loss
        if 'train_flow_loss' in results:
            axes[0, 1].plot(epochs, results['train_flow_loss'], 'g-', label='Flow Loss', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Flow Loss')
            axes[0, 1].set_title('Flow Matching Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Reconstruction loss
        if 'train_recon_loss' in results:
            axes[1, 0].plot(epochs, results['train_recon_loss'], 'm-', label='Recon Loss', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Reconstruction Loss')
            axes[1, 0].set_title('Reconstruction Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        if 'train_accuracies' in results and 'val_accuracies' in results:
            axes[1, 1].plot(epochs, results['train_accuracies'], 'b-', label='Train Acc', linewidth=2)
            axes[1, 1].plot(epochs, results['val_accuracies'], 'r-', label='Val Acc', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Classification Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])
        else:
            # If no accuracy data, show loss ratio
            if 'train_flow_loss' in results and 'train_recon_loss' in results:
                flow_recon_ratio = np.array(results['train_flow_loss']) / np.array(results['train_recon_loss'])
                axes[1, 1].plot(epochs, flow_recon_ratio, 'purple', linewidth=2)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Flow/Recon Loss Ratio')
                axes[1, 1].set_title('Flow vs Reconstruction Loss Ratio')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        vae_plot_path = output_path / "vae_training_analysis.png"
        plt.savefig(vae_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"VAE training analysis plot saved to: {vae_plot_path}")
        
        # 2. Latent space visualization (if 2D or 3D)
        if results['train_pred'].shape[1] == 2:
            self._create_latent_space_plot(results, output_path)
    
    def _create_latent_space_plot(self, results: Dict[str, Any], output_path):
        """Create latent space visualization for 2D data."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('VAE Flow Model - Latent Space Analysis', fontsize=16, fontweight='bold')
        
        # Original data
        train_x = results['train_x']
        train_y = results['train_y']
        val_x = results['val_x']
        val_y = results['val_y']
        
        # Plot original data
        axes[0].scatter(train_x[:, 0], train_x[:, 1], c=train_y.argmax(axis=1), 
                       cmap='viridis', alpha=0.6, s=20, label='Train')
        axes[0].scatter(val_x[:, 0], val_x[:, 1], c=val_y.argmax(axis=1), 
                       cmap='viridis', alpha=0.8, s=30, marker='x', label='Val')
        axes[0].set_xlabel('X1')
        axes[0].set_ylabel('X2')
        axes[0].set_title('Original Data')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot predictions
        train_pred = results['train_pred']
        val_pred = results['val_pred']
        
        axes[1].scatter(train_pred[:, 0], train_pred[:, 1], c=train_y.argmax(axis=1), 
                       cmap='viridis', alpha=0.6, s=20, label='Train Pred')
        axes[1].scatter(val_pred[:, 0], val_pred[:, 1], c=val_y.argmax(axis=1), 
                       cmap='viridis', alpha=0.8, s=30, marker='x', label='Val Pred')
        axes[1].set_xlabel('Predicted X1')
        axes[1].set_ylabel('Predicted X2')
        axes[1].set_title('Model Predictions')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot residuals
        train_residuals = train_pred - train_x
        val_residuals = val_pred - val_x
        
        axes[2].scatter(train_residuals[:, 0], train_residuals[:, 1], 
                       c=train_y.argmax(axis=1), cmap='viridis', alpha=0.6, s=20, label='Train Res')
        axes[2].scatter(val_residuals[:, 0], val_residuals[:, 1], 
                       c=val_y.argmax(axis=1), cmap='viridis', alpha=0.8, s=30, marker='x', label='Val Res')
        axes[2].set_xlabel('Residual X1')
        axes[2].set_ylabel('Residual X2')
        axes[2].set_title('Prediction Residuals')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        latent_plot_path = output_path / "latent_space_analysis.png"
        plt.savefig(latent_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Latent space analysis plot saved to: {latent_plot_path}")
    
    def _create_data_visualization_plot(self, results: Dict[str, Any], output_path):
        """Create data visualization plot showing true vs predicted class labels."""
        
        # Create a 1x2 subplot layout
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Two Moons: True vs Predicted Class Labels', fontsize=16, fontweight='bold')
        
        # Get the data
        train_x = results.get('train_x', results.get('x_data'))
        train_y = results.get('train_y', results.get('y_data'))
        train_pred = results.get('train_pred')
        
        if train_x is None or train_y is None or train_pred is None:
            print("Warning: No data available for visualization")
            return
        
        # Convert one-hot encoded y back to class labels if needed
        if train_y.ndim > 1 and train_y.shape[1] > 1:
            y_true_classes = np.argmax(train_y, axis=1)
        else:
            y_true_classes = train_y.astype(int)
        
        # Convert predictions to class labels
        if train_pred.ndim > 1 and train_pred.shape[1] > 1:
            y_pred_classes = np.argmax(train_pred, axis=1)
        else:
            y_pred_classes = (train_pred > 0.5).astype(int).flatten()
        
        # Colors and labels
        colors = ['red', 'blue']
        class_names = ['Class 0', 'Class 1']
        
        # Plot 1: True class labels
        for class_id in [0, 1]:
            mask = y_true_classes == class_id
            axes[0].scatter(train_x[mask, 0], train_x[mask, 1], 
                           c=colors[class_id], alpha=0.6, s=20, 
                           label=class_names[class_id])
        
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].set_title('True Class Labels')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal')
        
        # Plot 2: Predicted class labels
        for class_id in [0, 1]:
            mask = y_pred_classes == class_id
            axes[1].scatter(train_x[mask, 0], train_x[mask, 1], 
                           c=colors[class_id], alpha=0.6, s=20, 
                           label=class_names[class_id])
        
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')
        axes[1].set_title('Predicted Class Labels')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal')
        
        plt.tight_layout()
        
        # Save the plot
        data_plot_path = output_path / "data_visualization.png"
        plt.savefig(data_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Data visualization plot saved to: {data_plot_path}")
    
    def _create_trajectory_plots(self, results: Dict[str, Any], output_path):
        """Create trajectory plots showing learned dynamics."""
        
        # Get sample data for trajectory analysis
        train_x = results.get('train_x')
        train_y = results.get('train_y')
        
        if train_x is None or train_y is None:
            print("Warning: No training data available for trajectory plots")
            return
        
        # Use a small subset for trajectory visualization
        num_samples = min(10, train_x.shape[0])
        sample_indices = np.random.choice(train_x.shape[0], num_samples, replace=False)
        x_sample = train_x[sample_indices]
        y_sample = train_y[sample_indices]
        
        # Convert to JAX arrays
        x_jax = jnp.array(x_sample)
        y_jax = jnp.array(y_sample)
        
        print(f"Computing trajectories for {num_samples} samples...")
        
        # Get trajectory data using the model's predict method
        try:
            # Get full trajectories
            trajectories = self.model.predict(
                self.params, 
                x_jax, 
                num_steps=20,
                output_type="trajectory"
            )  # Shape: [num_steps, num_samples, output_dim]
            
            # Convert to numpy for plotting
            trajectories_np = np.array(trajectories)
            targets_np = np.array(y_jax)
            
            print(f"Trajectory shape: {trajectories_np.shape}")
            
            # Create trajectory diagnostic plot
            
            trajectory_plot_path = output_path / "trajectory_diagnostics.png"
            create_trajectory_diagnostic_plot(
                trajectories=trajectories_np.transpose(1, 0, 2),  # [num_samples, num_steps, output_dim]
                targets=targets_np,
                output_path=str(trajectory_plot_path),
                model_name="VAE Flow Model",
                num_samples=num_samples
            )
            
            # Create simple trajectory plot
            
            simple_trajectory_plot_path = output_path / "simple_trajectories.png"
            create_simple_trajectory_plot(
                trajectories=trajectories_np.transpose(1, 0, 2),  # [num_samples, num_steps, output_dim]
                targets=targets_np,
                output_path=str(simple_trajectory_plot_path),
                model_name="VAE Flow Model",
                num_samples=min(5, num_samples)
            )
            
            print(f"Trajectory plots saved to: {trajectory_plot_path} and {simple_trajectory_plot_path}")
            
        except Exception as e:
            print(f"Error computing trajectories: {e}")
            # Fallback: create a simple plot showing the flow field
            self._create_flow_field_plot(x_sample, y_sample, output_path)
    
    def _create_flow_field_plot(self, x_sample: np.ndarray, y_sample: np.ndarray, output_path):
        """Create a simple flow field visualization as fallback."""
        
        # Create a grid for flow field visualization
        x_min, x_max = x_sample[:, 0].min() - 0.5, x_sample[:, 0].max() + 0.5
        y_min, y_max = x_sample[:, 1].min() - 0.5, x_sample[:, 1].max() + 0.5
        
        # Create grid
        x_grid = np.linspace(x_min, x_max, 20)
        y_grid = np.linspace(y_min, y_max, 20)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Flatten grid for model evaluation
        grid_points = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1)
        grid_jax = jnp.array(grid_points)
        
        # Get predictions for grid points
        try:
            predictions = self.model.predict(self.params, grid_jax, num_steps=20)
            pred_classes = np.argmax(np.array(predictions), axis=1)
            
            # Reshape for plotting
            pred_grid = pred_classes.reshape(X_grid.shape)
            
            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot decision boundary
            ax.contourf(X_grid, Y_grid, pred_grid, levels=[-0.5, 0.5, 1.5], colors=['lightcoral', 'lightblue'], alpha=0.3)
            
            # Plot original data points
            colors = ['red', 'blue']
            for class_id in [0, 1]:
                mask = np.argmax(y_sample, axis=1) == class_id
                ax.scatter(x_sample[mask, 0], x_sample[mask, 1], 
                          c=colors[class_id], alpha=0.7, s=30, 
                          label=f'Class {class_id}')
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title('VAE Flow Model - Decision Boundary')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            plt.tight_layout()
            
            # Save the plot
            flow_field_plot_path = output_path / "flow_field_visualization.png"
            plt.savefig(flow_field_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Flow field visualization saved to: {flow_field_plot_path}")
            
        except Exception as e:
            print(f"Error creating flow field plot: {e}")
    
    def _compute_accuracy(self, x_data: jnp.ndarray, y_data: jnp.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = self._compute_predictions(x_data)
        
        # For classification, compare predicted vs true classes
        if y_data.shape[1] > 1:  # One-hot encoded
            pred_classes = jnp.argmax(predictions, axis=1)
            true_classes = jnp.argmax(y_data, axis=1)
        else:  # Binary classification
            pred_classes = (predictions > 0.5).astype(int).flatten()
            true_classes = y_data.astype(int).flatten()
        
        accuracy = jnp.mean(pred_classes == true_classes)
        return float(accuracy)
    
    def _compute_predictions(self, x_data: jnp.ndarray) -> jnp.ndarray:
        """Compute model predictions for given input data."""
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        # Use the model's predict method with the corrected signature
        predictions = self.model.predict(self.params, x_data, num_steps=20, integration_method="euler")
        return predictions
    
    def _create_latent_space_flow_plot(self, results: Dict[str, Any], output_path):
        """Create latent space flow visualization showing how data flows through the latent space."""
        
        # Get sample data
        train_x = results.get('train_x')
        train_y = results.get('train_y')
        
        if train_x is None or train_y is None:
            print("Warning: No training data available for latent space flow plot")
            return
        
        # Use representative samples from each moon for visualization
        true_classes = np.argmax(train_y, axis=1)
        num_samples_per_class = min(50, np.sum(true_classes == 0), np.sum(true_classes == 1))
        
        # Sample from each class
        class_0_indices = np.where(true_classes == 0)[0]
        class_1_indices = np.where(true_classes == 1)[0]
        
        class_0_sample = np.random.choice(class_0_indices, num_samples_per_class, replace=False)
        class_1_sample = np.random.choice(class_1_indices, num_samples_per_class, replace=False)
        
        # Combine samples
        sample_indices = np.concatenate([class_0_sample, class_1_sample])
        x_sample = train_x[sample_indices]
        y_sample = train_y[sample_indices]
        
        num_samples = len(sample_indices)
        
        # Convert to JAX arrays
        x_jax = jnp.array(x_sample)
        y_jax = jnp.array(y_sample)
        
        print(f"Computing latent space flow for {num_samples} samples...")
        
        try:
            # Get latent representations at different time points
            num_time_points = 10
            time_points = jnp.linspace(0.0, 1.0, num_time_points)
            
            # Sample initial latent states
            key = jr.PRNGKey(42)
            z_0 = jr.normal(key, (num_samples,) + self.model.z_shape)
            
            # Get target latent states from encoder
            mu_target, logvar_target = self.model.apply(
                self.params, y_jax, method='encoder', training=False, rngs={'dropout': key}
            )
            z_target = mu_target  # Use deterministic encoding
            
            # Compute latent trajectories
            latent_trajectories = []
            for t in time_points:
                # Linear interpolation: z_t = z_0 + t * (z_target - z_0)
                z_t = z_0 + t * (z_target - z_0)
                latent_trajectories.append(z_t)
            
            latent_trajectories = jnp.stack(latent_trajectories)  # [num_time_points, num_samples, latent_dim]
            
            # Convert to numpy for plotting
            trajectories_np = np.array(latent_trajectories)
            true_classes = np.argmax(y_sample, axis=1)
            
            # Skip the old latent space flow plot - not useful
            
            # Create latent trajectories over time for each condition
            self._create_latent_trajectories_over_time_plot(x_sample, y_sample, output_path)
            
        except Exception as e:
            print(f"Error creating latent space flow plot: {e}")
            traceback.print_exc()
    
    def _create_latent_flow_field_plot(self, x_sample: np.ndarray, y_sample: np.ndarray, output_path):
        """Create a flow field visualization in the latent space."""
        
        try:
            # Create a grid in the latent space
            latent_dim = self.model.config.config["latent_shape"][0]
            if latent_dim < 2:
                print("Latent dimension too small for flow field visualization")
                return
            
            # Create 2D grid for first two latent dimensions
            grid_size = 20
            z1_range = np.linspace(-3, 3, grid_size)
            z2_range = np.linspace(-3, 3, grid_size)
            Z1, Z2 = np.meshgrid(z1_range, z2_range)
            
            # Create dummy x values for the flow field computation
            batch_size = grid_size * grid_size
            x_dummy = jnp.zeros((batch_size, 2))  # Dummy input
            z_grid = jnp.stack([Z1.flatten(), Z2.flatten()], axis=1)
            
            # Pad with zeros if latent_dim > 2
            if latent_dim > 2:
                z_padding = jnp.zeros((batch_size, latent_dim - 2))
                z_grid = jnp.concatenate([z_grid, z_padding], axis=1)
            
            # Compute flow field
            t_dummy = jnp.ones(batch_size) * 0.5  # Midpoint in time
            key = jr.PRNGKey(42)
            
            dz_dt = self.model.apply(
                self.params, z_grid, x_dummy, t_dummy, method='flow_model', 
                training=False, rngs={'dropout': key}
            )
            
            # Extract first two dimensions of the flow field
            dz_dt_2d = np.array(dz_dt[:, :2])
            U = dz_dt_2d[:, 0].reshape(grid_size, grid_size)
            V = dz_dt_2d[:, 1].reshape(grid_size, grid_size)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot flow field
            magnitude = np.sqrt(U**2 + V**2)
            ax.quiver(Z1, Z2, U, V, magnitude, cmap='viridis', alpha=0.7)
            
            # Add some sample trajectories
            num_trajectories = 20
            for i in range(num_trajectories):
                # Random starting point
                z_start = jr.normal(jr.PRNGKey(i), (2,)) * 2
                if latent_dim > 2:
                    z_start_padded = jnp.concatenate([z_start, jnp.zeros(latent_dim - 2)])
                else:
                    z_start_padded = z_start
                
                # Simple Euler integration for trajectory
                trajectory = [z_start]
                z_current = z_start_padded
                x_dummy_single = jnp.zeros((1, 2))
                
                for step in range(10):
                    t_current = jnp.array([0.5])
                    dz_dt_current = self.model.apply(
                        self.params, z_current.reshape(1, -1), x_dummy_single, t_current,
                        method='flow_model', training=False, rngs={'dropout': key}
                    )
                    z_current = z_current + 0.1 * dz_dt_current[0]
                    trajectory.append(z_current[:2])
                
                trajectory = np.array(trajectory)
                ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Latent Dim 1')
            ax.set_ylabel('Latent Dim 2')
            ax.set_title('Latent Space Flow Field (2D Projection)')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            flow_field_path = output_path / "latent_flow_field.png"
            plt.savefig(flow_field_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Latent flow field plot saved to: {flow_field_path}")
            
        except Exception as e:
            print(f"Error creating latent flow field plot: {e}")
            traceback.print_exc()
    
    def _create_latent_trajectories_over_time_plot(self, x_sample: np.ndarray, y_sample: np.ndarray, output_path):
        """Create a visualization showing latent trajectories over time for each moon condition."""
        
        try:
            # Use exactly 4 samples from each condition
            true_classes = np.argmax(y_sample, axis=1)
            class_0_indices = np.where(true_classes == 0)[0]
            class_1_indices = np.where(true_classes == 1)[0]
            
            # Sample 4 from each class
            class_0_sample = np.random.choice(class_0_indices, min(4, len(class_0_indices)), replace=False)
            class_1_sample = np.random.choice(class_1_indices, min(4, len(class_1_indices)), replace=False)
            
            # Combine samples
            sample_indices = np.concatenate([class_0_sample, class_1_sample])
            x_subset = x_sample[sample_indices]
            y_subset = y_sample[sample_indices]
            
            # Convert to JAX arrays
            x_jax = jnp.array(x_subset)
            y_jax = jnp.array(y_subset)
            
            # Get latent representations from encoder
            key = jr.PRNGKey(42)
            mu_target, logvar_target = self.model.apply(
                self.params, y_jax, method='encoder', training=False, rngs={'dropout': key}
            )
            z_target = mu_target  # Use deterministic encoding
            
            # Sample initial latent states
            z_0 = jr.normal(key, (x_subset.shape[0],) + self.model.z_shape)
            
            # Create time points for trajectory
            num_time_points = 20
            time_points = jnp.linspace(0.0, 1.0, num_time_points)
            
            # Compute latent trajectories using the flow model
            latent_trajectories = []
            for t in time_points:
                # Linear interpolation: z_t = z_0 + t * (z_target - z_0)
                z_t = z_0 + t * (z_target - z_0)
                latent_trajectories.append(z_t)
            
            latent_trajectories = jnp.stack(latent_trajectories)  # [num_time_points, num_samples, latent_dim]
            
            # Convert to numpy
            trajectories_np = np.array(latent_trajectories)
            true_classes = np.argmax(y_subset, axis=1)
            
            # Create the plot: 2 rows (conditions) x 4 columns (data points)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('Latent Trajectories Over Time: All Dimensions for Each Data Point', fontsize=16, fontweight='bold')
            
            colors = ['red', 'blue']
            moon_names = ['Moon 0', 'Moon 1']
            
            # Plot trajectories for each condition and data point
            for condition_idx in range(2):
                condition_mask = true_classes == condition_idx
                if not np.any(condition_mask):
                    continue
                
                condition_trajectories = trajectories_np[:, condition_mask, :]  # [time, samples, latent_dim]
                num_condition_samples = condition_trajectories.shape[1]
                
                # Plot each data point from this condition
                for data_point_idx in range(min(4, num_condition_samples)):
                    ax = axes[condition_idx, data_point_idx]
                    
                    # Plot all latent dimensions for this data point
                    for dim in range(self.model.config.config["latent_shape"][0]):
                        traj = condition_trajectories[:, data_point_idx, dim]
                        ax.plot(time_points, traj, alpha=0.7, linewidth=1, 
                               label=f'Dim {dim+1}' if dim < 4 else None)  # Only label first 4 for clarity
                    
                    ax.set_xlabel('Time (t)')
                    ax.set_ylabel('Latent Value')
                    ax.set_title(f'{moon_names[condition_idx]} - Data Point {data_point_idx+1}')
                    ax.grid(True, alpha=0.3)
                    
                    # Add legend only to first subplot of each row
                    if data_point_idx == 0:
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            # Save the plot
            trajectories_plot_path = output_path / "latent_trajectories_over_time.png"
            plt.savefig(trajectories_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Latent trajectories over time plot saved to: {trajectories_plot_path}")
            
        except Exception as e:
            print(f"Error creating latent trajectories over time plot: {e}")
            traceback.print_exc()
