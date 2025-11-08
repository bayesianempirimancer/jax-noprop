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
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.flow_models.fm import VAE_flow as FlowMatchingModel
from src.flow_models.df import VAE_flow as DiffusionModel
from src.flow_models.ct import VAE_flow as CTModel
from src.flow_models.config import Config as FlowMatchingConfig, Config as DiffusionConfig, Config as CTConfig
from flax import traverse_util

# Optional plotting imports (with try-except for graceful degradation)
try:
    from src.utils.plotting.plot_regression import create_all_regression_plots
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class VAEFlowTrainer:
    """Streamlined trainer for VAE_flow model using built-in methods."""
    
    def __init__(
        self,
        config,
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
        
        # Initialize model based on config type
        if isinstance(config, DiffusionConfig):
            self.model = DiffusionModel(config=config)
            self.model_type = "diffusion"
        elif isinstance(config, CTConfig):
            self.model = CTModel(config=config)
            self.model_type = "ct"
        elif isinstance(config, FlowMatchingConfig):
            self.model = FlowMatchingModel(config=config)
            self.model_type = "flow_matching"
        else:
            raise ValueError(
                f"Unknown config type: {type(config)}. "
                f"Expected one of: DiffusionConfig, CTConfig, or FlowMatchingConfig"
            )
        
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
        
        # Debug: print parameter tree summary (module path -> shape, dtype)
        try:
            flat_params = traverse_util.flatten_dict(self.params, keep_empty_nodes=True)
            print("Parameter tree summary (path: shape dtype):")
            for path, value in flat_params.items():
                if hasattr(value, 'shape'):
                    key_path = "/".join(str(k) for k in path)
                    print(f"  {key_path}: {tuple(value.shape)} {value.dtype}")
        except Exception as e:
            print(f"Warning: could not summarize parameter tree: {e}")
        
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
        
        # Always use model's train_step method
        if not hasattr(self.model, 'train_step'):
            raise AttributeError("Model must implement train_step(params, x, y, opt_state, optimizer, key, training)")
        
        self.rng, train_rng = jr.split(self.rng)
        self.params, self.opt_state, loss, metrics = self.model.train_step(
            self.params, x_batch, y_batch, self.opt_state, self.optimizer, train_rng, training=use_dropout
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
    
    def predict(self, x_data: jnp.ndarray, num_steps: int = 20, integration_method: str = "midpoint", output_type: str = "end_point") -> jnp.ndarray:
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
            # Labels are now {0, 1}, argmax works directly
            pred_classes = np.argmax(pred_np, axis=1)
            true_classes = np.argmax(y_np, axis=1)
        else:  # Binary classification
            # Predictions: if pred > 0, class 1, else class 0
            pred_classes = (pred_np > 0).astype(int).flatten()
            # Labels are now {0, 1}, use directly
            true_classes = y_np.astype(int).flatten()
        
        accuracy = np.mean(pred_classes == true_classes)
        return float(accuracy)
    
    def save_params(self, filepath: str):
        """Save model parameters and config to file."""
        if self.params is None:
            raise ValueError("Model not initialized. No parameters to save.")
        
        # Save both parameters and config
        save_data = {
            'params': self.params,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Parameters and config saved to {filepath}")
    
    def load_params(self, filepath: str):
        """Load model parameters and config from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both old format (just params) and new format (params + config)
        if isinstance(data, dict) and 'params' in data:
            self.params = data['params']
            if 'config' in data:
                self.config = data['config']
                print(f"Parameters and config loaded from {filepath}")
            else:
                print(f"Parameters loaded from {filepath} (no config found)")
        else:
            # Old format - just parameters
            self.params = data
            print(f"Parameters loaded from {filepath} (old format, no config)")
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save training results and create plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        results_file = os.path.join(output_dir, "training_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {results_file}")
        
        # Save config as YAML using BaseConfig method
        config_yaml_path = os.path.join(output_dir, "config.yaml")
        config_json_path = os.path.join(output_dir, "config.json")
        if hasattr(self.config, 'save_yaml'):
            self.config.save_yaml(config_yaml_path)
            print(f"Config saved to {config_yaml_path}")
        elif hasattr(self.config, 'save_json'):
            self.config.save_json(config_json_path)
            print(f"Config saved to {config_json_path}")
        
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
            from src.utils.plotting.plot_regression import create_all_regression_plots
            create_all_regression_plots(results, self.model, self.params, output_dir)
        except Exception as e:
            print(f"Warning: Error creating plots: {e}")
            traceback.print_exc()


def main():
    """Example usage of the VAEFlowTrainer."""
    # This would be used for testing or as a standalone script
    pass


if __name__ == "__main__":
    main()
