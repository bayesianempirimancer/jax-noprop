"""
Minimal, professional JAX trainer for VAE flow models.

This trainer provides a clean, JAX-compliant interface for training
flow models with scan-optimized training loops.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from typing import Dict, Any, Tuple, Optional

from src.flow_models.fm import VAE_flow as FlowMatchingModel
from src.flow_models.df import VAE_flow as DiffusionModel
from src.flow_models.ct import VAE_flow as CTModel
from src.flow_models.config import Config as Config, Config as DiffusionConfig, Config as CTConfig


class Trainer:
    """Minimal trainer for regression/classification tasks."""
    
    def __init__(
        self,
        config,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        seed: int = 42,
        warmup_steps: int = 0,
        model_type: str = "flow_matching"
    ):
        self.config = config
        self.learning_rate = learning_rate
        self.seed = seed
        self.model_type = model_type
        
        if model_type == "diffusion":
            self.model = DiffusionModel(config=config)
        elif model_type == "flow_matching":
            self.model = FlowMatchingModel(config=config)
        elif model_type == "ct":
            self.model = CTModel(config=config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.optimizer = optax.adam(learning_rate) if optimizer_name.lower() == "adam" else optax.sgd(learning_rate)
        
        self.params = None
        self.opt_state = None
        self.rng = jr.PRNGKey(self.seed)
    
    def initialize(self, x_sample: jnp.ndarray, y_sample: jnp.ndarray):
        """Initialize model parameters.
        
        Args:
            x_sample: Sample input [input_dim] or [batch_size, input_dim]
            y_sample: Sample target [output_dim] or [batch_size, output_dim]
        """
        # Ensure we have batches with batch_size=1
        if x_sample.ndim == 1:
            x_sample = x_sample[None, :]
        elif x_sample.shape[0] > 1:
            x_sample = x_sample[0:1]  # Use only first sample
        
        if y_sample.ndim == 1:
            y_sample = y_sample[None, :]
        elif y_sample.shape[0] > 1:
            y_sample = y_sample[0:1]  # Use only first sample
        
        self.rng, init_rng = jr.split(self.rng)
        self.params = self.model.init(init_rng, x_sample, y_sample, init_rng)
        self.opt_state = self.optimizer.init(self.params)
    
    def train_epoch(
        self,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        batch_size: int = 256,
        use_dropout: bool = True
    ) -> Dict[str, float]:
        """Train for one epoch using regular for loop."""
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        x_data = jnp.asarray(x_data)
        y_data = jnp.asarray(y_data)
        
        # Shuffle and batch
        num_samples = y_data.shape[0]
        self.rng, shuffle_rng = jr.split(self.rng)
        perm = jr.permutation(shuffle_rng, num_samples)
        x_shuffled = x_data[perm]
        y_shuffled = y_data[perm]
        
        # Pre-batch data
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # Regular for loop over batches
        total_losses = []
        flow_losses = []
        recon_losses = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            x_batch = x_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Pad if needed
            if end_idx - start_idx < batch_size:
                pad_size = batch_size - (end_idx - start_idx)
                x_batch = jnp.concatenate([x_batch, jnp.zeros((pad_size, *x_batch.shape[1:]))])
                y_batch = jnp.concatenate([y_batch, jnp.zeros((pad_size, *y_batch.shape[1:]))])
            
            self.rng, step_rng = jr.split(self.rng)
            self.params, self.opt_state, loss, metrics = self.model.train_step(
                self.params, x_batch, y_batch, self.opt_state, self.optimizer, step_rng, training=use_dropout
            )
            
            # Scale loss by actual batch size
            actual_batch_size = end_idx - start_idx
            scale = batch_size / actual_batch_size if actual_batch_size < batch_size else 1.0
            
            total_losses.append(float(loss) * scale)
            flow_losses.append(float(metrics.get('flow_loss', 0.0)) * scale)
            recon_losses.append(float(metrics.get('recon_loss', 0.0)) * scale)
        
        return {
            'total_loss': sum(total_losses) / len(total_losses),
            'flow_loss': sum(flow_losses) / len(flow_losses),
            'recon_loss': sum(recon_losses) / len(recon_losses)
        }
    
    def train(
        self,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        num_epochs: int,
        batch_size: int = 256,
        validation_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        dropout_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train the model."""
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
        
        for epoch in range(num_epochs):
            use_dropout = epoch < dropout_epochs
            metrics = self.train_epoch(x_data, y_data, batch_size, use_dropout)
            
            history['train_losses'].append(metrics['total_loss'])
            history['train_flow_losses'].append(metrics['flow_loss'])
            history['train_recon_losses'].append(metrics['recon_loss'])
            
            if validation_data is not None:
                val_metrics = self.evaluate(validation_data[0], validation_data[1], batch_size)
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_flow_losses'].append(val_metrics['flow_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
        
        # Store predictions and data for plotting
        import numpy as np
        num_viz = min(2000, x_data.shape[0])
        history['train_pred'] = np.array(self.predict(x_data[:num_viz]))
        history['train_x'] = np.array(x_data[:num_viz])
        history['train_y'] = np.array(y_data[:num_viz])
        
        if validation_data is not None:
            num_val_viz = min(1000, validation_data[0].shape[0])
            history['val_pred'] = np.array(self.predict(validation_data[0][:num_val_viz]))
            history['val_x'] = np.array(validation_data[0][:num_val_viz])
            history['val_y'] = np.array(validation_data[1][:num_val_viz])
        
        return history
    
    def evaluate(
        self,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        batch_size: int = 256
    ) -> Dict[str, float]:
        """Evaluate the model."""
        if self.params is None:
            raise ValueError("Model not initialized.")
        
        x_data = jnp.asarray(x_data)
        y_data = jnp.asarray(y_data)
        
        num_samples = y_data.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        total_losses = []
        flow_losses = []
        recon_losses = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            x_batch = x_data[start_idx:end_idx]
            y_batch = y_data[start_idx:end_idx]
            
            # Pad if needed
            if end_idx - start_idx < batch_size:
                pad_size = batch_size - (end_idx - start_idx)
                x_batch = jnp.concatenate([x_batch, jnp.zeros((pad_size, *x_batch.shape[1:]))])
                y_batch = jnp.concatenate([y_batch, jnp.zeros((pad_size, *y_batch.shape[1:]))])
            
            self.rng, eval_rng = jr.split(self.rng)
            loss, metrics = self.model.loss(self.params, x_batch, y_batch, eval_rng, training=False)
            
            # Scale loss by actual batch size
            actual_batch_size = end_idx - start_idx
            scale = batch_size / actual_batch_size if actual_batch_size < batch_size else 1.0
            
            total_losses.append(float(loss) * scale)
            flow_losses.append(float(metrics.get('flow_loss', 0.0)) * scale)
            recon_losses.append(float(metrics.get('recon_loss', 0.0)) * scale)
        
        return {
            'total_loss': sum(total_losses) / len(total_losses),
            'flow_loss': sum(flow_losses) / len(flow_losses),
            'recon_loss': sum(recon_losses) / len(recon_losses)
        }
    
    def predict(self, x_data: jnp.ndarray, num_steps: int = 20) -> jnp.ndarray:
        """Make predictions."""
        if self.params is None:
            raise ValueError("Model not initialized.")
        return self.model.predict(self.params, x_data, num_steps, "midpoint", "end_point")
    
    def save_params(self, filepath: str):
        """Save model parameters."""
        import pickle
        if self.params is None:
            raise ValueError("Model not initialized. No parameters to save.")
        with open(filepath, 'wb') as f:
            pickle.dump(jax.device_get(self.params), f)
    
    def save_results(self, history: Dict[str, Any], output_dir: str):
        """Save results and create plots."""
        import os
        import pickle
        from pathlib import Path
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save history
        with open(f"{output_dir}/history.pkl", 'wb') as f:
            pickle.dump(history, f)
        
        # Save config
        if hasattr(self.config, 'save_yaml'):
            self.config.save_yaml(f"{output_dir}/config.yaml")
        
        # Save params
        self.save_params(f"{output_dir}/params.pkl")
        
        # Create plots
        try:
            from src.utils.plotting.plot_regression import create_all_regression_plots
            create_all_regression_plots(history, self.model, self.params, output_dir)
        except ImportError:
            pass
