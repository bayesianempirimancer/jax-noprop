"""
Minimal, professional JAX trainer for VAE flow models.

This trainer provides a clean, JAX-compliant interface for training
flow models with scan-optimized training loops.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import optax
from typing import Dict, Any, Tuple, Optional
from functools import partial
from flax import traverse_util
from src.flow_models.fm import VAE_flow as FlowMatchingModel
from src.flow_models.df import VAE_flow as DiffusionModel
from src.flow_models.ct import VAE_flow as CTModel
from src.flow_models.config import Config as Config


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
        
        # Create optimizer with warmup
        if warmup_steps > 0:
            lr_schedule = optax.join_schedules(
                [
                    optax.linear_schedule(0.0, learning_rate, warmup_steps),
                    optax.constant_schedule(learning_rate)
                ],
                [warmup_steps]
            )
        else:
            lr_schedule = optax.constant_schedule(learning_rate)
        
        self.optimizer = optax.adam(lr_schedule) if optimizer_name.lower() == "adam" else optax.sgd(lr_schedule)
        
        self.params = None
        self.opt_state = None
        self.rng = jr.PRNGKey(self.seed)
    
    def initialize(self, x_sample: jnp.ndarray, y_sample: jnp.ndarray):
        """Initialize model parameters.
        
        Args:
            x_sample: Sample input matching input_shape or [batch_size, *input_shape]
            y_sample: Sample target matching output_shape or [batch_size, *output_shape]
        """
        # Get expected shapes from config
        input_shape = self.config.main["input_shape"]
        output_shape = self.config.main["output_shape"]
        
        # Check if samples match expected shapes exactly (no batch dimension)
        # If so, add batch dimension
        if x_sample.shape == input_shape:
            x_sample = x_sample[None, ...]
        elif len(x_sample.shape) == len(input_shape) + 1 and x_sample.shape[1:] == input_shape:
            # Already has batch dimension, use first sample
            x_sample = x_sample[0:1]
        elif x_sample.shape != (1,) + input_shape:
            # Try to reshape or use first sample if shape doesn't match
            x_sample = x_sample[None, ...] if x_sample.shape == input_shape else x_sample[0:1]
        
        if y_sample.shape == output_shape:
            y_sample = y_sample[None, ...]
        elif len(y_sample.shape) == len(output_shape) + 1 and y_sample.shape[1:] == output_shape:
            # Already has batch dimension, use first sample
            y_sample = y_sample[0:1]
        elif y_sample.shape != (1,) + output_shape:
            # Try to reshape or use first sample if shape doesn't match
            y_sample = y_sample[None, ...] if y_sample.shape == output_shape else y_sample[0:1]
        
        self.rng, init_rng = jr.split(self.rng)
        self.params = self.model.init(init_rng, x_sample, y_sample, init_rng)
        self.opt_state = self.optimizer.init(self.params)
        print(f"Model initialized with {sum(x.size for x in jt.leaves(self.params))} parameters")
        
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
        
        x_batch = jnp.asarray(x_batch)
        y_batch = jnp.asarray(y_batch)
        
        # Run single training step
        self.rng, step_rng = jr.split(self.rng)
        self.params, self.opt_state, loss, metrics = self.model.train_step(
            self.params, x_batch, y_batch, self.opt_state, self.optimizer, step_rng, training=use_dropout
        )
        
        return {
            'loss': float(loss),
            'flow_loss': float(metrics.get('flow_loss', 0.0)),
            'recon_loss': float(metrics.get('recon_loss', 0.0)),
            'reg_loss': float(metrics.get('reg_loss', 0.0)),
            'vae_loss': float(metrics.get('vae_loss', 0.0)),
            'kl_z0_loss': float(metrics.get('kl_z0_loss', 0.0)),
        }
    
    def train_epoch(
        self,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        batch_size: int = 256,
        use_dropout: bool = True
    ) -> Dict[str, float]:
        """Train for one epoch using jax-dataloader."""
        # Lazy import to avoid JAX initialization conflicts
        import jax_dataloader as jdl
        
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized.")
        
        x_data = jnp.asarray(x_data)
        y_data = jnp.asarray(y_data)
        
        # Create dataset and dataloader
        dataset = jdl.ArrayDataset(x_data, y_data)
        self.rng, shuffle_rng = jr.split(self.rng)
        dataloader = jdl.DataLoader(
            dataset,
            backend='jax',
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            rng_key=shuffle_rng
        )
        
        total_losses = []
        flow_losses = []
        recon_losses = []
        reg_losses = []
        vae_losses = []
        kl_z0_losses = []
        
        for x_batch, y_batch in dataloader:
            metrics = self.train_step(x_batch, y_batch, use_dropout=use_dropout)
            
            total_losses.append(metrics['loss'])
            flow_losses.append(metrics['flow_loss'])
            recon_losses.append(metrics['recon_loss'])
            reg_losses.append(metrics.get('reg_loss', 0.0))
            vae_losses.append(metrics.get('vae_loss', 0.0))
            kl_z0_losses.append(metrics.get('kl_z0_loss', 0.0))
        
        num_batches = len(total_losses)
        return {
            'total_loss': sum(total_losses) / num_batches if num_batches > 0 else 0.0,
            'flow_loss': sum(flow_losses) / num_batches if num_batches > 0 else 0.0,
            'recon_loss': sum(recon_losses) / num_batches if num_batches > 0 else 0.0,
            'reg_loss': sum(reg_losses) / num_batches if num_batches > 0 else 0.0,
            'vae_loss': sum(vae_losses) / num_batches if num_batches > 0 else 0.0,
            'kl_z0_loss': sum(kl_z0_losses) / num_batches if num_batches > 0 else 0.0
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
            'train_reg_losses': [],
            'train_vae_losses': [],
            'train_kl_z0_losses': [],
            'val_losses': [],
            'val_flow_losses': [],
            'val_recon_losses': [],
            'val_reg_losses': [],
            'val_vae_losses': [],
            'val_kl_z0_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'train_pve': [],  # Percent variance explained per epoch
            'val_pve': [],  # Percent variance explained per epoch
            'train_mse': [],  # Mean squared error per epoch
            'val_mse': []  # Mean squared error per epoch
        }
        
        for epoch in range(num_epochs):
            use_dropout = epoch < dropout_epochs
            metrics = self.train_epoch(x_data, y_data, batch_size, use_dropout)
            
            history['train_losses'].append(metrics['total_loss'])
            history['train_flow_losses'].append(metrics['flow_loss'])
            history['train_recon_losses'].append(metrics['recon_loss'])
            history['train_reg_losses'].append(metrics.get('reg_loss', 0.0))
            history['train_vae_losses'].append(metrics.get('vae_loss', 0.0))
            history['train_kl_z0_losses'].append(metrics.get('kl_z0_loss', 0.0))
            
            if validation_data is not None:
                val_metrics = self.evaluate(validation_data[0], validation_data[1], batch_size)
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_flow_losses'].append(val_metrics['flow_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
                history['val_reg_losses'].append(val_metrics.get('reg_loss', 0.0))
                history['val_vae_losses'].append(val_metrics.get('vae_loss', 0.0))
                history['val_kl_z0_losses'].append(val_metrics.get('kl_z0_loss', 0.0))
            
            # Compute PVE and MSE for this epoch (using a small subset for efficiency)
            import numpy as np
            num_viz = min(2000, x_data.shape[0])
            train_pred = np.array(self.predict(x_data[:num_viz]))
            train_y_subset = np.array(y_data[:num_viz])
            
            # Calculate MSE
            mse_train = np.mean((train_y_subset - train_pred) ** 2)
            history['train_mse'].append(float(mse_train))
            
            # Calculate R² = 1 - (SS_res / SS_tot)
            ss_res = np.sum((train_y_subset - train_pred) ** 2)
            ss_tot = np.sum((train_y_subset - np.mean(train_y_subset, axis=0, keepdims=True)) ** 2)
            if ss_tot > 0:
                r2_train = 1 - (ss_res / ss_tot)
                history['train_pve'].append(r2_train * 100)
            else:
                history['train_pve'].append(0.0)
            
            if validation_data is not None:
                num_val_viz = min(1000, validation_data[0].shape[0])
                val_pred = np.array(self.predict(validation_data[0][:num_val_viz]))
                val_y_subset = np.array(validation_data[1][:num_val_viz])
                
                # Calculate MSE
                mse_val = np.mean((val_y_subset - val_pred) ** 2)
                history['val_mse'].append(float(mse_val))
                
                # Calculate R²
                ss_res = np.sum((val_y_subset - val_pred) ** 2)
                ss_tot = np.sum((val_y_subset - np.mean(val_y_subset, axis=0, keepdims=True)) ** 2)
                if ss_tot > 0:
                    r2_val = 1 - (ss_res / ss_tot)
                    history['val_pve'].append(r2_val * 100)
                else:
                    history['val_pve'].append(0.0)
        
        # Store final predictions and data for plotting (backward compatibility)
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
        """Evaluate the model using jax-dataloader."""
        # Lazy import to avoid JAX initialization conflicts
        import jax_dataloader as jdl
        
        if self.params is None:
            raise ValueError("Model not initialized.")
        
        x_data = jnp.asarray(x_data)
        y_data = jnp.asarray(y_data)
        
        # Create dataset and dataloader (no shuffling for evaluation)
        dataset = jdl.ArrayDataset(x_data, y_data)
        dataloader = jdl.DataLoader(
            dataset,
            backend='jax',
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        total_losses = []
        flow_losses = []
        recon_losses = []
        reg_losses = []
        vae_losses = []
        kl_z0_losses = []
        
        for x_batch, y_batch in dataloader:
            self.rng, eval_rng = jr.split(self.rng)
            loss, metrics = self.model.loss(self.params, x_batch, y_batch, eval_rng, training=False)
            
            total_losses.append(float(loss))
            flow_losses.append(float(metrics.get('flow_loss', 0.0)))
            recon_losses.append(float(metrics.get('recon_loss', 0.0)))
            reg_losses.append(float(metrics.get('reg_loss', 0.0)))
            vae_losses.append(float(metrics.get('vae_loss', 0.0)))
            kl_z0_losses.append(float(metrics.get('kl_z0_loss', 0.0)))
        
        num_batches = len(total_losses)
        return {
            'total_loss': sum(total_losses) / num_batches if num_batches > 0 else 0.0,
            'flow_loss': sum(flow_losses) / num_batches if num_batches > 0 else 0.0,
            'recon_loss': sum(recon_losses) / num_batches if num_batches > 0 else 0.0,
            'reg_loss': sum(reg_losses) / num_batches if num_batches > 0 else 0.0,
            'vae_loss': sum(vae_losses) / num_batches if num_batches > 0 else 0.0,
            'kl_z0_loss': sum(kl_z0_losses) / num_batches if num_batches > 0 else 0.0
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
            create_all_regression_plots(history, self.model, self.params, output_dir, model_type=self.model_type)
        except ImportError:
            pass
