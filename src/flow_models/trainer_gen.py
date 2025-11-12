"""
Minimal, professional JAX trainer for generation tasks.

This trainer provides a clean, JAX-compliant interface for training
flow models on conditional and unconditional generation tasks.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from typing import Dict, Any, Tuple, Optional
from functools import partial
from src.flow_models.fm import VAE_flow as FlowMatchingModel
from src.flow_models.df import VAE_flow as DiffusionModel
from src.flow_models.ct import VAE_flow as CTModel
from src.flow_models.fm_mix import VAE_flow_mix as FMMixModel
from src.flow_models.config import Config


class GenerationTrainer:
    """Minimal trainer for conditional/unconditional generation tasks."""
    
    def __init__(
        self,
        config,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        seed: int = 42,
        unconditional: bool = False,
        warmup_steps: int = 0,
        model_type: str = "flow_matching"
    ):
        self.config = config
        self.learning_rate = learning_rate
        self.unconditional = unconditional
        self.seed = seed
        self.model_type = model_type
        
        # Initialize model
        if model_type == "diffusion":
            self.model = DiffusionModel(config=config)
        elif model_type == "flow_matching":
            self.model = FlowMatchingModel(config=config)
        elif model_type == "ct":
            self.model = CTModel(config=config)
        elif model_type == "fm_mix":
            self.model = FMMixModel(config=config)
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
        
        self.optimizer = optax.adamw(lr_schedule) if optimizer_name.lower() == "adamw" else optax.sgd(lr_schedule)
        
        # State
        self.params = None
        self.opt_state = None
        self.rng = jr.PRNGKey(seed)
    
    def initialize(self, x_sample: Optional[jnp.ndarray], y_sample: jnp.ndarray):
        """Initialize model parameters.
        
        Args:
            x_sample: Sample input [input_dim] or [batch_size, input_dim] or None
            y_sample: Sample target [output_dim] or [batch_size, output_dim]
        """
        # Ensure we have batches with batch_size=1
        if x_sample is not None:
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
    
    # JIT removed to avoid nested JIT compilation issues with train_step
    def train_epoch(
        self,
        x_data: Optional[jnp.ndarray],
        y_data: jnp.ndarray,
        batch_size: int = 256,
        use_dropout: bool = True
    ) -> Dict[str, float]:
        """Train for one epoch using jax-dataloader."""
        # Lazy import to avoid JAX initialization conflicts
        import jax_dataloader as jdl
        
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        y_data = jnp.asarray(y_data)
        x_data = jnp.asarray(x_data) if x_data is not None else None
        
        # Create dataset and dataloader
        # For unconditional generation, x_data is None, so we only use y_data
        if x_data is not None:
            dataset = jdl.ArrayDataset(x_data, y_data)
        else:
            # For unconditional, create dataset with just y_data (x will be None in batches)
            dataset = jdl.ArrayDataset(y_data)
        
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
        
        for batch in dataloader:
            if x_data is not None:
                # Conditional: batch is (x_batch, y_batch) tuple
                x_batch, y_batch = batch
            else:
                # Unconditional: batch is just y_data (single array)
                y_batch = batch[0] if isinstance(batch, (tuple, list)) and len(batch) == 1 else batch
                x_batch = None
            
            self.rng, step_rng = jr.split(self.rng)
            x_input = None if (self.unconditional or x_batch is None) else x_batch
            self.params, self.opt_state, loss, metrics = self.model.train_step(
                self.params, x_input, y_batch, self.opt_state, self.optimizer, step_rng, training=use_dropout
            )
            
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
    
    def train(
        self,
        x_data: Optional[jnp.ndarray],
        y_data: jnp.ndarray,
        num_epochs: int,
        batch_size: int = 256,
        validation_data: Optional[Tuple[Optional[jnp.ndarray], jnp.ndarray]] = None,
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
            'val_chamfer_distances': []
        }
        
        for epoch in range(num_epochs):
            use_dropout = epoch < dropout_epochs
            metrics = self.train_epoch(x_data, y_data, batch_size, use_dropout)
            
            history['train_losses'].append(metrics['total_loss'])
            history['train_flow_losses'].append(metrics['flow_loss'])
            history['train_recon_losses'].append(metrics['recon_loss'])
            history['train_reg_losses'].append(metrics['reg_loss'])
            history['train_vae_losses'].append(metrics.get('vae_loss', 0.0))
            history['train_kl_z0_losses'].append(metrics.get('kl_z0_loss', 0.0))
            
            if validation_data is not None:
                vx, vy = validation_data
                val_metrics = self.evaluate(vx, vy, batch_size)
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_flow_losses'].append(val_metrics['flow_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
                history['val_reg_losses'].append(val_metrics['reg_loss'])
                history['val_vae_losses'].append(val_metrics.get('vae_loss', 0.0))
                history['val_kl_z0_losses'].append(val_metrics.get('kl_z0_loss', 0.0))
                
                # Compute Chamfer distance
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    num_eval = min(1000, vy.shape[0])
                    self.rng, gen_rng = jr.split(self.rng)
                    if self.unconditional:
                        x_gen = self.unconditional_generate((num_eval,), 20, gen_rng)
                    else:
                        cond = vx[:num_eval] if vx is not None else None
                        if cond is not None:
                            x_gen = self.conditional_generate(cond, num_steps=20, prng_key=gen_rng)
                        else: 
                            x_gen = self.unconditional_generate((num_eval,), num_steps=20, prng_key=gen_rng)
                    
                    from src.utils.metrics import chamfer_distance
                    chamfer_dist = chamfer_distance(x_gen, vy[:num_eval])
                    history['val_chamfer_distances'].append(chamfer_dist)
        
        return history
    
    def evaluate(
        self,
        x_data: Optional[jnp.ndarray],
        y_data: jnp.ndarray,
        batch_size: int = 256
    ) -> Dict[str, float]:
        """Evaluate the model using jax-dataloader."""
        # Lazy import to avoid JAX initialization conflicts
        import jax_dataloader as jdl
        
        if self.params is None:
            raise ValueError("Model not initialized.")
        
        y_data = jnp.asarray(y_data)
        x_data = jnp.asarray(x_data) if x_data is not None else None
        
        # Create dataset and dataloader (no shuffling for evaluation)
        if x_data is not None:
            dataset = jdl.ArrayDataset(x_data, y_data)
        else:
            dataset = jdl.ArrayDataset(y_data)
        
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
        
        for batch in dataloader:
            if x_data is not None:
                # Conditional: batch is (x_batch, y_batch) tuple
                x_batch, y_batch = batch
            else:
                # Unconditional: batch is just y_data (single array)
                y_batch = batch[0] if isinstance(batch, (tuple, list)) and len(batch) == 1 else batch
                x_batch = None
            
            self.rng, eval_rng = jr.split(self.rng)
            x_input = None if (self.unconditional or x_batch is None) else x_batch
            loss, metrics = self.model.loss(self.params, x_input, y_batch, eval_rng, training=False)
            
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
    
    def conditional_generate(self, cond_y: jnp.ndarray, num_steps: int = 20, integration_method: str = 'midpoint', prng_key: Optional[jr.PRNGKey] = None) -> jnp.ndarray:
        """Generate samples conditioned on y."""
        if self.params is None:
            raise ValueError("Model not initialized.")
        if self.unconditional:
            raise ValueError("Use unconditional_generate() for unconditional generation")
        if prng_key is None:
            self.rng, prng_key = jr.split(self.rng)
        return self.model.predict(self.params, cond_y, num_steps, integration_method, "end_point", prng_key=prng_key)
    
    def unconditional_generate(self, batch_shape: Tuple[int, ...], num_steps: int = 20, prng_key: Optional[jr.PRNGKey] = None) -> jnp.ndarray:
        """Generate samples unconditionally."""
        if self.params is None:
            raise ValueError("Model not initialized.")
        if not self.unconditional:
            raise ValueError("Model was trained conditionally. Use conditional_generate() instead")
        if prng_key is None:
            self.rng, prng_key = jr.split(self.rng)
        integration_method = "midpoint" if self.model_type == "ct" else "euler"
        batch_shape = tuple(int(x) for x in batch_shape) if isinstance(batch_shape, (list, tuple)) else batch_shape
        return self.model.sample(self.params, prng_key, batch_shape, num_steps, integration_method, "end_point")
    
    def save_params(self, filepath: str):
        """Save model parameters."""
        import pickle
        from pathlib import Path
        if self.params is None:
            raise ValueError("Model not initialized. No parameters to save.")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(jax.device_get(self.params), f)
    
    def save_results(self, history: Dict[str, Any], output_dir: str, x_real: Optional[jnp.ndarray] = None, 
                     x_gen: Optional[jnp.ndarray] = None, y_labels: Optional[jnp.ndarray] = None):
        """Save results and create plots.
        
        Args:
            history: Training history dictionary
            output_dir: Directory to save results
            x_real: Real samples for generation plot [optional]
            x_gen: Generated samples for generation plot [optional]
            y_labels: Labels for conditional generation plot [optional]
        """
        import os
        import pickle
        import numpy as np
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
            from src.utils.plotting.plot_generation import (
                create_generation_plot,
                create_loss_trends_plot,
                create_latent_trajectories_plot
            )
            
            # Loss trends plot
            create_loss_trends_plot(history, self.model_type, output_dir)
            
            # Generation plot (if data provided)
            if x_gen is not None and x_real is not None:
                create_generation_plot(
                    np.array(x_real), 
                    np.array(y_labels) if y_labels is not None else None, 
                    np.array(x_gen), 
                    output_dir, 
                    self.unconditional
                )
            
            # Latent trajectories plot
            self.rng, traj_rng = jr.split(self.rng)
            cond_y = None if self.unconditional else (y_labels[:20] if y_labels is not None else None)
            create_latent_trajectories_plot(
                model=self.model,
                params=self.params,
                model_type=self.model_type,
                unconditional=self.unconditional,
                output_dir=output_dir,
                cond_y=cond_y,
                num_trajectories=20,
                num_steps=20,
                prng_key=traj_rng,
                rng=self.rng
            )
        except ImportError:
            pass
        except Exception as e:
            import traceback
            print(f"Warning: Error creating plots: {e}")
            traceback.print_exc()
