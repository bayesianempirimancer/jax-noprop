"""
Minimal, professional JAX trainer for generation tasks with VAE_flow_mix model.

This trainer provides a clean, JAX-compliant interface for training
VAE_flow_mix models on conditional and unconditional generation tasks.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import numpy as np
from typing import Dict, Any, Tuple, Optional
from functools import partial
from flax.core import unfreeze, freeze

from src.flow_models.fm_mix import VAE_flow_mix
from src.flow_models.config_mix import Config


class GenerationMixTrainer:
    """Minimal trainer for conditional/unconditional generation tasks with VAE_flow_mix."""
    
    def __init__(
        self,
        config,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        seed: int = 42,
        unconditional: bool = False,
        warmup_steps: int = 0,
        update_gmm: bool = True,
        gmm_lr: float = 0.2,
        gmm_N_eff: float = 2000.0
    ):
        self.config = config
        self.learning_rate = learning_rate
        self.unconditional = unconditional
        self.seed = seed
        self.update_gmm = update_gmm
        self.gmm_lr = gmm_lr
        self.gmm_N_eff = gmm_N_eff
        
        # Initialize model
        self.model = VAE_flow_mix(config=config)
        
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
        
        if optimizer_name.lower() == "adamw":
            self.optimizer = optax.adamw(lr_schedule)
        elif optimizer_name.lower() == "adam":
            self.optimizer = optax.adam(lr_schedule)
        else:
            self.optimizer = optax.sgd(lr_schedule)
        
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
        gmm_losses = []
        
        # Check if we should update GMM
        sample_method = self.config.flow_planner.get('sample_method', 'mixture')
        update_gmm_epoch = (sample_method == "mixture") and self.update_gmm
        
        for batch in dataloader:
            if x_data is not None:
                # Conditional: batch is (x_batch, y_batch) tuple
                x_batch, y_batch = batch
            else:
                # Unconditional: batch is just y_data (single array)
                y_batch = batch[0] if isinstance(batch, (tuple, list)) and len(batch) == 1 else batch
                x_batch = None
            
            # Update GMM parameters using VBEM (if requested) - done outside JIT context
            if update_gmm_epoch:
                # Encode y to get z_target for GMM update
                self.rng, encode_key = jr.split(self.rng)
                mu_z_target, logvar_z_target = self.model.apply(
                    self.params, y_batch, method='encode', training=False, rngs={'dropout': encode_key}
                )
                # Use mean for GMM update (or could sample, but mean is more stable)
                z_target = mu_z_target
                
                # Flatten z_target for GMM update
                z_target_flat = z_target.reshape(-1, self.model.z_dim)
                
                # Compute updated GMM parameters (returns dict, doesn't modify params)
                updated_gmm_params = self.model.apply(
                    self.params,
                    z_target_flat,
                    method='update_gmm_params',
                    N_eff=self.gmm_N_eff,
                    lr=self.gmm_lr,
                    training=use_dropout
                )
                
                # Apply GMM parameter updates to params structure (outside JIT context)
                params_unfrozen = unfreeze(self.params)
                params_unfrozen['params']['flow_planner']['gmm'] = freeze(updated_gmm_params)
                self.params = freeze(params_unfrozen)
            
            # Training step: Use model's train_step method from fm_mix.py
            self.rng, step_rng = jr.split(self.rng)
            x_input = None if (self.unconditional or x_batch is None) else x_batch
            self.params, self.opt_state, loss, metrics, _ = self.model.train_step(
                self.params, x_input, y_batch, self.opt_state, self.optimizer, step_rng,
                training=use_dropout, update_gmm=False,  # GMM already updated above
                gmm_lr=self.gmm_lr, N_eff=self.gmm_N_eff
            )
            
            total_losses.append(float(loss))
            flow_losses.append(float(metrics.get('flow_loss', 0.0)))
            recon_losses.append(float(metrics.get('recon_loss', 0.0)))
            reg_losses.append(float(metrics.get('reg_loss', 0.0)))
            vae_losses.append(float(metrics.get('vae_loss', 0.0)))
            gmm_losses.append(float(metrics.get('gmm_loss', 0.0)))
        
        num_batches = len(total_losses)
        return {
            'total_loss': sum(total_losses) / num_batches if num_batches > 0 else 0.0,
            'flow_loss': sum(flow_losses) / num_batches if num_batches > 0 else 0.0,
            'recon_loss': sum(recon_losses) / num_batches if num_batches > 0 else 0.0,
            'reg_loss': sum(reg_losses) / num_batches if num_batches > 0 else 0.0,
            'vae_loss': sum(vae_losses) / num_batches if num_batches > 0 else 0.0,
            'gmm_loss': sum(gmm_losses) / num_batches if num_batches > 0 else 0.0
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
            'train_gmm_losses': [],
            'val_losses': [],
            'val_flow_losses': [],
            'val_recon_losses': [],
            'val_reg_losses': [],
            'val_vae_losses': [],
            'val_gmm_losses': [],
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
            history['train_gmm_losses'].append(metrics.get('gmm_loss', 0.0))
            
            if validation_data is not None:
                vx, vy = validation_data
                val_metrics = self.evaluate(vx, vy, batch_size)
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_flow_losses'].append(val_metrics['flow_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
                history['val_reg_losses'].append(val_metrics['reg_loss'])
                history['val_vae_losses'].append(val_metrics.get('vae_loss', 0.0))
                history['val_gmm_losses'].append(val_metrics.get('gmm_loss', 0.0))
                
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
        gmm_losses = []
        
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
            gmm_losses.append(float(metrics.get('gmm_loss', 0.0)))
        
        num_batches = len(total_losses)
        return {
            'total_loss': sum(total_losses) / num_batches if num_batches > 0 else 0.0,
            'flow_loss': sum(flow_losses) / num_batches if num_batches > 0 else 0.0,
            'recon_loss': sum(recon_losses) / num_batches if num_batches > 0 else 0.0,
            'reg_loss': sum(reg_losses) / num_batches if num_batches > 0 else 0.0,
            'vae_loss': sum(vae_losses) / num_batches if num_batches > 0 else 0.0,
            'gmm_loss': sum(gmm_losses) / num_batches if num_batches > 0 else 0.0
        }
    
    def conditional_generate(self, cond_y: jnp.ndarray, num_steps: int = 20, integration_method: str = 'euler', prng_key: Optional[jr.PRNGKey] = None) -> jnp.ndarray:
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
        batch_shape = tuple(int(x) for x in batch_shape) if isinstance(batch_shape, (list, tuple)) else batch_shape
        return self.model.sample(self.params, prng_key, batch_shape, num_steps, "euler", "end_point")
    
    def _create_gmm_fit_plot(self, output_dir: str, y_data: Optional[jnp.ndarray] = None, max_samples: int = 2000):
        """Create a visualization of the GMM fit in latent space.
        
        Args:
            output_dir: Directory to save the plot
            y_data: Training data to encode and visualize (optional)
            max_samples: Maximum number of samples to visualize
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        try:
            # Get GMM parameters
            gmm_params = self.params['params']['flow_planner']['gmm']
            cluster_means = np.array(gmm_params['mu_n'])  # [num_clusters, latent_dim]
            num_clusters = cluster_means.shape[0]
            latent_dim = cluster_means.shape[1]
            
            # Only plot if latent_dim is 2D (for 2D scatter plot)
            if latent_dim != 2:
                print(f"Skipping GMM fit plot: latent_dim={latent_dim} (only 2D supported)")
                return
            
            # For visualization, we want to show the GMM fit on the actual data points
            # So we'll encode the data to get latent representations for cluster assignment,
            # but plot the actual data points (y_data) colored by their cluster assignments
            if y_data is not None:
                # Sample a subset for visualization
                n_viz = min(max_samples, y_data.shape[0])
                if n_viz < y_data.shape[0]:
                    self.rng, sample_key = jr.split(self.rng)
                    indices = jr.choice(sample_key, y_data.shape[0], shape=(n_viz,), replace=False)
                    y_viz = y_data[indices]
                else:
                    y_viz = y_data
                
                # Encode to latent space to get cluster assignments
                self.rng, encode_key = jr.split(self.rng)
                mu_z_target, _ = self.model.apply(
                    self.params, y_viz, method='encode', training=False, rngs={'dropout': encode_key}
                )
                z_target = np.array(mu_z_target)  # [n_viz, latent_dim]
                
                # Use actual data points for plotting (y_viz), not encoded representations
                data_points = np.array(y_viz)  # [n_viz, data_dim]
            else:
                # If no data provided, just plot cluster means
                z_target = None
                data_points = None
            
            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot data points if available
            if z_target is not None and data_points is not None:
                # Get cluster assignments for coloring (using latent space)
                z_target_jax = jnp.array(z_target)
                log_p_tilde = self.model.flow_planner.gmm.apply(
                    freeze({'params': gmm_params}),
                    z_target_jax,
                    training=False,
                    method='log_p_tilde'
                )
                assignments = np.argmax(np.array(log_p_tilde), axis=1)
                
                # Plot actual data points (y_viz) with colors based on cluster assignments
                colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))
                for k in range(num_clusters):
                    mask = assignments == k
                    if np.any(mask):
                        ax.scatter(
                            data_points[mask, 0], 
                            data_points[mask, 1], 
                            alpha=0.4, 
                            s=15, 
                            c=[colors[k]], 
                            label=f'Cluster {k}' if k < 10 else None
                        )
            
            # Decode cluster means back to data space for visualization
            cluster_means_jax = jnp.array(cluster_means)  # [num_clusters, latent_dim]
            # Decode cluster means to data space
            decoded_means = self.model.apply(
                self.params,
                cluster_means_jax,
                method='decode',
                training=False
            )
            decoded_means_np = np.array(decoded_means)  # [num_clusters, data_dim]
            
            # Plot decoded cluster means
            ax.scatter(
                decoded_means_np[:, 0], 
                decoded_means_np[:, 1], 
                s=300, 
                c='red', 
                marker='x', 
                linewidths=4,
                label='GMM Cluster Means (decoded)',
                zorder=10
            )
            
            ax.set_xlabel('Data Dimension 1', fontsize=12)
            ax.set_ylabel('Data Dimension 2', fontsize=12)
            ax.set_title('GMM Fit on Data (cluster means decoded from latent space)', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            save_path = Path(output_dir) / "gmm_fit.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved GMM fit visualization to {save_path}")
        except Exception as e:
            import traceback
            print(f"Warning: Error creating GMM fit plot: {e}")
            traceback.print_exc()
    
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
            create_loss_trends_plot(history, "flow_matching", output_dir)
            
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
                model_type="flow_matching",
                unconditional=self.unconditional,
                output_dir=output_dir,
                cond_y=cond_y,
                num_trajectories=20,
                num_steps=20,
                prng_key=traj_rng,
                rng=self.rng
            )
            
            # GMM fit visualization (if using mixture sampling)
            sample_method = self.config.flow_planner.get('sample_method', 'mixture')
            if sample_method == "mixture":
                # Use x_real (validation data) for visualization, or y_labels if available
                viz_data = x_real if x_real is not None else (y_labels if y_labels is not None else None)
                if viz_data is not None:
                    self._create_gmm_fit_plot(output_dir, y_data=jnp.array(viz_data))
        except ImportError:
            pass
        except Exception as e:
            import traceback
            print(f"Warning: Error creating plots: {e}")
            traceback.print_exc()


def _create_gmm_fit_plot_standalone(model: VAE_flow_mix, params: dict, output_dir: str, y_data: Optional[jnp.ndarray] = None, max_samples: int = 2000):
    """Standalone version of GMM fit plot for use in save_results_fm_mix.
    
    Args:
        model: VAE_flow_mix model instance
        params: Model parameters
        output_dir: Directory to save the plot
        y_data: Training data to encode and visualize (optional)
        max_samples: Maximum number of samples to visualize
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from pathlib import Path
    from flax.core import freeze
    
    try:
        # Get GMM parameters
        gmm_params = params['params']['flow_planner']['gmm']
        cluster_means = np.array(gmm_params['mu_n'])  # [num_clusters, latent_dim]
        num_clusters = cluster_means.shape[0]
        latent_dim = cluster_means.shape[1]
        
        # Only plot if latent_dim is 2D (for 2D scatter plot)
        if latent_dim != 2:
            print(f"Skipping GMM fit plot: latent_dim={latent_dim} (only 2D supported)")
            return
        
        # For visualization, we want to show the GMM fit on the actual data points
        # So we'll encode the data to get latent representations for cluster assignment,
        # but plot the actual data points (y_data) colored by their cluster assignments
        if y_data is not None:
            # Sample a subset for visualization
            n_viz = min(max_samples, y_data.shape[0])
            key = jr.PRNGKey(42)  # Use fixed key for reproducibility
            if n_viz < y_data.shape[0]:
                key, sample_key = jr.split(key)
                indices = jr.choice(sample_key, y_data.shape[0], shape=(n_viz,), replace=False)
                y_viz = y_data[indices]
            else:
                y_viz = y_data
                indices = None
            
            # Encode to latent space to get cluster assignments
            key, encode_key = jr.split(key)
            mu_z_target, _ = model.apply(
                params, y_viz, method='encode', training=False, rngs={'dropout': encode_key}
            )
            z_target = np.array(mu_z_target)  # [n_viz, latent_dim]
            
            # Use actual data points for plotting (y_viz), not encoded representations
            data_points = np.array(y_viz)  # [n_viz, data_dim]
        else:
            # If no data provided, just plot cluster means
            z_target = None
            data_points = None
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot data points if available
        if z_target is not None and data_points is not None:
            # Get cluster assignments for coloring (using latent space)
            z_target_jax = jnp.array(z_target)
            # Create a temporary GMM instance to call log_p_tilde
            from src.vae.vb_gmm import create_gmm_vbem
            gmm_config = model.config.flow_planner.get('gmm', {})
            from src.vae.vb_gmm import GMMVBEMConfig
            gmm_vbem_config = GMMVBEMConfig(
                num_clusters=gmm_config.get('num_clusters', 8),
                latent_dim=latent_dim
            )
            gmm_temp = create_gmm_vbem(gmm_vbem_config)
            log_p_tilde = gmm_temp.apply(
                freeze({'params': gmm_params}),
                z_target_jax,
                training=False,
                method='log_p_tilde'
            )
            assignments = np.argmax(np.array(log_p_tilde), axis=1)
            
            # Plot actual data points (y_viz) with colors based on cluster assignments
            colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))
            for k in range(num_clusters):
                mask = assignments == k
                if np.any(mask):
                    ax.scatter(
                        data_points[mask, 0], 
                        data_points[mask, 1], 
                        alpha=0.4, 
                        s=15, 
                        c=[colors[k]], 
                        label=f'Cluster {k}' if k < 10 else None
                    )
        
        # Decode cluster means back to data space for visualization
        cluster_means_jax = jnp.array(cluster_means)  # [num_clusters, latent_dim]
        # Decode cluster means to data space
        decoded_means = model.apply(
            params,
            cluster_means_jax,
            method='decode',
            training=False
        )
        decoded_means_np = np.array(decoded_means)  # [num_clusters, data_dim]
        
        # Plot decoded cluster means
        ax.scatter(
            decoded_means_np[:, 0], 
            decoded_means_np[:, 1], 
            s=300, 
            c='red', 
            marker='x', 
            linewidths=4,
            label='GMM Cluster Means (decoded)',
            zorder=10
        )
        
        ax.set_xlabel('Data Dimension 1', fontsize=12)
        ax.set_ylabel('Data Dimension 2', fontsize=12)
        ax.set_title('GMM Fit on Data (cluster means decoded from latent space)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        save_path = Path(output_dir) / "gmm_fit.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved GMM fit visualization to {save_path}")
    except Exception as e:
        import traceback
        print(f"Warning: Error creating GMM fit plot: {e}")
        traceback.print_exc()


# Keep the standalone functions for backward compatibility
def train_epoch_fm_mix(
    model: VAE_flow_mix,
    params: dict,
    x_data: Optional[jnp.ndarray],
    y_data: jnp.ndarray,
    opt_state: dict,
    optimizer,
    key: jr.PRNGKey,
    batch_size: int = 256,
    training: bool = True,
    x_mask: Optional[jnp.ndarray] = None,
    update_gmm: bool = True,
    gmm_lr: float = 0.2,
    N_eff: float = 2000.0
) -> Tuple[dict, dict, float, dict]:
    """
    Train for one epoch using the model's train_step method.
    
    This function handles both GMM parameter updates (via VBEM) and flow model
    parameter updates (via gradient descent). GMM updates are performed outside
    of the JIT-compiled train_step to avoid JAX tracing issues with lambda functions.
    
    Args:
        model: VAE_flow_mix model instance
        params: Current model parameters
        x_data: Input data [num_samples, input_dim] or [num_samples, seq_len, embed_dim]
        y_data: Target data [num_samples, output_dim] or [num_samples, seq_len, embed_dim]
        opt_state: Optimizer state
        optimizer: Optax optimizer
        key: Random key
        batch_size: Batch size for training
        training: Whether in training mode
        x_mask: Boolean mask [num_samples, x_seq_len] for sequence models
        update_gmm: Whether to update GMM parameters in each step
        gmm_lr: Learning rate for GMM VBEM updates
        N_eff: Effective number of data points for GMM updates
        
    Returns:
        params: Updated model parameters
        opt_state: Updated optimizer state
        avg_loss: Average training loss for the epoch
        metrics: Dictionary of average metrics
    """
    num_samples = y_data.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Shuffle data
    key, shuffle_key = jr.split(key)
    indices = jr.permutation(shuffle_key, num_samples)
    x_shuffled = x_data[indices] if x_data is not None else None
    y_shuffled = y_data[indices]
    mask_shuffled = x_mask[indices] if x_mask is not None else None
    
    losses = []
    all_metrics = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        x_batch = x_shuffled[start_idx:end_idx] if x_shuffled is not None else None
        y_batch = y_shuffled[start_idx:end_idx]
        mask_batch = mask_shuffled[start_idx:end_idx] if mask_shuffled is not None else None
        
        # Update GMM parameters using VBEM (if requested) - done outside JIT context
        # This computes updated GMM params dict without modifying params structure
        # The updated params are then applied to the params structure after train_step
        if update_gmm:
            # Encode y to get z_target for GMM update
            key, encode_key = jr.split(key)
            mu_z_target, logvar_z_target = model.apply(
                params, y_batch, method='encode', training=False, rngs={'dropout': encode_key}
            )
            # Use mean for GMM update (or could sample, but mean is more stable)
            z_target = mu_z_target
            
            # Flatten z_target for GMM update
            z_target_flat = z_target.reshape(-1, model.z_dim)
            
            # Compute updated GMM parameters (returns dict, doesn't modify params)
            updated_gmm_params = model.apply(
                params,
                z_target_flat,
                method='update_gmm_params',
                N_eff=N_eff,
                lr=gmm_lr,
                training=training
            )
            
            # Apply GMM parameter updates to params structure (outside JIT context)
            params_unfrozen = unfreeze(params)
            params_unfrozen['params']['flow_planner']['gmm'] = freeze(updated_gmm_params)
            params = freeze(params_unfrozen)
        
        # Training step: Use model's train_step method from fm_mix.py
        # This method handles flow model parameter updates via gradient descent
        # GMM params are excluded from gradients via stop_gradient in extract_params
        key, step_key = jr.split(key)
        params, opt_state, loss, metrics, _ = model.train_step(
            params, x_batch, y_batch, opt_state, optimizer, step_key,
            training=training, x_mask=mask_batch, update_gmm=False,
            gmm_lr=gmm_lr, N_eff=N_eff
        )
        
        losses.append(float(loss))
        all_metrics.append(metrics)
    
    # Compute average metrics
    avg_loss = np.mean(losses)
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([float(m.get(key, 0.0)) for m in all_metrics])
    
    return params, opt_state, avg_loss, avg_metrics


def save_results_fm_mix(
    model: VAE_flow_mix,
    params: dict,
    history: Dict[str, Any],
    output_dir: str,
    x_real: Optional[jnp.ndarray] = None,
    x_gen: Optional[jnp.ndarray] = None,
    y_labels: Optional[jnp.ndarray] = None,
    x_real_labels: Optional[jnp.ndarray] = None,
    key: Optional[jr.PRNGKey] = None
):
    """Save results and create plots for VAE_flow_mix model.
    
    Args:
        model: VAE_flow_mix model instance
        params: Model parameters
        history: Training history dictionary
        output_dir: Directory to save results
        x_real: Real samples for generation plot [optional]
        x_gen: Generated samples for generation plot [optional]
        y_labels: Labels for conditional generation plot [optional]
        key: Random key for trajectory plots [optional]
    """
    import pickle
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save history
    with open(f"{output_dir}/history.pkl", 'wb') as f:
        pickle.dump(history, f)
    
    # Save params
    import jax
    with open(f"{output_dir}/params.pkl", 'wb') as f:
        pickle.dump(jax.device_get(params), f)
    
    # Create plots
    try:
        from src.utils.plotting.plot_generation import (
            create_generation_plot,
            create_loss_trends_plot,
            create_latent_trajectories_plot
        )
        
        # Loss trends plot
        create_loss_trends_plot(history, "flow_matching", output_dir)
        
        # Generation plot (if data provided)
        if x_gen is not None and x_real is not None:
            # For conditional generation:
            # - x_real: real data points (should be colored by their actual labels)
            # - x_gen: generated data (should be colored by the condition labels used for generation)
            # - y_labels: condition labels used for generation (should match x_gen)
            # - x_real_labels: actual labels of x_real (should be passed separately if different from y_labels)
            # For now, assume y_labels are the actual labels of x_real (they should be aligned)
            # Use x_real_labels if provided, otherwise fall back to y_labels
            # x_real_labels should be the actual labels of x_real
            # y_labels should be the condition labels used for generation (for x_gen)
            real_labels = np.array(x_real_labels) if x_real_labels is not None else (np.array(y_labels) if y_labels is not None else None)
            create_generation_plot(
                np.array(x_real), 
                np.array(y_labels) if y_labels is not None else None,  # Labels for generated data (condition)
                np.array(x_gen), 
                output_dir, 
                unconditional=False,  # fm_mix is typically conditional
                x_real_labels=real_labels  # Labels for real data
            )
        
        # Latent trajectories plot
        if key is None:
            key = jr.PRNGKey(42)
        key, traj_key = jr.split(key)
        cond_y = y_labels[:20] if y_labels is not None and y_labels.shape[0] >= 20 else y_labels
        create_latent_trajectories_plot(
            model=model,
            params=params,
            model_type="flow_matching",
            unconditional=False,
            output_dir=output_dir,
            cond_y=cond_y,
            num_trajectories=20,
            num_steps=20,
            prng_key=traj_key,
            rng=key
        )
        
        # GMM fit visualization (if using mixture sampling)
        # Check config for sample_method (fallback to checking model if config not available)
        sample_method = None
        if hasattr(model, 'config') and hasattr(model.config, 'flow_planner'):
            sample_method = model.config.flow_planner.get('sample_method', 'mixture')
        elif hasattr(model, 'flow_planner'):
            sample_method = model.flow_planner.sample_method
        
        if sample_method == "mixture":
            # Use x_real (validation data) for visualization, or y_labels if available
            viz_data = x_real if x_real is not None else (y_labels if y_labels is not None else None)
            if viz_data is not None:
                _create_gmm_fit_plot_standalone(model, params, output_dir, y_data=jnp.array(viz_data))
    except ImportError as e:
        print(f"Warning: Could not import plotting utilities: {e}")
    except Exception as e:
        import traceback
        print(f"Warning: Error creating plots: {e}")
        traceback.print_exc()
