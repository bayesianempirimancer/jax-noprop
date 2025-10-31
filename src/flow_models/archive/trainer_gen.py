"""
Trainer for conditional generation (x | y) on Two Moons.

This trainer focuses on training the selected model with reversed mapping
(inputs=y, targets=x) and evaluating conditional generation by sampling
stochastic trajectories using a provided PRNGKey.
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from src.flow_models.fm import VAE_flow as FlowMatchingModel, VAEFlowConfig as FlowMatchingConfig
from src.flow_models.df import VAE_flow as DiffusionModel, VAEFlowConfig as DiffusionConfig
from src.flow_models.ct import VAE_flow as CTModel, VAEFlowConfig as CTConfig


@dataclass
class GenerationTrainer:
    config: Any
    learning_rate: float = 1e-3
    optimizer_name: str = "adam"
    seed: int = 42
    unconditional: bool = False  # If True, use unconditional generation (x=None)

    def __post_init__(self):
        if isinstance(self.config, DiffusionConfig):
            self.model = DiffusionModel(config=self.config)
            self.model_type = "diffusion"
        elif isinstance(self.config, CTConfig):
            self.model = CTModel(config=self.config)
            self.model_type = "ct"
        else:
            self.model = FlowMatchingModel(config=self.config)
            self.model_type = "flow_matching"

        if self.optimizer_name.lower() == "adam":
            self.optimizer = optax.adam(self.learning_rate)
        elif self.optimizer_name.lower() == "sgd":
            self.optimizer = optax.sgd(self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        self.params = None
        self.opt_state = None
        self.rng = jr.PRNGKey(self.seed)

    def initialize(self, x_sample: Optional[jnp.ndarray], y_sample: jnp.ndarray, z_sample: jnp.ndarray, t_sample: jnp.ndarray):
        self.rng, init_rng = jr.split(self.rng)
        # x_sample can be None for unconditional generation
        self.params = self.model.init(init_rng, x_sample, y_sample, init_rng)
        self.opt_state = self.optimizer.init(self.params)

    def train_step(self, x_batch: Optional[jnp.ndarray], y_batch: jnp.ndarray, use_dropout: bool = True) -> Dict[str, float]:
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        self.rng, train_rng = jr.split(self.rng)
        # For unconditional generation, pass None for x_batch
        x_input = None if (self.unconditional or x_batch is None) else x_batch
        self.params, self.opt_state, loss, metrics = self.model.train_step(
            self.params, x_input, y_batch, self.opt_state, self.optimizer, train_rng, training=use_dropout
        )
        return metrics

    def train(
        self,
        x_data: Optional[jnp.ndarray],
        y_data: jnp.ndarray,
        num_epochs: int = 50,
        batch_size: int = 256,
        validation_data: Optional[Tuple[Optional[jnp.ndarray], jnp.ndarray]] = None,
        dropout_epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")

        if dropout_epochs is None:
            dropout_epochs = num_epochs

        history: Dict[str, Any] = {
            'train_losses': [],
            'train_flow_losses': [],
            'train_recon_losses': [],
            'train_reg_losses': [],
            'val_losses': [],
            'val_flow_losses': [],
            'val_recon_losses': [],
            'val_reg_losses': [],
        }

        num_samples = y_data.shape[0]
        for epoch in range(num_epochs):
            use_dropout = epoch < dropout_epochs
            # shuffle
            self.rng, shuf = jr.split(self.rng)
            perm = jr.permutation(shuf, num_samples)
            x_shuf = x_data[perm] if x_data is not None else None
            y_shuf = y_data[perm]

            # minibatches
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                x_batch = x_shuf[start:end] if x_shuf is not None else None
                metrics = self.train_step(x_batch, y_shuf[start:end], use_dropout=use_dropout)
            
            # Store detailed loss metrics from last batch of epoch
            history['train_losses'].append(float(metrics.get('total_loss', 0.0)))
            history['train_flow_losses'].append(float(metrics.get('flow_loss', 0.0)))
            history['train_recon_losses'].append(float(metrics.get('recon_loss', 0.0)))
            history['train_reg_losses'].append(float(metrics.get('reg_loss', 0.0)))

            if validation_data is not None:
                vx, vy = validation_data
                val_metrics = self.evaluate_detailed(vx, vy, batch_size)
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_flow_losses'].append(val_metrics['flow_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
                history['val_reg_losses'].append(val_metrics['reg_loss'])

        return history

    def evaluate(self, x_data: Optional[jnp.ndarray], y_data: jnp.ndarray, batch_size: int = 256) -> float:
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        num_samples = y_data.shape[0]
        total = 0.0
        steps = 0
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            self.rng, eval_rng = jr.split(self.rng)
            x_input = None if (self.unconditional or x_data is None) else x_data[start:end]
            loss, _ = self.model.loss(self.params, x_input, y_data[start:end], eval_rng, training=False)
            total += float(loss)
            steps += 1
        return total / max(steps, 1)
    
    def evaluate_detailed(self, x_data: Optional[jnp.ndarray], y_data: jnp.ndarray, batch_size: int = 256) -> Dict[str, float]:
        """Evaluate model and return detailed loss metrics."""
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        num_samples = y_data.shape[0]
        metrics_sum = {'total_loss': 0.0, 'flow_loss': 0.0, 'recon_loss': 0.0, 'reg_loss': 0.0}
        steps = 0
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            self.rng, eval_rng = jr.split(self.rng)
            x_input = None if (self.unconditional or x_data is None) else x_data[start:end]
            _, metrics = self.model.loss(self.params, x_input, y_data[start:end], eval_rng, training=False)
            for key in metrics_sum:
                metrics_sum[key] += float(metrics.get(key, 0.0))
            steps += 1
        return {key: val / max(steps, 1) for key, val in metrics_sum.items()}

    def conditional_generate(
        self,
        cond_y: jnp.ndarray,
        num_steps: int = 20,
        prng_key: Optional[jr.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Generate x samples conditioned on labels y using stochastic z_0.
        For unconditional generation, use unconditional_generate instead.
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        if self.unconditional:
            raise ValueError("Use unconditional_generate() for unconditional generation")
        # Model predict expects x as conditional input; since we trained reversed, x is y
        return self.model.predict(self.params, cond_y, num_steps=num_steps, integration_method="midpoint", output_type="end_point", prng_key=prng_key)
    
    def unconditional_generate(
        self,
        batch_shape: Tuple[int, ...],
        num_steps: int = 20,
        prng_key: Optional[jr.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Generate x samples unconditionally using stochastic z_0.
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        if not self.unconditional:
            raise ValueError("Model was trained conditionally. Use conditional_generate() instead")
        if prng_key is None:
            self.rng, prng_key = jr.split(self.rng)
        
        integration_method = "midpoint" if self.model_type == "ct" else "euler"
        # Ensure batch_shape is a tuple of Python integers for static argument
        if isinstance(batch_shape, (list, tuple)):
            batch_shape = tuple(int(x) for x in batch_shape)
        return self.model.sample(self.params, prng_key, batch_shape, num_steps=num_steps, integration_method=integration_method, output_type="end_point")

    def save_params(self, output_path: str):
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(jax.device_get(self.params), f)

    def save_generation_plot(self, x_real: np.ndarray, y_labels: Optional[np.ndarray], x_gen: np.ndarray, output_dir: str):
        import matplotlib.pyplot as plt
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        if y_labels is not None and not self.unconditional:
            # Conditional generation: color by class labels
            # Real
            ax[0].scatter(x_real[:, 0], x_real[:, 1], c=(y_labels[:, 0] > 0).astype(int), s=6, cmap='coolwarm', alpha=0.8)
            ax[0].set_title('Real samples (x)')
            ax[0].set_aspect('equal', 'box')
            # Generated
            ax[1].scatter(x_gen[:, 0], x_gen[:, 1], c=(y_labels[:, 0] > 0).astype(int), s=6, cmap='coolwarm', alpha=0.8)
            ax[1].set_title('Generated samples (x | y)')
        else:
            # Unconditional generation: single color
            # Real
            ax[0].scatter(x_real[:, 0], x_real[:, 1], c='blue', s=6, alpha=0.8)
            ax[0].set_title('Real samples (x)')
            ax[0].set_aspect('equal', 'box')
            # Generated
            ax[1].scatter(x_gen[:, 0], x_gen[:, 1], c='purple', s=6, alpha=0.8)
            ax[1].set_title('Generated samples (x) - Unconditional')
        
        ax[1].set_aspect('equal', 'box')
        for a in ax:
            a.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_name = 'unconditional_generation.png' if self.unconditional else 'conditional_generation.png'
        fig.savefig(os.path.join(output_dir, plot_name), dpi=200, bbox_inches='tight')
        plt.close(fig)
    
    def save_loss_trends_plot(self, history: Dict[str, Any], output_dir: str):
        """Plot loss terms over training epochs to diagnose training issues."""
        import matplotlib.pyplot as plt
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Loss Trends - {self.model_type.title()} Model', fontsize=16, fontweight='bold')
        
        epochs = range(len(history['train_losses']))
        
        # Total Loss
        ax = axes[0, 0]
        ax.plot(epochs, history['train_losses'], label='Train Total', color='blue', linewidth=2)
        if history.get('val_losses') and len(history['val_losses']) > 0:
            ax.plot(epochs, history['val_losses'], label='Val Total', color='red', linewidth=2, linestyle='--')
        ax.set_title('Total Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Flow Loss
        ax = axes[0, 1]
        ax.plot(epochs, history['train_flow_losses'], label='Train Flow', color='green', linewidth=2)
        if history.get('val_flow_losses') and len(history['val_flow_losses']) > 0:
            ax.plot(epochs, history['val_flow_losses'], label='Val Flow', color='orange', linewidth=2, linestyle='--')
        ax.set_title('Flow Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Reconstruction Loss
        ax = axes[1, 0]
        ax.plot(epochs, history['train_recon_losses'], label='Train Recon', color='purple', linewidth=2)
        if history.get('val_recon_losses') and len(history['val_recon_losses']) > 0:
            ax.plot(epochs, history['val_recon_losses'], label='Val Recon', color='brown', linewidth=2, linestyle='--')
        ax.set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Regularization Loss
        ax = axes[1, 1]
        ax.plot(epochs, history['train_reg_losses'], label='Train Reg', color='cyan', linewidth=2)
        if history.get('val_reg_losses') and len(history['val_reg_losses']) > 0:
            ax.plot(epochs, history['val_reg_losses'], label='Val Reg', color='magenta', linewidth=2, linestyle='--')
        ax.set_title('Regularization Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'loss_trends.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
    
    def save_trajectory_plot(self, cond_y: Optional[jnp.ndarray] = None, num_trajectories: int = 20, num_steps: int = 20, prng_key: Optional[jr.PRNGKey] = None, output_dir: str = None):
        """Generate and plot latent z trajectories during integration."""
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        import matplotlib.pyplot as plt
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        n_samples = num_trajectories
        
        # Generate trajectories
        if prng_key is None:
            self.rng, prng_key = jr.split(self.rng)
        
        # Split keys for each trajectory
        prng_keys = jr.split(prng_key, n_samples)
        trajectories = []
        
        integration_method = "midpoint" if self.model_type == "ct" else "euler"
        
        for i in range(n_samples):
            if self.unconditional:
                # Use sample() for unconditional generation
                traj = self.model.sample(
                    self.params,
                    prng_keys[i],
                    batch_shape=(1,),
                    num_steps=num_steps,
                    integration_method=integration_method,
                    output_type="trajectory"
                )
            else:
                # Use predict() for conditional generation
                if cond_y is None:
                    raise ValueError("cond_y must be provided for conditional generation")
                cond_subset = cond_y[:n_samples]
                traj = self.model.predict(
                    self.params,
                    cond_subset[i:i+1],  # Single condition with batch dim
                    num_steps=num_steps,
                    integration_method=integration_method,
                    output_type="trajectory",
                    prng_key=prng_keys[i]
                )
            
            # Remove batch dimension: [num_steps, 1, output_dim] -> [num_steps, output_dim]
            if traj.ndim == 3:
                traj = traj[:, 0, :]
            trajectories.append(np.array(traj))
        
        trajectories = np.array(trajectories)  # [n_samples, num_steps, output_dim]
        
        # Plot trajectories
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.unconditional:
            # Unconditional: all trajectories same color
            for i in range(n_samples):
                traj = trajectories[i]  # [num_steps, 2]
                ax.plot(traj[:, 0], traj[:, 1], color='purple', alpha=0.6, linewidth=1.5)
                # Mark end point
                ax.scatter(traj[-1, 0], traj[-1, 1], color='purple', s=50, marker='s', edgecolors='black', linewidths=1, zorder=5)
            
            ax.set_title(f'Latent z Trajectories During Integration - Unconditional ({n_samples} samples)', fontsize=14, fontweight='bold')
            legend_elements = [
                Line2D([0], [0], color='purple', linewidth=2, label='Unconditional'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='End', markeredgecolor='black')
            ]
        else:
            # Conditional: color by class
            cond_subset = cond_y[:n_samples]
            class_labels = np.array((cond_subset[:, 0] > 0).astype(int))  # 0 for class -1, 1 for class +1
            class_colors = {0: 'blue', 1: 'red'}  # Discrete colors for each class
            
            for i in range(n_samples):
                traj = trajectories[i]  # [num_steps, 2]
                color = class_colors[int(class_labels[i])]
                ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.6, linewidth=1.5)
                # Mark end point
                ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=50, marker='s', edgecolors='black', linewidths=1, zorder=5)
            
            ax.set_title(f'Latent z Trajectories During Integration ({n_samples} samples)', fontsize=14, fontweight='bold')
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=2, label='Class -1'),
                Line2D([0], [0], color='red', linewidth=2, label='Class +1'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='End', markeredgecolor='black')
            ]
        
        ax.set_xlabel('z[0]', fontsize=12)
        ax.set_ylabel('z[1]', fontsize=12)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_elements, loc='upper right')
        
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'latent_trajectories.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)


