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

from src.flow_models.fm import VAE_flow as FlowMatchingModel
from src.flow_models.df import VAE_flow as DiffusionModel
from src.flow_models.ct import VAE_flow as CTModel
from src.flow_models.config import Config as FlowMatchingConfig, Config as DiffusionConfig, Config as CTConfig


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
        
        # Use dropout-free train step when dropout is disabled for efficiency
        if use_dropout:
            self.params, self.opt_state, loss, metrics = self.model.train_step(
                self.params, x_input, y_batch, self.opt_state, self.optimizer, train_rng, training=True
            )
        else:
            # Use the optimized dropout-free method
            if hasattr(self.model, 'train_step_without_dropout'):
                self.params, self.opt_state, loss, metrics = self.model.train_step_without_dropout(
                    self.params, x_input, y_batch, self.opt_state, self.optimizer, train_rng
                )
            else:
                # Fallback to regular train_step with training=False
                self.params, self.opt_state, loss, metrics = self.model.train_step(
                    self.params, x_input, y_batch, self.opt_state, self.optimizer, train_rng, training=False
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
            'val_chamfer_distances': [],
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
                
                # Compute Chamfer Distance: generate samples and compare with real validation data
                num_eval_samples = min(1000, vy.shape[0])  # Limit to 1000 samples for efficiency
                self.rng, gen_rng = jr.split(self.rng)
                if self.unconditional:
                    # Unconditional generation
                    x_gen_eval = self.unconditional_generate(
                        batch_shape=(num_eval_samples,),
                        num_steps=20,
                        prng_key=gen_rng
                    )
                else:
                    # Conditional generation: use validation conditions
                    cond_eval = vx[:num_eval_samples] if vx is not None else None
                    if cond_eval is None:
                        x_gen_eval = self.unconditional_generate(
                            batch_shape=(num_eval_samples,),
                            num_steps=20,
                            prng_key=gen_rng
                        )
                    else:
                        x_gen_eval = self.conditional_generate(
                            cond_eval,
                            num_steps=20,
                            prng_key=gen_rng
                        )
                x_real_eval = vy[:num_eval_samples]
                chamfer_dist = self.compute_chamfer_distance(x_gen_eval, x_real_eval)
                history['val_chamfer_distances'].append(chamfer_dist)

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

    def compute_chamfer_distance(self, generated_samples: jnp.ndarray, real_samples: jnp.ndarray) -> float:
        """
        Compute Chamfer Distance between generated and real point clouds.
        
        Chamfer Distance measures the average distance from each generated point to its
        nearest neighbor in the real data, and from each real point to its nearest neighbor
        in the generated data.
        
        Args:
            generated_samples: Generated samples [num_gen, feature_dim]
            real_samples: Real samples [num_real, feature_dim]
            
        Returns:
            Chamfer Distance (scalar), or float('inf') if generation failed (NaN/Inf present)
        """
        # Check for NaN or Inf in generated samples - indicates generation failure
        gen_has_invalid = jnp.any(~jnp.isfinite(generated_samples))
        real_has_invalid = jnp.any(~jnp.isfinite(real_samples))
        
        if gen_has_invalid or real_has_invalid:
            # Return inf to indicate failure (we want to minimize, so inf is worst case)
            return float('inf')
        
        # Compute pairwise squared distances: [num_gen, num_real]
        # ||g_i - r_j||^2 = ||g_i||^2 - 2*g_i*r_j + ||r_j||^2
        gen_norm_sq = jnp.sum(generated_samples ** 2, axis=1, keepdims=True)  # [num_gen, 1]
        real_norm_sq = jnp.sum(real_samples ** 2, axis=1)  # [num_real,]
        dot_product = jnp.dot(generated_samples, real_samples.T)  # [num_gen, num_real]
        pairwise_dist_sq = gen_norm_sq - 2 * dot_product + real_norm_sq  # [num_gen, num_real]
        
        # Check for negative values due to numerical errors and clip
        pairwise_dist_sq = jnp.maximum(pairwise_dist_sq, 0.0)
        
        # Distance from each generated point to nearest real point
        min_dist_gen_to_real = jnp.sqrt(jnp.min(pairwise_dist_sq, axis=1))  # [num_gen,]
        chamfer_gen_to_real = jnp.mean(min_dist_gen_to_real)
        
        # Distance from each real point to nearest generated point
        min_dist_real_to_gen = jnp.sqrt(jnp.min(pairwise_dist_sq, axis=0))  # [num_real,]
        chamfer_real_to_gen = jnp.mean(min_dist_real_to_gen)
        
        # Bidirectional Chamfer Distance (average of both directions)
        chamfer_distance = (chamfer_gen_to_real + chamfer_real_to_gen) / 2.0
        
        # Final check for NaN/Inf (shouldn't happen now, but safety check)
        if not jnp.isfinite(chamfer_distance):
            return float('inf')
        
        return float(chamfer_distance)

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
        """Create generation comparison plot showing real vs generated samples."""
        from src.utils.plotting.plot_generation import create_generation_plot
        create_generation_plot(x_real, y_labels, x_gen, output_dir, self.unconditional)
    
    def save_loss_trends_plot(self, history: Dict[str, Any], output_dir: str):
        """Plot loss terms over training epochs to diagnose training issues."""
        from src.utils.plotting.plot_generation import create_loss_trends_plot
        create_loss_trends_plot(history, self.model_type, output_dir)
    
    def save_trajectory_plot(self, cond_y: Optional[jnp.ndarray] = None, num_trajectories: int = 20, num_steps: int = 20, prng_key: Optional[jr.PRNGKey] = None, output_dir: str = None):
        """Generate and plot latent z trajectories during integration."""
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        from src.utils.plotting.plot_generation import create_latent_trajectories_plot
        
        # Generate PRNG key if needed
        if prng_key is None:
            self.rng, prng_key = jr.split(self.rng)
        
        create_latent_trajectories_plot(
            model=self.model,
            params=self.params,
            model_type=self.model_type,
            unconditional=self.unconditional,
            output_dir=output_dir,
            cond_y=cond_y,
            num_trajectories=num_trajectories,
            num_steps=num_steps,
            prng_key=prng_key,
            rng=self.rng
        )


