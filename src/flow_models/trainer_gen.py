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

    def initialize(self, x_sample: jnp.ndarray, y_sample: jnp.ndarray, z_sample: jnp.ndarray, t_sample: jnp.ndarray):
        self.rng, init_rng = jr.split(self.rng)
        self.params = self.model.init(init_rng, x_sample, y_sample, init_rng)
        self.opt_state = self.optimizer.init(self.params)

    def train_step(self, x_batch: jnp.ndarray, y_batch: jnp.ndarray, use_dropout: bool = True) -> Dict[str, float]:
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        self.rng, train_rng = jr.split(self.rng)
        self.params, self.opt_state, loss, metrics = self.model.train_step(
            self.params, x_batch, y_batch, self.opt_state, self.optimizer, train_rng, training=use_dropout
        )
        return metrics

    def train(
        self,
        x_data: jnp.ndarray,
        y_data: jnp.ndarray,
        num_epochs: int = 50,
        batch_size: int = 256,
        validation_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        dropout_epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")

        if dropout_epochs is None:
            dropout_epochs = num_epochs

        history: Dict[str, Any] = {
            'train_losses': [],
            'val_losses': [],
        }

        num_samples = x_data.shape[0]
        for epoch in range(num_epochs):
            use_dropout = epoch < dropout_epochs
            # shuffle
            self.rng, shuf = jr.split(self.rng)
            perm = jr.permutation(shuf, num_samples)
            x_shuf = x_data[perm]
            y_shuf = y_data[perm]

            # minibatches
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                metrics = self.train_step(x_shuf[start:end], y_shuf[start:end], use_dropout=use_dropout)
            history['train_losses'].append(float(metrics.get('total_loss', 0.0)))

            if validation_data is not None:
                vx, vy = validation_data
                vloss = self.evaluate(vx, vy, batch_size)
                history['val_losses'].append(vloss)

        return history

    def evaluate(self, x_data: jnp.ndarray, y_data: jnp.ndarray, batch_size: int = 256) -> float:
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        num_samples = x_data.shape[0]
        total = 0.0
        steps = 0
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            self.rng, eval_rng = jr.split(self.rng)
            loss, _ = self.model.loss(self.params, x_data[start:end], y_data[start:end], eval_rng, training=False)
            total += float(loss)
            steps += 1
        return total / max(steps, 1)

    def conditional_generate(
        self,
        cond_y: jnp.ndarray,
        num_steps: int = 20,
        prng_key: Optional[jr.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Generate x samples conditioned on labels y using stochastic z_0.
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        # Model predict expects x as conditional input; since we trained reversed, x is y
        return self.model.predict(self.params, cond_y, num_steps=num_steps, integration_method="midpoint", output_type="end_point", prng_key=prng_key)

    def save_params(self, output_path: str):
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(jax.device_get(self.params), f)

    def save_generation_plot(self, x_real: np.ndarray, y_labels: np.ndarray, x_gen: np.ndarray, output_dir: str):
        import matplotlib.pyplot as plt
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        # Real
        ax[0].scatter(x_real[:, 0], x_real[:, 1], c=(y_labels[:, 0] > 0).astype(int), s=6, cmap='coolwarm', alpha=0.8)
        ax[0].set_title('Real samples (x)')
        ax[0].set_aspect('equal', 'box')
        # Generated
        ax[1].scatter(x_gen[:, 0], x_gen[:, 1], c=(y_labels[:, 0] > 0).astype(int), s=6, cmap='coolwarm', alpha=0.8)
        ax[1].set_title('Generated samples (x | y)')
        ax[1].set_aspect('equal', 'box')
        for a in ax:
            a.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'conditional_generation.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)


