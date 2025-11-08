"""
Trainer for conditional generation on sequence data (x | y).

This trainer focuses on training the selected model (FM, CT, or DF) with sequence data
where both inputs and outputs are sequences of shape (batch, seq_len, embed_dim).
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

from src.utils.plotting.plot_loss_trends import plot_loss_trends
from experiments.stock_prediction.plotting import (
    plot_direct_comparison,
    plot_trajectory_comparison,
    plot_sequence_comparison,
    plot_price_comparison,
    plot_latent_trajectories,
)


@dataclass
class SequenceTrainer:
    config: Any
    learning_rate: float = 1e-3
    optimizer_name: str = "adam"
    seed: int = 42
    unconditional: bool = False  # If True, use unconditional generation (x=None)

    def __post_init__(self):

        if(self.config.model_type == "diffusion"):
            self.model = DiffusionModel(config=config)
            self.model_type = "diffusion"
        elif(self.config.model_type == "flow_matching"):
            self.model = FlowMatchingModel(config=config)
            self.model_type = "flow_matching"
        elif(self.config.model_type == "ct"):
            self.model = CTModel(config=config)
            self.model_type = "ct"
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

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

    def _create_minibatch_with_random_splits(
        self, 
        sequences: list, 
        indices: jnp.ndarray, 
        y_seq_len: int = 12,
        min_x_len: int = 12
    ) -> Tuple[Optional[jnp.ndarray], jnp.ndarray]:
        """
        Create minibatch by randomly splitting full-day sequences.
        
        Args:
            sequences: List of full-day sequences (variable length)
            indices: Batch indices to select
            y_seq_len: Length of target sequence (12 = 1 hour)
            min_x_len: Minimum length for x sequence (default: 12 = 1 hour)
        
        Returns:
            x_batch: Padded x sequences (batch, max_x_len, features)
            y_batch: Target sequences (batch, y_seq_len, features)
        """
        batch_seqs = [sequences[int(i)] for i in indices]
        
        # Randomly sample split points for each sequence
        x_batches = []
        y_batches = []
        split_points = []
        
        self.rng, split_rng = jr.split(self.rng)
        split_keys = jr.split(split_rng, len(batch_seqs))
        
        for seq, key in zip(batch_seqs, split_keys):
            seq_len = len(seq)
            # Split point must allow for at least min_x_len for x and y_seq_len for y
            # Minimum required sequence length: min_x_len + y_seq_len
            min_required_len = min_x_len + y_seq_len
            if seq_len < min_required_len:
                # Sequence too short - skip or use available data
                # Use all but last y_seq_len as x (may be less than min_x_len)
                split_point = max(0, seq_len - y_seq_len)
            else:
                # Random split between min_x_len and max_split (inclusive)
                max_split = seq_len - y_seq_len
                split_point = int(jr.randint(key, (), minval=min_x_len, maxval=max_split + 1))
            
            x_seq = seq[:split_point]
            y_seq = seq[split_point:split_point + y_seq_len]
            
            x_batches.append(x_seq)
            y_batches.append(y_seq)
            split_points.append(split_point)
        
        # Pad x sequences to fixed length for batching
        max_x_len = max(len(x) for x in x_batches) if x_batches else 0
        x_padded = []
        for x_seq in x_batches:
            if len(x_seq) < max_x_len:
                padding = np.zeros((max_x_len - len(x_seq), x_seq.shape[1]))
                x_padded.append(np.concatenate([x_seq, padding], axis=0))
            else:
                x_padded.append(x_seq)
        
        x_batch = jnp.array(x_padded)  # (batch, max_x_len, features)
        y_batch = jnp.array(y_batches)  # (batch, y_seq_len, features)
        
        return x_batch, y_batch

    def train(
        self,
        sequences_data: Optional[list],  # List of full-day sequences
        num_epochs: int = 50,
        batch_size: int = 256,
        y_seq_len: int = 12,  # Target sequence length (1 hour)
        validation_data: Optional[list] = None,  # List of validation sequences
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
            'val_seq_metrics': [],
        }

        num_samples = len(sequences_data) if sequences_data is not None else 0
        for epoch in range(num_epochs):
            use_dropout = epoch < dropout_epochs
            # shuffle
            self.rng, shuf = jr.split(self.rng)
            perm = jr.permutation(shuf, num_samples)

            # minibatches
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = perm[start:end]
                x_batch, y_batch = self._create_minibatch_with_random_splits(
                    sequences_data, batch_indices, y_seq_len=y_seq_len
                )
                metrics = self.train_step(x_batch, y_batch, use_dropout=use_dropout)
            
            # Store detailed loss metrics from last batch of epoch
            history['train_losses'].append(float(metrics.get('total_loss', 0.0)))
            history['train_flow_losses'].append(float(metrics.get('flow_loss', 0.0)))
            history['train_recon_losses'].append(float(metrics.get('recon_loss', 0.0)))
            history['train_reg_losses'].append(float(metrics.get('reg_loss', 0.0)))

            if validation_data is not None:
                val_metrics = self.evaluate_detailed_with_random_splits(
                    validation_data, batch_size, y_seq_len=y_seq_len
                )
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_flow_losses'].append(val_metrics['flow_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
                history['val_reg_losses'].append(val_metrics['reg_loss'])
                
                # Compute sequence metrics: generate samples and compare with real validation data
                num_eval_samples = min(100, len(validation_data))
                self.rng, gen_rng = jr.split(self.rng)
                # Randomly select validation sequences and create splits for generation
                eval_indices = jr.permutation(gen_rng, len(validation_data))[:num_eval_samples]
                eval_x, eval_y_real = self._create_minibatch_with_random_splits(
                    validation_data, eval_indices, y_seq_len=y_seq_len
                )
                
                if self.unconditional:
                    # Unconditional generation
                    y_gen_eval = self.unconditional_generate(
                        batch_shape=(num_eval_samples,),
                        num_steps=20,
                        prng_key=gen_rng
                    )
                else:
                    # Conditional generation: use validation conditions
                    y_gen_eval = self.conditional_generate(
                        cond_x=eval_x,
                        num_steps=20
                    )
                seq_metrics = self.compute_sequence_metrics(y_gen_eval, eval_y_real)
                history['val_seq_metrics'].append(seq_metrics)

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
    
    def evaluate_detailed_with_random_splits(
        self, 
        sequences_data: list, 
        batch_size: int = 256,
        y_seq_len: int = 12
    ) -> Dict[str, float]:
        """Evaluate model with random splits and return detailed loss metrics."""
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        num_samples = len(sequences_data)
        metrics_sum = {'total_loss': 0.0, 'flow_loss': 0.0, 'recon_loss': 0.0, 'reg_loss': 0.0}
        num_batches = 0
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = jnp.arange(start, end)
            x_batch, y_batch = self._create_minibatch_with_random_splits(
                sequences_data, batch_indices, y_seq_len=y_seq_len
            )
            self.rng, eval_rng = jr.split(self.rng)
            _, metrics = self.model.loss(self.params, x_batch, y_batch, eval_rng, training=False)
            for key in metrics_sum:
                metrics_sum[key] += metrics.get(key, 0.0)
            num_batches += 1
        # Average metrics
        for key in metrics_sum:
            metrics_sum[key] /= num_batches
        return metrics_sum

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

    def compute_sequence_metrics(self, generated_sequences: jnp.ndarray, real_sequences: jnp.ndarray) -> Dict[str, float]:
        """
        Compute metrics between generated and real sequences.
        
        Args:
            generated_sequences: Generated sequences [num_gen, seq_len, embed_dim]
            real_sequences: Real sequences [num_real, seq_len, embed_dim]
            
        Returns:
            Dictionary of metrics including MSE, MAE, cosine similarity, and R² (percent variance explained)
        """
        from src.utils.metrics import sequence_metrics
        return sequence_metrics(generated_sequences, real_sequences, price_dim=0)

    def conditional_generate(
        self,
        cond_x: jnp.ndarray,
        num_steps: int = 20,
    ) -> jnp.ndarray:
        """
        Generate y sequences conditioned on x sequences using stochastic z_0.
        For unconditional generation, use unconditional_generate instead.
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        if self.unconditional:
            raise ValueError("Use unconditional_generate() for unconditional generation")
        # Model predict expects x as conditional input
        return self.model.predict(self.params, cond_x, num_steps=num_steps, integration_method="midpoint", output_type="end_point")
    
    def unconditional_generate(
        self,
        batch_shape: Tuple[int, ...],
        num_steps: int = 20,
        prng_key: Optional[jr.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Generate y sequences unconditionally using stochastic z_0.
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

    def save_sequence_plot(self, y_real: np.ndarray, x_labels: Optional[np.ndarray], y_gen: np.ndarray, output_dir: str, 
                          data_path: Optional[str] = None):
        """Plot generated sequences vs real sequences.
        
        If data_path is provided, removes positional embeddings and projects back to 2D
        for meaningful visualization (shows actual price/volume features).
        Otherwise, plots the raw embedding dimensions (may show positional embedding patterns).
        """
        plot_sequence_comparison(y_real, y_gen, output_dir, data_path)
    
    def save_direct_comparison_plot(
        self,
        y_real: np.ndarray,
        y_pred: np.ndarray,
        output_dir: str,
        num_samples: int = 100
    ):
        """
        Direct comparison plot of predictions vs ground truth in model input/output space (2D).
        
        This plots predictions vs ground truth directly without any transformations,
        showing the model's performance in the standardized 2D space (price, volume).
        
        Args:
            y_real: Real sequences [batch, seq_len, 2] in standardized 2D space
            y_pred: Predicted sequences [batch, seq_len, 2] in standardized 2D space
            output_dir: Directory to save the plot
            num_samples: Number of samples to plot (will use first num_samples)
        """
        plot_direct_comparison(y_real, y_pred, output_dir, num_samples)
    
    def save_trajectory_comparison_plot(
        self,
        y_real: np.ndarray,
        y_pred: np.ndarray,
        output_dir: str,
        num_samples: int = 20
    ):
        """
        Plot raw prediction vs ground truth trajectories over time.
        
        Shows time series plots for a random selection of sequences, comparing
        predicted and ground truth trajectories for each dimension.
        
        Args:
            y_real: Real sequences [batch, seq_len, 2] in standardized 2D space
            y_pred: Predicted sequences [batch, seq_len, 2] in standardized 2D space
            output_dir: Directory to save the plot
            num_samples: Number of random sequences to plot
        """
        plot_trajectory_comparison(y_real, y_pred, output_dir, num_samples)
    
    def save_price_comparison_plot(
        self,
        y_real: np.ndarray,
        y_pred: np.ndarray,
        data_path: str,
        output_dir: str,
        num_samples: int = 8,
        start_time: str = "10:30",
        end_time: str = "14:30"
    ):
        """
        Plot actual vs predicted prices during trading hours (10:30 AM - 2:30 PM).
        
        Args:
            y_real: Real sequences [batch, seq_len, embed_dim] (20D)
            y_pred: Predicted sequences [batch, seq_len, embed_dim] (20D)
            data_path: Path to processed data file containing projection matrix
            output_dir: Directory to save the plot
            num_samples: Number of samples to plot
            start_time: Start time in format "HH:MM"
            end_time: End time in format "HH:MM"
        """
        plot_price_comparison(y_real, y_pred, data_path, output_dir, num_samples, start_time, end_time)
    
    def save_loss_trends_plot(self, history: Dict[str, Any], output_dir: str):
        """Plot loss terms over training epochs to diagnose training issues."""
        plot_loss_trends(history, self.model_type, output_dir)
    
    def save_trajectory_plot(self, cond_x: Optional[jnp.ndarray] = None, num_trajectories: int = 20, num_steps: int = 20, prng_key: Optional[jr.PRNGKey] = None, output_dir: str = None):
        """Generate and plot latent z trajectories during integration for sequence data."""
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        plot_latent_trajectories(
            model=self.model,
            params=self.params,
            model_type=self.model_type,
            unconditional=self.unconditional,
            output_dir=output_dir,
            cond_x=cond_x,
            num_trajectories=num_trajectories,
            num_steps=num_steps,
            prng_key=None,  # Not used - always use MLE (predict with no prng_key)
            rng=None  # Not used - always use MLE (predict with no prng_key)
        )

