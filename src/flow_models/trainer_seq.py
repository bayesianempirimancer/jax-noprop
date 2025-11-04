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

from src.flow_models.fm import VAE_flow as FlowMatchingModel, VAEFlowConfig as FlowMatchingConfig
from src.flow_models.df import VAE_flow as DiffusionModel, VAEFlowConfig as DiffusionConfig
from src.flow_models.ct import VAE_flow as CTModel, VAEFlowConfig as CTConfig


@dataclass
class SequenceTrainer:
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
                        num_steps=20,
                        prng_key=gen_rng
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
            Dictionary of metrics including MSE, MAE, and cosine similarity
        """
        # Check for NaN or Inf in generated sequences - indicates generation failure
        gen_has_invalid = jnp.any(~jnp.isfinite(generated_sequences))
        real_has_invalid = jnp.any(~jnp.isfinite(real_sequences))
        
        if gen_has_invalid or real_has_invalid:
            # Return inf to indicate failure
            return {'mse': float('inf'), 'mae': float('inf'), 'cosine_sim': -1.0}
        
        # Flatten sequences for comparison
        gen_flat = generated_sequences.reshape(generated_sequences.shape[0], -1)
        real_flat = real_sequences.reshape(real_sequences.shape[0], -1)
        
        # Compute MSE (Mean Squared Error)
        mse = jnp.mean((gen_flat - real_flat) ** 2)
        
        # Compute MAE (Mean Absolute Error)
        mae = jnp.mean(jnp.abs(gen_flat - real_flat))
        
        # Compute cosine similarity (average across samples)
        # Normalize vectors
        gen_norm = gen_flat / (jnp.linalg.norm(gen_flat, axis=1, keepdims=True) + 1e-8)
        real_norm = real_flat / (jnp.linalg.norm(real_flat, axis=1, keepdims=True) + 1e-8)
        cosine_sim = jnp.mean(jnp.sum(gen_norm * real_norm, axis=1))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'cosine_sim': float(cosine_sim)
        }

    def conditional_generate(
        self,
        cond_x: jnp.ndarray,
        num_steps: int = 20,
        prng_key: Optional[jr.PRNGKey] = None,
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
        return self.model.predict(self.params, cond_x, num_steps=num_steps, integration_method="midpoint", output_type="end_point", prng_key=prng_key)
    
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
        
        If data_path is provided, removes positional embeddings and projects back to 4D
        for meaningful visualization (shows actual price/volume/bid/ask features).
        Otherwise, plots the raw embedding dimensions (may show positional embedding patterns).
        """
        import matplotlib.pyplot as plt
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        seq_len = y_real.shape[1]
        feature_dim = y_real.shape[2]
        batch_size = y_real.shape[0]
        
        # If data_path is provided, handle 2D data or remove embeddings (old format)
        if data_path is not None:
            try:
                import pickle
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Check if data has projection info (old format) or is 2D (new format)
                has_projection = 'projection' in data and data['projection'] is not None
                if has_projection:
                    projection_matrix = data['projection']['matrix']  # [20, 4]
                    input_dim = data['projection']['input_dim']  # 2 (price, volume)
                    rope_base = data.get('rope', {}).get('base', 10000.0)
                else:
                    # New format: data is already 2D, no projection needed
                    input_dim = 2
                    projection_matrix = None
                    rope_base = 10000.0
                
                # Check if outputs are already 2D (new format) or need projection (old format)
                feature_dim = y_real.shape[2]
                if has_projection and feature_dim != input_dim:
                    # Old format: outputs are in embed_dim, need to remove embeddings and project
                    embed_dim = feature_dim
                    
                    # Get day-of-week embeddings and labels
                    day_embeddings_dict = data.get('day_of_week', {}).get('embeddings', {})
                    day_embeddings = {int(k): np.array(v) for k, v in day_embeddings_dict.items()}
                    all_days_of_week = data.get('day_of_week', {}).get('days_of_week_val', None)
                    if all_days_of_week is None:
                        all_days_of_week = np.zeros(batch_size, dtype=np.int32)
                    if len(all_days_of_week) < batch_size:
                        all_days_of_week = np.concatenate([
                            all_days_of_week,
                            np.zeros(batch_size - len(all_days_of_week), dtype=np.int32)
                        ])
                    sample_days_of_week = all_days_of_week[:batch_size]
                    
                    # Remove RoPE positional encodings (y sequences use shifted positions)
                    from src.embeddings.positional_encoding import rotary_positional_encoding
                    position_offset = -(seq_len - 1)
                    max_pos_needed = abs(position_offset) + seq_len
                    rope_encoding_full = np.array(rotary_positional_encoding(max_pos_needed, embed_dim, base=rope_base))
                    
                    start_idx = abs(position_offset)
                    end_idx = start_idx + seq_len
                    extracted = rope_encoding_full[start_idx:end_idx]
                    rope_encoding = np.flip(extracted, axis=0)
                    rope_encoding[:, 0::2] = -rope_encoding[:, 0::2]
                    
                    norms = np.linalg.norm(rope_encoding, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-8)
                    rope_encoding = rope_encoding / norms
                    
                    y_real_no_rope = y_real - rope_encoding[None, :, :]
                    y_gen_no_rope = y_gen - rope_encoding[None, :, :]
                    
                    sample_embeddings = np.array([day_embeddings.get(day, day_embeddings.get(0, np.zeros(embed_dim))) 
                                                for day in sample_days_of_week])
                    y_real_no_pos = y_real_no_rope - sample_embeddings[:, None, :]
                    y_gen_no_pos = y_gen_no_rope - sample_embeddings[:, None, :]
                    
                    proj_pinv = np.linalg.pinv(projection_matrix)
                    y_real_4d = y_real_no_pos.reshape(-1, embed_dim) @ proj_pinv.T
                    y_real_4d = y_real_4d.reshape(batch_size, seq_len, input_dim)
                    y_gen_4d = y_gen_no_pos.reshape(-1, embed_dim) @ proj_pinv.T
                    y_gen_4d = y_gen_4d.reshape(batch_size, seq_len, input_dim)
                else:
                    # New format: outputs are already 2D (price, volume)
                    # No need to remove embeddings or project - CRN handles this internally
                    y_real_4d = y_real  # Already 2D
                    y_gen_4d = y_gen  # Already 2D
                
                # Convert log-normalized values back to original domain
                # Get previous closes and avg volumes for validation set
                metadata = data.get('metadata', {})
                previous_closes = metadata.get('previous_closes_val', None)
                if previous_closes is None:
                    previous_closes = data.get('previous_closes', {}).get('val', None)
                
                previous_avg_volumes = metadata.get('previous_avg_volumes_val', None)
                if previous_avg_volumes is None:
                    previous_avg_volumes = data.get('previous_avg_volumes', {}).get('val', None)
                
                # Get standardization parameters from metadata (needed to reverse standardization)
                std_log_price = metadata.get('std_log_price', 1.0)
                std_log_volume_diff = metadata.get('std_log_volume_diff', 1.0)
                volume_scale_factor = metadata.get('volume_scale_factor', 0.05)
                
                if previous_closes is not None and len(previous_closes) >= batch_size:
                    sample_previous_closes = previous_closes[:batch_size]
                    sample_previous_avg_volumes = previous_avg_volumes[:batch_size] if previous_avg_volumes is not None else None
                    
                    # y_real_4d and y_gen_4d are currently standardized:
                    # - Price/bid/ask: log(price/prev_close) / std_log_price
                    # - Volume: (log(1+vol) - log(1+prev_avg_vol)) / std_log_volume_diff * volume_scale_factor
                    
                    y_real_4d_original = y_real_4d.copy()
                    y_gen_4d_original = y_gen_4d.copy()
                    
                    for i in range(batch_size):
                        prev_close = sample_previous_closes[i]
                        
                        # Convert dim0 (price) from standardized to original
                        # Step 1: Reverse standardization: multiply by std_log_price
                        log_norm_real = y_real_4d[i, :, 0] * std_log_price  # log(price / prev_close)
                        log_norm_gen = y_gen_4d[i, :, 0] * std_log_price    # log(price / prev_close)
                        
                        # Step 2: Convert: price = prev_close * exp(log_norm)
                        exp_real = np.clip(np.exp(log_norm_real), 1e-10, 1e10)
                        exp_gen = np.clip(np.exp(log_norm_gen), 1e-10, 1e10)
                        y_real_4d_original[i, :, 0] = prev_close * exp_real
                        y_gen_4d_original[i, :, 0] = prev_close * exp_gen
                        
                        # Ensure prices are positive (safety check)
                        y_real_4d_original[i, :, 0] = np.maximum(y_real_4d_original[i, :, 0], 1e-6)
                        y_gen_4d_original[i, :, 0] = np.maximum(y_gen_4d_original[i, :, 0], 1e-6)
                        
                        # Convert dim1 (volume) from standardized to original
                        if sample_previous_avg_volumes is not None and input_dim > 1:
                            prev_avg_vol = sample_previous_avg_volumes[i]
                            # Step 1: Reverse volume scaling: divide by volume_scale_factor
                            vol_scaled = y_real_4d[i, :, 1] / volume_scale_factor
                            vol_scaled_gen = y_gen_4d[i, :, 1] / volume_scale_factor
                            
                            # Step 2: Reverse standardization: multiply by std_log_volume_diff
                            vol_diff_real = vol_scaled * std_log_volume_diff  # log(1+vol) - log(1+prev_avg_vol)
                            vol_diff_gen = vol_scaled_gen * std_log_volume_diff
                            
                            # Step 3: Convert: exp(vol_diff) = (1+vol) / (1+prev_avg_vol)
                            # So: vol = (1+prev_avg_vol) * exp(vol_diff) - 1
                            y_real_4d_original[i, :, 1] = (1.0 + prev_avg_vol) * np.exp(vol_diff_real) - 1.0
                            y_gen_4d_original[i, :, 1] = (1.0 + prev_avg_vol) * np.exp(vol_diff_gen) - 1.0
                    
                    plot_data_real = y_real_4d_original
                    plot_data_gen = y_gen_4d_original
                else:
                    # If no previous closes available, plot log-normalized values
                    plot_data_real = y_real_4d
                    plot_data_gen = y_gen_4d
                
                # Use 4D features for plotting - only plot dim0 (Price) and dim1 (Volume)
                num_dims_to_plot = min(2, input_dim)  # Only Price and Volume
                dim_names = ['Price ($)', 'Volume'][:num_dims_to_plot]
            except Exception as e:
                print(f"  Warning: Could not load data_path for embedding removal: {e}")
                print(f"  Plotting raw feature dimensions instead")
                plot_data_real = y_real
                plot_data_gen = y_gen
                num_dims_to_plot = min(2, feature_dim)  # Price and volume
                dim_names = ['Price', 'Volume'][:num_dims_to_plot]
        else:
            # Plot raw feature dimensions (should be 2D: price, volume)
            plot_data_real = y_real
            plot_data_gen = y_gen
            num_dims_to_plot = min(2, feature_dim)  # Price and volume
            dim_names = ['Price', 'Volume'][:num_dims_to_plot]
        
        # Plot each dimension separately with many subplots
        # Each subplot shows one true vs one predicted sequence
        num_samples_to_plot = min(20, plot_data_real.shape[0])
        
        # Create separate figures for each dimension (dim0 and dim1)
        for dim_idx in range(num_dims_to_plot):
            # Calculate subplot layout: 4 rows x 5 cols for 20 samples
            n_cols = 5
            n_rows = (num_samples_to_plot + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1) if axes.ndim > 1 else axes.reshape(-1)
            
            for sample_idx in range(num_samples_to_plot):
                row = sample_idx // n_cols
                col = sample_idx % n_cols
                
                if n_rows == 1:
                    ax = axes[col]
                else:
                    ax = axes[row, col]
                
                # Plot single true and predicted sequence
                ax.plot(range(seq_len), plot_data_real[sample_idx, :, dim_idx], 
                       label='True', color='blue', linewidth=2, alpha=0.7)
                ax.plot(range(seq_len), plot_data_gen[sample_idx, :, dim_idx], 
                       label='Predicted', color='red', linewidth=2, linestyle='--', alpha=0.7)
                
                ax.set_title(f'Sample {sample_idx+1}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Time Step', fontsize=9)
                ax.set_ylabel(dim_names[dim_idx], fontsize=9)
                ax.grid(True, alpha=0.3)
                if sample_idx == 0:
                    ax.legend(fontsize=8)
            
            # Hide unused subplots
            for sample_idx in range(num_samples_to_plot, n_rows * n_cols):
                row = sample_idx // n_cols
                col = sample_idx % n_cols
                if n_rows == 1:
                    axes[col].axis('off')
                else:
                    axes[row, col].axis('off')
            
            fig.suptitle(f'True vs Predicted {dim_names[dim_idx]} Sequences ({num_samples_to_plot} samples)', 
                        fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            # Save separate file for each dimension
            plot_name = f'sequence_comparison_{dim_names[dim_idx].lower()}.png'
            fig.savefig(os.path.join(output_dir, plot_name), dpi=200, bbox_inches='tight')
            plt.close(fig)
    
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
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime, timedelta
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load data file to get preprocessing parameters
        # Note: Since CRN handles embeddings internally, outputs are already 2D
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if data has projection info (old format) or is 2D (new format)
        has_projection = 'projection' in data and data['projection'] is not None
        if has_projection:
            projection_matrix = data['projection']['matrix']  # [20, 4]
            input_dim = data['projection']['input_dim']  # 2 (price, volume)
        else:
            # New format: data is already 2D, no projection needed
            # y_real and y_pred are already in 2D format (price, volume)
            input_dim = 2
            projection_matrix = None
        
        # Get RoPE parameters
        rope_base = data.get('rope', {}).get('base', 10000.0)
        
        # Get sequence dimensions first
        batch_size, seq_len, feature_dim = y_real.shape
        
        # Check if outputs are already 2D (new format) or need projection (old format)
        if has_projection and feature_dim != input_dim:
            # Old format: outputs are in embed_dim, need to remove embeddings and project
            embed_dim = feature_dim
            
            # Get day-of-week embeddings (if available)
            day_embeddings_dict = data.get('day_of_week', {}).get('embeddings', {})
            day_embeddings = {int(k): np.array(v) for k, v in day_embeddings_dict.items()}
            
            # Get days_of_week labels for validation set
            all_days_of_week = data.get('day_of_week', {}).get('days_of_week_val', None)
            if all_days_of_week is None:
                all_days_of_week = np.zeros(batch_size, dtype=np.int32)
            if len(all_days_of_week) < batch_size:
                all_days_of_week = np.concatenate([
                    all_days_of_week,
                    np.zeros(batch_size - len(all_days_of_week), dtype=np.int32)
                ])
            sample_days_of_week = all_days_of_week[:batch_size]
            
            # Remove RoPE and day-of-week embeddings, then project back to 2D
            from src.embeddings.positional_encoding import rotary_positional_encoding
            
            position_offset = -(seq_len - 1)
            max_pos_needed = abs(position_offset) + seq_len
            rope_encoding_full = np.array(rotary_positional_encoding(max_pos_needed, embed_dim, base=rope_base))
            
            start_idx = abs(position_offset)
            end_idx = start_idx + seq_len
            extracted = rope_encoding_full[start_idx:end_idx]
            rope_encoding = np.flip(extracted, axis=0)
            rope_encoding[:, 0::2] = -rope_encoding[:, 0::2]
            
            norms = np.linalg.norm(rope_encoding, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            rope_encoding = rope_encoding / norms
            
            y_real_no_rope = y_real - rope_encoding[None, :, :]
            y_pred_no_rope = y_pred - rope_encoding[None, :, :]
            
            sample_embeddings = np.array([day_embeddings.get(day, day_embeddings.get(0, np.zeros(embed_dim))) 
                                         for day in sample_days_of_week])
            y_real_no_pos = y_real_no_rope - sample_embeddings[:, None, :]
            y_pred_no_pos = y_pred_no_rope - sample_embeddings[:, None, :]
            
            proj_pinv = np.linalg.pinv(projection_matrix)
            y_real_4d = y_real_no_pos.reshape(-1, embed_dim) @ proj_pinv.T
            y_real_4d = y_real_4d.reshape(batch_size, seq_len, input_dim)
            
            y_pred_4d = y_pred_no_pos.reshape(-1, embed_dim) @ proj_pinv.T
            y_pred_4d = y_pred_4d.reshape(batch_size, seq_len, input_dim)
        else:
            # New format: outputs are already 2D (price, volume)
            # No need to remove embeddings or project - CRN handles this internally
            y_real_4d = y_real  # Already 2D
            y_pred_4d = y_pred  # Already 2D
        
        # Debug: Check 2D values
        print(f"  DEBUG: Data shape (should be 2D): {y_real_4d.shape}")
        print(f"    y_real_4d[0, :, 0] (price) range: [{y_real_4d[0, :, 0].min():.6f}, {y_real_4d[0, :, 0].max():.6f}], std: {y_real_4d[0, :, 0].std():.6f}")
        if y_real_4d.shape[2] > 1:
            print(f"    y_real_4d[0, :, 1] (volume) range: [{y_real_4d[0, :, 1].min():.6f}, {y_real_4d[0, :, 1].max():.6f}], std: {y_real_4d[0, :, 1].std():.6f}")
        
        # Convert log-normalized prices back to original domain
        # Get previous closes for validation set
        metadata = data.get('metadata', {})
        previous_closes = metadata.get('previous_closes_val', None)
        if previous_closes is None:
            # Try old format
            previous_closes = data.get('previous_closes', {}).get('val', None)
        
        # Get standardization parameters from metadata (needed to reverse standardization)
        std_log_price = metadata.get('std_log_price', 1.0)
        std_log_volume_diff = metadata.get('std_log_volume_diff', 1.0)
        volume_scale_factor = metadata.get('volume_scale_factor', 0.05)
        
        if previous_closes is not None and len(previous_closes) >= batch_size:
            # Note: y_real should be the first batch_size samples from validation set
            # so we use previous_closes[:batch_size] to match
            sample_previous_closes = previous_closes[:batch_size]
            
            # Debug: Check if previous_closes are all the same
            if len(sample_previous_closes) > 1:
                if np.allclose(sample_previous_closes, sample_previous_closes[0]):
                    print(f"  WARNING: All previous_closes in batch are the same: {sample_previous_closes[0]:.2f}")
                else:
                    print(f"  DEBUG: previous_closes vary: min={sample_previous_closes.min():.2f}, max={sample_previous_closes.max():.2f}, std={sample_previous_closes.std():.2f}")
            
            # y_real_4d and y_pred_4d are currently standardized:
            # - Price: log(price/prev_close) / std_log_price
            # Inverse: price = prev_close * exp(log_normalized * std_log_price)
            y_real_price = np.zeros((batch_size, seq_len))
            y_pred_price = np.zeros((batch_size, seq_len))
            
            for i in range(batch_size):
                prev_close = sample_previous_closes[i]
                # Step 1: Reverse standardization: multiply by std_log_price
                log_norm_real = y_real_4d[i, :, 0] * std_log_price  # log(price / prev_close)
                log_norm_pred = y_pred_4d[i, :, 0] * std_log_price  # log(price / prev_close)
                
                # Step 2: Convert: price = prev_close * exp(log_norm)
                exp_real = np.clip(np.exp(log_norm_real), 1e-10, 1e10)
                exp_pred = np.clip(np.exp(log_norm_pred), 1e-10, 1e10)
                y_real_price[i, :] = prev_close * exp_real
                y_pred_price[i, :] = prev_close * exp_pred
                
                # Debug: Check if prices have variation (only for first sample)
                if i == 0 and batch_size > 0:
                    price_std = y_real_price[i].std()
                    price_range = y_real_price[i].max() - y_real_price[i].min()
                    print(f"  DEBUG (sample {i}): Converted prices - range: [{y_real_price[i].min():.2f}, {y_real_price[i].max():.2f}], std: {price_std:.2f}, range: {price_range:.2f}")
                    if price_std < 0.01:
                        print(f"    ⚠️  WARNING: Prices appear constant (std={price_std:.6f})")
                        print(f"       This is likely because log_norm values are very small (std={log_norm_real.std():.6f})")
                        print(f"       Small log_norm -> exp(log_norm) ≈ 1 -> prices ≈ prev_close (constant)")
                
                # Ensure prices are positive (safety check)
                y_real_price[i, :] = np.maximum(y_real_price[i, :], 1e-6)
                y_pred_price[i, :] = np.maximum(y_pred_price[i, :], 1e-6)
        else:
            # If no previous closes available, plot log-normalized values
            y_real_price = y_real_4d[:, :, 0]  # [batch, seq_len]
            y_pred_price = y_pred_4d[:, :, 0]  # [batch, seq_len]
        
        # Create time labels (5-minute intervals from start_time to end_time)
        start_dt = datetime.strptime(start_time, "%H:%M")
        time_delta = timedelta(minutes=5)
        time_labels = [start_dt + i * time_delta for i in range(seq_len)]
        time_strs = [t.strftime("%H:%M") for t in time_labels]
        
        # Select samples to plot
        num_samples = min(num_samples, batch_size)
        n_cols = 2
        n_rows = (num_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Actual vs Predicted Price (10:30 AM - 2:30 PM Window)', 
                     fontsize=16, fontweight='bold')
        
        for idx in range(num_samples):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Plot actual and predicted
            ax.plot(range(seq_len), y_real_price[idx], 
                   label='Actual', color='blue', linewidth=2, marker='o', markersize=4, alpha=0.7)
            ax.plot(range(seq_len), y_pred_price[idx], 
                   label='Predicted', color='red', linewidth=2, marker='s', markersize=4, 
                   linestyle='--', alpha=0.7)
            
            # Formatting
            ax.set_title(f'Sample {idx+1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (5-min intervals)', fontsize=10)
            ax.set_ylabel('Price ($)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set x-axis ticks (every 6 timesteps = 30 minutes)
            tick_step = max(1, seq_len // 8)
            ax.set_xticks(range(0, seq_len, tick_step))
            ax.set_xticklabels([time_strs[i] for i in range(0, seq_len, tick_step)], 
                              rotation=45, ha='right', fontsize=8)
        
        # Hide unused subplots
        for idx in range(num_samples, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        fig.tight_layout()
        plot_path = os.path.join(output_dir, 'price_comparison_10_30_to_14_30.png')
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Saved price comparison plot to {plot_path}")
    
    def save_loss_trends_plot(self, history: Dict[str, Any], output_dir: str):
        """Plot loss terms over training epochs to diagnose training issues."""
        import matplotlib.pyplot as plt
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if we have sequence metrics to determine subplot layout
        has_seq_metrics = history.get('val_seq_metrics') and len(history['val_seq_metrics']) > 0
        if has_seq_metrics:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Loss Trends - {self.model_type.title()} Model (Sequence Data)', fontsize=16, fontweight='bold')
        
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
        
        # Sequence Metrics (if available)
        if has_seq_metrics:
            ax = axes[1, 2]
            seq_epochs = range(len(history['val_seq_metrics']))
            mse_vals = [m['mse'] for m in history['val_seq_metrics'] if 'mse' in m]
            if mse_vals:
                ax.plot(seq_epochs[:len(mse_vals)], mse_vals, label='MSE', color='darkorange', linewidth=2, linestyle='--')
            ax.set_title('Sequence Metrics (MSE)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'loss_trends.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
    
    def save_trajectory_plot(self, cond_x: Optional[jnp.ndarray] = None, num_trajectories: int = 20, num_steps: int = 20, prng_key: Optional[jr.PRNGKey] = None, output_dir: str = None):
        """Generate and plot latent z trajectories during integration for sequence data."""
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
                if cond_x is None:
                    raise ValueError("cond_x must be provided for conditional generation")
                cond_subset = cond_x[:n_samples]
                traj = self.model.predict(
                    self.params,
                    cond_subset[i:i+1],  # Single condition with batch dim
                    num_steps=num_steps,
                    integration_method=integration_method,
                    output_type="trajectory",
                    prng_key=prng_keys[i]
                )
            
            # For sequences, trajectories are [num_steps, 1, seq_len, embed_dim]
            # We'll flatten to show sequence evolution: [num_steps, seq_len * embed_dim]
            if traj.ndim >= 4:
                # Flatten sequence dimensions: [num_steps, 1, seq_len, embed_dim] -> [num_steps, seq_len * embed_dim]
                traj = traj.reshape(traj.shape[0], -1)
            elif traj.ndim == 3:
                traj = traj[:, 0, :]  # Remove batch dim
            trajectories.append(np.array(traj))
        
        trajectories = np.array(trajectories)  # [n_samples, num_steps, flattened_dim]
        
        # Plot trajectories - show first 2 principal components or first 2 dims
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use PCA or just first 2 dims for visualization
        # For simplicity, just use first 2 dimensions of flattened space
        for i in range(n_samples):
            traj = trajectories[i]  # [num_steps, flattened_dim]
            if traj.shape[1] >= 2:
                ax.plot(traj[:, 0], traj[:, 1], color='purple', alpha=0.6, linewidth=1.5)
                # Mark end point
                ax.scatter(traj[-1, 0], traj[-1, 1], color='purple', s=50, marker='s', edgecolors='black', linewidths=1, zorder=5)
        
        ax.set_title(f'Latent z Trajectories During Integration - Sequences ({n_samples} samples)', fontsize=14, fontweight='bold')
        ax.set_xlabel('z[0]', fontsize=12)
        ax.set_ylabel('z[1]', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        legend_elements = [
            Line2D([0], [0], color='purple', linewidth=2, label='Trajectory'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='End', markeredgecolor='black')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'latent_trajectories.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

