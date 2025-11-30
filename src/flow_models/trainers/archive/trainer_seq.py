"""
Minimal, professional JAX trainer for sequence generation tasks.

This trainer provides a clean, JAX-compliant interface for training
flow models on conditional and unconditional sequence generation tasks.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from typing import Dict, Any, Tuple, Optional, List
import pickle
from pathlib import Path
import numpy as np

from src.flow_models.fm import VAE_flow as FlowMatchingModel
from src.flow_models.df import VAE_flow as DiffusionModel
from src.flow_models.ct import VAE_flow as CTModel
from src.flow_models.config import Config


class SequenceTrainer:
    """Minimal trainer for conditional/unconditional sequence generation tasks."""
    
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
        
        # State
        self.params = None
        self.opt_state = None
        self.rng = jr.PRNGKey(seed)
    
    def initialize(self, x_sample: Optional[jnp.ndarray], y_sample: jnp.ndarray, z_sample: jnp.ndarray, t_sample: jnp.ndarray):
        """Initialize model parameters.
        
        Args:
            x_sample: Sample input sequence [seq_len, embed_dim] or [batch_size, seq_len, embed_dim] or None
            y_sample: Sample target sequence [seq_len, embed_dim] or [batch_size, seq_len, embed_dim]
            z_sample: Sample latent [latent_seq_len, embed_dim] or [batch_size, latent_seq_len, embed_dim]
            t_sample: Sample time [batch_size] or scalar
        """
        # Ensure we have batches with batch_size=1
        if x_sample is not None:
            if x_sample.ndim == 2:
                x_sample = x_sample[None, :, :]
            elif x_sample.shape[0] > 1:
                x_sample = x_sample[0:1]  # Use only first sample
        
        if y_sample.ndim == 2:
            y_sample = y_sample[None, :, :]
        elif y_sample.shape[0] > 1:
            y_sample = y_sample[0:1]  # Use only first sample
        
        if z_sample.ndim == 2:
            z_sample = z_sample[None, :, :]
        elif z_sample.shape[0] > 1:
            z_sample = z_sample[0:1]  # Use only first sample
        
        if t_sample.ndim == 0:
            t_sample = t_sample[None]
        elif t_sample.shape[0] > 1:
            t_sample = t_sample[0:1]  # Use only first sample
        
        self.rng, init_rng = jr.split(self.rng)
        self.params = self.model.init(init_rng, x_sample, y_sample, init_rng)
        self.opt_state = self.optimizer.init(self.params)
    
    def train_step(self, x_batch: Optional[jnp.ndarray], y_batch: jnp.ndarray, x_mask: Optional[jnp.ndarray] = None, use_dropout: bool = True) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            x_batch: Input sequences (batch, seq_len, features) or None
            y_batch: Target sequences (batch, seq_len, features)
            x_mask: Boolean mask [batch, x_seq_len] where True=valid, False=masked, or None
            use_dropout: Whether to use dropout
        """
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        self.rng, train_rng = jr.split(self.rng)
        # For unconditional generation, pass None for x_batch
        x_input = None if (self.unconditional or x_batch is None) else x_batch
        
        # Store x_mask for use in model (will be passed through loss to CRN)
        # Note: This requires the flow model to support x_mask parameter
        # For now, we'll need to modify the flow model's loss/train_step to accept x_mask
        # Use dropout-free train step when dropout is disabled for efficiency
        if use_dropout:
            self.params, self.opt_state, loss, metrics = self.model.train_step(
                self.params, x_input, y_batch, self.opt_state, self.optimizer, train_rng, training=True, x_mask=x_mask
            )
        else:
            # Use the optimized dropout-free method
            if hasattr(self.model, 'train_step_without_dropout'):
                self.params, self.opt_state, loss, metrics = self.model.train_step_without_dropout(
                    self.params, x_input, y_batch, self.opt_state, self.optimizer, train_rng, x_mask=x_mask
                )
            else:
                # Fallback to regular train_step with training=False
                self.params, self.opt_state, loss, metrics = self.model.train_step(
                    self.params, x_input, y_batch, self.opt_state, self.optimizer, train_rng, training=False, x_mask=x_mask
                )
        return metrics
    
    def _create_minibatch_with_masked_training(
        self,
        x_sequences: Optional[List[jnp.ndarray]],
        y_sequences: List[jnp.ndarray],
        indices: jnp.ndarray,
        mask_ratio: float = 0.5,
        min_visible_len: int = 1
    ) -> Tuple[Optional[jnp.ndarray], jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Create minibatch with masked training protocol.
        
        Randomly masks out entire timesteps of input sequences during training (like dropout).
        This makes the model robust to partial inputs. The mask is passed to the CRN to prevent
        attention to/from masked positions.
        
        Args:
            x_sequences: List of input sequences (fixed or variable length), None for unconditional
            y_sequences: List of target sequences (fixed length)
            indices: Batch indices to select
            mask_ratio: Ratio of timesteps to mask (0.0 = no masking, 1.0 = full masking)
            min_visible_len: Minimum number of visible timesteps (must be at least 1)
        
        Returns:
            x_batch: Input sequences (batch, seq_len, features) or None (NOT masked - mask applied in CRN)
            y_batch: Target sequences (batch, seq_len, features)
            x_mask: Boolean mask [batch, x_seq_len] where True=valid, False=masked, or None
        """
        batch_y = [y_sequences[int(i)] for i in indices]
        y_batch = jnp.array(batch_y)  # (batch, y_seq_len, features)
        
        # Handle unconditional case
        if self.unconditional or x_sequences is None:
            return None, y_batch, None
        
        batch_x = [x_sequences[int(i)] for i in indices]
        x_batch = jnp.array(batch_x)  # (batch, x_seq_len, features)
        
        # Generate random boolean mask for entire timesteps (like dropout)
        x_mask = None
        if mask_ratio > 0.0:
            self.rng, mask_rng = jr.split(self.rng)
            batch_size, x_seq_len, features = x_batch.shape
            
            # Randomly mask entire timesteps
            # Each timestep has probability (1 - mask_ratio) of being visible
            # But ensure at least min_visible_len timesteps are visible
            mask_probs = jr.uniform(mask_rng, (batch_size, x_seq_len))
            
            # Create mask: True = visible, False = masked
            # Start with all timesteps having probability (1 - mask_ratio) of being visible
            x_mask = mask_probs > mask_ratio  # (batch, x_seq_len)
            
            # Ensure at least min_visible_len timesteps are visible per sample
            num_visible = x_mask.sum(axis=1)  # (batch,)
            min_visible_mask = num_visible < min_visible_len
            
            if jnp.any(min_visible_mask):
                # For samples with too few visible timesteps, force the last min_visible_len to be visible
                for i in range(batch_size):
                    if num_visible[i] < min_visible_len:
                        # Force last min_visible_len timesteps to be visible
                        x_mask = x_mask.at[i, -min_visible_len:].set(True)
        
        return x_batch, y_batch, x_mask
    
    def train_epoch(
        self,
        x_sequences: Optional[List[jnp.ndarray]],
        y_sequences: List[jnp.ndarray],
        batch_size: int = 256,
        use_dropout: bool = True,
        mask_ratio: float = 0.5,
        min_visible_len: int = 1
    ) -> Dict[str, float]:
        """Train for one epoch with masked training protocol."""
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        num_samples = len(y_sequences)
        if x_sequences is not None:
            assert len(x_sequences) == num_samples, "x_sequences and y_sequences must have same length"
        
        # Shuffle
        self.rng, shuf = jr.split(self.rng)
        perm = jr.permutation(shuf, num_samples)
        
        # Train over batches
        total_losses = []
        flow_losses = []
        recon_losses = []
        reg_losses = []
        vae_losses = []
        kl_z0_losses = []
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = perm[start:end]
            x_batch, y_batch, x_mask = self._create_minibatch_with_masked_training(
                x_sequences, y_sequences, batch_indices,
                mask_ratio=mask_ratio, min_visible_len=min_visible_len
            )
            metrics = self.train_step(x_batch, y_batch, x_mask=x_mask, use_dropout=use_dropout)
            
            total_losses.append(float(metrics.get('total_loss', 0.0)))
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
        x_sequences: Optional[List[jnp.ndarray]],
        y_sequences: Optional[List[jnp.ndarray]],
        num_epochs: int = 50,
        batch_size: int = 256,
        validation_data: Optional[Tuple[List[jnp.ndarray], List[jnp.ndarray]]] = None,
        dropout_epochs: Optional[int] = None,
        mask_ratio: float = 0.5,
        min_visible_len: int = 1
    ) -> Dict[str, Any]:
        """Train the model with masked training protocol.
        
        Args:
            x_sequences: List of input sequences (can be None for unconditional)
            y_sequences: List of target sequences
            num_epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Optional tuple of (val_x_sequences, val_y_sequences)
            dropout_epochs: Number of epochs to use dropout
            mask_ratio: Ratio of input sequence to mask during training (0.0 = no masking)
            min_visible_len: Minimum number of visible timesteps in masked sequences
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
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
            'val_seq_metrics': []
        }
        
        for epoch in range(num_epochs):
            use_dropout = epoch < dropout_epochs
            metrics = self.train_epoch(
                x_sequences, y_sequences, batch_size, use_dropout,
                mask_ratio=mask_ratio, min_visible_len=min_visible_len
            )
            
            history['train_losses'].append(metrics['total_loss'])
            history['train_flow_losses'].append(metrics['flow_loss'])
            history['train_recon_losses'].append(metrics['recon_loss'])
            history['train_reg_losses'].append(metrics.get('reg_loss', 0.0))
            history['train_vae_losses'].append(metrics.get('vae_loss', 0.0))
            history['train_kl_z0_losses'].append(metrics.get('kl_z0_loss', 0.0))
            
            if validation_data is not None:
                val_x, val_y = validation_data
                val_metrics = self.evaluate(val_x, val_y, batch_size, mask_ratio=0.0)  # No masking for validation
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_flow_losses'].append(val_metrics['flow_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
                history['val_reg_losses'].append(val_metrics.get('reg_loss', 0.0))
                history['val_vae_losses'].append(val_metrics.get('vae_loss', 0.0))
                history['val_kl_z0_losses'].append(val_metrics.get('kl_z0_loss', 0.0))
                
                # Compute sequence metrics: generate samples and compare with real validation data
                # Compute metrics every epoch for better tracking (was: every 10 epochs)
                if True:  # Always compute metrics (changed from: epoch % 10 == 0 or epoch == num_epochs - 1)
                    num_eval_samples = min(100, len(val_y))
                    self.rng, gen_rng = jr.split(self.rng)
                    # Randomly select validation sequences
                    eval_indices = jr.permutation(gen_rng, len(val_y))[:num_eval_samples]
                    eval_x, eval_y_real, _ = self._create_minibatch_with_masked_training(
                        val_x, val_y, eval_indices, mask_ratio=0.0  # No masking for generation
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
    
    def evaluate(
        self,
        x_sequences: Optional[List[jnp.ndarray]],
        y_sequences: List[jnp.ndarray],
        batch_size: int = 256,
        mask_ratio: float = 0.0
    ) -> Dict[str, float]:
        """Evaluate the model with masked sequences."""
        if self.params is None:
            raise ValueError("Model not initialized.")
        
        num_samples = len(y_sequences)
        if x_sequences is not None:
            assert len(x_sequences) == num_samples, "x_sequences and y_sequences must have same length"
        
        metrics_sum = {
            'total_loss': 0.0,
            'flow_loss': 0.0,
            'recon_loss': 0.0,
            'reg_loss': 0.0,
            'vae_loss': 0.0,
            'kl_z0_loss': 0.0
        }
        num_batches = 0
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = jnp.arange(start, end)
            x_batch, y_batch, x_mask = self._create_minibatch_with_masked_training(
                x_sequences, y_sequences, batch_indices, mask_ratio=mask_ratio
            )
            self.rng, eval_rng = jr.split(self.rng)
            x_input = None if (self.unconditional or x_batch is None) else x_batch
            # Note: x_mask would need to be passed to model.loss if it supports it
            # For now, evaluation uses mask_ratio=0.0 so x_mask will be all True or None
            _, metrics = self.model.loss(self.params, x_input, y_batch, eval_rng, training=False, x_mask=x_mask)
            
            for key in metrics_sum:
                metrics_sum[key] += float(metrics.get(key, 0.0))
            num_batches += 1
        
        # Average metrics
        return {key: val / num_batches if num_batches > 0 else 0.0 for key, val in metrics_sum.items()}
    
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
        prng_key: Optional[jr.PRNGKey] = None
    ) -> jnp.ndarray:
        """Generate y sequences conditioned on x sequences."""
        if self.params is None:
            raise ValueError("Model not initialized.")
        if self.unconditional:
            raise ValueError("Use unconditional_generate() for unconditional generation")
        if prng_key is None:
            self.rng, prng_key = jr.split(self.rng)
        return self.model.predict(self.params, cond_x, num_steps, "midpoint", "end_point", prng_key=prng_key)
    
    def unconditional_generate(
        self,
        batch_shape: Tuple[int, ...],
        num_steps: int = 20,
        prng_key: Optional[jr.PRNGKey] = None
    ) -> jnp.ndarray:
        """Generate y sequences unconditionally."""
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
        if self.params is None:
            raise ValueError("Model not initialized. No parameters to save.")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(jax.device_get(self.params), f)
    
    def save_results(self, history: Dict[str, Any], output_dir: str, y_real: Optional[np.ndarray] = None,
                     y_gen: Optional[np.ndarray] = None, x_labels: Optional[np.ndarray] = None):
        """Save results and create plots.
        
        Args:
            history: Training history dictionary
            output_dir: Directory to save results
            y_real: Real sequences for generation plot [optional]
            y_gen: Generated sequences for generation plot [optional]
            x_labels: Labels for conditional generation plot [optional]
        """
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
            from src.utils.plotting.plot_loss_trends import create_loss_trends_plot
            from src.utils.plotting.plot_lorenz_sequence_comparison import plot_lorenz_sequence_comparison
            try:
                from experiments.stock_prediction.plotting import plot_latent_trajectories
            except ImportError:
                # Fallback if experiments module doesn't exist
                plot_latent_trajectories = None
            
            # Loss trends plot
            create_loss_trends_plot(history, self.model_type, output_dir)
            
            # Sequence comparison plot (if data provided)
            if y_gen is not None and y_real is not None:
                # Check if this is Lorenz data (3D coordinates)
                if y_real.shape[-1] == 3 and y_gen.shape[-1] == 3:
                    # Use Lorenz-specific plot
                    plot_lorenz_sequence_comparison(y_real, y_gen, output_dir, num_samples=12)
            
            # Latent trajectories plot (if available)
            if plot_latent_trajectories is not None:
                self.rng, traj_rng = jr.split(self.rng)
                cond_x = None if self.unconditional else (x_labels[:20] if x_labels is not None else None)
                plot_latent_trajectories(
                    model=self.model,
                    params=self.params,
                    model_type=self.model_type,
                    unconditional=self.unconditional,
                    output_dir=output_dir,
                    cond_x=cond_x,
                    num_trajectories=20,
                    num_steps=20,
                    prng_key=None,  # Not used - always use MLE (predict with no prng_key)
                    rng=None  # Not used - always use MLE (predict with no prng_key)
                )
        except ImportError:
            pass
        except Exception as e:
            import traceback
            print(f"Warning: Error creating plots: {e}")
            traceback.print_exc()
