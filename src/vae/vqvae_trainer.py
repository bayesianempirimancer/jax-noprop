"""Trainer for Vector Quantized VAE (VQ-VAE) model."""

from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from typing import Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import traceback

from src.vae.vqvae import VQVAE, VQVAEConfig
from flax import traverse_util

# Disable JAX optimizations that can cause slowdowns
jax.config.update('jax_disable_jit', False)


class VQVAETrainer:
    """Trainer for Vector Quantized VAE model."""
    
    def __init__(
        self,
        config: VQVAEConfig,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        seed: int = 42
    ):
        """
        Initialize VQ-VAE trainer.
        
        Args:
            config: VQVAEConfig configuration
            learning_rate: Learning rate for optimizer
            optimizer_name: Name of optimizer ("adam" or "sgd")
            seed: Random seed
        """
        self.config = config
        self.model = VQVAE(config=config)
        
        # Create optimizer
        if optimizer_name.lower() == "adam":
            self.optimizer = optax.adam(learning_rate)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optax.sgd(learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Initialize state
        self.params = None
        self.opt_state = None
        self.rng = jr.PRNGKey(seed)
    
    def initialize(self, x_sample: jnp.ndarray, x_data: Optional[jnp.ndarray] = None):
        """
        Initialize model parameters and optimizer state.
        
        Args:
            x_sample: Sample input data [batch_size, *input_shape] for parameter initialization
            x_data: Optional full dataset for codebook initialization [num_samples, *input_shape]
        """
        self.rng, init_rng = jr.split(self.rng)
        # Use the model's __call__ method for initialization
        self.params = self.model.init(init_rng, x_sample, init_rng)
        
        # Initialize codebook from random data points
        if x_data is not None:
            print("Initializing codebook from random data points...")
            self._initialize_codebook_from_data(x_data)
        else:
            print("Warning: No full dataset provided for codebook initialization, using default init")
        
        self.opt_state = self.optimizer.init(self.params)
        
        num_params = sum(x.size for x in jax.tree_leaves(self.params))
        print(f"Model initialized with {num_params:,} parameters")
        
        # Debug: print parameter tree summary
        try:
            flat_params = traverse_util.flatten_dict(self.params, keep_empty_nodes=True)
            print("Parameter tree summary (path: shape dtype):")
            for path, value in list(flat_params.items())[:10]:  # Show first 10
                if hasattr(value, 'shape'):
                    key_path = "/".join(str(k) for k in path)
                    print(f"  {key_path}: {tuple(value.shape)} {value.dtype}")
            if len(flat_params) > 10:
                print(f"  ... and {len(flat_params) - 10} more parameters")
        except Exception as e:
            print(f"Warning: could not summarize parameter tree: {e}")
    
    def _initialize_codebook_from_data(self, x_data: jnp.ndarray):
        """
        Initialize codebook embeddings by sampling random data points and encoding them.
        
        Args:
            x_data: Full dataset [num_samples, *input_shape]
        """
        codebook_size = self.config.main.get("codebook_size", 512)
        embedding_dim = self.config.main.get("embedding_dim", 32)
        
        # Sample random indices from dataset
        n_samples = x_data.shape[0]
        n_samples_to_use = min(codebook_size, n_samples)
        
        self.rng, sample_rng = jr.split(self.rng)
        sample_indices = jr.choice(sample_rng, n_samples, shape=(n_samples_to_use,), replace=False)
        x_sampled = x_data[sample_indices]  # [n_samples_to_use, *input_shape]
        
        # Encode sampled data points
        self.rng, encode_rng = jr.split(self.rng)
        z_e, _, _ = self.model.apply(self.params, x_sampled, method='encode', training=False)
        
        # Flatten encoder outputs (handle multi-dimensional spatial dimensions)
        z_e_flat = z_e.reshape(-1, z_e.shape[-1])  # [N, embedding_dim]
        
        # If we have fewer samples than codebook size, pad with additional random samples
        if z_e_flat.shape[0] < codebook_size:
            n_additional = codebook_size - z_e_flat.shape[0]
            # Sample more data points
            self.rng, sample_rng2 = jr.split(self.rng)
            remaining_indices = jnp.setdiff1d(jnp.arange(n_samples), sample_indices)
            if len(remaining_indices) > 0:
                n_to_sample = min(n_additional, len(remaining_indices))
                additional_indices = jr.choice(sample_rng2, remaining_indices, shape=(n_to_sample,), replace=False)
                x_additional = x_data[additional_indices]
                
                # Encode additional samples
                self.rng, encode_rng2 = jr.split(self.rng)
                z_e_additional, _, _ = self.model.apply(self.params, x_additional, method='encode', training=False)
                z_e_additional_flat = z_e_additional.reshape(-1, z_e_additional.shape[-1])
                
                z_e_flat = jnp.vstack([z_e_flat, z_e_additional_flat])
            
            # If still not enough, pad with random vectors from the distribution
            if z_e_flat.shape[0] < codebook_size:
                n_remaining = codebook_size - z_e_flat.shape[0]
                z_e_mean = jnp.mean(z_e_flat, axis=0)
                z_e_std = jnp.std(z_e_flat, axis=0) + 1e-6
                self.rng, noise_rng = jr.split(self.rng)
                z_e_random = z_e_mean + z_e_std * jr.normal(noise_rng, (n_remaining, embedding_dim))
                z_e_flat = jnp.vstack([z_e_flat, z_e_random])
        
        # Truncate to exact codebook size
        codebook_init = z_e_flat[:codebook_size]
        
        # Update codebook in params
        from flax.core import freeze, unfreeze
        params_unfrozen = unfreeze(self.params)
        params_unfrozen['params']['vq']['embedding'] = codebook_init
        self.params = freeze(params_unfrozen)
        
        print(f"  Codebook initialized with {codebook_init.shape[0]} embeddings from random data points")
    
    def _reset_unused_codebook_entries(self, x_data: jnp.ndarray, used_indices: np.ndarray):
        """
        Reset unused codebook entries by sampling random data points.
        
        Args:
            x_data: Full dataset [num_samples, *input_shape]
            used_indices: Array of codebook indices that are being used
        """
        codebook_size = self.config.main.get("codebook_size", 512)
        embedding_dim = self.config.main.get("embedding_dim", 32)
        
        # Get unique used indices
        used_set = set(np.unique(used_indices).tolist())
        unused_indices = [i for i in range(codebook_size) if i not in used_set]
        
        if len(unused_indices) == 0:
            return  # All codebook entries are being used
        
        print(f"  Resetting {len(unused_indices)} unused codebook entries...")
        
        # Sample random data points (limit to reasonable batch size)
        n_samples = x_data.shape[0]
        n_to_sample = min(len(unused_indices), 512)  # Limit batch size
        
        self.rng, sample_rng = jr.split(self.rng)
        sample_indices = jr.choice(sample_rng, n_samples, shape=(n_to_sample,), replace=True)
        x_sampled = x_data[sample_indices]
        
        # Encode sampled data points
        self.rng, encode_rng = jr.split(self.rng)
        z_e, _, _ = self.model.apply(self.params, x_sampled, method='encode', training=False)
        z_e_flat = z_e.reshape(-1, z_e.shape[-1])  # [n_to_sample, embedding_dim]
        
        # Truncate to number of unused indices
        if z_e_flat.shape[0] > len(unused_indices):
            z_e_flat = z_e_flat[:len(unused_indices)]
        
        # Update unused codebook entries
        from flax.core import freeze, unfreeze
        params_unfrozen = unfreeze(self.params)
        codebook = np.array(params_unfrozen['params']['vq']['embedding'])  # Convert to numpy for easier indexing
        
        # Update unused entries
        for i, unused_idx in enumerate(unused_indices):
            if i < z_e_flat.shape[0]:
                codebook[unused_idx] = np.array(z_e_flat[i])
        
        # Convert back to JAX array and update params
        params_unfrozen['params']['vq']['embedding'] = jnp.array(codebook)
        self.params = freeze(params_unfrozen)
    
    @partial(jax.jit, static_argnums=(0, 5))
    def train_step(self, params: dict, x_batch: jnp.ndarray, opt_state: dict, key: jr.PRNGKey, training: bool = True) -> Tuple[dict, dict, jnp.ndarray, dict]:
        """
        Single training step.
        
        Args:
            params: Model parameters
            x_batch: Input batch [batch_size, *input_shape]
            opt_state: Optimizer state
            key: Random key
            training: Whether in training mode
            
        Returns:
            Tuple of (params, opt_state, loss, metrics)
        """
        # Compute loss and gradients
        def loss_fn(params):
            return self.model.loss(params, x_batch, key, training=training)
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        
        # Update parameters using optimizer
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, metrics
    
    def train_epoch(self, x_data: jnp.ndarray, batch_size: int = 256, use_dropout: bool = True) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            x_data: Training input data [num_samples, *input_shape]
            batch_size: Batch size for training
            use_dropout: Whether to use dropout during training
            
        Returns:
            Dictionary of epoch metrics
        """
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        num_samples = x_data.shape[0]
        # Truncate to ensure all batches are exactly the same size (avoid recompilation)
        num_full_batches = num_samples // batch_size
        num_samples_used = num_full_batches * batch_size
        num_batches = num_full_batches
        
        epoch_metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'vq_loss': 0.0,
            'commitment_loss': 0.0,
            'step': num_batches,
            'codebook_usage': 0.0,
            'num_tokens_used': 0,
            'pve_by_dim': []  # Will store PVE for each dimension
        }
        
        # Shuffle data once (more efficient)
        self.rng, shuffle_rng = jr.split(self.rng)
        perm = jr.permutation(shuffle_rng, num_samples)
        x_data_shuffled = x_data[perm][:num_samples_used]  # Truncate to exact multiple of batch_size
        
        # Accumulate metrics as JAX arrays (avoid host-device sync)
        total_loss_acc = jnp.array(0.0)
        recon_loss_acc = jnp.array(0.0)
        vq_loss_acc = jnp.array(0.0)
        commitment_loss_acc = jnp.array(0.0)
        
        # Track all used indices for codebook reset (sample from multiple batches)
        all_used_indices = []
        
        # Train on batches (all batches are now exactly batch_size)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size  # Always exactly batch_size
            
            x_batch = x_data_shuffled[start_idx:end_idx]
            
            # Training step
            self.rng, train_rng = jr.split(self.rng)
            self.params, self.opt_state, loss, metrics = self.train_step(
                self.params, x_batch, self.opt_state, train_rng, training=use_dropout
            )
            
            # Accumulate metrics (keep as JAX arrays until end - no host-device sync)
            total_loss_acc = total_loss_acc + loss
            recon_loss_acc = recon_loss_acc + metrics.get('recon_loss', jnp.array(0.0))
            vq_loss_acc = vq_loss_acc + metrics.get('vq_loss', jnp.array(0.0))
            commitment_loss_acc = commitment_loss_acc + metrics.get('commitment_loss', jnp.array(0.0))
            
            # Track codebook usage (sample from every 10th batch for efficiency)
            if i % 10 == 0 and 'indices' in metrics:
                indices_batch = metrics['indices']
                indices_flat = indices_batch.flatten()
                # Convert to numpy for unique computation (outside JIT)
                indices_np = np.array(indices_flat)
                all_used_indices.append(indices_np)
                
                # Compute usage on first tracked batch
                if i == 0:
                    unique_count = len(np.unique(indices_np))
                    codebook_size = self.config.main.get("codebook_size", 512)
                    epoch_metrics['codebook_usage'] = unique_count / codebook_size
                    epoch_metrics['num_tokens_used'] = unique_count
        
        # Store all used indices for codebook reset
        if all_used_indices:
            epoch_metrics['used_indices'] = np.concatenate(all_used_indices)
        
        # Compute PVE for each dimension on a sample batch
        # Sample one batch for PVE computation (to avoid host-device sync on every batch)
        if num_batches > 0:
            sample_batch = x_data_shuffled[:batch_size]
            self.rng, recon_rng = jr.split(self.rng)
            x_recon = self.reconstruct(sample_batch)
            
            # Compute PVE for each dimension
            # Data shape is [batch, seq_len, feature_dim] or [batch, feature_dim]
            # Extract each feature dimension
            x_true = np.array(sample_batch)
            x_recon_np = np.array(x_recon)
            
            # Handle different input shapes
            if len(x_true.shape) == 3:  # [batch, seq_len, feature_dim]
                num_dims = x_true.shape[2]
                pve_by_dim = []
                for dim in range(num_dims):
                    x_true_dim = x_true[:, :, dim].flatten()  # Flatten to 1D
                    x_recon_dim = x_recon_np[:, :, dim].flatten()
                    pve = self._compute_pve(x_true_dim, x_recon_dim)
                    pve_by_dim.append(pve)
            elif len(x_true.shape) == 2:  # [batch, feature_dim]
                num_dims = x_true.shape[1]
                pve_by_dim = []
                for dim in range(num_dims):
                    x_true_dim = x_true[:, dim]
                    x_recon_dim = x_recon_np[:, dim]
                    pve = self._compute_pve(x_true_dim, x_recon_dim)
                    pve_by_dim.append(pve)
            else:
                # Flatten and compute overall PVE
                x_true_flat = x_true.flatten()
                x_recon_flat = x_recon_np.flatten()
                pve_overall = self._compute_pve(x_true_flat, x_recon_flat)
                pve_by_dim = [pve_overall]
            
            epoch_metrics['pve_by_dim'] = pve_by_dim
        
        # Convert to Python float only once at the end (single host-device sync)
        epoch_metrics['total_loss'] = float(total_loss_acc) / num_batches
        epoch_metrics['recon_loss'] = float(recon_loss_acc) / num_batches
        epoch_metrics['vq_loss'] = float(vq_loss_acc) / num_batches
        epoch_metrics['commitment_loss'] = float(commitment_loss_acc) / num_batches
        
        return epoch_metrics
    
    def _compute_pve(self, x_true: np.ndarray, x_recon: np.ndarray) -> float:
        """
        Compute Percent Variance Explained (PVE) = R² * 100.
        
        Args:
            x_true: True values [N]
            x_recon: Reconstructed values [N]
            
        Returns:
            PVE as percentage (0-100)
        """
        # Compute sum of squared residuals
        ss_res = np.sum((x_true - x_recon) ** 2)
        
        # Compute total sum of squares
        x_mean = np.mean(x_true)
        ss_tot = np.sum((x_true - x_mean) ** 2)
        
        # Avoid division by zero
        if ss_tot > 1e-10:
            r2 = 1.0 - (ss_res / ss_tot)
            pve = r2 * 100.0
        else:
            # If variance is zero, PVE is undefined
            pve = float('nan')
        
        return pve
    
    def train(
        self,
        x_data: jnp.ndarray,
        num_epochs: int = 100,
        batch_size: int = 256,
        validation_data: Optional[jnp.ndarray] = None,
        dropout_epochs: Optional[int] = None,
        verbose: bool = True,
        reset_unused_codes: bool = True
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            x_data: Training input data [num_samples, *input_shape]
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Optional validation data [num_val_samples, *input_shape]
            dropout_epochs: Number of epochs with dropout (if None, uses all epochs)
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training history
        """
        if self.params is None or self.opt_state is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        # Set dropout epochs
        if dropout_epochs is None:
            dropout_epochs = num_epochs
        
        # Initialize history
        history = {
            'train_losses': [],
            'train_recon_losses': [],
            'train_vq_losses': [],
            'train_commitment_losses': [],
            'val_losses': [],
            'val_recon_losses': [],
            'val_vq_losses': [],
            'val_commitment_losses': [],
            'train_codebook_usage': [],
            'train_num_tokens_used': [],
            'train_pve_by_dim': [],  # List of lists, each inner list is PVE for each dimension
            'val_pve_by_dim': [],
        }
        
        if verbose:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Dropout epochs: {dropout_epochs}")
            print(f"Training data shape: {x_data.shape}")
            if validation_data is not None:
                print(f"Validation data shape: {validation_data.shape}")
        
        # Training loop
        # Note: We always use training=True in train_step (static arg), and evaluation
        # uses a separate _eval_batch function with training=False hardcoded.
        # This ensures no recompilation when switching between train and eval modes.
        for epoch in tqdm(range(num_epochs), desc="Training", disable=not verbose):
            # Determine if we should use dropout
            use_dropout = epoch < dropout_epochs
            
            # Train epoch
            train_metrics = self.train_epoch(x_data, batch_size, use_dropout)
            
            # Store training metrics
            history['train_losses'].append(train_metrics['total_loss'])
            history['train_recon_losses'].append(train_metrics['recon_loss'])
            history['train_vq_losses'].append(train_metrics['vq_loss'])
            history['train_commitment_losses'].append(train_metrics['commitment_loss'])
            
            # Track codebook usage if available
            if 'train_codebook_usage' not in history:
                history['train_codebook_usage'] = []
                history['train_num_tokens_used'] = []
            history['train_codebook_usage'].append(train_metrics.get('codebook_usage', 0.0))
            history['train_num_tokens_used'].append(train_metrics.get('num_tokens_used', 0))
            
            # Track PVE by dimension
            if 'train_pve_by_dim' not in history:
                history['train_pve_by_dim'] = []
            history['train_pve_by_dim'].append(train_metrics.get('pve_by_dim', []))
            
            # Reset unused codebook entries after each epoch
            if reset_unused_codes and 'used_indices' in train_metrics:
                used_indices = train_metrics['used_indices']
                self._reset_unused_codebook_entries(x_data, used_indices)
            
            # Validation (only every 10 epochs to save time and avoid recompilation overhead)
            if validation_data is not None and (epoch % 10 == 0 or epoch == num_epochs - 1):
                val_metrics = self.evaluate(validation_data, batch_size)
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
                history['val_vq_losses'].append(val_metrics['vq_loss'])
                history['val_commitment_losses'].append(val_metrics['commitment_loss'])
                history['val_pve_by_dim'].append(val_metrics.get('pve_by_dim', []))
                
                if verbose:
                    tokens_used = train_metrics.get('num_tokens_used', 0)
                    codebook_usage = train_metrics.get('codebook_usage', 0.0)
                    pve_str = ""
                    if val_metrics.get('pve_by_dim'):
                        pve_vals = val_metrics['pve_by_dim']
                        pve_str = ", PVE=" + ", ".join([f"{p:.1f}%" if np.isfinite(p) else "N/A" for p in pve_vals])
                    print(f"Epoch {epoch}: train_loss={train_metrics['total_loss']:.4f}, "
                          f"val_loss={val_metrics['total_loss']:.4f}, "
                          f"tokens_used={tokens_used}, usage={codebook_usage:.3f}{pve_str}")
            elif validation_data is not None:
                # Append previous validation loss to keep list lengths consistent
                if len(history['val_losses']) > 0:
                    history['val_losses'].append(history['val_losses'][-1])
                    history['val_recon_losses'].append(history['val_recon_losses'][-1])
                    history['val_vq_losses'].append(history['val_vq_losses'][-1])
                    history['val_commitment_losses'].append(history['val_commitment_losses'][-1])
                    if len(history['val_pve_by_dim']) > 0:
                        history['val_pve_by_dim'].append(history['val_pve_by_dim'][-1])
                    else:
                        history['val_pve_by_dim'].append([])
        
        if verbose:
            print("Training completed!")
        
        return history
    
    @partial(jax.jit, static_argnums=(0,))
    def _eval_batch(self, params: dict, x_batch: jnp.ndarray, key: jr.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """JIT-compiled evaluation of a single batch."""
        loss, metrics = self.model.loss(params, x_batch, key, training=False)
        return loss, metrics.get('recon_loss', jnp.array(0.0)), metrics.get('vq_loss', jnp.array(0.0)), metrics.get('commitment_loss', jnp.array(0.0))
    
    def evaluate(self, x_data: jnp.ndarray, batch_size: int = 256) -> Dict[str, float]:
        """
        Evaluate the model on given data.
        
        Args:
            x_data: Input data [num_samples, *input_shape]
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics including PVE by dimension
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        num_samples = x_data.shape[0]
        # Truncate to ensure all batches are exactly the same size (avoid recompilation)
        num_full_batches = num_samples // batch_size
        num_samples_used = num_full_batches * batch_size
        num_batches = num_full_batches
        
        # Truncate data to exact multiple of batch_size
        x_data_eval = x_data[:num_samples_used]
        
        # Accumulate as JAX arrays (avoid host-device sync)
        total_loss_acc = jnp.array(0.0)
        recon_loss_acc = jnp.array(0.0)
        vq_loss_acc = jnp.array(0.0)
        commitment_loss_acc = jnp.array(0.0)
        
        # Evaluate on batches (all batches are now exactly batch_size)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size  # Always exactly batch_size
            
            x_batch = x_data_eval[start_idx:end_idx]
            
            # Use JIT-compiled evaluation
            self.rng, eval_rng = jr.split(self.rng)
            loss, recon_loss, vq_loss, commitment_loss = self._eval_batch(self.params, x_batch, eval_rng)
            
            # Accumulate as JAX arrays (no host-device sync)
            total_loss_acc = total_loss_acc + loss
            recon_loss_acc = recon_loss_acc + recon_loss
            vq_loss_acc = vq_loss_acc + vq_loss
            commitment_loss_acc = commitment_loss_acc + commitment_loss
        
        # Compute PVE for each dimension on a sample batch
        # Use first batch for PVE computation
        if num_batches > 0:
            sample_batch = x_data_eval[:batch_size]
            x_recon = self.reconstruct(sample_batch)
            
            # Compute PVE for each dimension
            x_true = np.array(sample_batch)
            x_recon_np = np.array(x_recon)
            
            # Handle different input shapes
            if len(x_true.shape) == 3:  # [batch, seq_len, feature_dim]
                num_dims = x_true.shape[2]
                pve_by_dim = []
                for dim in range(num_dims):
                    x_true_dim = x_true[:, :, dim].flatten()
                    x_recon_dim = x_recon_np[:, :, dim].flatten()
                    pve = self._compute_pve(x_true_dim, x_recon_dim)
                    pve_by_dim.append(pve)
            elif len(x_true.shape) == 2:  # [batch, feature_dim]
                num_dims = x_true.shape[1]
                pve_by_dim = []
                for dim in range(num_dims):
                    x_true_dim = x_true[:, dim]
                    x_recon_dim = x_recon_np[:, dim]
                    pve = self._compute_pve(x_true_dim, x_recon_dim)
                    pve_by_dim.append(pve)
            else:
                x_true_flat = x_true.flatten()
                x_recon_flat = x_recon_np.flatten()
                pve_overall = self._compute_pve(x_true_flat, x_recon_flat)
                pve_by_dim = [pve_overall]
        else:
            pve_by_dim = []
        
        # Convert to Python float only once at the end (single host-device sync)
        avg_loss = float(total_loss_acc) / num_batches
        avg_recon_loss = float(recon_loss_acc) / num_batches
        avg_vq_loss = float(vq_loss_acc) / num_batches
        avg_commitment_loss = float(commitment_loss_acc) / num_batches
        
        return {
            'total_loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'vq_loss': avg_vq_loss,
            'commitment_loss': avg_commitment_loss,
            'pve_by_dim': pve_by_dim
        }
    
    def encode(self, x_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Encode input data to discrete tokens.
        
        Args:
            x_data: Input data [num_samples, *input_shape]
            
        Returns:
            Tuple of (z_e, z_q, indices) where:
            - z_e: Encoder output (continuous) [num_samples, ..., embedding_dim]
            - z_q: Quantized vectors [num_samples, ..., embedding_dim]
            - indices: Discrete token indices [num_samples, ...]
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        return self.model.apply(self.params, x_data, method='encode', training=False)
    
    def decode(self, z_q: jnp.ndarray) -> jnp.ndarray:
        """
        Decode quantized vectors to output space.
        
        Args:
            z_q: Quantized vectors [num_samples, ..., embedding_dim]
            
        Returns:
            Reconstructed output [num_samples, *output_shape]
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        return self.model.apply(self.params, z_q, method='decode', training=False)
    
    def reconstruct(self, x_data: jnp.ndarray) -> jnp.ndarray:
        """
        Reconstruct input data by encoding and decoding.
        
        Args:
            x_data: Input data [num_samples, *input_shape]
            
        Returns:
            Reconstructed output [num_samples, *output_shape]
        """
        # Encode and quantize
        z_e, z_q, indices = self.encode(x_data)
        # Decode
        return self.decode(z_q)
    
    def save_params(self, filepath: str):
        """Save model parameters and config to file."""
        if self.params is None:
            raise ValueError("Model not initialized. Cannot save.")
        
        save_dict = {
            'params': self.params,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load_params(self, filepath: str):
        """Load model parameters and config from file."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.params = save_dict['params']
        self.config = save_dict['config']
        self.model = VQVAE(config=self.config)
    
    def save_results(self, history: Dict[str, list], save_dir: str):
        """
        Save training results including history, parameters, and plots.
        
        Args:
            history: Training history dictionary
            save_dir: Directory to save results
        """
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save training history
        history_path = save_dir_path / 'training_results.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        
        # Save parameters
        params_path = save_dir_path / 'model_params.pkl'
        self.save_params(params_path)
        
        # Save config as YAML using BaseConfig method
        if hasattr(self.config, 'save_yaml'):
            self.config.save_yaml(save_dir_path / "config.yaml")
            print(f"Config saved to {save_dir_path / 'config.yaml'}")
        elif hasattr(self.config, 'save_json'):
            self.config.save_json(save_dir_path / "config.json")
            print(f"Config saved to {save_dir_path / 'config.json'}")
        
        # Create and save training progress plot
        self._plot_training_progress(history, save_dir_path / 'training_progress.png')
    
    def _plot_training_progress(self, history: Dict[str, list], save_path: Path):
        """Create training progress plot."""
        from src.utils.plotting.plot_vae_loss_trends import create_vae_loss_trends_plot
        create_vae_loss_trends_plot(history, save_path.parent, save_name=save_path.name)

