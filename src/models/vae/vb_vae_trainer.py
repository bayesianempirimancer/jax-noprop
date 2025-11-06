"""Trainer for Variational Bayesian VAE (VBVAE) model."""

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
from flax.core import freeze, unfreeze

from src.models.vae.vb_vae import VBVAE, VBVAEConfig
from flax import traverse_util

# Disable JAX optimizations that can cause slowdowns
jax.config.update('jax_disable_jit', False)


class VBVAETrainer:
    """Trainer for Variational Bayesian VAE model.
    
    This trainer handles two types of parameter updates:
    1. Encoder/Decoder parameters: Updated via gradient descent (standard optimizer)
    2. GMM-VBEM parameters: Updated via Variational Bayesian EM updates (not gradient descent)
    """
    
    def __init__(
        self,
        config: VBVAEConfig,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        N_eff: Optional[float] = None,  # Effective number of data points (inverse temperature). If None, defaults to total number of training data points.
        seed: int = 42
    ):
        """
        Initialize VBVAE trainer.
        
        Args:
            config: VBVAEConfig configuration
            learning_rate: Learning rate for encoder/decoder optimizer
            optimizer_name: Name of optimizer ("adam" or "sgd")
            N_eff: Effective number of data points (inverse temperature). If None, defaults to total number of training data points.
            seed: Random seed
        """
        self.config = config
        # Set prior_beta to 0.1 in config
        config_main = unfreeze(config.main)
        config_main["prior_beta"] = 0.1
        config.main = freeze(config_main)
        self.model = VBVAE(config=config)
        self.N_eff = N_eff
        
        # Create optimizer for encoder/decoder only (GMM params updated separately)
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
    
    def initialize(self, x_sample: jnp.ndarray):
        """
        Initialize model parameters and optimizer state.
        
        Args:
            x_sample: Sample input data [batch_size, *input_shape] for parameter initialization
        """
        self.rng, init_rng = jr.split(self.rng)
        # Use the model's __call__ method for initialization
        self.params = self.model.init(init_rng, x_sample, init_rng)
        
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
    
    def _separate_gmm_params(self, params: dict) -> Tuple[dict, dict]:
        """
        Separate encoder/decoder parameters from GMM parameters.
        
        Args:
            params: Full parameter dictionary
            
        Returns:
            Tuple of (encoder_decoder_params, gmm_params)
        """
        params_unfrozen = unfreeze(params)
        
        # Extract GMM params
        gmm_params = params_unfrozen['params'].pop('gmm_vbem')
        
        # Remaining params are encoder/decoder
        encoder_decoder_params = freeze(params_unfrozen)
        
        return encoder_decoder_params, gmm_params
    
    def _combine_params(self, encoder_decoder_params: dict, gmm_params: dict) -> dict:
        """
        Combine encoder/decoder parameters with GMM parameters.
        
        Args:
            encoder_decoder_params: Encoder/decoder parameters
            gmm_params: GMM parameters
            
        Returns:
            Combined parameter dictionary
        """
        params_unfrozen = unfreeze(encoder_decoder_params)
        params_unfrozen['params']['gmm_vbem'] = gmm_params
        return freeze(params_unfrozen)
    
    def _update_gmm_vbem(
        self,
        z_e_batch: jnp.ndarray,
        cluster_probs_batch: jnp.ndarray,
        logZ_batch: jnp.ndarray,
        gmm_params: dict,
        N_eff: float
    ) -> dict:
        """
        Update GMM parameters using Variational Bayesian EM.
        
        Args:
            z_e_batch: Encoder outputs [batch, ..., latent_dim]
            cluster_probs_batch: Cluster probabilities [batch, ..., num_clusters]
            gmm_params: Current GMM parameters
            N_eff: Effective number of data points (inverse temperature)
            
        Returns:
            Updated GMM parameters
        """
        # Call the GMM-VBEM update method
        # Prior parameters are accessed via self in the GMMVBEM class
        updated_gmm_params = self.model.gmm_vbem.update(
            params=gmm_params,
            z_e=z_e_batch,
            cluster_probs=cluster_probs_batch,
            logZ=logZ_batch,
            N_eff=N_eff
        )
        
        return updated_gmm_params
    
    def train_epoch(
        self,
        x_data: jnp.ndarray,
        batch_size: int = 256,
        use_dropout: bool = True,
        update_gmm_every_n_batches: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            x_data: Training input data [num_samples, *input_shape]
            batch_size: Batch size for training
            use_dropout: Whether to use dropout during training
            update_gmm_every_n_batches: Update GMM params every N batches (to reduce overhead)
            
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
            'gmm_loss': 0.0,
            'step': num_batches,
        }
        
        # Separate encoder/decoder and GMM params
        encoder_decoder_params, gmm_params = self._separate_gmm_params(self.params)
        
        # Shuffle data once (more efficient)
        self.rng, shuffle_rng = jr.split(self.rng)
        perm = jr.permutation(shuffle_rng, num_samples)
        x_data_shuffled = x_data[perm][:num_samples_used]  # Truncate to exact multiple of batch_size
        
        # Accumulate metrics and data for GMM updates
        total_loss_acc = jnp.array(0.0)
        recon_loss_acc = jnp.array(0.0)
        gmm_loss_acc = jnp.array(0.0)
        
        # Accumulate encoder outputs and cluster probs for VBEM updates
        z_e_accumulator = []
        cluster_probs_accumulator = []
        logZ_accumulator = []
        
        # Train on batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size  # Always exactly batch_size
            
            x_batch = x_data_shuffled[start_idx:end_idx]
            
            # Training step (updates encoder/decoder, keeps GMM fixed)
            self.rng, train_rng = jr.split(self.rng)
            encoder_decoder_params, gmm_params, self.opt_state, loss, metrics = self.model.train_step(
                encoder_decoder_params, gmm_params, x_batch, self.opt_state, train_rng, 
                training=use_dropout, optimizer=self.optimizer
            )
            
            # Accumulate metrics (keep as JAX arrays until end - no host-device sync)
            total_loss_acc = total_loss_acc + loss
            recon_loss_acc = recon_loss_acc + metrics.get('recon_loss', jnp.array(0.0))
            gmm_loss_acc = gmm_loss_acc + metrics.get('gmm_loss', jnp.array(0.0))
            
            # Collect encoder outputs and cluster probs for VBEM updates
            # Compute these on the current batch (with current params)
            if i % update_gmm_every_n_batches == 0:
                # Recombine params for encoding
                params_combined = self._combine_params(encoder_decoder_params, gmm_params)
                
                # Encode to get z_e and cluster_probs
                self.rng, encode_rng = jr.split(self.rng)
                z_e = self.model.apply(params_combined, x_batch, method='encode', training=False)
                
                # Get cluster probabilities and logZ
                gmm_params_for_apply = {'params': gmm_params}
                _, cluster_probs, logZ = self.model.gmm_vbem.apply(
                    gmm_params_for_apply,
                    z_e
                )
                
                # Store for VBEM update
                z_e_accumulator.append(np.array(z_e))
                cluster_probs_accumulator.append(np.array(cluster_probs))
                logZ_accumulator.append(np.array(logZ))
        
        # Update GMM parameters using accumulated data (VBEM update)
        if len(z_e_accumulator) > 0:
            z_e_combined = np.concatenate(z_e_accumulator, axis=0)
            cluster_probs_combined = np.concatenate(cluster_probs_accumulator, axis=0)
            logZ_combined = np.concatenate(logZ_accumulator, axis=0)
            
            # Convert to JAX arrays
            z_e_jax = jnp.array(z_e_combined)
            cluster_probs_jax = jnp.array(cluster_probs_combined)
            logZ_jax = jnp.array(logZ_combined)
            
            # Determine N_eff (effective number of data points)
            # If not specified, use total number of training data points
            N_eff = self.N_eff if self.N_eff is not None else float(num_samples_used)
            
            # Update GMM parameters
            gmm_params = self._update_gmm_vbem(z_e_jax, cluster_probs_jax, logZ_jax, gmm_params, N_eff)
        
        # Recombine all params
        self.params = self._combine_params(encoder_decoder_params, gmm_params)
        
        # Convert to Python float only once at the end (single host-device sync)
        epoch_metrics['total_loss'] = float(total_loss_acc) / num_batches
        epoch_metrics['recon_loss'] = float(recon_loss_acc) / num_batches
        epoch_metrics['gmm_loss'] = float(gmm_loss_acc) / num_batches
        
        return epoch_metrics
    
    def train(
        self,
        x_data: jnp.ndarray,
        num_epochs: int = 100,
        batch_size: int = 256,
        validation_data: Optional[jnp.ndarray] = None,
        dropout_epochs: Optional[int] = None,
        verbose: bool = True,
        update_gmm_every_n_batches: int = 10
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
            update_gmm_every_n_batches: Update GMM params every N batches
            
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
            'train_gmm_losses': [],
            'val_losses': [],
            'val_recon_losses': [],
            'val_gmm_losses': [],
        }
        
        if verbose:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Dropout epochs: {dropout_epochs}")
            print(f"Training data shape: {x_data.shape}")
            print(f"N_eff (effective data points): {self.N_eff if self.N_eff is not None else 'auto (total training data points)'}")
            print(f"GMM updates every {update_gmm_every_n_batches} batches")
        
        for epoch in tqdm(range(num_epochs), desc="Training", disable=not verbose):
            use_dropout = epoch < dropout_epochs
            train_metrics = self.train_epoch(
                x_data,
                batch_size,
                use_dropout=use_dropout,
                update_gmm_every_n_batches=update_gmm_every_n_batches
            )
            
            # Store metrics
            history['train_losses'].append(train_metrics['total_loss'])
            history['train_recon_losses'].append(train_metrics['recon_loss'])
            history['train_gmm_losses'].append(train_metrics['gmm_loss'])
            
            # Validation (only every 10 epochs to save time)
            if validation_data is not None and (epoch % 10 == 0 or epoch == num_epochs - 1):
                val_metrics = self.evaluate(validation_data, batch_size)
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
                history['val_gmm_losses'].append(val_metrics['gmm_loss'])
                
                if verbose:
                    print(f"Epoch {epoch}: train_loss={train_metrics['total_loss']:.4f}, "
                          f"val_loss={val_metrics['total_loss']:.4f}")
            elif validation_data is not None:
                # Append previous validation loss to keep list lengths consistent
                if len(history['val_losses']) > 0:
                    history['val_losses'].append(history['val_losses'][-1])
                    history['val_recon_losses'].append(history['val_recon_losses'][-1])
                    history['val_gmm_losses'].append(history['val_gmm_losses'][-1])
        
        if verbose:
            print("Training completed!")
        
        return history
    
    @partial(jax.jit, static_argnums=(0,))
    def _eval_batch(self, params: dict, x_batch: jnp.ndarray, key: jr.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """JIT-compiled evaluation of a single batch."""
        loss, metrics = self.model.loss(params, x_batch, key, training=False)
        return loss, metrics.get('recon_loss', jnp.array(0.0)), metrics.get('gmm_loss', jnp.array(0.0))
    
    def evaluate(self, x_data: jnp.ndarray, batch_size: int = 256) -> Dict[str, float]:
        """
        Evaluate the model on given data.
        
        Args:
            x_data: Input data [num_samples, *input_shape]
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
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
        gmm_loss_acc = jnp.array(0.0)
        
        # Evaluate on batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            x_batch = x_data_eval[start_idx:end_idx]
            
            # Use JIT-compiled evaluation
            self.rng, eval_rng = jr.split(self.rng)
            loss, recon_loss, gmm_loss = self._eval_batch(self.params, x_batch, eval_rng)
            
            # Accumulate as JAX arrays
            total_loss_acc = total_loss_acc + loss
            recon_loss_acc = recon_loss_acc + recon_loss
            gmm_loss_acc = gmm_loss_acc + gmm_loss
        
        # Convert to Python float only once at the end
        avg_loss = float(total_loss_acc) / num_batches
        avg_recon_loss = float(recon_loss_acc) / num_batches
        avg_gmm_loss = float(gmm_loss_acc) / num_batches
        
        return {
            'total_loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'gmm_loss': avg_gmm_loss
        }
    
    def encode(self, x_data: jnp.ndarray) -> jnp.ndarray:
        """
        Encode input data to continuous latent representation.
        
        Args:
            x_data: Input data [num_samples, *input_shape]
            
        Returns:
            Encoder output [num_samples, ..., latent_dim]
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        return self.model.apply(self.params, x_data, method='encode', training=False)
    
    def decode(self, z_q: jnp.ndarray) -> jnp.ndarray:
        """
        Decode quantized representation to output space.
        
        Args:
            z_q: Quantized representation [num_samples, ..., latent_dim]
            
        Returns:
            Reconstructed output [num_samples, *output_shape]
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        return self.model.apply(self.params, z_q, method='decode', training=False)
    
    def reconstruct(self, x_data: jnp.ndarray) -> jnp.ndarray:
        """
        Reconstruct input data: encode -> quantize -> decode.
        
        Args:
            x_data: Input data [num_samples, *input_shape]
            
        Returns:
            Reconstructed output [num_samples, *output_shape]
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        # Encode
        z_e = self.encode(x_data)
        
        # Get quantized representation
        top_k = self.config.main.get("top_k", 1)
        gmm_params = {'params': self.params['params']['gmm_vbem']}
        self.rng, sample_rng = jr.split(self.rng)
        z_q, _, _ = self.model.gmm_vbem.apply(
            gmm_params,
            z_e,
            top_k,
            sample_rng
        )
        
        # Decode
        x_recon = self.decode(z_q)
        
        return x_recon
    
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
        self.model = VBVAE(config=self.config)
    
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
        
        # Create and save training progress plot
        self._plot_training_progress(history, save_dir_path / 'training_progress.png')
    
    def _plot_training_progress(self, history: Dict[str, list], save_path: Path):
        """Create training progress plot."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(len(history['train_losses']))
        
        # Calculate validation epochs
        val_epochs = []
        if len(history['val_losses']) > 0:
            val_epochs = [i for i in range(len(history['train_losses'])) 
                         if i % 10 == 0 or i == len(history['train_losses']) - 1]
            val_epochs = val_epochs[:len(history['val_losses'])]
        
        # Total loss
        axes[0, 0].plot(epochs, history['train_losses'], label='Train', alpha=0.7)
        if len(history['val_losses']) > 0 and len(val_epochs) == len(history['val_losses']):
            axes[0, 0].plot(val_epochs, history['val_losses'], label='Val', marker='o', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[0, 1].plot(epochs, history['train_recon_losses'], label='Train', alpha=0.7)
        if len(history['val_recon_losses']) > 0 and len(val_epochs) == len(history['val_recon_losses']):
            axes[0, 1].plot(val_epochs, history['val_recon_losses'], label='Val', marker='o', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Reconstruction Loss')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # GMM loss
        axes[1, 0].plot(epochs, history['train_gmm_losses'], label='Train', alpha=0.7)
        if len(history['val_gmm_losses']) > 0 and len(val_epochs) == len(history['val_gmm_losses']):
            axes[1, 0].plot(val_epochs, history['val_gmm_losses'], label='Val', marker='o', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('GMM Loss')
        axes[1, 0].set_title('GMM Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Empty subplot (can be used for additional metrics)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

