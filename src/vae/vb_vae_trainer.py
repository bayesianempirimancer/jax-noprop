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
import time

from src.vae.vb_vae import VBVAE, VBVAEConfig
from src.vae.vb_gmm import GMMVBEM
from src.utils.math_utils import logsumexp
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
        gmm_learning_rate: float = 0.2,  # Learning rate for GMM VBEM updates (mixing parameter between 0 and 1, different from gradient descent LR)
        seed: int = 42
    ):
        """
        Initialize VBVAE trainer.
        
        Args:
            config: VBVAEConfig configuration
            learning_rate: Learning rate for encoder/decoder optimizer (gradient descent)
            optimizer_name: Name of optimizer ("adam" or "sgd")
            N_eff: Effective number of data points (inverse temperature). If None, defaults to total number of training data points.
            gmm_learning_rate: Learning rate for GMM VBEM updates (mixing parameter between 0 and 1, NOT a gradient descent LR)
            seed: Random seed
        """
        self.config = config
        self.model = VBVAE(config=config)
        self.N_eff = N_eff
        self.gmm_learning_rate = gmm_learning_rate
        
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
        
        # Initialize cluster means from encoder outputs
        print("Initializing cluster means from encoder outputs...")
        self.rng, init_key = jr.split(self.rng)
        
        # Encode sample data to get latent representations
        z_e = self.model.apply(self.params, x_sample, method='encode', training=False)
        
        # Extract GMM params and initialize cluster means
        _, gmm_params = self._separate_gmm_params(self.params)
        
        # Get configuration values
        num_clusters = self.config.main["num_clusters"]
        latent_dim = self.config.main["latent_dim"]
        
        # Initialize cluster means from data using class method
        mu_n = GMMVBEM.get_initial_cluster_means(
            num_clusters=num_clusters,
            latent_dim=latent_dim,
            x=z_e,
            key=init_key
        )
        
        # Update gmm_params with initialized cluster means
        gmm_params['mu_n'] = mu_n
        
        # Update self.params with initialized GMM params
        params_unfrozen = unfreeze(self.params)
        params_unfrozen['params']['gmm_vbem'] = gmm_params
        self.params = freeze(params_unfrozen)
        
        # Initialize optimizer state with only encoder/decoder params (not GMM params)
        encoder_decoder_params, _ = self._separate_gmm_params(self.params)
        self.opt_state = self.optimizer.init(encoder_decoder_params)
        
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
        # Create a GMMVBEM instance to call update method
        # Get config values from the model config
        num_clusters = self.config.main["num_clusters"]
        latent_dim = self.config.main["latent_dim"]
        prior_mu = self.config.main.get("prior_mu", 0.0)
        prior_alpha = self.config.main.get("prior_alpha", 0.5)
        prior_beta = self.config.main.get("prior_beta", 0.5)
        prior_alpha_mix = self.config.main.get("prior_alpha_mix", 0.5)
        
        gmm_vbem = GMMVBEM(
            num_clusters=num_clusters,
            latent_dim=latent_dim,
            prior_mu=prior_mu,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            prior_alpha_mix=prior_alpha_mix,
            beta_mix=self.config.main.get("beta_mix", 0.0)
        )
        
        # Call the GMM-VBEM update method with GMM-specific learning rate
        # Note: gmm_learning_rate is a mixing parameter (0-1), not a gradient descent LR
        # Call update via apply since it's now @nn.compact
        from flax.core import freeze
        gmm_params_frozen = freeze({'params': gmm_params})
        updated_gmm_params = gmm_vbem.apply(
            gmm_params_frozen,
            z_e_batch,
            N_eff=N_eff,
            lr=self.gmm_learning_rate,
            training=True,
            method='update'
        )
        
        return updated_gmm_params
    
    def train_epoch(
        self,
        x_data: jnp.ndarray,
        batch_size: int = 256,
        use_dropout: bool = True,
        update_gmm_every_n_batches: int = 1
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
            'num_batches': num_batches,  # Store for batch time calculation
        }
        
        # Separate encoder/decoder and GMM params
        encoder_decoder_params, gmm_params = self._separate_gmm_params(self.params)
        
        # Shuffle data once (more efficient)
        self.rng, shuffle_rng = jr.split(self.rng)
        perm = jr.permutation(shuffle_rng, num_samples)
        x_data_shuffled = x_data[perm][:num_samples_used]  # Truncate to exact multiple of batch_size
        
        # Accumulate metrics
        total_loss_acc = jnp.array(0.0)
        recon_loss_acc = jnp.array(0.0)
        gmm_loss_acc = jnp.array(0.0)
        
        # Determine N_eff (effective number of data points)
        # If not specified, use total number of training data points
        N_eff = self.N_eff if self.N_eff is not None else float(num_samples_used)
        
        # Train on batches
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size  # Always exactly batch_size
            
            x_batch = x_data_shuffled[start_idx:end_idx]
            
            # Update GMM parameters FIRST (before computing loss for encoder/decoder)
            if i % update_gmm_every_n_batches == 0:
                # Recombine params for encoding
                params_combined = self._combine_params(encoder_decoder_params, gmm_params)
                
                # Encode to get z_e and cluster_probs
                self.rng, encode_rng = jr.split(self.rng)
                z_e = self.model.apply(params_combined, x_batch, method='encode', training=False)
                
                # Flatten z_e for GMM update
                z_e_flat = z_e.reshape(-1, z_e.shape[-1])  # [N_batch, latent_dim]
                
                # Update GMM parameters on this minibatch
                gmm_params = self._update_gmm_vbem(
                    z_e_flat,
                    gmm_params,
                    N_eff
                )
            
            # Training step (updates encoder/decoder, uses updated GMM params)
            self.rng, train_rng = jr.split(self.rng)
            encoder_decoder_params, gmm_params, self.opt_state, loss, metrics = self.model.train_step(
                encoder_decoder_params, gmm_params, x_batch, self.opt_state, train_rng, 
                training=use_dropout, optimizer=self.optimizer
            )
            
            # Accumulate metrics (keep as JAX arrays until end - no host-device sync)
            total_loss_acc = total_loss_acc + loss
            recon_loss_acc = recon_loss_acc + metrics.get('recon_loss', jnp.array(0.0))
            gmm_loss_acc = gmm_loss_acc + metrics.get('gmm_loss', jnp.array(0.0))
        
        # Count active clusters and store normalized mixing weights (from final GMM params)
        gmm_params_dict = gmm_params if isinstance(gmm_params, dict) else dict(gmm_params)
        alpha_mix = np.array(gmm_params_dict['alpha_mix'])
        num_active_clusters = np.sum(alpha_mix > 1.5)
        epoch_metrics['active_clusters'] = int(num_active_clusters)
        
        # Store normalized mixing weights (E[π] = alpha_mix / sum(alpha_mix))
        normalized_pi = alpha_mix / (np.sum(alpha_mix) + 1e-8)
        epoch_metrics['normalized_pi'] = normalized_pi.tolist()
        
        # Store cluster means (first two dimensions for plotting)
        mu_n = np.array(gmm_params_dict['mu_n'])
        epoch_metrics['cluster_means'] = mu_n[:, :2].tolist()  # [num_clusters, 2]
        
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
        update_gmm_every_n_batches: int = 1
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
            'active_clusters': [],
            'normalized_pi': [],  # Normalized mixing weights over time
            'cluster_means': [],  # Cluster means (first 2 dims) over time
            'val_losses': [],
            'val_recon_losses': [],
            'val_gmm_losses': [],
            'epoch_times': [],  # Time per epoch
        }
        
        if verbose:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Dropout epochs: {dropout_epochs}")
            print(f"Training data shape: {x_data.shape}")
            print(f"N_eff (effective data points): {self.N_eff if self.N_eff is not None else 'auto (total training data points)'}")
            print(f"GMM updates every {update_gmm_every_n_batches} batches")
        
        for epoch in tqdm(range(num_epochs), desc="Training", disable=not verbose):
            epoch_start_time = time.perf_counter()
            use_dropout = epoch < dropout_epochs
            train_metrics = self.train_epoch(
                x_data,
                batch_size,
                use_dropout=use_dropout,
                update_gmm_every_n_batches=update_gmm_every_n_batches
            )
            epoch_end_time = time.perf_counter()
            epoch_time = epoch_end_time - epoch_start_time
            
            # Store metrics
            history['train_losses'].append(train_metrics['total_loss'])
            history['train_recon_losses'].append(train_metrics['recon_loss'])
            history['train_gmm_losses'].append(train_metrics['gmm_loss'])
            active_clusters = train_metrics.get('active_clusters', 0)
            history['active_clusters'].append(active_clusters)
            
            # Store timing metrics
            history['epoch_times'].append(epoch_time)
            
            # Store normalized mixing weights and cluster means
            if 'normalized_pi' in train_metrics:
                history['normalized_pi'].append(train_metrics['normalized_pi'])
            if 'cluster_means' in train_metrics:
                history['cluster_means'].append(train_metrics['cluster_means'])
            
            # Validation (only every 10 epochs to save time)
            if validation_data is not None and (epoch % 10 == 0 or epoch == num_epochs - 1):
                val_metrics = self.evaluate(validation_data, batch_size)
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
                history['val_gmm_losses'].append(val_metrics['gmm_loss'])
                
                if verbose:
                    avg_epoch_time = np.mean(history['epoch_times']) if history['epoch_times'] else epoch_time
                    num_batches = train_metrics.get('num_batches', 1)
                    avg_step_time = avg_epoch_time / num_batches if num_batches > 0 else 0.0
                    print(f"Epoch {epoch}: train_loss={train_metrics['total_loss']:.4f}, "
                          f"val_loss={val_metrics['total_loss']:.4f}, "
                          f"active_clusters={active_clusters}, "
                          f"epoch_time={epoch_time:.3f}s (avg={avg_epoch_time:.3f}s), "
                          f"step_time={avg_step_time*1000:.2f}ms")
            elif validation_data is not None:
                # Append previous validation loss to keep list lengths consistent
                if len(history['val_losses']) > 0:
                    history['val_losses'].append(history['val_losses'][-1])
                    history['val_recon_losses'].append(history['val_recon_losses'][-1])
                    history['val_gmm_losses'].append(history['val_gmm_losses'][-1])
        
        if verbose:
            if history['epoch_times']:
                total_time = np.sum(history['epoch_times'])
                avg_epoch_time = np.mean(history['epoch_times'])
                # Get num_batches from the last train_metrics (all epochs should have same num_batches)
                # We need to get it from the last epoch - store it in history or get from last train_metrics
                # For now, calculate from data shape
                num_samples = x_data.shape[0]
                batch_size = batch_size  # Use the batch_size parameter
                num_batches = num_samples // batch_size
                avg_step_time = avg_epoch_time / num_batches if num_batches > 0 else 0.0
                print(f"\nTraining completed!")
                print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
                print(f"Average time per epoch: {avg_epoch_time:.3f}s")
                print(f"Average time per step (batch): {avg_step_time*1000:.2f}ms")
        
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
        
        # Get quantized representation using apply
        z_q, _ = self.model.apply(
            self.params,
            z_e,
            method=lambda mdl, x: mdl.gmm_vbem.quantize(x, training=False)
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
        
        # Save config as YAML using BaseConfig method
        if hasattr(self.config, 'save_yaml'):
            self.config.save_yaml(save_dir_path / "config.yaml")
            print(f"Config saved to {save_dir_path / 'config.yaml'}")
        elif hasattr(self.config, 'save_json'):
            self.config.save_json(save_dir_path / "config.json")
            print(f"Config saved to {save_dir_path / 'config.json'}")
        
        # Create and save training progress plot
        self._plot_training_progress(history, save_dir_path / 'training_progress.png')
        
        # Create and save cluster means over time plot
        if 'cluster_means' in history and len(history['cluster_means']) > 0:
            self.plot_cluster_means_over_time(
                history, 
                save_path=save_dir_path / 'cluster_means_over_time.png'
            )
    
    def _plot_training_progress(self, history: Dict[str, list], save_path: Path):
        """Create training progress plot."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
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
        
        # Empty panel in top row (can be used for additional metrics)
        axes[0, 2].axis('off')
        
        # GMM loss
        axes[1, 0].plot(epochs, history['train_gmm_losses'], label='Train', alpha=0.7)
        if len(history['val_gmm_losses']) > 0 and len(val_epochs) == len(history['val_gmm_losses']):
            axes[1, 0].plot(val_epochs, history['val_gmm_losses'], label='Val', marker='o', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('GMM Loss')
        axes[1, 0].set_title('GMM Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Active clusters over time (if tracked in history)
        if 'active_clusters' in history and len(history['active_clusters']) > 0:
            axes[1, 1].plot(epochs, history['active_clusters'], label='Active Clusters', color='green', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Number of Active Clusters')
            axes[1, 1].set_title('Active Clusters Over Time')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        # Normalized mixing weights (alpha_mix normalized) over time
        if 'normalized_pi' in history and len(history['normalized_pi']) > 0:
            ax = axes[1, 2]
            # Get number of clusters
            num_clusters = len(history['normalized_pi'][0]) if len(history['normalized_pi']) > 0 else 0
            
            # Plot top clusters by final mixing weight
            if num_clusters > 0:
                final_pi = np.array(history['normalized_pi'][-1])
                top_cluster_indices = np.argsort(final_pi)[-min(10, num_clusters):][::-1]  # Top 10 clusters
                
                # Plot mixing weights over time for top clusters
                pi_over_time = np.array(history['normalized_pi'])  # [num_epochs, num_clusters]
                colors = plt.cm.tab10(np.linspace(0, 1, len(top_cluster_indices)))
                
                for idx, k in enumerate(top_cluster_indices):
                    ax.plot(epochs, pi_over_time[:, k], label=f'Cluster {k}', 
                           color=colors[idx], alpha=0.7, linewidth=1.5)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Normalized Mixing Weight (E[π])')
                ax.set_title('Top Cluster Mixing Weights Over Time')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
                ax.grid(True, alpha=0.3)
            else:
                ax.axis('off')
        else:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_cluster_means_over_time(
        self,
        history: Dict[str, list],
        save_path: Optional[str] = None,
        top_n_clusters: int = 10
    ):
        """
        Plot cluster means (first two dimensions) over time for most commonly used clusters.
        
        Args:
            history: Training history dictionary
            save_path: Optional path to save the plot
            top_n_clusters: Number of top clusters to plot
        """
        if 'cluster_means' not in history or len(history['cluster_means']) == 0:
            print("Warning: No cluster means tracked in history")
            return
        
        if 'normalized_pi' not in history or len(history['normalized_pi']) == 0:
            print("Warning: No mixing weights tracked in history")
            return
        
        # Get final mixing weights to determine top clusters
        final_pi = np.array(history['normalized_pi'][-1])
        top_cluster_indices = np.argsort(final_pi)[-top_n_clusters:][::-1]  # Top N clusters
        
        # Get cluster means over time
        cluster_means_over_time = np.array(history['cluster_means'])  # [num_epochs, num_clusters, 2]
        epochs = range(len(cluster_means_over_time))
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Cluster Means Over Time (Top Clusters)', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_cluster_indices)))
        
        # Plot 1: Cluster means trajectory (x1, x2) over epochs
        ax = axes[0]
        for idx, k in enumerate(top_cluster_indices):
            means_k = cluster_means_over_time[:, k, :]  # [num_epochs, 2]
            ax.plot(means_k[:, 0], means_k[:, 1], 
                   color=colors[idx], alpha=0.6, linewidth=2, label=f'Cluster {k}')
            # Mark start and end points
            ax.scatter(means_k[0, 0], means_k[0, 1], 
                      color=colors[idx], marker='o', s=100, alpha=0.8, zorder=5)
            ax.scatter(means_k[-1, 0], means_k[-1, 1], 
                      color=colors[idx], marker='s', s=100, alpha=0.8, zorder=5)
        
        ax.set_xlabel('Latent Dimension 0 (x1)')
        ax.set_ylabel('Latent Dimension 1 (x2)')
        ax.set_title(f'Cluster Mean Trajectories (Top {len(top_cluster_indices)} Clusters)\nCircles=Start, Squares=End')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Plot 2: Cluster means over epochs (separate plots for x1 and x2)
        ax = axes[1]
        for idx, k in enumerate(top_cluster_indices):
            means_k = cluster_means_over_time[:, k, :]  # [num_epochs, 2]
            ax.plot(epochs, means_k[:, 0], 
                   color=colors[idx], linestyle='-', alpha=0.7, linewidth=1.5, 
                   label=f'Cluster {k} (x1)')
            ax.plot(epochs, means_k[:, 1], 
                   color=colors[idx], linestyle='--', alpha=0.7, linewidth=1.5, 
                   label=f'Cluster {k} (x2)')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cluster Mean Value')
        ax.set_title(f'Cluster Means Over Time\nSolid=x1, Dashed=x2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Cluster means plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_cluster_diagnostics(
        self,
        x_sample: jnp.ndarray,
        save_path: Optional[str] = None,
        max_clusters_to_plot: int = 20
    ):
        """
        Generate diagnostic plots for GMM clusters.
        
        Args:
            x_sample: Sample input data [num_samples, *input_shape]
            save_path: Optional path to save the plot
            max_clusters_to_plot: Maximum number of clusters to visualize
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        # Encode to get latent representations
        z_e = self.encode(x_sample)  # [num_samples, ..., latent_dim]
        z_e_flat = z_e.reshape(-1, z_e.shape[-1])  # [N, latent_dim]
        
        # Get cluster assignments
        gmm_params = {'params': self.params['params']['gmm_vbem']}
        _, log_p_tilde = self.model.apply(
            self.params,
            z_e,
            method=lambda mdl, x: mdl.gmm_vbem.quantize(x, training=False)
        )
        log_p_tilde_flat = log_p_tilde.reshape(-1, log_p_tilde.shape[-1])  # [N, num_clusters]
        cluster_assignments = jnp.argmax(log_p_tilde_flat, axis=-1)  # [N]
        # Use numerically stable softmax
        from src.utils.math_utils import stable_softmax
        cluster_probs_flat = stable_softmax(log_p_tilde_flat, axis=-1)
        
        # Get GMM parameters
        gmm_params_dict = self.params['params']['gmm_vbem']
        mu_n = np.array(gmm_params_dict['mu_n'])  # [num_clusters, latent_dim]
        alpha_n = np.array(gmm_params_dict['alpha_n'])  # [num_clusters, 1]
        beta_n = np.array(gmm_params_dict['beta_n'])  # [num_clusters, latent_dim]
        alpha_mix = np.array(gmm_params_dict['alpha_mix'])  # [num_clusters]
        
        # Compute expected statistics
        E_mu = mu_n  # [num_clusters, latent_dim]
        E_var = beta_n / (alpha_n + 1e-8)  # [num_clusters, latent_dim]
        E_pi = alpha_mix / (np.sum(alpha_mix) + 1e-8)  # [num_clusters]
        
        # Find active clusters (alpha_mix > threshold)
        active_mask = alpha_mix > 1.5
        active_indices = np.where(active_mask)[0]
        num_active = len(active_indices)
        
        # Only plot if latent_dim is 2D
        if z_e_flat.shape[-1] != 2:
            print(f"Warning: Cluster diagnostics plot only supports 2D latent space. Current dimension: {z_e_flat.shape[-1]}")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('VBVAE Cluster Diagnostics', fontsize=16, fontweight='bold')
        
        # Plot 1: Data colored by cluster assignment
        ax = axes[0, 0]
        z_e_np = np.array(z_e_flat)
        cluster_assignments_np = np.array(cluster_assignments)
        
        # Plot top active clusters
        top_clusters = active_indices[np.argsort(E_pi[active_indices])[-max_clusters_to_plot:]][::-1]
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_clusters)))
        
        for idx, k in enumerate(top_clusters):
            mask = cluster_assignments_np == k
            if np.any(mask):
                ax.scatter(z_e_np[mask, 0], z_e_np[mask, 1], 
                          c=[colors[idx]], alpha=0.4, s=20, label=f'Cluster {k}')
        
        # Plot cluster means
        ax.scatter(E_mu[top_clusters, 0], E_mu[top_clusters, 1],
                  c='black', marker='x', s=200, linewidths=3, label='Cluster means')
        ax.set_title(f'Data Colored by Cluster Assignment\n({num_active} active clusters)')
        ax.set_xlabel('Latent Dimension 0')
        ax.set_ylabel('Latent Dimension 1')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mixing weights (top clusters)
        ax = axes[0, 1]
        top_pi = E_pi[top_clusters]
        ax.barh(range(len(top_clusters)), top_pi, color=colors)
        ax.set_yticks(range(len(top_clusters)))
        ax.set_yticklabels([f'Cluster {k}' for k in top_clusters])
        ax.set_xlabel('Mixing Weight')
        ax.set_title(f'Top {len(top_clusters)} Mixing Weights')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Cluster sizes (number of assigned points)
        ax = axes[1, 0]
        cluster_sizes = np.bincount(cluster_assignments_np, minlength=len(E_pi))
        top_sizes = cluster_sizes[top_clusters]
        ax.barh(range(len(top_clusters)), top_sizes, color=colors)
        ax.set_yticks(range(len(top_clusters)))
        ax.set_yticklabels([f'Cluster {k}' for k in top_clusters])
        ax.set_xlabel('Number of Assigned Points')
        ax.set_title('Cluster Sizes')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Alpha_mix values (posterior Dirichlet parameters)
        ax = axes[1, 1]
        top_alpha_mix = alpha_mix[top_clusters]
        ax.barh(range(len(top_clusters)), top_alpha_mix, color=colors)
        ax.set_yticks(range(len(top_clusters)))
        ax.set_yticklabels([f'Cluster {k}' for k in top_clusters])
        ax.set_xlabel('Alpha_mix (Posterior Dirichlet Parameter)')
        ax.set_title('Posterior Dirichlet Parameters')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Cluster diagnostics plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

