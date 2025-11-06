"""
Streamlined trainer for standard VAE model.

This trainer provides a simple interface for training the VAE model
with standard functionality like training, evaluation, and saving.
"""

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

from src.models.vae.vae import VAE, VAEConfig
from flax import traverse_util

# Disable JAX optimizations that can cause slowdowns
jax.config.update('jax_disable_jit', False)


class VAETrainer:
    """Trainer for standard VAE model."""
    
    def __init__(
        self,
        config: VAEConfig,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        seed: int = 42
    ):
        """
        Initialize the trainer.
        
        Args:
            config: VAE configuration
            learning_rate: Learning rate for optimizer
            optimizer_name: Name of optimizer ('adam', 'sgd', etc.)
            seed: Random seed for reproducibility
        """
        self.config = config
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.seed = seed
        
        # Initialize model
        self.model = VAE(config=config)
        
        # Initialize optimizer
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
            x_sample: Sample input data [batch_size, *input_shape]
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
        # Compute loss and gradients - loss function is now JIT-compiled
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
            'kl_loss': 0.0,
            'step': 0
        }
        
        # Shuffle data once (more efficient)
        self.rng, shuffle_rng = jr.split(self.rng)
        perm = jr.permutation(shuffle_rng, num_samples)
        x_data_shuffled = x_data[perm][:num_samples_used]  # Truncate to exact multiple of batch_size
        
        # Accumulate metrics as JAX arrays (avoid host-device sync)
        total_loss_acc = jnp.array(0.0)
        recon_loss_acc = jnp.array(0.0)
        kl_loss_acc = jnp.array(0.0)
        
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
            kl_loss_acc = kl_loss_acc + metrics.get('kl_loss', jnp.array(0.0))
        
        # Convert to Python float only once at the end (single host-device sync)
        epoch_metrics['total_loss'] = float(total_loss_acc) / num_batches
        epoch_metrics['recon_loss'] = float(recon_loss_acc) / num_batches
        epoch_metrics['kl_loss'] = float(kl_loss_acc) / num_batches
        epoch_metrics['step'] = num_batches
        
        return epoch_metrics
    
    def train(
        self,
        x_data: jnp.ndarray,
        num_epochs: int = 100,
        batch_size: int = 256,
        validation_data: Optional[jnp.ndarray] = None,
        dropout_epochs: Optional[int] = None,
        verbose: bool = True
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
            'train_kl_losses': [],
            'val_losses': [],
            'val_recon_losses': [],
            'val_kl_losses': [],
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
            history['train_kl_losses'].append(train_metrics['kl_loss'])
            
            # Validation (only every 10 epochs to save time and avoid recompilation overhead)
            if validation_data is not None and (epoch % 10 == 0 or epoch == num_epochs - 1):
                val_metrics = self.evaluate(validation_data, batch_size)
                history['val_losses'].append(val_metrics['total_loss'])
                history['val_recon_losses'].append(val_metrics['recon_loss'])
                history['val_kl_losses'].append(val_metrics['kl_loss'])
                
                if verbose:
                    print(f"Epoch {epoch}: train_loss={train_metrics['total_loss']:.4f}, "
                          f"val_loss={val_metrics['total_loss']:.4f}")
            elif validation_data is not None:
                # Append previous validation loss to keep list lengths consistent
                if len(history['val_losses']) > 0:
                    history['val_losses'].append(history['val_losses'][-1])
                    history['val_recon_losses'].append(history['val_recon_losses'][-1])
                    history['val_kl_losses'].append(history['val_kl_losses'][-1])
        
        if verbose:
            print("Training completed!")
        
        return history
    
    @partial(jax.jit, static_argnums=(0,))
    def _eval_batch(self, params: dict, x_batch: jnp.ndarray, key: jr.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """JIT-compiled evaluation of a single batch.
        
        This uses training=False and is completely separate from train_step to avoid
        recompilation when switching between training and evaluation modes.
        """
        # Use training=False explicitly - this is a separate compiled function
        # so it won't cause train_step to recompile
        loss, metrics = self.model.loss(params, x_batch, key, training=False)
        return loss, metrics.get('recon_loss', jnp.array(0.0)), metrics.get('kl_loss', jnp.array(0.0))
    
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
        kl_loss_acc = jnp.array(0.0)
        
        # Evaluate on batches (all batches are now exactly batch_size)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size  # Always exactly batch_size
            
            x_batch = x_data_eval[start_idx:end_idx]
            
            # Use JIT-compiled evaluation
            self.rng, eval_rng = jr.split(self.rng)
            loss, recon_loss, kl_loss = self._eval_batch(self.params, x_batch, eval_rng)
            
            # Accumulate as JAX arrays (no host-device sync)
            total_loss_acc = total_loss_acc + loss
            recon_loss_acc = recon_loss_acc + recon_loss
            kl_loss_acc = kl_loss_acc + kl_loss
        
        # Convert to Python float only once at the end (single host-device sync)
        avg_loss = float(total_loss_acc) / num_batches
        avg_recon_loss = float(recon_loss_acc) / num_batches
        avg_kl_loss = float(kl_loss_acc) / num_batches
        
        return {
            'total_loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def encode(self, x_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode input data to latent space.
        
        Args:
            x_data: Input data [num_samples, *input_shape]
            
        Returns:
            Tuple of (mu, logvar) [num_samples, latent_dim]
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        return self.model.apply(self.params, x_data, method='encode', training=False)
    
    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Decode latent representation to output space.
        
        Args:
            z: Latent representation [num_samples, latent_dim]
            
        Returns:
            Reconstructed output [num_samples, *output_shape]
        """
        if self.params is None:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        return self.model.apply(self.params, z, method='decode', training=False)
    
    def reconstruct(self, x_data: jnp.ndarray) -> jnp.ndarray:
        """
        Reconstruct input data by encoding and decoding.
        
        Args:
            x_data: Input data [num_samples, *input_shape]
            
        Returns:
            Reconstructed output [num_samples, *output_shape]
        """
        # Encode
        mu, logvar = self.encode(x_data)
        # Sample z
        self.rng, sample_rng = jr.split(self.rng)
        std = jnp.exp(0.5 * logvar)
        z = mu + std * jr.normal(sample_rng, mu.shape)
        # Decode
        return self.decode(z)
    
    def save_params(self, filepath: str):
        """Save model parameters and config to file."""
        if self.params is None:
            raise ValueError("Model not initialized. No parameters to save.")
        
        # Save both parameters and config
        save_data = {
            'params': self.params,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Parameters and config saved to {filepath}")
    
    def load_params(self, filepath: str):
        """Load model parameters and config from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both old format (just params) and new format (params + config)
        if isinstance(data, dict) and 'params' in data:
            self.params = data['params']
            if 'config' in data:
                self.config = data['config']
                print(f"Parameters and config loaded from {filepath}")
            else:
                print(f"Parameters loaded from {filepath} (no config found)")
        else:
            # Old format - just parameters
            self.params = data
            print(f"Parameters loaded from {filepath} (old format, no config)")
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save training results and create plots."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        results_file = Path(output_dir) / "training_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {results_file}")
        
        # Save parameters
        params_file = Path(output_dir) / "model_params.pkl"
        self.save_params(str(params_file))
        
        # Create plots
        self._create_plots(results, output_dir)
    
    def _create_plots(self, results: Dict[str, Any], output_dir: str):
        """Create diagnostic plots."""
        try:
            # Training progress plot
            if 'train_losses' in results and len(results['train_losses']) > 0:
                self._create_training_progress_plot(results, output_dir)
        except Exception as e:
            print(f"Warning: Error creating plots: {e}")
            traceback.print_exc()
    
    def _create_training_progress_plot(self, results: Dict[str, Any], output_dir: str):
        """Create training progress plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('VAE Training Progress', fontsize=16, fontweight='bold')
        
        epochs = range(len(results['train_losses']))
        
        # Panel 1: Total loss
        axes[0, 0].plot(epochs, results['train_losses'], label='Train', color='blue', linewidth=2)
        if 'val_losses' in results and len(results['val_losses']) > 0:
            val_epochs = range(len(results['val_losses']))
            axes[0, 0].plot(val_epochs, results['val_losses'], label='Validation', color='red', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Panel 2: Reconstruction loss
        if 'train_recon_losses' in results and len(results['train_recon_losses']) > 0:
            axes[0, 1].plot(epochs, results['train_recon_losses'], label='Train', color='blue', linewidth=2)
            if 'val_recon_losses' in results and len(results['val_recon_losses']) > 0:
                val_epochs = range(len(results['val_recon_losses']))
                axes[0, 1].plot(val_epochs, results['val_recon_losses'], label='Validation', color='red', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Panel 3: KL loss
        if 'train_kl_losses' in results and len(results['train_kl_losses']) > 0:
            axes[1, 0].plot(epochs, results['train_kl_losses'], label='Train', color='blue', linewidth=2)
            if 'val_kl_losses' in results and len(results['val_kl_losses']) > 0:
                val_epochs = range(len(results['val_kl_losses']))
                axes[1, 0].plot(val_epochs, results['val_kl_losses'], label='Validation', color='red', linewidth=2)
        axes[1, 0].set_title('KL Divergence Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Panel 4: Loss components comparison
        axes[1, 1].plot(epochs, results['train_recon_losses'], label='Recon', color='blue', linewidth=2)
        axes[1, 1].plot(epochs, results['train_kl_losses'], label='KL', color='green', linewidth=2)
        axes[1, 1].set_title('Loss Components', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = Path(output_dir) / "training_progress.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training progress plot saved to {plot_file}")

