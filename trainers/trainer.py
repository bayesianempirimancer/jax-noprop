#!/usr/bin/env python3
"""
Final working trainers for NoProp models using the train_step methods from the models.
Based on the working two_moons example.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from typing import Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
import os
import warnings

# Suppress JAX buffer comparator warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'  # Use GPU for better performance
warnings.filterwarnings('ignore', category=UserWarning, module='jax')
warnings.filterwarnings('ignore', message='.*buffer_comparator.*')
warnings.filterwarnings('ignore', message='.*Results do not match the reference.*')
warnings.filterwarnings('ignore', message='.*gemm_fusion_autotuner.*')

# Suppress JAX logging
import logging
logging.getLogger('jax').setLevel(logging.ERROR)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.jax_noprop.noprop_ct import NoPropCT
from src.jax_noprop.noprop_fm import NoPropFM
from src.jax_noprop.models import SimpleMLP
from src.jax_noprop.noise_schedules import LinearNoiseSchedule, CosineNoiseSchedule


class NoPropTrainer:
    """Working trainer for NoProp models using the built-in train_step methods."""
    
    def __init__(
        self,
        model_type: str = "ct",  # "ct" or "fm"
        input_dim: int = 2,
        target_dim: int = 2,
        hidden_dims: Tuple[int, ...] = (64, 64, 64),
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 100,
        noise_schedule: str = "linear",  # for CT
        sigma_t: float = 0.05,  # for FM
        integration_method: str = "euler",
        num_steps: int = 10,
        random_seed: int = 42
    ):
        """Initialize the trainer."""
        self.model_type = model_type
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.integration_method = integration_method
        self.num_steps = num_steps
        self.random_seed = random_seed
        
        # Create model
        if model_type == "ct":
            if noise_schedule == "linear":
                noise_sched = LinearNoiseSchedule()
            elif noise_schedule == "cosine":
                noise_sched = CosineNoiseSchedule()
            else:
                raise ValueError(f"Unknown noise schedule: {noise_schedule}")
            
            self.model = NoPropCT(
                model=SimpleMLP(hidden_dims=hidden_dims),
                noise_schedule=noise_sched,
                target_dim=target_dim
            )
        elif model_type == "fm":
            self.model = NoPropFM(
                model=SimpleMLP(hidden_dims=hidden_dims),
                sigma_t=sigma_t,
                target_dim=target_dim
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        
        # Training state
        self.params = None
        self.opt_state = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epoch_times = []
        self.inference_times = []
    
    def _prepare_data(self, data, labels) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prepare data for training."""
        # Convert to numpy if needed
        if hasattr(data, 'numpy'):
            x = jnp.array(data.numpy())
        elif hasattr(data, 'cpu'):
            x = jnp.array(data.cpu().numpy())
        else:
            x = jnp.array(data)
            
        if hasattr(labels, 'numpy'):
            y = jnp.array(labels.numpy())
        elif hasattr(labels, 'cpu'):
            y = jnp.array(labels.cpu().numpy())
        else:
            y = jnp.array(labels)
        
        # Ensure correct shapes
        if x.ndim == 4:  # Image data (B, H, W, C)
            x = x.reshape(x.shape[0], -1)  # Flatten to (B, H*W*C)
        
        # Convert labels to one-hot for NoProp models
        y_onehot = jax.nn.one_hot(y, self.target_dim)
        
        return x, y_onehot
    
    def _initialize_params(self, x_sample: jnp.ndarray) -> Tuple[Dict[str, Any], Any]:
        """Initialize model parameters and optimizer state."""
        key = jr.PRNGKey(self.random_seed)
        
        # Initialize model parameters
        params = self.model.init(
            key, 
            jnp.zeros((1, self.target_dim)), 
            x_sample[:1], 
            jnp.array([0.0])
        )
        
        # Initialize optimizer state
        opt_state = self.optimizer.init(params)
        
        return params, opt_state
    
    def fit(
        self, 
        train_data, 
        train_labels, 
        val_data=None, 
        val_labels=None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Train the model using the built-in train_step method."""
        # Prepare data
        x_train, z_train = self._prepare_data(train_data, train_labels)
        
        if val_data is not None and val_labels is not None:
            x_val, z_val = self._prepare_data(val_data, val_labels)
        else:
            x_val, z_val = None, None
        
        # Initialize parameters
        self.params, self.opt_state = self._initialize_params(x_train)
        
        # Training loop
        for epoch in tqdm(range(self.num_epochs), desc="Training", disable=not verbose):
            epoch_start = time.time()
            
            # Training
            epoch_train_losses = []
            epoch_train_mses = []
            
            # Shuffle data
            key = jr.PRNGKey(self.random_seed + epoch)
            indices = jr.permutation(key, len(x_train))
            x_train_shuffled = x_train[indices]
            z_train_shuffled = z_train[indices]
            
            # Process batches
            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train_shuffled[i:i+self.batch_size]
                z_batch = z_train_shuffled[i:i+self.batch_size]
                
                # Pad with random samples if needed
                if len(x_batch) < self.batch_size:
                    n_pad = self.batch_size - len(x_batch)
                    key, subkey = jr.split(key)
                    pad_indices = jr.choice(subkey, len(x_train), (n_pad,))
                    x_batch = jnp.concatenate([x_batch, x_train[pad_indices]], axis=0)
                    z_batch = jnp.concatenate([z_batch, z_train[pad_indices]], axis=0)
                
                # Training step using the model's built-in train_step
                key, subkey = jr.split(key)
                self.params, self.opt_state, loss, metrics = self.model.train_step(
                    self.params, self.opt_state, x_batch, z_batch, subkey, self.optimizer
                )
                
                epoch_train_losses.append(loss)
                epoch_train_mses.append(metrics['mse'])
            
            # Validation
            if x_val is not None:
                val_loss, val_metrics = self.model.compute_loss(
                    self.params, x_val, z_val, jr.PRNGKey(42)
                )
                epoch_val_loss = val_loss
                epoch_val_mse = val_metrics['mse']
            else:
                epoch_val_loss = 0.0
                epoch_val_mse = 0.0
            
            # Compute accuracies
            if self.model_type == "ct":
                # For CT, we need to predict and compare
                z_train_pred = self.model.predict(
                    self.params, x_train, self.target_dim, self.num_steps, self.integration_method
                )
                train_pred_classes = jnp.argmax(z_train_pred, axis=-1)
                train_true_classes = jnp.argmax(z_train, axis=-1)
                epoch_train_acc = jnp.mean(train_pred_classes == train_true_classes)
                
                if x_val is not None:
                    z_val_pred = self.model.predict(
                        self.params, x_val, self.target_dim, self.num_steps, self.integration_method
                    )
                    val_pred_classes = jnp.argmax(z_val_pred, axis=-1)
                    val_true_classes = jnp.argmax(z_val, axis=-1)
                    epoch_val_acc = jnp.mean(val_pred_classes == val_true_classes)
                else:
                    epoch_val_acc = 0.0
            else:
                # For FM, use the metrics from compute_loss
                epoch_train_acc = metrics.get('accuracy', 0.0)
                epoch_val_acc = val_metrics.get('accuracy', 0.0) if x_val is not None else 0.0
            
            # Record epoch timing
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            # Store history
            self.train_losses.append(np.mean(epoch_train_losses))
            self.val_losses.append(epoch_val_loss)
            self.train_accs.append(epoch_train_acc)
            self.val_accs.append(epoch_val_acc)
            
            # Print progress
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {self.train_losses[-1]:.4f}, "
                      f"Val Loss = {self.val_losses[-1]:.4f}, Train Acc = {self.train_accs[-1]:.4f}, "
                      f"Val Acc = {self.val_accs[-1]:.4f}, Time = {epoch_time:.3f}s")
        
        # Measure inference time
        if x_val is not None:
            inference_start = time.time()
            _ = self.predict(x_val[:10])  # Predict on small batch
            inference_time = (time.time() - inference_start) / 10  # Per sample
            self.inference_times.append(inference_time)
        else:
            self.inference_times.append(0.0)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'epoch_times': self.epoch_times,
            'inference_times': self.inference_times,
            'params': self.params
        }
    
    def predict(self, x_data) -> jnp.ndarray:
        """Make predictions on new data."""
        if self.params is None:
            raise ValueError("Model must be trained before making predictions")
        
        x, _ = self._prepare_data(x_data, np.zeros(len(x_data)))
        
        return self.model.predict(
            self.params, x, self.target_dim, self.num_steps, self.integration_method
        )
    
    def evaluate(self, x_data, y_data) -> Tuple[float, float]:
        """Evaluate model on data."""
        if self.params is None:
            raise ValueError("Model must be trained before evaluation")
        
        x, z = self._prepare_data(x_data, y_data)
        
        # Get predictions
        z_pred = self.model.predict(
            self.params, x, self.target_dim, self.num_steps, self.integration_method
        )
        
        # Compute accuracy
        pred_classes = jnp.argmax(z_pred, axis=-1)
        true_classes = jnp.argmax(z, axis=-1)
        accuracy = jnp.mean(pred_classes == true_classes)
        
        # Compute loss
        loss, metrics = self.model.compute_loss(self.params, x, z, jr.PRNGKey(42))
        
        return loss, accuracy


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    
    # Generate data
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🌙 Training NoProp-CT")
    print("=" * 50)
    
    # Train NoProp-CT
    ct_trainer = NoPropTrainer(
        model_type="ct",
        input_dim=2,
        target_dim=2,
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=50
    )
    
    ct_history = ct_trainer.fit(x_train, y_train, x_val, y_val)
    
    # Evaluate
    ct_loss, ct_acc = ct_trainer.evaluate(x_val, y_val)
    print(f"\nNoProp-CT Results:")
    print(f"  Final validation loss: {ct_loss:.4f}")
    print(f"  Final validation accuracy: {ct_acc:.4f}")
    print(f"  Average epoch time: {np.mean(ct_history['epoch_times']):.3f}s")
    print(f"  Inference time per sample: {ct_history['inference_times'][-1]*1000:.2f}ms")
    
    print("\n🌙 Training NoProp-FM")
    print("=" * 50)
    
    # Train NoProp-FM
    fm_trainer = NoPropTrainer(
        model_type="fm",
        input_dim=2,
        target_dim=2,
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=50
    )
    
    fm_history = fm_trainer.fit(x_train, y_train, x_val, y_val)
    
    # Evaluate
    fm_loss, fm_acc = fm_trainer.evaluate(x_val, y_val)
    print(f"\nNoProp-FM Results:")
    print(f"  Final validation loss: {fm_loss:.4f}")
    print(f"  Final validation accuracy: {fm_acc:.4f}")
    print(f"  Average epoch time: {np.mean(fm_history['epoch_times']):.3f}s")
    print(f"  Inference time per sample: {fm_history['inference_times'][-1]*1000:.2f}ms")
    
    print("\n🎉 Training completed successfully!")
