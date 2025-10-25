"""
Unified trainer for NoProp models (FM, CT, DF).

This trainer can handle all three training protocols with identical training procedures
to ensure fair comparison.
"""

from typing import Any, Dict, Tuple, Optional
import time
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

# from src.configs.base_trainer import BaseETTrainer
# from src.configs.base_training_config import BaseTrainingConfig
from .fm import NoPropFM, Config as FMConfig
from .ct import NoPropCT, Config as CTConfig
from .df import NoPropDF, Config as DFConfig  

# Trajectory plotting functions are imported in save_results method
import pickle
from typing import Dict, Any, Tuple


def load_training_data(data_path: str) -> Tuple[Dict[str, Any], int, int]:
    """
    Load training data from pickle file.
    
    Args:
        data_path: Path to the pickle file containing training data
        
    Returns:
        Tuple of (data_dict, x_dim, y_dim)
    """
    print(f"Loading training data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract dimensions
    x_dim = data['train']['x'].shape[-1]
    y_dim = data['train']['y'].shape[-1]
    
    return data, x_dim, y_dim


class NoPropTrainer:
    """
    Unified trainer for NoProp models (FM, CT, DF) with CRN architectures.
    
    This trainer supports all training protocols (FM, CT, DF) and all model architectures
    (potential_flow, geometric_flow, natural_flow, conditional_resnet_mlp).
    """
    
    def __init__(self, model):
        """
        Initialize the trainer.
        
        Args:
            model: The NoProp model to train (NoPropFM, NoPropCT, or NoPropDF)
        """
        self.model = model
        # Determine model type from the model instance
        if isinstance(model, NoPropFM):
            self.model_type = "FM"
        elif isinstance(model, NoPropCT):
            self.model_type = "CT"
        elif isinstance(model, NoPropDF):
            self.model_type = "DF"
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        
    def _get_model_type(self) -> str:
        """Return model type."""
        return self.model_type
    
    def train(self, 
              train_x: jnp.ndarray, 
              train_y: jnp.ndarray,
              val_x: jnp.ndarray, 
              val_y: jnp.ndarray,
              num_epochs: int,
              test_x: jnp.ndarray = None,
              test_y: jnp.ndarray = None,
              dropout_epochs: Optional[int] = None,
              learning_rate: Optional[float] = None,
              batch_size: Optional[int] = None,
              eval_steps: int = 10,
              save_steps: Optional[int] = None,
              output_dir: str = None) -> Dict[str, Any]:
        """
        Train the NoProp model using the unified protocol.
        
        This ensures identical training procedures across all model types.
        
        Args:
            train_x: Training input data [N_train, x_dim]
            train_y: Training target data [N_train, y_dim]
            val_x: Validation input data [N_val, x_dim]
            val_y: Validation target data [N_val, y_dim]
            num_epochs: Number of training epochs
            test_x: Test input data [N_test, x_dim] (optional)
            test_y: Test target data [N_test, y_dim] (optional)
            dropout_epochs: Number of epochs with dropout (if None, uses all epochs)
            learning_rate: Learning rate (if None, uses config default)
            batch_size: Batch size (if None, uses config default)
            eval_steps: Steps between detailed evaluation prints
            save_steps: Steps between model saves (if None, saves only at end)
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing training results
        """
        print(f"Unified NoProp {self.model_type} Training")
        print("="*60)
        
        # Use config defaults if not provided
        if learning_rate is None:
            learning_rate = self.model.config.learning_rate
        if batch_size is None:
            batch_size = self.model.config.batch_size
        if dropout_epochs is None:
            dropout_epochs = num_epochs
        
        # Print model-specific information
        if hasattr(self.model, 'model'):
            print(f"{self.model_type} Model: {self.model.model}")
        else:
            print(f"{self.model_type} Model: {type(self.model).__name__}")
        print(f"Epochs: {num_epochs}")
        print(f"Dropout epochs: {dropout_epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        
        # Initialize model parameters
        print("Initializing model parameters...")
        rng = jr.PRNGKey(42)
        rng, init_key = jr.split(rng)
        
        # Create dummy batch for initialization (following individual trainer pattern)
        z_sample = jnp.zeros_like(train_y[:1])
        t_sample = jnp.array([0.0])  # Make it an array with batch dimension
        x_sample = train_x[:1]
        
        # Initialize parameters using the standard NoProp pattern
        params = self.model.init(init_key, z_sample, x_sample, t_sample, training=True)
        print(f"Model parameters initialized: {sum(x.size for x in jax.tree.leaves(params)):,} parameters")
        
        # Set up optimizer
        optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
        opt_state = optimizer.init(params)
        
        # Use the model's pre-compiled train_step method for efficiency
        
        # Training loop
        print(f"\nStarting training for {num_epochs} epochs...")
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        # Start timing the training
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            # Determine if we should use dropout
            use_dropout = epoch < dropout_epochs
            
            # Training step using model's JIT-compiled train_step method
            rng, train_key = jr.split(rng)
            
            # Shuffle data for this epoch
            key, shuffle_key = jr.split(train_key)
            indices = jr.permutation(shuffle_key, len(train_x))
            x_shuffled = train_x[indices]
            y_shuffled = train_y[indices]
            
            # Process in batches using the model's train_step method
            num_batches = len(train_x) // batch_size
            epoch_losses = []
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_x = x_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]
                
                # Use the model's optimized train_step method
                key, batch_key = jr.split(key)
                params, opt_state, loss, metrics = self.model.train_step(
                    params, batch_x, batch_y, opt_state, optimizer, batch_key
                )
                
                epoch_losses.append(float(loss))
            
            # Average loss for this epoch
            avg_loss = jnp.mean(jnp.array(epoch_losses))
            train_losses.append(float(avg_loss))
            
            # Validation - compute every epoch
            val_rng = jr.PRNGKey(42)  # Fixed seed for validation
            val_loss, val_metrics = self.model.compute_loss(params, val_x, val_y, val_rng)
            val_losses.append(float(val_loss))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Print progress
            if epoch % eval_steps == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss = {loss:.6f}, Val Loss = {val_loss:.6f}")
            else:
                print(f"Epoch {epoch:3d}: Train Loss = {loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Calculate total training time
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        # Compute inference time
        start_time = time.time()
        _ = self.model.predict(params, val_x[:100], num_steps=20)
        inference_time = time.time() - start_time
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree.leaves(params))
        
        # Compute final MSE metrics for plotting using predict routine for consistency
        print("\nComputing final predictions...")
        train_pred = self.model.predict(params, train_x, num_steps=20)
        val_pred = self.model.predict(params, val_x, num_steps=20)
        
        # Compute MSE from predictions (consistent method for all datasets)
        final_train_mse = float(jnp.mean((train_pred - train_y) ** 2))
        final_val_mse = float(jnp.mean((val_pred - val_y) ** 2))
        
        # Compute test MSE if test data is provided
        if test_x is not None and test_y is not None:
            test_pred = self.model.predict(params, test_x, num_steps=20)
            final_test_mse = float(jnp.mean((test_pred - test_y) ** 2))
        else:
            final_test_mse = None
        
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'params': params,
            'optimizer_state': opt_state,
            'epochs': num_epochs,
            'total_training_time': total_training_time,
            'inference_time': inference_time,
            'inference_time_per_sample': inference_time / 100,
            'param_count': param_count,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1] if train_losses else 0.0,
            'final_val_loss': val_losses[-1] if val_losses else 0.0,
            'final_train_mse': final_train_mse,
            'final_val_mse': final_val_mse,
            'final_test_mse': final_test_mse,
            'train_x': train_x,
            'val_x': val_x,
            'test_x': test_x if test_x is not None else None,
            'train_y': train_y,
            'val_y': val_y,
            'test_y': test_y if test_y is not None else None,
            'train_pred': train_pred,
            'val_pred': val_pred,
            'test_pred': test_pred if test_x is not None and test_y is not None else None,
        }
        
        print(f"\nTraining completed!")
        print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
        print(f"Final train loss: {results['final_train_loss']:.6f}")
        print(f"Final val loss: {results['final_val_loss']:.6f}")
        print(f"Best val loss: {results['best_val_loss']:.6f}")
        print(f"Final train MSE: {results['final_train_mse']:.6f}")
        print(f"Final val MSE: {results['final_val_mse']:.6f}")
        if results['final_test_mse'] is not None:
            print(f"Final test MSE: {results['final_test_mse']:.6f}")
        print(f"Model parameters: {param_count:,}")
        
        return results
    
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save training results and generate plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training results
        import pickle
        with open(output_path / "training_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Generate plots
        print("Generating plots...")
        
        # Import plotting functions
        from src.utils.plotting.plot_learning_curves import create_enhanced_learning_plot
        from src.utils.plotting.plot_trajectories import (
            create_trajectory_diagnostic_plot,
            create_model_output_trajectory_plot,
            create_dzdt_trajectory_plot
        )
        
        # 1. Learning curve plot
        print("Generating learning curve plot...")
        try:
            learning_plot_path = output_path / "learning_analysis.png"
            create_enhanced_learning_plot(
                results=results,
                train_pred=results['train_pred'],
                val_pred=results['val_pred'],
                test_pred=results['test_pred'],
                train_y=results['train_y'],
                val_y=results['val_y'],
                test_y=results['test_y'],
                output_path=str(learning_plot_path),
                model_name=f"NoProp {self.model_type} Model",
                skip_epochs=4
            )
            print(f"Learning curve plot saved to: {learning_plot_path}")
        except Exception as e:
            print(f"Warning: Could not generate learning curve plot: {e}")
        
        # 2. Trajectory diagnostic plot
        print("Generating trajectory diagnostic plot...")
        try:
            # Get trajectories for a small sample of test data
            num_trajectory_samples = 10
            test_x_sample = results['test_x'][:num_trajectory_samples] if results['test_x'] is not None else results['val_x'][:num_trajectory_samples]
            test_y_sample = results['test_y'][:num_trajectory_samples] if results['test_y'] is not None else results['val_y'][:num_trajectory_samples]
            
            # Get trajectories using predict with output_type="trajectory"
            trajectories = self.model.predict(
                results['params'], 
                test_x_sample, 
                num_steps=20, 
                output_type="trajectory"
            )
            
            # Transpose trajectories from [num_steps, num_samples, output_dim] to [num_samples, num_steps, output_dim]
            import numpy as np
            trajectories_transposed = np.transpose(trajectories, (1, 0, 2))
            
            trajectory_plot_path = output_path / "trajectory_diagnostics.png"
            create_trajectory_diagnostic_plot(
                trajectories=trajectories_transposed,
                targets=test_y_sample,
                output_path=str(trajectory_plot_path),
                model_name=f"NoProp {self.model_type} Model",
                num_samples=num_trajectory_samples
            )
            print(f"Trajectory diagnostic plot saved to: {trajectory_plot_path}")
        except Exception as e:
            print(f"Warning: Could not generate trajectory diagnostic plot: {e}")
        
        # 3. Model output trajectory plot
        print("Generating model output trajectory plot...")
        try:
            model_output_plot_path = output_path / "model_output_trajectories.png"
            create_model_output_trajectory_plot(
                model=self.model,
                params=results['params'],
                x_sample=test_x_sample,
                target_sample=test_y_sample,
                output_path=str(model_output_plot_path),
                model_name=f"NoProp {self.model_type} Model",
                num_steps=20,
                num_samples=num_trajectory_samples
            )
            print(f"Model output trajectory plot saved to: {model_output_plot_path}")
        except Exception as e:
            print(f"Warning: Could not generate model output trajectory plot: {e}")
        
        # 4. dz/dt trajectory plot
        print("Generating dz/dt trajectory plot...")
        try:
            dzdt_plot_path = output_path / "dzdt_trajectories.png"
            create_dzdt_trajectory_plot(
                model=self.model,
                params=results['params'],
                x_sample=test_x_sample,
                target_sample=test_y_sample,
                output_path=str(dzdt_plot_path),
                model_name=f"NoProp {self.model_type} Model",
                num_steps=20,
                num_samples=num_trajectory_samples
            )
            print(f"dz/dt trajectory plot saved to: {dzdt_plot_path}")
        except Exception as e:
            print(f"Warning: Could not generate dz/dt trajectory plot: {e}")
        
        print(f"Results saved to: {output_path}")


# create_unified_config function removed - flow_models directory only uses FM models
