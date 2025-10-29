#!/usr/bin/env python3
"""
Parameter sweep script for testing different recon_weight and decoder_type configurations.

This script systematically tests different parameter combinations and saves results
for comparison.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import argparse
import os
import pickle
import json
from itertools import product
from flax.core import FrozenDict
import time

from .df import VAEFlowConfig as DiffusionConfig
from .trainer import VAEFlowTrainer


def load_two_moons_data(data_path: str = "data/two_moons_formatted.pkl") -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load the two moons dataset."""
    print(f"Loading two moons dataset from {data_path}...")
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract train and validation data
        x_train = jnp.array(data['train']['x'])
        y_train = jnp.array(data['train']['y'])
        x_val = jnp.array(data['val']['x'])
        y_val = jnp.array(data['val']['y'])
        
        # Convert binary labels from 0/1 to -1/1
        y_train = 2 * y_train - 1
        y_val = 2 * y_val - 1
        
        print(f"Loaded dataset:")
        print(f"  Training: x={x_train.shape}, y={y_train.shape}")
        print(f"  Validation: x={x_val.shape}, y={y_val.shape}")
        
        return x_train, y_train, x_val, y_val
        
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {data_path}")
        raise


def create_config(
    recon_weight: float,
    decoder_type: str,
    input_shape: Tuple[int, ...] = (2,),
    output_shape: Tuple[int, ...] = (2,),
    latent_shape: Tuple[int, ...] = (2,),
    crn_type: str = "vanilla",
    network_type: str = "mlp",
    hidden_dims: Tuple[int, ...] = (64, 64, 64)
) -> DiffusionConfig:
    """Create VAE flow configuration with specified parameters."""
    
    base_config = FrozenDict({
        "input_shape": input_shape,
        "output_shape": output_shape,
        "latent_shape": latent_shape,
        "recon_loss_type": "mse" if decoder_type != "none" else "none",
        "recon_weight": recon_weight,
        "reg_weight": 0.0,
        "integration_method": "midpoint",
    })
    
    crn_config = FrozenDict({
        "model_type": crn_type,
        "network_type": network_type,
        "hidden_dims": hidden_dims,
        "time_embed_dim": 32,
        "time_embed_method": "sinusoidal",
        "dropout_rate": 0.1,
        "activation_fn": "swish",
        "use_batch_norm": False
    })
    
    encoder_config = FrozenDict({
        "model_type": "identity",
        "encoder_type": "deterministic",
        "input_shape": input_shape,
        "latent_shape": latent_shape,
        "hidden_dims": (16, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.0
    })
    
    decoder_config = FrozenDict({
        "model_type": "mlp" if decoder_type != "none" else "identity",
        "decoder_type": decoder_type,
        "latent_shape": latent_shape,
        "output_shape": output_shape,
        "hidden_dims": (16, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.0
    })
    
    return DiffusionConfig(
        config=base_config,
        crn_config=crn_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config
    )


def run_single_experiment(
    recon_weight: float,
    decoder_type: str,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    num_epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42
) -> Dict:
    """Run a single experiment with given parameters."""
    
    print(f"\n{'='*60}")
    print(f"Running experiment: recon_weight={recon_weight}, decoder_type={decoder_type}")
    print(f"{'='*60}")
    
    # Create configuration
    config = create_config(
        recon_weight=recon_weight,
        decoder_type=decoder_type,
        input_shape=(x_train.shape[1],),
        output_shape=(y_train.shape[1],),
        latent_shape=(2,)
    )
    
    # Create trainer
    trainer = VAEFlowTrainer(
        config=config,
        learning_rate=learning_rate,
        optimizer_name="adam",
        seed=seed
    )
    
    # Initialize model
    batch_size_init = min(batch_size, x_train.shape[0])
    x_sample = x_train[:batch_size_init]
    y_sample = y_train[:batch_size_init]
    z_sample = jr.normal(jr.PRNGKey(seed), (batch_size_init, 2))
    t_sample = jr.uniform(jr.PRNGKey(seed + 1), (batch_size_init,), minval=0.0, maxval=1.0)
    
    trainer.initialize(x_sample, y_sample, z_sample, t_sample)
    
    # Train model
    start_time = time.time()
    history = trainer.train(
        x_data=x_train,
        y_data=y_train,
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        dropout_epochs=num_epochs,
        verbose=False
    )
    training_time = time.time() - start_time
    
    # Extract final metrics
    final_train_loss = history['train_losses'][-1]
    final_val_loss = history['val_losses'][-1] if history['val_losses'] else None
    final_flow_loss = history['train_flow_losses'][-1]
    final_recon_loss = history['train_recon_losses'][-1]
    
    # Compute accuracy
    train_acc = trainer.compute_accuracy(x_train[:500], y_train[:500])
    val_acc = trainer.compute_accuracy(x_val[:500], y_val[:500])
    
    # Generate predictions for analysis
    train_pred = trainer.predict(x_train[:100])
    val_pred = trainer.predict(x_val[:100])
    
    results = {
        'recon_weight': recon_weight,
        'decoder_type': decoder_type,
        'final_train_loss': float(final_train_loss),
        'final_val_loss': float(final_val_loss) if final_val_loss is not None else None,
        'final_flow_loss': float(final_flow_loss),
        'final_recon_loss': float(final_recon_loss),
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'training_time': training_time,
        'num_epochs': num_epochs,
        'history': history,
        'train_pred': np.array(train_pred),
        'val_pred': np.array(val_pred)
    }
    
    print(f"Results:")
    print(f"  Final train loss: {final_train_loss:.6f}")
    print(f"  Final val loss: {final_val_loss:.6f}" if final_val_loss else "  Final val loss: N/A")
    print(f"  Final flow loss: {final_flow_loss:.6f}")
    print(f"  Final recon loss: {final_recon_loss:.6f}")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Val accuracy: {val_acc:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    
    return results


def run_parameter_sweep(
    recon_weights: List[float],
    decoder_types: List[str],
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: jnp.ndarray,
    num_epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42,
    save_dir: str = "artifacts/parameter_sweep"
) -> Dict:
    """Run parameter sweep across all combinations."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate all parameter combinations
    param_combinations = list(product(recon_weights, decoder_types))
    total_experiments = len(param_combinations)
    
    print(f"Starting parameter sweep with {total_experiments} experiments")
    print(f"Recon weights: {recon_weights}")
    print(f"Decoder types: {decoder_types}")
    print(f"Save directory: {save_dir}")
    
    all_results = []
    
    for i, (recon_weight, decoder_type) in enumerate(param_combinations):
        print(f"\nExperiment {i+1}/{total_experiments}")
        
        try:
            results = run_single_experiment(
                recon_weight=recon_weight,
                decoder_type=decoder_type,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed + i  # Different seed for each experiment
            )
            all_results.append(results)
            
            # Save individual results
            exp_name = f"recon_{recon_weight}_decoder_{decoder_type}"
            exp_dir = os.path.join(save_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Save results (without history to save space)
            results_to_save = {k: v for k, v in results.items() if k != 'history'}
            with open(os.path.join(exp_dir, "results.json"), 'w') as f:
                json.dump(results_to_save, f, indent=2)
            
        except Exception as e:
            print(f"Error in experiment {i+1}: {e}")
            continue
    
    # Save summary results
    summary = {
        'experiments': all_results,
        'parameter_combinations': param_combinations,
        'total_experiments': len(all_results),
        'successful_experiments': len(all_results)
    }
    
    with open(os.path.join(save_dir, "sweep_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create comparison plots
    create_comparison_plots(all_results, save_dir)
    
    return summary


def create_comparison_plots(results: List[Dict], save_dir: str):
    """Create comparison plots across different parameter settings."""
    
    if not results:
        print("No results to plot")
        return
    
    # Extract data for plotting
    recon_weights = [r['recon_weight'] for r in results]
    decoder_types = [r['decoder_type'] for r in results]
    train_losses = [r['final_train_loss'] for r in results]
    val_losses = [r['final_val_loss'] for r in results if r['final_val_loss'] is not None]
    train_accs = [r['train_accuracy'] for r in results]
    val_accs = [r['val_accuracy'] for r in results]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Parameter Sweep Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Train Loss vs Recon Weight (by decoder type)
    decoder_type_unique = list(set(decoder_types))
    colors = plt.cm.tab10(np.linspace(0, 1, len(decoder_type_unique)))
    
    for i, dt in enumerate(decoder_type_unique):
        mask = [dt == d for d in decoder_types]
        rw_subset = [rw for j, rw in enumerate(recon_weights) if mask[j]]
        tl_subset = [tl for j, tl in enumerate(train_losses) if mask[j]]
        
        axes[0, 0].plot(rw_subset, tl_subset, 'o-', color=colors[i], label=f'decoder={dt}', linewidth=2, markersize=8)
    
    axes[0, 0].set_xlabel('Reconstruction Weight')
    axes[0, 0].set_ylabel('Final Train Loss')
    axes[0, 0].set_title('Train Loss vs Recon Weight')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Plot 2: Validation Loss vs Recon Weight (by decoder type)
    for i, dt in enumerate(decoder_type_unique):
        mask = [dt == d for d in decoder_types]
        rw_subset = [rw for j, rw in enumerate(recon_weights) if mask[j]]
        vl_subset = [vl for j, vl in enumerate(val_losses) if mask[j]]
        
        if vl_subset:  # Only plot if we have validation losses
            axes[0, 1].plot(rw_subset, vl_subset, 'o-', color=colors[i], label=f'decoder={dt}', linewidth=2, markersize=8)
    
    axes[0, 1].set_xlabel('Reconstruction Weight')
    axes[0, 1].set_ylabel('Final Val Loss')
    axes[0, 1].set_title('Validation Loss vs Recon Weight')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # Plot 3: Train Accuracy vs Recon Weight (by decoder type)
    for i, dt in enumerate(decoder_type_unique):
        mask = [dt == d for d in decoder_types]
        rw_subset = [rw for j, rw in enumerate(recon_weights) if mask[j]]
        ta_subset = [ta for j, ta in enumerate(train_accs) if mask[j]]
        
        axes[1, 0].plot(rw_subset, ta_subset, 'o-', color=colors[i], label=f'decoder={dt}', linewidth=2, markersize=8)
    
    axes[1, 0].set_xlabel('Reconstruction Weight')
    axes[1, 0].set_ylabel('Train Accuracy')
    axes[1, 0].set_title('Train Accuracy vs Recon Weight')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Plot 4: Validation Accuracy vs Recon Weight (by decoder type)
    for i, dt in enumerate(decoder_type_unique):
        mask = [dt == d for d in decoder_types]
        rw_subset = [rw for j, rw in enumerate(recon_weights) if mask[j]]
        va_subset = [va for j, va in enumerate(val_accs) if mask[j]]
        
        axes[1, 1].plot(rw_subset, va_subset, 'o-', color=colors[i], label=f'decoder={dt}', linewidth=2, markersize=8)
    
    axes[1, 1].set_xlabel('Reconstruction Weight')
    axes[1, 1].set_ylabel('Val Accuracy')
    axes[1, 1].set_title('Validation Accuracy vs Recon Weight')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    plot_file = os.path.join(save_dir, "parameter_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {plot_file}")


def main():
    """Main function for parameter sweep."""
    parser = argparse.ArgumentParser(description='Parameter sweep for recon_weight and decoder_type')
    parser.add_argument('--data_path', type=str, default='data/two_moons_formatted.pkl',
                       help='Path to two moons dataset')
    parser.add_argument('--recon_weights', type=float, nargs='+', 
                       default=[0.0, 0.01, 0.1, 1.0, 10.0],
                       help='Reconstruction weights to test')
    parser.add_argument('--decoder_types', type=str, nargs='+',
                       default=['none', 'linear', 'softmax'],
                       help='Decoder types to test')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='artifacts/parameter_sweep',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Parameter Sweep for Diffusion Model")
    print("=" * 80)
    print(f"Recon weights: {args.recon_weights}")
    print(f"Decoder types: {args.decoder_types}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 80)
    
    # Load data
    x_train, y_train, x_val, y_val = load_two_moons_data(args.data_path)
    
    # Run parameter sweep
    results = run_parameter_sweep(
        recon_weights=args.recon_weights,
        decoder_types=args.decoder_types,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        save_dir=args.save_dir
    )
    
    print(f"\nParameter sweep completed!")
    print(f"Successful experiments: {results['successful_experiments']}/{results['total_experiments']}")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
