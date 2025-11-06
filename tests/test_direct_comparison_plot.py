"""
Test script to load saved model and generate direct comparison plot.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pickle
import numpy as np
import jax.numpy as jnp
import jax.random as jr

from src.flow_models.trainer_seq import SequenceTrainer
from src.flow_models.fm import VAEFlowConfig as FMConfig
from src.flow_models.df import VAEFlowConfig as DFConfig
from src.flow_models.ct import VAEFlowConfig as CTConfig


def main():
    # Paths
    save_dir = "artifacts/stock_sequences"
    data_path = "data/stock_sequences_full_day_2d.pkl"
    model_params_path = os.path.join(save_dir, "model_params.pkl")
    results_path = os.path.join(save_dir, "results.pkl")
    
    # Check if files exist
    if not os.path.exists(model_params_path):
        print(f"Error: Model params not found at {model_params_path}")
        return
    if not os.path.exists(results_path):
        print(f"Error: Results not found at {results_path}")
        return
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return
    
    # Load results to get config
    print(f"Loading results from {results_path}...")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    config = results['config']
    model_type = results['model_type']
    print(f"Model type: {model_type}")
    
    # Convert config dict back to config object
    if model_type == 'diffusion':
        config = DFConfig(**config)
    elif model_type == 'ct':
        config = CTConfig(**config)
    else:
        config = FMConfig(**config)
    
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    val_sequences = data['val']['sequences']
    y_seq_len = data.get('metadata', {}).get('y_seq_len', 12)
    print(f"Loaded {len(val_sequences)} validation sequences")
    print(f"Target sequence length: {y_seq_len}")
    
    # Create trainer
    trainer = SequenceTrainer(
        config=config,
        learning_rate=results.get('learning_rate', 1e-3),
        seed=42
    )
    
    # Load model params
    print(f"Loading model params from {model_params_path}...")
    with open(model_params_path, 'rb') as f:
        trainer.params = pickle.load(f)
    
    # Generate predictions
    print("\nGenerating predictions...")
    num_gen = min(100, len(val_sequences))
    key = jr.PRNGKey(42)
    key, gen_key = jr.split(key)
    
    # Create a sample batch with random splits for generation
    eval_indices = jnp.arange(num_gen)
    cond_x, y_real = trainer._create_minibatch_with_random_splits(
        val_sequences, eval_indices, y_seq_len=y_seq_len
    )
    y_gen = np.array(trainer.conditional_generate(cond_x, num_steps=20))
    y_real_np = np.array(y_real)
    
    print(f"Generated sequences: {y_gen.shape}")
    print(f"Real sequences: {y_real_np.shape}")
    
    # Compute metrics
    metrics = trainer.compute_sequence_metrics(jnp.array(y_gen), jnp.array(y_real_np))
    print(f"\nMetrics:")
    print(f"  MSE: {metrics.get('mse', 'N/A')}")
    print(f"  MAE: {metrics.get('mae', 'N/A')}")
    if 'percent_variance_explained' in metrics:
        pve = metrics['percent_variance_explained']
        if np.isfinite(pve):
            print(f"  Percent Variance Explained: {pve:.2f}%")
        else:
            print(f"  Percent Variance Explained: N/A")
    
    # Test the direct comparison plot
    print("\nGenerating direct comparison plot...")
    trainer.save_direct_comparison_plot(
        y_real=y_real_np,
        y_pred=y_gen,
        output_dir=save_dir,
        num_samples=100
    )
    
    # Test the trajectory comparison plot
    print("\nGenerating trajectory comparison plot...")
    trainer.save_trajectory_comparison_plot(
        y_real=y_real_np,
        y_pred=y_gen,
        output_dir=save_dir,
        num_samples=20
    )
    
    print(f"\n✓ Direct comparison plot saved to {save_dir}/direct_comparison_model_space.png")
    print(f"✓ Trajectory comparison plot saved to {save_dir}/trajectory_comparison_model_space.png")


if __name__ == '__main__':
    main()

