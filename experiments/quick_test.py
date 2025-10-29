#!/usr/bin/env python3
"""
Quick test script to demonstrate parameter testing with a few combinations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from parameter_sweep import run_parameter_sweep, load_two_moons_data
import jax

def main():
    """Run a quick test with a few parameter combinations."""
    
    print("Quick Parameter Test for Diffusion Model")
    print("=" * 50)
    
    # Load data
    try:
        x_train, y_train, x_val, y_val = load_two_moons_data("data/two_moons_formatted.pkl")
    except FileNotFoundError:
        print("Two moons dataset not found. Please ensure it's available at data/two_moons_formatted.pkl")
        return
    
    # Test a few parameter combinations
    recon_weights = [0.0, 0.1, 1.0]  # Small set for quick testing
    decoder_types = ['none', 'linear']  # Test two decoder types
    
    print(f"Testing {len(recon_weights) * len(decoder_types)} combinations:")
    print(f"  Recon weights: {recon_weights}")
    print(f"  Decoder types: {decoder_types}")
    
    # Run parameter sweep
    results = run_parameter_sweep(
        recon_weights=recon_weights,
        decoder_types=decoder_types,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        num_epochs=10,  # Short training for quick test
        batch_size=256,
        learning_rate=1e-3,
        seed=42,
        save_dir="artifacts/quick_test"
    )
    
    print("\nQuick test completed!")
    print(f"Results saved to: artifacts/quick_test")
    
    # Print summary
    print("\nSummary of results:")
    for i, exp in enumerate(results['experiments']):
        print(f"  {i+1}. recon_weight={exp['recon_weight']}, decoder_type={exp['decoder_type']}")
        print(f"     Train loss: {exp['final_train_loss']:.6f}, Val loss: {exp['final_val_loss']:.6f}")
        print(f"     Train acc: {exp['train_accuracy']:.4f}, Val acc: {exp['val_accuracy']:.4f}")

if __name__ == "__main__":
    main()
