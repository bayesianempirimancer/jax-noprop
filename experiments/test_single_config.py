#!/usr/bin/env python3
"""
Test a single parameter configuration easily.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from parameter_sweep import run_single_experiment, load_two_moons_data
import argparse

def main():
    """Test a single parameter configuration."""
    
    parser = argparse.ArgumentParser(description='Test single parameter configuration')
    parser.add_argument('--recon_weight', type=float, default=1.0,
                       help='Reconstruction weight')
    parser.add_argument('--decoder_type', type=str, default='none',
                       choices=['none', 'linear', 'softmax'],
                       help='Decoder type')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--data_path', type=str, default='data/two_moons_formatted.pkl',
                       help='Path to two moons dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("Single Configuration Test")
    print("=" * 40)
    print(f"Recon weight: {args.recon_weight}")
    print(f"Decoder type: {args.decoder_type}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 40)
    
    # Load data
    try:
        x_train, y_train, x_val, y_val = load_two_moons_data(args.data_path)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {args.data_path}")
        return
    
    # Run single experiment
    results = run_single_experiment(
        recon_weight=args.recon_weight,
        decoder_type=args.decoder_type,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    print(f"\nExperiment completed!")
    print(f"Final train loss: {results['final_train_loss']:.6f}")
    print(f"Final val loss: {results['final_val_loss']:.6f}")
    print(f"Train accuracy: {results['train_accuracy']:.4f}")
    print(f"Val accuracy: {results['val_accuracy']:.4f}")
    print(f"Training time: {results['training_time']:.2f}s")

if __name__ == "__main__":
    main()
