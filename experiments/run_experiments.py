#!/usr/bin/env python3
"""
Experiment runner script for systematic parameter testing.

This script runs multiple experiments with different parameter configurations
and tracks all results in a comprehensive markdown report.
"""

import subprocess
import sys
import os
import time
from typing import List, Dict, Any
import argparse
from itertools import product


def run_experiment(
    recon_weight: float,
    decoder_type: str,
    recon_loss_type: str = "mse",
    reg_weight: float = 0.0,
    model_type: str = "diffusion",
    num_epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    data_path: str = "data/two_moons_formatted.pkl",
    results_file: str = "experiment_results.md",
    verbose: bool = True
) -> bool:
    """
    Run a single experiment with given parameters.
    
    Returns:
        bool: True if experiment succeeded, False otherwise
    """
    cmd = [
        sys.executable, "-m", "src.flow_models.train",
        "--model_type", model_type,
        "--recon_weight", str(recon_weight),
        "--decoder_type", decoder_type,
        "--recon_loss_type", recon_loss_type,
        "--reg_weight", str(reg_weight),
        "--num_epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--data_path", data_path,
        "--results_file", results_file
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    print(f"\n{'='*80}")
    print(f"Running experiment: recon_weight={recon_weight}, decoder_type={decoder_type}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Experiment completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        return False


def run_parameter_sweep(
    recon_weights: List[float],
    decoder_types: List[str],
    recon_loss_types: List[str] = ["mse"],
    reg_weights: List[float] = [0.0],
    model_type: str = "diffusion",
    num_epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    data_path: str = "data/two_moons_formatted.pkl",
    results_file: str = "experiment_results.md",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run parameter sweep across all combinations.
    
    Returns:
        Dictionary with experiment results summary
    """
    # Generate all parameter combinations
    param_combinations = list(product(recon_weights, decoder_types, recon_loss_types, reg_weights))
    total_experiments = len(param_combinations)
    
    print(f"Starting parameter sweep with {total_experiments} experiments")
    print(f"Recon weights: {recon_weights}")
    print(f"Decoder types: {decoder_types}")
    print(f"Recon loss types: {recon_loss_types}")
    print(f"Reg weights: {reg_weights}")
    print(f"Results file: {results_file}")
    
    results = {
        'total_experiments': total_experiments,
        'successful_experiments': 0,
        'failed_experiments': 0,
        'experiments': []
    }
    
    start_time = time.time()
    
    for i, (recon_weight, decoder_type, recon_loss_type, reg_weight) in enumerate(param_combinations):
        print(f"\n🔄 Experiment {i+1}/{total_experiments}")
        
        experiment_start = time.time()
        success = run_experiment(
            recon_weight=recon_weight,
            decoder_type=decoder_type,
            recon_loss_type=recon_loss_type,
            reg_weight=reg_weight,
            model_type=model_type,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            data_path=data_path,
            results_file=results_file,
            verbose=verbose
        )
        experiment_time = time.time() - experiment_start
        
        experiment_result = {
            'recon_weight': recon_weight,
            'decoder_type': decoder_type,
            'recon_loss_type': recon_loss_type,
            'reg_weight': reg_weight,
            'success': success,
            'time': experiment_time
        }
        
        results['experiments'].append(experiment_result)
        
        if success:
            results['successful_experiments'] += 1
        else:
            results['failed_experiments'] += 1
        
        # Print progress
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = (total_experiments - i - 1) * avg_time
        
        print(f"Progress: {i+1}/{total_experiments} ({100*(i+1)/total_experiments:.1f}%)")
        print(f"Elapsed: {elapsed/60:.1f}m, ETA: {remaining/60:.1f}m")
    
    total_time = time.time() - start_time
    results['total_time'] = total_time
    results['avg_time_per_experiment'] = total_time / total_experiments
    
    print(f"\n🎉 Parameter sweep completed!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {results['successful_experiments']}/{total_experiments}")
    print(f"Failed: {results['failed_experiments']}/{total_experiments}")
    print(f"Average time per experiment: {results['avg_time_per_experiment']/60:.1f} minutes")
    
    return results


def main():
    """Main function for experiment runner."""
    parser = argparse.ArgumentParser(description='Run systematic parameter experiments')
    
    # Parameter ranges
    parser.add_argument('--recon_weights', type=float, nargs='+', 
                       default=[0.0, 0.01, 0.1, 1.0, 10.0],
                       help='Reconstruction weights to test')
    parser.add_argument('--decoder_types', type=str, nargs='+',
                       default=['none', 'linear', 'softmax'],
                       help='Decoder types to test')
    parser.add_argument('--recon_loss_types', type=str, nargs='+',
                       default=['mse'],
                       help='Reconstruction loss types to test')
    parser.add_argument('--reg_weights', type=float, nargs='+',
                       default=[0.0],
                       help='Regularization weights to test')
    
    # Training parameters
    parser.add_argument('--model_type', type=str, default='diffusion',
                       choices=['diffusion', 'flow_matching'],
                       help='Model type')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    
    # Data and output
    parser.add_argument('--data_path', type=str, default='data/two_moons_formatted.pkl',
                       help='Path to dataset')
    parser.add_argument('--results_file', type=str, default='experiment_results.md',
                       help='Markdown file for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    # Quick test options
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with fewer parameters')
    parser.add_argument('--single_test', action='store_true',
                       help='Run single test with default parameters')
    
    args = parser.parse_args()
    
    # Quick test configuration
    if args.quick_test:
        args.recon_weights = [0.0, 0.1, 1.0]
        args.decoder_types = ['none', 'linear']
        args.num_epochs = 10
        print("🚀 Running quick test with reduced parameters")
    
    # Single test configuration
    if args.single_test:
        args.recon_weights = [1.0]
        args.decoder_types = ['none']
        args.num_epochs = 20
        print("🧪 Running single test")
    
    # Run experiments
    results = run_parameter_sweep(
        recon_weights=args.recon_weights,
        decoder_types=args.decoder_types,
        recon_loss_types=args.recon_loss_types,
        reg_weights=args.reg_weights,
        model_type=args.model_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_path=args.data_path,
        results_file=args.results_file,
        verbose=args.verbose
    )
    
    # Generate detailed report
    print(f"\n📊 Generating detailed report...")
    try:
        from results_tracker import ResultsTracker
        tracker = ResultsTracker(args.results_file)
        report_file = tracker.generate_detailed_report()
        print(f"📄 Detailed report saved to: {report_file}")
    except ImportError:
        try:
            from src.flow_models.results_tracker import ResultsTracker
            tracker = ResultsTracker(args.results_file)
            report_file = tracker.generate_detailed_report()
            print(f"📄 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"⚠️  Could not generate detailed report: {e}")
    except Exception as e:
        print(f"⚠️  Could not generate detailed report: {e}")
    
    print(f"\n✅ All experiments completed!")
    print(f"Results summary saved to: {args.results_file}")


if __name__ == "__main__":
    main()
