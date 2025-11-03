#!/usr/bin/env python3
"""Script to test all noise schedules for Diffusion model and find the best based on Chamfer distance."""

import subprocess
import sys
from pathlib import Path
import pickle
import re

# Noise schedules to test (excluding neural network ones: monotonic_nn, network, and learnable)
NOISE_SCHEDULES = [
    'linear',
    'cosine',
    'sigmoid',
    'exponential',
    'cauchy',
    'laplace',
    'logistic',
    'quadratic',
    'polynomial',
]

RESULTS = {}

def run_training(noise_schedule: str):
    """Run training for a specific noise schedule."""
    print(f"\n{'='*60}")
    print(f"Testing noise schedule: {noise_schedule}")
    print(f"{'='*60}\n")
    
    cmd = [
        'conda', 'run', '-n', 'numpyro',
        'python', '-m', 'src.flow_models.train_gen',
        '--model_type', 'diffusion',
        '--latent_dim', '8',
        '--num_epochs', '100',
        '--dropout_epochs', '80',
        '--noise_schedule', noise_schedule,
        '--encoder_model_type', 'linear',
        '--decoder_model_type', 'identity',
        '--decoder_type', 'linear',  # Linear output transformation for 8D -> 2D
        '--seed', '42',
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for {noise_schedule}")
        print(result.stderr)
        return None
    
    # Extract Chamfer distance from output
    chamfer_dist = None
    
    # Try to find in stdout
    chamfer_match = re.search(r'Final Chamfer Distance:\s+([\d\.]+)', result.stdout)
    if chamfer_match:
        chamfer_dist = float(chamfer_match.group(1))
    
    # Also try to extract from validation Chamfer distances
    if chamfer_dist is None:
        val_match = re.search(r'Final Validation Chamfer Distance:\s+([\d\.]+)', result.stdout)
        if val_match:
            chamfer_dist = float(val_match.group(1))
    
    # Try to load from saved results file
    if chamfer_dist is None:
        # Find the most recent output directory
        artifacts_dir = Path('artifacts')
        if artifacts_dir.exists():
            dirs = sorted([d for d in artifacts_dir.iterdir() if d.is_dir()], 
                        key=lambda x: x.stat().st_mtime, reverse=True)
            for dir_path in dirs:
                diffusion_dir = dir_path / 'diffusion'
                results_file = diffusion_dir / 'training_results.pkl'
                if results_file.exists():
                    try:
                        with open(results_file, 'rb') as f:
                            history = pickle.load(f)
                            if 'val_chamfer_distances' in history and len(history['val_chamfer_distances']) > 0:
                                chamfer_dist = history['val_chamfer_distances'][-1]
                                break
                    except Exception as e:
                        print(f"Warning: Could not load results from {results_file}: {e}")
                        continue
    
    print(f"Chamfer Distance: {chamfer_dist}")
    
    return chamfer_dist


def main():
    """Test all noise schedules and report results."""
    print("="*60)
    print("Testing all noise schedules for Diffusion model")
    print("="*60)
    print(f"Schedules to test: {', '.join(NOISE_SCHEDULES)}")
    print(f"\nConfig:")
    print(f"  - Model: Diffusion")
    print(f"  - Latent dim: 8")
    print(f"  - Encoder: linear")
    print(f"  - Decoder: identity (linear output)")
    print(f"  - Epochs: 100")
    print(f"  - Dropout epochs: 80")
    print("="*60)
    
    for schedule in NOISE_SCHEDULES:
        chamfer_dist = run_training(schedule)
        RESULTS[schedule] = chamfer_dist
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Sort by Chamfer distance (lower is better)
    sorted_results = sorted(RESULTS.items(), key=lambda x: (x[1] is None, x[1] or float('inf')))
    
    print(f"\n{'Noise Schedule':<20} {'Chamfer Distance':<20}")
    print("-" * 40)
    for schedule, chamfer in sorted_results:
        if chamfer is None:
            print(f"{schedule:<20} {'FAILED':<20}")
        else:
            print(f"{schedule:<20} {chamfer:<20.6f}")
    
    # Find best
    valid_results = [(s, c) for s, c in sorted_results if c is not None]
    if valid_results:
        best_schedule, best_chamfer = valid_results[0]
        print(f"\n{'='*60}")
        print(f"BEST NOISE SCHEDULE: {best_schedule}")
        print(f"Chamfer Distance: {best_chamfer:.6f}")
        print(f"{'='*60}\n")
    else:
        print("\nWARNING: No valid results found!\n")


if __name__ == '__main__':
    main()

