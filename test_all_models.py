#!/usr/bin/env python3
"""
Test script to verify all three models (df, ct, fm) work on all three tasks
(classification, conditional generation, unconditional generation) on two moons dataset.

This runs minimal training (1-2 epochs) just to verify nothing is broken.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"Command: conda run -n numpyro {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Use conda run to execute in numpyro environment
    full_cmd = ["conda", "run", "-n", "numpyro"] + cmd
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ SUCCESS: {description}")
        return True
    else:
        print(f"✗ FAILED: {description}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return False

def main():
    """Run all training combinations."""
    base_dir = Path(__file__).parent
    config_file = base_dir / "examples" / "two_moons" / "config.yaml"
    data_path = base_dir / "data" / "two_moons.pkl"
    
    # Check if dataset exists
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("Please generate it first with:")
        print("  python examples/two_moons/generate_two_moons.py --output_dir data --filename two_moons.pkl")
        sys.exit(1)
    
    if not config_file.exists():
        print(f"ERROR: Config file not found at {config_file}")
        sys.exit(1)
    
    results = []
    
    # Task 1: Classification (x -> y)
    print("\n" + "="*80)
    print("TASK 1: CLASSIFICATION (x -> y)")
    print("="*80)
    
    for model_type in ["flow_matching", "diffusion", "ct"]:
        cmd = [
            sys.executable, "-m", "src.flow_models.train",
            "--config_file", str(config_file),
            "--data_path", str(data_path),
            "--model_type", model_type,
            "--num_epochs", "1",  # Minimal epochs for testing
            "--batch_size", "64",  # Smaller batch for faster testing
        ]
        success = run_command(cmd, f"Classification - {model_type}")
        results.append(("Classification", model_type, success))
    
    # Task 2: Conditional Generation (x | y)
    print("\n" + "="*80)
    print("TASK 2: CONDITIONAL GENERATION (x | y)")
    print("="*80)
    
    for model_type in ["flow_matching", "diffusion", "ct"]:
        cmd = [
            sys.executable, "-m", "src.flow_models.train_gen",
            "--config_file", str(config_file),
            "--data_path", str(data_path),
            "--model_type", model_type,
            "--num_epochs", "1",  # Minimal epochs for testing
            "--batch_size", "64",  # Smaller batch for faster testing
        ]
        success = run_command(cmd, f"Conditional Generation - {model_type}")
        results.append(("Conditional Generation", model_type, success))
    
    # Task 3: Unconditional Generation
    print("\n" + "="*80)
    print("TASK 3: UNCONDITIONAL GENERATION")
    print("="*80)
    
    for model_type in ["flow_matching", "diffusion", "ct"]:
        cmd = [
            sys.executable, "-m", "src.flow_models.train_gen",
            "--config_file", str(config_file),
            "--data_path", str(data_path),
            "--model_type", model_type,
            "--unconditional",
            "--num_epochs", "1",  # Minimal epochs for testing
            "--batch_size", "64",  # Smaller batch for faster testing
        ]
        success = run_command(cmd, f"Unconditional Generation - {model_type}")
        results.append(("Unconditional Generation", model_type, success))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, _, success in results if success)
    total = len(results)
    
    for task, model, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {task} - {model}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Core functionality is intact.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

