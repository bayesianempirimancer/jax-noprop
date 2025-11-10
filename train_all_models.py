#!/usr/bin/env python3
"""Train all model types on all tasks for the two moons dataset."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"\n✓ Successfully completed: {description}\n")
        return True
    else:
        print(f"\n✗ Failed: {description} (exit code: {result.returncode})\n")
        return False

def main():
    """Run all training combinations."""
    
    # Check if data file exists
    data_path = Path("data/two_moons.pkl")
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found!")
        print("Please generate it first using:")
        print("  python examples/two_moons/generate_two_moons.py")
        sys.exit(1)
    
    config_file = "examples/two_moons/config.yaml"
    if not Path(config_file).exists():
        print(f"Error: Config file {config_file} not found!")
        sys.exit(1)
    
    model_types = ["flow_matching", "diffusion", "ct"]
    tasks = [
        ("classification", "train"),
        ("conditional_generation", "train_gen"),
        ("unconditional_generation", "train_gen_unconditional")
    ]
    
    results = []
    
    for model_type in model_types:
        for task_name, script_name in tasks:
            description = f"{model_type} - {task_name}"
            
            if script_name == "train":
                # Classification task
                cmd = [
                    "conda", "run", "-n", "numpyro", "python", "-m", "src.flow_models.train",
                    "--config_file", config_file,
                    "--data_path", str(data_path),
                    "--model_type", model_type,
                    "--num_epochs", "50"
                ]
            elif script_name == "train_gen":
                # Conditional generation
                cmd = [
                    "conda", "run", "-n", "numpyro", "python", "-m", "src.flow_models.train_gen",
                    "--config_file", config_file,
                    "--data_path", str(data_path),
                    "--model_type", model_type,
                    "--num_epochs", "50"
                ]
            else:  # train_gen_unconditional
                # Unconditional generation
                cmd = [
                    "conda", "run", "-n", "numpyro", "python", "-m", "src.flow_models.train_gen",
                    "--config_file", config_file,
                    "--data_path", str(data_path),
                    "--model_type", model_type,
                    "--num_epochs", "50",
                    "--unconditional"
                ]
            
            success = run_command(cmd, description)
            results.append((description, success))
    
    # Print summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}\n")
    
    for description, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {description}")
    
    print(f"\n{'='*80}")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Total: {passed}/{total} passed")
    print(f"{'='*80}\n")
    
    if passed < total:
        sys.exit(1)

if __name__ == "__main__":
    main()

