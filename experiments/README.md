# Experiments Directory

This directory contains all the scripts and documentation for running parameter experiments with the flow models.

## Core Scripts

- **`run_experiments.py`** - Main script for running systematic parameter sweeps
- **`train.py`** - Individual training script (moved from src/flow_models/)
- **`results_tracker.py`** - Results tracking and markdown generation
- **`generate_comprehensive_report.py`** - Generate detailed analysis reports

## Utility Scripts

- **`quick_test.py`** - Quick parameter testing
- **`parameter_sweep.py`** - Advanced parameter sweeping
- **`test_single_config.py`** - Test individual configurations
- **`generate_report.py`** - Generate reports from results

## Documentation

- **`README_experiments.md`** - Comprehensive guide to the experiment system
- **`README_parameter_testing.md`** - Parameter testing documentation

## Usage

Run experiments from the project root:

```bash
# Quick test
python experiments/run_experiments.py --quick_test

# Full parameter sweep
python experiments/run_experiments.py --recon_weights 0.0 0.1 1.0 --decoder_types none linear

# Generate comprehensive report
python experiments/generate_comprehensive_report.py
```

## Output

All experiment results are saved to the `artifacts/` directory (ignored by git) with unique timestamps for each experiment.
