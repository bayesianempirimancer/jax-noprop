# Parameter Testing for Diffusion Model

This directory contains scripts to systematically test different parameter configurations for the diffusion model, specifically focusing on `recon_weight` and `decoder_type` parameters.

## Files Overview

- `df.py` - Main diffusion model implementation
- `train.py` - Training script for single experiments
- `trainer.py` - Trainer class with plotting and saving functionality
- `parameter_sweep.py` - Systematic parameter sweep across multiple combinations
- `quick_test.py` - Quick test with a few parameter combinations
- `test_single_config.py` - Test a single parameter configuration

## Key Parameters

### recon_weight
Controls the weight of reconstruction loss in the total loss function:
- `0.0` - No reconstruction loss (pure diffusion)
- `0.01` - Very small reconstruction weight
- `0.1` - Small reconstruction weight
- `1.0` - Equal weight with diffusion loss
- `10.0` - High reconstruction weight

### decoder_type
Controls how the latent representation is transformed to output:
- `"none"` - Identity transformation (current default)
- `"linear"` - Linear transformation
- `"softmax"` - Softmax activation for classification

## Usage Examples

### 1. Test a Single Configuration

```bash
# Test with default parameters
python src/flow_models/test_single_config.py

# Test with specific parameters
python src/flow_models/test_single_config.py --recon_weight 0.1 --decoder_type linear --num_epochs 50
```

### 2. Quick Test (Few Combinations)

```bash
# Run quick test with 6 combinations
python src/flow_models/quick_test.py
```

### 3. Full Parameter Sweep

```bash
# Run comprehensive parameter sweep
python src/flow_models/parameter_sweep.py \
    --recon_weights 0.0 0.01 0.1 1.0 10.0 \
    --decoder_types none linear softmax \
    --num_epochs 50 \
    --save_dir artifacts/full_sweep
```

### 4. Custom Parameter Ranges

```bash
# Test specific parameter ranges
python src/flow_models/parameter_sweep.py \
    --recon_weights 0.0 0.1 1.0 \
    --decoder_types none linear \
    --num_epochs 30 \
    --batch_size 128 \
    --learning_rate 0.001
```

## Output

The scripts generate:

1. **Individual experiment results** - Saved in `artifacts/[experiment_name]/results.json`
2. **Comparison plots** - `parameter_comparison.png` showing performance across parameters
3. **Summary file** - `sweep_summary.json` with all results
4. **Training plots** - Individual training progress plots for each experiment

## Key Metrics Tracked

- **Final train loss** - Training loss at end of training
- **Final validation loss** - Validation loss at end of training
- **Final flow loss** - Diffusion/flow loss component
- **Final reconstruction loss** - Reconstruction loss component
- **Train accuracy** - Classification accuracy on training data
- **Validation accuracy** - Classification accuracy on validation data
- **Training time** - Time taken for training

## Interpreting Results

1. **Low reconstruction loss** - Model is good at reconstructing targets
2. **High accuracy** - Model is good at classification
3. **Balanced losses** - Both diffusion and reconstruction components are working
4. **Validation vs training** - Check for overfitting

## Tips for Parameter Exploration

1. **Start with quick_test.py** - Get a feel for the parameter space
2. **Use single config testing** - Deep dive into interesting configurations
3. **Monitor validation performance** - Avoid overfitting to training data
4. **Compare training times** - Some configurations may be slower
5. **Check reconstruction quality** - Visualize predictions vs targets

## Example Workflow

```bash
# 1. Quick test to see parameter effects
python src/flow_models/quick_test.py

# 2. Test specific interesting configurations
python src/flow_models/test_single_config.py --recon_weight 0.1 --decoder_type linear
python src/flow_models/test_single_config.py --recon_weight 1.0 --decoder_type softmax

# 3. Run full sweep if needed
python src/flow_models/parameter_sweep.py --num_epochs 100
```

## Notes

- The two moons dataset should be available at `data/two_moons_formatted.pkl`
- Results are saved in the `artifacts/` directory
- Each experiment uses a different random seed for reproducibility
- Training plots are generated automatically for each experiment
