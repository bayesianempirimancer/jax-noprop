# Diffusion Model Parameter Experimentation System

This system provides a comprehensive framework for testing different parameter configurations of the diffusion model and tracking results in markdown reports.

## 🚀 Quick Start

### 1. Single Experiment
```bash
# Test with default parameters
python src/flow_models/train.py

# Test with specific parameters
python src/flow_models/train.py --recon_weight 0.1 --decoder_type linear --num_epochs 50
```

### 2. Quick Test (6 experiments)
```bash
python src/flow_models/run_experiments.py --quick_test
```

### 3. Full Parameter Sweep
```bash
python src/flow_models/run_experiments.py --recon_weights 0.0 0.01 0.1 1.0 10.0 --decoder_types none linear softmax
```

## 📁 Files Overview

### Core Scripts
- **`train.py`** - Enhanced training script with command-line parameter control
- **`run_experiments.py`** - Systematic experiment runner
- **`generate_report.py`** - Generate comprehensive reports from results
- **`results_tracker.py`** - Results tracking and markdown generation

### Model Files
- **`df.py`** - Diffusion model implementation
- **`trainer.py`** - Training logic and plotting

## 🎛️ Key Parameters

### Primary Parameters (Most Important)
- **`--recon_weight`** - Reconstruction loss weight (0.0 = no reconstruction, 1.0 = equal weight)
- **`--decoder_type`** - Decoder type: `none` (identity), `linear`, or `softmax`

### Secondary Parameters
- **`--recon_loss_type`** - Reconstruction loss type: `mse`, `cross_entropy`, or `none`
- **`--reg_weight`** - Regularization loss weight
- **`--model_type`** - Model type: `diffusion` or `flow_matching`

### Training Parameters
- **`--num_epochs`** - Number of training epochs
- **`--batch_size`** - Batch size
- **`--learning_rate`** - Learning rate
- **`--optimizer`** - Optimizer: `adam`, `sgd`, or `adagrad`

## 📊 Usage Examples

### Basic Parameter Testing
```bash
# Test different reconstruction weights
python src/flow_models/train.py --recon_weight 0.0
python src/flow_models/train.py --recon_weight 0.1
python src/flow_models/train.py --recon_weight 1.0

# Test different decoder types
python src/flow_models/train.py --decoder_type none
python src/flow_models/train.py --decoder_type linear
python src/flow_models/train.py --decoder_type softmax
```

### Systematic Testing
```bash
# Quick test (6 experiments, ~10 minutes)
python src/flow_models/run_experiments.py --quick_test

# Medium test (15 experiments, ~30 minutes)
python src/flow_models/run_experiments.py --recon_weights 0.0 0.1 1.0 --decoder_types none linear softmax

# Full test (45 experiments, ~90 minutes)
python src/flow_models/run_experiments.py --recon_weights 0.0 0.01 0.1 1.0 10.0 --decoder_types none linear softmax
```

### Custom Experiments
```bash
# Test specific parameter combinations
python src/flow_models/run_experiments.py \
    --recon_weights 0.0 0.1 1.0 \
    --decoder_types none linear \
    --recon_loss_types mse cross_entropy \
    --num_epochs 50 \
    --batch_size 128
```

## 📈 Results Tracking

### Automatic Tracking
- All experiments are automatically tracked in `experiment_results.md`
- Each experiment gets a unique name based on parameters
- Results include performance metrics and training details

### Markdown Reports
The system generates two types of reports:

1. **Summary Table** - Quick overview of all experiments
2. **Detailed Report** - Comprehensive analysis with recommendations

### Generated Files
- `experiment_results.md` - Live summary of all experiments
- `detailed_report_YYYYMMDD_HHMMSS.md` - Comprehensive analysis
- `experiment_data.json` - Raw data for further analysis

### Output Directory Structure
Each experiment creates a unique directory with timestamp:
- **Format**: `artifacts/{experiment_name}_{timestamp}/`
- **Example**: `artifacts/recon1.0_decnone_lossmse_20251028_224249/`
- **Contents**: All figures, model files, and results for that specific experiment
- **Benefits**: No file overwrites, easy to compare experiments, organized storage

## 🔍 Understanding Results

### Key Metrics
- **Final Train Loss** - Training loss at end of training
- **Final Val Loss** - Validation loss at end of training
- **Train Accuracy** - Classification accuracy on training data
- **Val Accuracy** - Classification accuracy on validation data

### Parameter Effects
- **`recon_weight=0.0`** - Pure diffusion, may have lower reconstruction quality
- **`recon_weight=0.1-1.0`** - Balanced diffusion + reconstruction
- **`decoder_type="linear"`** - May improve reconstruction for continuous outputs
- **`decoder_type="softmax"`** - Better for classification tasks

## 🛠️ Advanced Usage

### Custom Experiment Names
```bash
python src/flow_models/train.py --experiment_name "my_custom_experiment" --recon_weight 0.5
```

### Different Results Files
```bash
python src/flow_models/run_experiments.py --results_file "my_experiments.md"
```

### Generate Reports
```bash
# Generate comprehensive report from existing results
python src/flow_models/generate_report.py --results_file experiment_results.md
```

## 📋 Example Workflow

### 1. Initial Exploration
```bash
# Quick test to understand parameter effects
python src/flow_models/run_experiments.py --quick_test
```

### 2. Focused Testing
```bash
# Test promising parameter ranges
python src/flow_models/run_experiments.py \
    --recon_weights 0.0 0.05 0.1 0.5 1.0 \
    --decoder_types none linear \
    --num_epochs 50
```

### 3. Deep Dive
```bash
# Test specific interesting configurations
python src/flow_models/train.py --recon_weight 0.1 --decoder_type linear --num_epochs 100 --verbose
```

### 4. Generate Report
```bash
# Create comprehensive analysis
python src/flow_models/generate_report.py
```

## 🎯 Expected Insights

### Parameter Interactions
- **High `recon_weight` + `decoder_type="linear"`** - Good for reconstruction tasks
- **Low `recon_weight` + `decoder_type="none"`** - Pure diffusion, good for generation
- **`decoder_type="softmax"`** - Best for classification tasks

### Performance Patterns
- **Overfitting** - High train accuracy, low val accuracy
- **Underfitting** - Low train and val accuracy
- **Good generalization** - Similar train and val performance

## 🔧 Troubleshooting

### Common Issues
1. **Dataset not found** - Ensure `data/two_moons_formatted.pkl` exists
2. **Out of memory** - Reduce `--batch_size`
3. **Slow training** - Reduce `--num_epochs` for quick tests

### Debug Mode
```bash
# Run with verbose output
python src/flow_models/train.py --verbose --recon_weight 0.1
```

## 📊 Sample Results

After running experiments, you'll see results like:

| Experiment | Parameters | Train Loss | Val Loss | Train Acc | Val Acc | Time | Status |
|------------|------------|------------|----------|-----------|---------|------|--------|
| recon0.0_decnone | recon_weight=0.0, decoder_type=none | 0.1234 | 0.1456 | 0.8500 | 0.8200 | 45.2s | completed |
| recon0.1_declinear | recon_weight=0.1, decoder_type=linear | 0.0987 | 0.1123 | 0.9200 | 0.8900 | 47.8s | completed |

This system makes it easy to systematically explore the parameter space and find optimal configurations for your diffusion model!
