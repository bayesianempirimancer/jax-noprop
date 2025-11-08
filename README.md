# JAX/Flax Flow Model Comparison

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-0.7.0+-green.svg)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A JAX/Flax implementation for **comparing Flow Matching, Diffusion, and Continuous-Time diffusion models**. These three approaches solve the same continuous-time problem using different objective functions, each resulting in networks that predict different quantities.

## Overview

This repository implements three variants of continuous-time generative models that address the same fundamental problem through different parameterizations:

### Three Approaches to Continuous-Time Generative Modeling

1. **Flow Matching (FM)**: Predicts the **denoising flow directly**
   - Network learns: `dz/dt = f(z, x, t)` where `f` directly predicts the flow field
   - Objective: Match the flow field that connects noise to data
   - **Most robust approach** - direct flow prediction tends to be stable across different data distributions

2. **Diffusion (DF)**: Predicts **noise** at each time step
   - Network learns: `noise_prediction = f(z, x, t)` where `f` predicts the noise component
   - Objective: Predict noise to be removed, reparameterized to avoid singularities
   - Uses noise prediction for denoising trajectory

3. **Continuous-Time (CT)**: Predicts the **target** at each time step
   - Network learns: `target_prediction = f(z, x, t)` where `f` predicts the clean target
   - Objective: Predict the target value at each time point with SNR-weighted loss
   - Uses target prediction to guide the denoising process

### Key Insight

All three methods solve the **same continuous-time problem** but parameterize it differently:
- **Flow Matching**: Direct flow field prediction
- **Diffusion**: Noise prediction → denoising trajectory
- **CT**: Target prediction → denoising trajectory

Each parameterization leads to different training dynamics and performance characteristics.

## Installation

```bash
git clone https://github.com/yourusername/jax-noprop.git
cd jax-noprop
pip install -e .
```

Make sure you have JAX installed (with CUDA support for GPU acceleration).

## Quick Start

### Training Scripts

This repository provides three main training scripts for different tasks:

1. **`train.py`** - Regression/Classification (x → y)
2. **`train_gen.py`** - Conditional/Unconditional Generation (y → x or x generation)
3. **`train_seq.py`** - Sequence Generation (for sequence data)

### Two Moons Dataset Example

See the [Two Moons Example README](examples/two_moons/README.md) for a complete walkthrough. Quick start:

```bash
# Generate the dataset
python examples/two_moons/generate_two_moons.py

# Regression: Predict labels from coordinates
python -m src.flow_models.train \
    --config_file examples/two_moons/config.yaml \
    --data_path data/two_moons.pkl \
    --model_type flow_matching

# Conditional Generation: Generate coordinates from labels
python -m src.flow_models.train_gen \
    --config_file examples/two_moons/config.yaml \
    --data_path data/two_moons.pkl \
    --model_type flow_matching

# Unconditional Generation: Generate coordinates without labels
python -m src.flow_models.train_gen \
    --config_file examples/two_moons/config.yaml \
    --data_path data/two_moons.pkl \
    --model_type flow_matching \
    --unconditional
```

### Output Structure

After training, results are saved to `artifacts/{model_type}_{task}/{YYYYMMDD_HHMM}/`:

- **Regression**: `artifacts/{model_type}_reg/{timestamp}/`
- **Conditional Generation**: `artifacts/{model_type}_gen/{timestamp}/`
- **Unconditional Generation**: `artifacts/{model_type}_uncond_gen/{timestamp}/`

Each directory contains:
- `training_results.pkl` - Training history
- `model_params.pkl` - Trained model parameters
- `config.yaml` - Configuration used for training
- Various plots (generation visualizations, loss trends, trajectories, etc.)

## Usage Examples

### Configuration Files

All training scripts support YAML configuration files:

```bash
# Use a YAML config file (recommended)
python -m src.flow_models.train_gen \
    --config_file examples/two_moons/config.yaml \
    --data_path data/two_moons.pkl \
    --model_type flow_matching

# Use a custom config class
python -m src.flow_models.train_gen \
    --config_file examples/two_moons/config.yaml \
    --config_class examples.two_moons.config.Config \
    --data_path data/two_moons.pkl \
    --model_type flow_matching
```

Command-line arguments override values in config files.

### Unconditional Generation

```bash
# Generate samples without conditioning
python -m src.flow_models.train_gen \
    --config_file examples/two_moons/config.yaml \
    --data_path data/two_moons.pkl \
    --model_type flow_matching \
    --unconditional \
    --num_epochs 100
```

### Custom Architecture

```bash
# Override config file values via command line
python -m src.flow_models.train_gen \
    --config_file examples/two_moons/config.yaml \
    --data_path data/two_moons.pkl \
    --model_type flow_matching \
    --latent_dim 8 \
    --encoder_model_type linear \
    --decoder_model_type identity \
    --decoder_type linear
```

### Noise Schedule Selection

```bash
# Override noise schedule (for diffusion/CT models)
python -m src.flow_models.train_gen \
    --config_file examples/two_moons/config.yaml \
    --data_path data/two_moons.pkl \
    --model_type diffusion \
    --noise_schedule cosine

# Available schedules: linear, cosine, sigmoid, exponential, cauchy, laplace, logistic, quadratic, polynomial
```

### Training with Dropout Schedule

```bash
# Use dropout for first 80 epochs, then disable it
python -m src.flow_models.train_gen \
    --config_file examples/two_moons/config.yaml \
    --data_path data/two_moons.pkl \
    --model_type flow_matching \
    --num_epochs 100 \
    --dropout_epochs 80
```

## Model Comparison

### When to Use Each Approach

**Flow Matching (Recommended for most cases)**
- ✅ **Most robust** - Stable across different data distributions
- ✅ Simple objective - direct flow prediction
- ✅ Fast training and inference
- ✅ Works well as baseline for comparison

**Diffusion**
- Good for noise-focused applications
- Uses noise prediction parameterization
- Requires careful noise schedule tuning

**Continuous-Time**
- Good when target prediction is natural
- Uses SNR-weighted loss for training stability
- Can learn optimal noise schedules

### Performance Characteristics

- **Flow Matching**: Fastest inference, most stable training
- **Diffusion**: Good for applications where noise structure matters
- **CT**: Can adapt to data with learnable schedules

## Python API

### Training a Model

```python
from src.flow_models.trainer_gen import GenerationTrainer
from src.flow_models.config import Config

# Load configuration from YAML or create default
config = Config.load_yaml('examples/two_moons/config.yaml')
# Or create a default config
# config = Config()

# Create trainer
trainer = GenerationTrainer(
    config=config,
    learning_rate=1e-3,
    optimizer_name='adam',
    seed=42,
    unconditional=False  # Set to True for unconditional generation
)

# Initialize and train
trainer.initialize(x_sample, y_sample, z_sample, t_sample)
history = trainer.train(
    x_data=x_train,  # None for unconditional generation
    y_data=y_train,
    num_epochs=100,
    batch_size=256,
    validation_data=(x_val, y_val)
)
```

### Generating Samples

```python
# Conditional generation
x_gen = trainer.conditional_generate(
    cond_y=conditions,  # conditional inputs
    num_steps=20,
    prng_key=key
)

# Unconditional generation
x_gen = trainer.unconditional_generate(
    batch_shape=(100,),  # number of samples
    num_steps=20,
    prng_key=key
)
```

## Command-Line Arguments

### Core Arguments

- `--config_file`: Path to YAML config file (recommended)
- `--config_class`: Optional custom config class (e.g., `examples.two_moons.config.Config`)
- `--model_type`: `flow_matching`, `diffusion`, or `ct`
- `--data_path`: Path to data file (required for most tasks)
- `--num_epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 256)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--optimizer`: `adam`, `sgd`, or `adagrad` (default: `adam`)

### Shape Arguments

- `--input_shape` or `--input_dim`: Input shape/dimension
- `--output_shape` or `--output_dim`: Output shape/dimension
- `--latent_shape` or `--latent_dim`: Latent shape/dimension

**Note:** If no config file is provided, you must specify these shapes.

### Architecture Arguments

- `--encoder_model_type`: `identity`, `linear`, `mlp`, `mlp_normal`, `resnet`, `resnet_normal`
- `--decoder_model_type`: `identity`, `mlp`, `resnet`
- `--decoder_type`: `linear`, `softmax`, `none`
- `--crn_type`: CRN type (e.g., `vanilla`, `geometric`, `potential`)
- `--network_type`: Network backbone (e.g., `mlp`, `bilinear`, `convex`)
- `--hidden_dims`: Hidden layer dimensions (space-separated integers)

### Noise Schedule Arguments

- `--noise_schedule`: `linear`, `cosine`, `sigmoid`, `exponential`, `cauchy`, `laplace`, `logistic`, `quadratic`, `polynomial`, `monotonic_nn`, `learnable`, `network`
- `--noise_schedule_learnable`: Make noise schedule learnable

### Training Arguments

- `--dropout_epochs`: Number of epochs to use dropout (default: all epochs)
- `--recon_weight`: Reconstruction loss weight
- `--reg_weight`: Regularization loss weight
- `--vae_weight`: VAE loss weight (for some models)
- `--use_snr_weight`: Apply SNR weighting

### Generation Arguments

- `--unconditional`: Train for unconditional generation (only for `train_gen.py`)

See `python -m src.flow_models.train --help`, `python -m src.flow_models.train_gen --help`, or `python -m src.flow_models.train_seq --help` for full lists of options.

## Project Structure

```
jax-noprop/
├── src/
│   ├── flow_models/
│   │   ├── fm.py              # Flow Matching implementation
│   │   ├── df.py              # Diffusion implementation
│   │   ├── ct.py              # Continuous-Time implementation
│   │   ├── config.py          # Unified Config class
│   │   ├── train.py           # Regression/classification training CLI
│   │   ├── train_gen.py       # Generation training CLI
│   │   ├── train_seq.py       # Sequence training CLI
│   │   ├── trainer.py         # Regression trainer
│   │   ├── trainer_gen.py     # Generation trainer
│   │   ├── trainer_seq.py      # Sequence trainer
│   │   └── training_utils.py  # Shared training utilities
│   ├── configs/
│   │   └── base_config.py     # BaseConfig class with YAML support
│   ├── embeddings/
│   │   └── noise_schedules.py # Noise schedule implementations
│   └── vae/                   # Encoder/decoder architectures
├── examples/
│   └── two_moons/             # Two moons dataset example
│       ├── config.py          # Example config class
│       ├── config.yaml        # Example YAML config
│       ├── generate_two_moons.py
│       └── README.md
├── data/                      # Dataset files
├── artifacts/                 # Training outputs
│   └── {model_type}_{task}/   # Organized by model and task
│       └── {YYYYMMDD_HHMM}/   # Timestamped runs
└── README.md
```

## Technical Details

### Noise Schedules

All models support multiple noise schedules:
- **Linear, Cosine, Sigmoid**: Standard fixed schedules
- **Exponential, Cauchy, Laplace**: Distribution-based schedules
- **Quadratic, Polynomial**: Power-based schedules
- **Learnable**: Neural network-based adaptive schedule

Schedules are parameterized to avoid singularities at boundaries.

### Training Objectives

**Flow Matching**:
```
Loss = E[||dz/dt(z_t, x, t) - (target - z_0)||²]
```
Direct flow field matching.

**Diffusion**:
```
Loss = E[SNR'(t) * ||noise_prediction - actual_noise||²] / E[SNR'(t)]
```
SNR-weighted noise prediction.

**Continuous-Time**:
```
Loss = E[SNR'(t) * ||target_prediction - target||²] / E[SNR'(t)]
```
SNR-weighted target prediction.

### Configuration System

The repository uses a unified configuration system:

- **YAML Config Files**: Human-readable configuration files (recommended)
- **Python Config Classes**: Custom config classes that extend `BaseConfig`
- **Command-Line Overrides**: All config values can be overridden via command-line arguments

The unified `Config` class in `src/flow_models/config.py` works for all three model types (Flow Matching, Diffusion, CT) and all tasks (regression, generation, sequences).

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{Li2025NoProp,
  title={{NoProp: Training Neural Networks without Full Back-propagation or Full Forward-propagation}},
  author={Qinyu Li and Yee Whye Teh and Razvan Pascanu},
  booktitle={Conference on Lifelong Learning Agents (CoLLAs)},
  year={2025},
  url={https://arxiv.org/abs/2503.24322}
}
```
