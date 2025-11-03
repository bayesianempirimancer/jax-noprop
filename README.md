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

### Basic Usage: Train All Three Models

```bash
# Train Flow Matching (most robust)
python -m src.flow_models.train_gen --model_type flow_matching --num_epochs 100

# Train Diffusion  
python -m src.flow_models.train_gen --model_type diffusion --num_epochs 100

# Train Continuous-Time
python -m src.flow_models.train_gen --model_type ct --num_epochs 100
```

### Two Moons Dataset Example

```bash
# Train Flow Matching on Two Moons (conditional generation)
python -m src.flow_models.train_gen \
    --model_type flow_matching \
    --data_path data/two_moons_formatted.pkl \
    --num_epochs 100 \
    --batch_size 256

# Train Diffusion on Two Moons
python -m src.flow_models.train_gen \
    --model_type diffusion \
    --data_path data/two_moons_formatted.pkl \
    --num_epochs 100 \
    --batch_size 256

# Train CT on Two Moons
python -m src.flow_models.train_gen \
    --model_type ct \
    --data_path data/two_moons_formatted.pkl \
    --num_epochs 100 \
    --batch_size 256
```

### Compare Results

After training, check the `artifacts/` directory for:
- `conditional_generation.png` - Generated samples vs real data
- `loss_trends.png` - Training and validation losses
- `latent_trajectories.png` - ODE integration trajectories

## Usage Examples

### Unconditional Generation

```bash
# Generate samples without conditioning
python -m src.flow_models.train_gen \
    --model_type flow_matching \
    --unconditional \
    --num_epochs 100
```

### Custom Architecture

```bash
# Use custom latent dimension and encoder/decoder
python -m src.flow_models.train_gen \
    --model_type flow_matching \
    --latent_dim 8 \
    --encoder_model_type linear \
    --decoder_model_type identity \
    --decoder_type linear
```

### Noise Schedule Selection

```bash
# Use exponential noise schedule (default)
python -m src.flow_models.train_gen \
    --model_type diffusion \
    --noise_schedule exponential

# Try different schedules: linear, cosine, sigmoid, exponential, cauchy, laplace, etc.
python -m src.flow_models.train_gen \
    --model_type diffusion \
    --noise_schedule cosine
```

### Training with Dropout Schedule

```bash
# Use dropout for first 80 epochs, then disable it
python -m src.flow_models.train_gen \
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
from src.flow_models.train_gen import build_config

# Build configuration
config = build_config(
    model='flow_matching',  # or 'diffusion' or 'ct'
    input_shape=(2,),  # conditional input dimension
    output_shape=(2,),  # output dimension
    latent_shape=(2,),  # latent dimension
    noise_schedule='exponential',
    # ... other config options
)

# Create trainer
trainer = GenerationTrainer(
    config=config,
    learning_rate=1e-3,
    optimizer_name='adam',
    seed=42
)

# Initialize and train
trainer.initialize(x_sample, y_sample, z_sample, t_sample)
history = trainer.train(
    x_data=x_train,
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

- `--model_type`: `flow_matching`, `diffusion`, or `ct`
- `--num_epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 256)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--latent_dim`: Latent space dimension (default: 2)

### Architecture Arguments

- `--encoder_model_type`: `linear`, `mlp`, `identity`, etc.
- `--decoder_model_type`: `identity`, `mlp`, `resnet`
- `--decoder_type`: `linear`, `softmax`, `none`

### Noise Schedule Arguments

- `--noise_schedule`: `linear`, `cosine`, `sigmoid`, `exponential`, `cauchy`, `laplace`, `logistic`, `quadratic`, `polynomial` (default: `exponential`)
- `--noise_schedule_learnable`: Make noise schedule learnable (default: False)

### Training Arguments

- `--dropout_epochs`: Number of epochs to use dropout (default: all epochs)
- `--recon_weight`: Reconstruction loss weight (default: 1.0)
- `--reg_weight`: Regularization loss weight (default: 0.0)

### Data Arguments

- `--data_path`: Path to data file (default: `data/two_moons_formatted.pkl`)
- `--unconditional`: Train for unconditional generation

See `python -m src.flow_models.train_gen --help` for full list of options.

## Project Structure

```
jax-noprop/
├── src/
│   ├── flow_models/
│   │   ├── fm.py              # Flow Matching implementation
│   │   ├── df.py              # Diffusion implementation
│   │   ├── ct.py              # Continuous-Time implementation
│   │   ├── train_gen.py       # Generative training CLI
│   │   └── trainer_gen.py     # Generative trainer
│   ├── embeddings/
│   │   └── noise_schedules.py # Noise schedule implementations
│   └── models/
│       └── vae/               # Encoder/decoder architectures
├── data/                      # Dataset files
├── artifacts/                 # Training outputs (plots, models)
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
