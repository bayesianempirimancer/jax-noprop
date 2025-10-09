# JAX/Flax NoProp Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-0.7.0+-green.svg)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/yourusername/jax-noprop/workflows/Tests/badge.svg)](https://github.com/yourusername/jax-noprop/actions)

A JAX/Flax implementation of the NoProp algorithm from the paper "NoProp: Training Neural Networks without Back-propagation or Forward-propagation" by Li et al. (arXiv:2503.24322v1).

## Overview

NoProp is a novel approach for training neural networks without relying on standard back-propagation or forward-propagation steps, taking inspiration from diffusion and flow matching methods. This repository provides:

- **NoProp-DT**: Discrete-time implementation
- **NoProp-CT**: Continuous-time implementation with neural ODEs
- **NoProp-FM**: Flow matching variant

## Key Features

- **Modular Design**: Easy to extend with different model architectures
- **ResNet Integration**: Works with any ResNet that takes (z, x) inputs for discrete case or (z, x, t) for continuous cases
- **Flexible Noise Scheduling**: Configurable noise schedules for all variants
- **JAX/Flax**: High-performance implementation with automatic differentiation
- **Multiple Datasets**: Support for MNIST, CIFAR-10, CIFAR-100
- **Comprehensive Examples**: Training scripts and quick start guide

## Installation

### From GitHub (Recommended)

```bash
git clone https://github.com/yourusername/jax-noprop.git
cd jax-noprop
pip install -e .
```

### From PyPI (Coming Soon)

```bash
pip install jax-noprop
```

### Development Installation

```bash
git clone https://github.com/yourusername/jax-noprop.git
cd jax-noprop
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax_noprop import NoPropDT, NoPropCT, NoPropFM
from jax_noprop.models import ResNetWrapper

# Create a ResNet wrapper
model = ResNetWrapper(num_classes=10, depth=18)

# Initialize NoProp variants
noprop_dt = NoPropDT(model, num_timesteps=10)
noprop_ct = NoPropCT(model, num_timesteps=1000)
noprop_fm = NoPropFM(model, num_timesteps=1000)

# Training loop
# ... (see examples/)
```

### Run Examples

```bash
# Quick start example
python examples/quick_start.py

# Train on MNIST
python examples/train_mnist.py --epochs 10 --variants DT CT FM

# Train on CIFAR-10
python examples/train_cifar.py --dataset cifar10 --epochs 50 --resnet_depth 18

# Train on CIFAR-100
python examples/train_cifar.py --dataset cifar100 --epochs 100 --resnet_depth 50
```

## Architecture

The implementation follows the paper's architecture with some modifications:

### Model Wrapper

The `ResNetWrapper` class provides a unified interface for all NoProp variants:

- **Discrete-time (DT)**: Takes `(z, x)` inputs where `z` is the noisy target
- **Continuous-time (CT/FM)**: Takes `(z, x, t)` inputs where `t` is the time step

### Noise Schedules

Three noise scheduling strategies are implemented:

- **Linear**: `alpha_t = 1 - t`, `sigma_t = sqrt(t)`
- **Cosine**: `alpha_t = cos(π/2 * t)`, `sigma_t = sin(π/2 * t)`
- **Sigmoid**: `alpha_t = σ(-γt)`, `sigma_t = σ(γt)`

### Training Process

Each NoProp variant implements a different training strategy:

1. **NoProp-DT**: Each layer learns to denoise independently with discrete timesteps
2. **NoProp-CT**: Learns a vector field for continuous-time denoising via neural ODEs
3. **NoProp-FM**: Learns a flow that transforms base distribution to target distribution

## API Reference

### Core Classes

#### `NoPropDT`
Discrete-time NoProp implementation.

```python
noprop_dt = NoPropDT(
    model=ResNetWrapper(num_classes=10),
    num_timesteps=10,
    noise_schedule=LinearNoiseSchedule(),
    eta=0.1
)
```

#### `NoPropCT`
Continuous-time NoProp with neural ODE integration.

```python
noprop_ct = NoPropCT(
    model=ResNetWrapper(num_classes=10),
    num_timesteps=1000,
    integration_method="euler",  # or "heun"
    eta=1.0
)
```

#### `NoPropFM`
Flow matching NoProp implementation.

```python
noprop_fm = NoPropFM(
    model=ResNetWrapper(num_classes=10),
    num_timesteps=1000,
    integration_method="euler",  # or "heun"
    eta=1.0
)
```

### Model Architectures

#### `ResNetWrapper`
Wrapper for ResNet backbones that handles NoProp-specific inputs.

```python
model = ResNetWrapper(
    num_classes=10,
    depth=18,  # 18, 50, 152
    width=64,
    time_embed_dim=128
)
```

#### `SimpleCNN`
Lightweight CNN for smaller datasets like MNIST.

```python
model = SimpleCNN(
    num_classes=10,
    time_embed_dim=64
)
```

### Noise Schedules

```python
from jax_noprop.noise_schedules import LinearNoiseSchedule, CosineNoiseSchedule, SigmoidNoiseSchedule

# Linear schedule (default)
schedule = LinearNoiseSchedule()

# Cosine schedule
schedule = CosineNoiseSchedule()

# Sigmoid schedule with learnable gamma
schedule = SigmoidNoiseSchedule(gamma=1.0)
```

## Training

### Basic Training Loop

```python
from jax_noprop.utils import create_train_state, train_step, eval_step

# Create training state
state = create_train_state(
    model=noprop_dt,
    params=params,
    learning_rate=1e-3,
    optimizer="adam"
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        state, loss, metrics = train_step(state, x, y, key)
    
    # Evaluation
    for batch in test_loader:
        metrics = eval_step(state, x, y, key)
```

### Hyperparameters

The paper suggests the following hyperparameters:

| Dataset | Method | Epochs | Learning Rate | Batch Size | Timesteps |
|---------|--------|--------|---------------|------------|-----------|
| MNIST | NoProp-DT | 100 | 1e-3 | 128 | 10 |
| MNIST | NoProp-CT | 100 | 1e-3 | 128 | 1000 |
| MNIST | NoProp-FM | 100 | 1e-3 | 128 | 1000 |
| CIFAR-10 | NoProp-DT | 150 | 1e-3 | 128 | 10 |
| CIFAR-10 | NoProp-CT | 500 | 1e-3 | 128 | 1000 |
| CIFAR-10 | NoProp-FM | 500 | 1e-3 | 128 | 1000 |

## Performance

The implementation aims to reproduce the results from the original paper:

- **MNIST**: ~99% accuracy with NoProp-CT
- **CIFAR-10**: ~90% accuracy with NoProp-CT
- **CIFAR-100**: ~70% accuracy with NoProp-CT

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

### Ways to Contribute

- 🐛 **Bug Reports**: Found a bug? Please report it in [Issues](https://github.com/yourusername/jax-noprop/issues)
- 💡 **Feature Requests**: Have an idea? Open a [Feature Request](https://github.com/yourusername/jax-noprop/issues/new?template=feature_request.md)
- 🔧 **Code Contributions**: Submit a [Pull Request](https://github.com/yourusername/jax-noprop/pulls)
- 📚 **Documentation**: Help improve our docs and examples
- 🧪 **Testing**: Add tests or improve test coverage

### Development

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/jax-noprop.git
cd jax-noprop
pip install -e ".[dev]"

# Run tests
python test_implementation.py

# Run linting
black src/ examples/ test_implementation.py
isort src/ examples/ test_implementation.py
flake8 src/ examples/ test_implementation.py
```

## Community

- 💬 **Discussions**: Join our [GitHub Discussions](https://github.com/yourusername/jax-noprop/discussions)
- 🐛 **Issues**: Report bugs and request features in [Issues](https://github.com/yourusername/jax-noprop/issues)
- 📖 **Wiki**: Check out our [Wiki](https://github.com/yourusername/jax-noprop/wiki) for additional resources

## Citation

```bibtex
@misc{li2025noprop,
  title={NoProp: Training Neural Networks without Back-propagation or Forward-propagation},
  author={Li, Qinyu and Teh, Yee Whye and Pascanu, Razvan},
  year={2025},
  eprint={2503.24322v1},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## License

MIT License
