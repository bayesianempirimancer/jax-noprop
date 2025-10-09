# JAX/Flax NoProp Implementation Summary

## Overview

This repository provides a complete JAX/Flax implementation of the NoProp algorithm from the paper "NoProp: Training Neural Networks without Back-propagation or Forward-propagation" by Li et al. (arXiv:2503.24322v1).

## What We've Built

### 1. Core NoProp Variants

- **NoProp-DT (Discrete-time)**: Implements the discrete-time variant with fixed timesteps
- **NoProp-CT (Continuous-time)**: Implements the continuous-time variant with neural ODE integration
- **NoProp-FM (Flow Matching)**: Implements the flow matching variant

### 2. Model Architectures

- **ResNetWrapper**: Flexible wrapper for ResNet backbones that handles NoProp-specific inputs
- **SimpleCNN**: Lightweight CNN for smaller datasets like MNIST
- Support for ResNet-18, ResNet-50, and ResNet-152 architectures

### 3. Noise Scheduling

- **LinearNoiseSchedule**: Linear noise schedule as used in the paper
- **CosineNoiseSchedule**: Cosine schedule for smoother transitions
- **SigmoidNoiseSchedule**: Sigmoid schedule with learnable parameters

### 4. Training Infrastructure

- **Training utilities**: Functions for creating training states, training steps, and evaluation
- **Data handling**: Utilities for data loading, normalization, and iteration
- **Loss functions**: Implemented for all three NoProp variants

### 5. Examples and Documentation

- **Quick start example**: Basic usage demonstration
- **MNIST training script**: Complete training pipeline for MNIST
- **CIFAR training script**: Training scripts for CIFAR-10 and CIFAR-100
- **Comprehensive documentation**: API reference and usage examples

## Key Features

### Modular Design
- Easy to extend with different model architectures
- Clean separation between NoProp variants and model backbones
- Flexible noise scheduling system

### JAX/Flax Integration
- High-performance implementation with automatic differentiation
- GPU acceleration support
- Functional programming paradigm

### Research-Ready
- Follows the paper's architecture and hyperparameters
- Supports all three variants described in the paper
- Includes evaluation metrics and training loops

## File Structure

```
jax-noprop/
├── src/jax_noprop/
│   ├── __init__.py              # Main package exports
│   ├── noprop_dt.py            # Discrete-time NoProp
│   ├── noprop_ct.py            # Continuous-time NoProp
│   ├── noprop_fm.py            # Flow matching NoProp
│   ├── models.py               # Model architectures
│   ├── noise_schedules.py      # Noise scheduling utilities
│   └── utils.py                # Training utilities
├── examples/
│   ├── quick_start.py          # Basic usage example
│   ├── train_mnist.py          # MNIST training script
│   └── train_cifar.py          # CIFAR training script
├── test_implementation.py      # Test suite
├── pyproject.toml              # Package configuration
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Usage

### Basic Usage

```python
from jax_noprop import NoPropDT, NoPropCT, NoPropFM, ResNetWrapper

# Create model
model = ResNetWrapper(num_classes=10, depth=18)

# Initialize NoProp variants
noprop_dt = NoPropDT(model, num_timesteps=10)
noprop_ct = NoPropCT(model, num_timesteps=1000)
noprop_fm = NoPropFM(model, num_timesteps=1000)
```

### Training

```python
from jax_noprop.utils import create_train_state, train_step

# Create training state
state = create_train_state(model, params, learning_rate=1e-3)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        state, loss, metrics = train_step(state, x, y, key)
```

### Running Examples

```bash
# Quick start
python examples/quick_start.py

# Train on MNIST
python examples/train_mnist.py --epochs 10 --variants DT CT FM

# Train on CIFAR-10
python examples/train_cifar.py --dataset cifar10 --epochs 50
```

## Testing

The implementation includes a comprehensive test suite that verifies:

- ✅ All imports work correctly
- ✅ Noise schedules function properly
- ✅ Models can be initialized and run forward passes
- ✅ NoProp variants can be created and used
- ✅ Training steps execute without errors

Run tests with:
```bash
python test_implementation.py
```

## Performance Expectations

Based on the original paper, the implementation should achieve:

- **MNIST**: ~99% accuracy with NoProp-CT
- **CIFAR-10**: ~90% accuracy with NoProp-CT  
- **CIFAR-100**: ~70% accuracy with NoProp-CT

## Next Steps

1. **Training**: Run the example scripts to train models on real datasets
2. **Experimentation**: Modify hyperparameters and architectures
3. **Extension**: Add new model architectures or noise schedules
4. **Research**: Use as a foundation for NoProp research

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
