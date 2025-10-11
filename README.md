# JAX/Flax NoProp Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-0.7.0+-green.svg)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A JAX/Flax implementation of the NoProp algorithm from the paper "NoProp: Training Neural Networks without Back-propagation or Forward-propagation" by Li et al. (arXiv:2503.24322v1).

## Overview

NoProp is a novel approach for training neural networks without relying on standard back-propagation or forward-propagation steps, taking inspiration from diffusion and flow matching methods. This repository provides:

- **NoProp-DT**: Discrete-time implementation
- **NoProp-CT**: Continuous-time implementation with neural ODEs  
- **NoProp-FM**: Flow matching variant

## Key Features

- **Modular Design**: Easy to extend with different model architectures
- **Flexible Model Interface**: Works with any model that takes `(z, x, t)` inputs and outputs `z'` with same shape as `z`
- **Advanced Noise Scheduling**: Multiple noise schedule types including learnable neural network-based schedules
- **Efficient Gamma Parameterization**: Uses `γ(t)` parameterization for numerical stability with `α(t) = sigmoid(γ(t))`
- **JAX/Flax**: High-performance implementation with automatic differentiation
- **Neural ODE Integration**: Built-in ODE solvers for continuous-time variants
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
from jax_noprop.models import ConditionalResNet
from jax_noprop.noise_schedules import CosineNoiseSchedule, LearnableNoiseSchedule

# Create a model that takes (z, x, t) inputs and outputs z' with same shape as z
model = ConditionalResNet(num_classes=10, z_dim=10, depth=18)

# Initialize NoProp variants with different noise schedules
noprop_dt = NoPropDT(model, num_timesteps=10)
noprop_ct = NoPropCT(
    model=model, 
    noise_schedule=CosineNoiseSchedule(),  # or LearnableNoiseSchedule()
    num_timesteps=1000
)
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

The implementation follows the paper's architecture with several key improvements:

### Model Requirements

**Critical**: Any model used with NoProp must satisfy these requirements:

- **Input signature**: `model(z, x, t)` where:
  - `z`: Noisy target tensor `[batch_size, z_dim]`
  - `x`: Input data tensor `[batch_size, height, width, channels]` 
  - `t`: Time step tensor `[batch_size]` (can be `None` for discrete-time variants)
- **Output**: Must return `z'` with **exactly the same shape** as input `z`
- **Time handling**: For discrete-time variants, `t=None` is allowed

The `ConditionalResNet` class provides a reference implementation that meets these requirements.

### Noise Schedules

The implementation uses a **gamma parameterization** for numerical stability:

- **Core relationship**: `α(t) = sigmoid(γ(t))` where `γ(t)` is an increasing function
- **Derived quantities**:
  - `σ(t) = sqrt(1 - α(t))` (noise coefficient)
  - `SNR(t) = α(t) / (1 - α(t)) = exp(γ(t))` (signal-to-noise ratio)
  - `SNR'(t) = γ'(t) * exp(γ(t))` (SNR derivative for loss weighting)

**Available schedules**:

1. **LinearNoiseSchedule**: `γ(t) = logit(t)`, `γ'(t) = 1/(t*(1-t))`
2. **CosineNoiseSchedule**: `γ(t) = logit(sin(π/2 * t))`, smooth transitions
3. **SigmoidNoiseSchedule**: `γ(t) = γ * (t - 0.5)`, `γ'(t) = γ` (constant)
4. **LearnableNoiseSchedule**: Neural network learns `γ(t)` with guaranteed monotonicity

### Training Process

Each NoProp variant implements a different training strategy:

1. **NoProp-DT**: Each layer learns to denoise independently with discrete timesteps
2. **NoProp-CT**: Learns a vector field `dz/dt = f(z, x, t)` for continuous-time denoising via neural ODEs
3. **NoProp-FM**: Learns a flow that transforms base distribution to target distribution

### Key Implementation Details

- **Efficient computation**: Single `get_gamma_gamma_prime_t()` method computes both `γ(t)` and `γ'(t)` to avoid redundant calculations
- **Learnable schedules**: Neural network with positive weights and ReLU activations ensures monotonic `γ(t)`
- **Boundary conditions**: Learnable schedules enforce exact `γ(0) = γ_min` and `γ(1) = γ_max`
- **ODE integration**: Built-in Euler, Heun, and adaptive step methods for continuous-time variants

## API Reference

### Core Classes

#### `NoPropDT`
Discrete-time NoProp implementation.

```python
noprop_dt = NoPropDT(
    model=ConditionalResNet(num_classes=10, z_dim=10),
    num_timesteps=10,
    noise_schedule=LinearNoiseSchedule(),
    eta=0.1
)
```

#### `NoPropCT`
Continuous-time NoProp with neural ODE integration.

```python
noprop_ct = NoPropCT(
    model=ConditionalResNet(num_classes=10, z_dim=10),
    noise_schedule=CosineNoiseSchedule(),  # or LearnableNoiseSchedule()
    num_timesteps=1000,
    integration_method="euler",  # or "heun", "rk4"
    reg_weight=1.0  # regularization weight
)
```

#### `NoPropFM`
Flow matching NoProp implementation.

```python
noprop_fm = NoPropFM(
    model=ConditionalResNet(num_classes=10, z_dim=10),
    num_timesteps=1000,
    integration_method="euler",  # or "heun", "rk4"
    eta=1.0
)
```

### Model Architectures

#### `ConditionalResNet`
Wrapper for ResNet backbones that handles NoProp-specific inputs.

```python
model = ConditionalResNet(
    num_classes=10,
    z_dim=10,  # Output dimension (defaults to num_classes if None)
    depth=18,  # 18, 50, 152
    width=64,
    time_embed_dim=128
)
```

**Key features**:
- Takes `(z, x, t)` inputs where `t` can be `None` for discrete-time variants
- Outputs `z'` with exactly the same shape as input `z`
- Handles time embedding automatically for continuous-time variants
- Projects input features to match `z_dim`

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
from jax_noprop.noise_schedules import (
    LinearNoiseSchedule, 
    CosineNoiseSchedule, 
    SigmoidNoiseSchedule,
    LearnableNoiseSchedule
)

# Linear schedule: γ(t) = logit(t)
schedule = LinearNoiseSchedule()

# Cosine schedule: γ(t) = logit(sin(π/2 * t)) - smoother transitions
schedule = CosineNoiseSchedule()

# Sigmoid schedule: γ(t) = γ * (t - 0.5) with constant derivative
schedule = SigmoidNoiseSchedule(gamma=1.0)

# Learnable schedule: Neural network learns γ(t) with guaranteed monotonicity
schedule = LearnableNoiseSchedule(
    hidden_dims=(64, 64),  # Network architecture
    gamma_min=-5.0,        # γ(0) boundary condition
    gamma_max=5.0          # γ(1) boundary condition
)
```

**Key features**:
- All schedules use gamma parameterization: `α(t) = sigmoid(γ(t))`
- Efficient computation: Single method returns both `γ(t)` and `γ'(t)`
- Learnable schedules ensure monotonicity through positive weights and ReLU activations
- Boundary conditions are enforced exactly for learnable schedules

## Training

### Basic Training Loop

```python
import jax
import jax.numpy as jnp
import optax
from jax_noprop.models import ConditionalResNet
from jax_noprop.noprop_ct import NoPropCT
from jax_noprop.noise_schedules import CosineNoiseSchedule

# Create model and NoProp instance
model = ConditionalResNet(num_classes=10, z_dim=10)
noprop_ct = NoPropCT(
    model=model,
    noise_schedule=CosineNoiseSchedule(),
    num_timesteps=1000
)

# Initialize parameters
key = jax.random.PRNGKey(42)
dummy_z = jnp.ones((batch_size, 10))
dummy_x = jnp.ones((batch_size, 28, 28, 1))
dummy_t = jnp.ones((batch_size,))
params = noprop_ct.init(key, dummy_z, dummy_x, dummy_t)

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# Training step
def train_step(params, opt_state, x, y, key):
    # Sample noisy targets and timesteps
    z_t, t = noprop_ct.sample_zt(key, y, noprop_ct.timesteps)
    
    # Compute loss
    loss, metrics = noprop_ct.compute_loss(params, z_t, x, y, t, key)
    
    # Compute gradients and update
    grads = jax.grad(lambda p: noprop_ct.compute_loss(p, z_t, x, y, t, key)[0])(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss, metrics

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        x, y = batch
        key, subkey = jax.random.split(key)
        params, opt_state, loss, metrics = train_step(params, opt_state, x, y, subkey)
```

### Key Mathematical Concepts

#### Noise Schedule Mathematics

The implementation uses a **gamma parameterization** for numerical stability:

- **Core relationship**: `α(t) = sigmoid(γ(t))` where `γ(t)` is monotonically increasing
- **Backward process**: `z_t = sqrt(α(t)) * z_1 + sqrt(1-α(t)) * ε`
- **Signal-to-Noise Ratio**: `SNR(t) = α(t)/(1-α(t)) = exp(γ(t))`
- **SNR derivative**: `SNR'(t) = γ'(t) * exp(γ(t))` (used for loss weighting)

#### NoProp-CT Vector Field

The continuous-time variant learns a vector field:

```
dz/dt = τ⁻¹(t) * (sqrt(α(t)) * target - (1+α(t))/2 * z)
```

where `τ⁻¹(t) = γ'(t)` is the inverse time constant.

#### Loss Function

The NoProp-CT loss is weighted by the SNR derivative:

```
L = E[SNR'(t) * ||model(z_t, x, t) - target||²] + λ * E[||model(z_t, x, t)||²]
```

This ensures the model learns to denoise more aggressively when the SNR changes rapidly.

### Hyperparameters

The paper suggests the following hyperparameters:

| Dataset | Method | Epochs | Learning Rate | Batch Size | Timesteps | Reg Weight |
|---------|--------|--------|---------------|------------|-----------|------------|
| MNIST | NoProp-DT | 100 | 1e-3 | 128 | 10 | - |
| MNIST | NoProp-CT | 100 | 1e-3 | 128 | 1000 | 1.0 |
| MNIST | NoProp-FM | 100 | 1e-3 | 128 | 1000 | - |
| CIFAR-10 | NoProp-DT | 150 | 1e-3 | 128 | 10 | - |
| CIFAR-10 | NoProp-CT | 500 | 1e-3 | 128 | 1000 | 1.0 |
| CIFAR-10 | NoProp-FM | 500 | 1e-3 | 128 | 1000 | - |

## Implementation Details

### Model Requirements Summary

**Critical**: Any model used with NoProp must satisfy these exact requirements:

1. **Input signature**: `model(z, x, t)` where:
   - `z`: Noisy target tensor `[batch_size, z_dim]`
   - `x`: Input data tensor `[batch_size, height, width, channels]` 
   - `t`: Time step tensor `[batch_size]` (can be `None` for discrete-time variants)

2. **Output**: Must return `z'` with **exactly the same shape** as input `z`

3. **Time handling**: For discrete-time variants, `t=None` is allowed

### Noise Schedule Architecture

The noise schedules are implemented as `nn.Module` instances with the following key features:

- **Gamma parameterization**: All schedules use `α(t) = sigmoid(γ(t))` for numerical stability
- **Efficient computation**: Single `get_gamma_gamma_prime_t()` method computes both `γ(t)` and `γ'(t)`
- **Learnable schedules**: Neural network with positive weights and ReLU activations ensures monotonicity
- **Boundary conditions**: Learnable schedules enforce exact `γ(0) = γ_min` and `γ(1) = γ_max`

### Best Practices

1. **Model Design**: Use `ConditionalResNet` as a reference implementation
2. **Noise Schedules**: Start with `CosineNoiseSchedule()` for most applications
3. **Learnable Schedules**: Use for complex datasets where fixed schedules don't work well
4. **Time Embedding**: Ensure your model properly handles time information for continuous-time variants
5. **Shape Consistency**: Always verify that model output has the same shape as input `z`

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
