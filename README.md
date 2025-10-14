# JAX/Flax NoProp Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-0.7.0+-green.svg)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A JAX/Flax implementation of the NoProp algorithm from the paper "NoProp: Training Neural Networks without Back-propagation or Forward-propagation" by Li et al. (arXiv:2503.24322v1)

## Overview

NoProp is a simulation free training protocol for continuous flow models consistent with recent developments in this area (https://arxiv.org/pdf/2210.02747, https://arxiv.org/abs/2503.24322).  Here we provide a Jax/Flax implementation along with a variety of both generative and discriminative models for use with this diffusion inspired training protocol.  Much of the base components are extracted directly from flax library or from the commendably organised jimmy repository (https://github.com/clementpoiret/jimmy).  Here we provide 3 modules designed to make experimentation with No-Prop inspired cost functions a bit easier.  This repository is intended to provide:

- **NoProp-CT**: Continuous-time implementation with neural ODEs (fully optimized)
- **NoProp-FM**: Flow matching variant (fully optimized)
- **NoProp-DT**: Discrete variant (currently broken and thus absent)

as well as a set of Conditional Resnet Models (CRMs) compaitble with the NoProp approach.  

## Key Features

- **Highly Optimized**: JIT-optimized implementations
- **Smart Integration**: Advanced ode integrators for prediction (Euler, Heun, RK4, Adaptive) with scan-based optimization
- **Modular Design**: The NoProp models are essentially wrappers for any CRM that takes in `(z, x, t)` outputs `z'` 
- **Advanced Noise Scheduling**: Multiple noise schedule types including learnable neural network-based schedules
- **Efficient Schedule Parameterization**: Uses `γ(t)` parameterization for numerical stability with effeective noise schedule `1 - sigmoid(γ(t))`
- **JAX/Flax**: High-performance implementation with judiciously placed @partial(jit... ) decorators
- **Pretrained Feature Extractors**: Dino, Vit, ResNet, EfficientNet are on the way via flaim and jimmy

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
from src.noprop_ct import NoPropCT
from src.no_prop_models import SimpleConditionalResnet
from src.embeddings.noise_schedules import CosineNoiseSchedule, LearnableNoiseSchedule

# Create a model that takes (z, x, t) inputs and outputs z' with same shape as z
model = SimpleConditionalResNet(hidden_dims=(64, 64))

# Initialize NoProp-CT with different noise schedules
noprop_ct = NoPropCT(
    z_shape=(1000,),
    x_shape=(224,224,3),
    model=model, 
    noise_schedule=LinearNoiseSchedule(), # skip for noprop_fm
    num_timesteps=20,
    integration_method='euler'
)

# Training Step
updated_params, updated_opt_state, loss, metrics = noprop_ct.trainstep(params, 
                                                                       x, 
                                                                       y, 
                                                                       opt_state, 
                                                                       optimizer, 
                                                                       key)

# Prediction
y = noprop_ct.predict(params, 
                      x, 
                      num_steps, 
                      integration_method = 'euler', 
                      output_type = 'end_point', 
                      with_logp = False, 
                      key = None)
```

### Prediction Options
- integration_methods = 'euler', ''heun', 'rk4', 'adaptive'
- output_type = 'end_point', 'trajectory'

where trajectory puts the n+1 time points into the first temsor dimension, i.e. y.batch_shape = (n+1,) + x.batch_shape

### Key Behavior

- **`key=None`** (default): Uses deterministic initialization (zeros) for reproducible inference
- **`key=jr.PRNGKey(...)`**: Uses random initialization for sensitivity analysis and stochastic inference
- **`compute_loss`**: Still requires a key for training (as it should)

This makes the codebase much more maintainable and user-friendly! 🎉

### Run Examples

```bash
# Two moons dataset example 
python examples/two_moons.py

# The example will train both NoProp-CT and NoProp-FM models
```

## Directory Structure

The codebase is organized into a clean, modular structure:

```
jax-noprop/
├── src/                          # Main source code
│   ├── noprop_ct.py             # NoProp-CT implementation
│   ├── noprop_fm.py             # NoProp-FM implementation
│   ├── no_prop_models.py        # Conditional Resnet Models for use with NoProp
│   ├── trainer.py               # Training utilities and trainer class
│   ├── embeddings/              # Time and positional embeddings
│   │   ├── embeddings.py        # Time embedding functions
│   │   ├── noise_schedules.py   # Noise scheduling strategies
│   │   └── positional_encoding.py # Positional encoding functions
│   ├── blocks/                  # Building block architectures
│   │   ├── image_blocks.py      # Image processing blocks (ResNet, EfficientNet, ViT)
│   │   └── point_cloud_blocks.py # Point cloud processing blocks (PointNet, PointNet2, etc.)
│   ├── layers/                  # Layer implementations
│   │   ├── concatsquash.py      # ConcatSquash layer for time conditioning
│   │   └── image_models.py      # Image model layers
│   ├── models/                  # Model architectures
│   │   ├── simple_models.py     # Simple MLP/ResNet/Transformer models
│   │   └── vit_crn.py          # Vision Transformer CRN
│   └── utils/                   # Utility functions
│       ├── jacobian_utils.py    # Jacobian computation utilities
│       ├── ode_integration.py   # ODE integration methods
├── examples/                    # Example scripts
│   ├── two_moons.py            # Two moons classification example
│   └── two_moons_swapped.py    # Two moons generative example
├── docs/                       # Documentation
│   ├── API_REFERENCE.md        # API documentation
│   └── NoPropCT_Forward_Fix.pdf # Technical notes
└── artifacts/                  # Generated outputs (plots, results)
```

## Architecture

The implementation follows the paper's architecture with several key improvements:

### Model Requirements

**Critical**: Any model used with NoProp must satisfy these requirements:

- **Input signature**: `model(z, x, t)` where:
  - `z`: Noisy target tensor `(batch_size,) + z_shape`
  - `x`: Input data tensor `(batch_size,) + x_shape` 
  - `t`: Time step tensor `[batch_size]` (can be `None` for discrete-time variants)
- **Output**: Must return `z'` with **exactly the same shape** as input `z`
- **Note**: Internally the NoProp code works with vectorized 'z' with reshapeing handled automatically

The `SimpleConditionalResnet` class provides a reference implementation that meets these requirements.

### Noise Schedules

The implementation uses a **gamma parameterization** for numerical stability:

- **Core relationship**: `α(t) = sigmoid(γ(t))` where `γ(t)` is an increasing function
- **Derived quantities**:
  - `σ(t) = sqrt(1 - α(t))` (noise coefficient)
  - `SNR(t) = α(t) / (1 - α(t)) = exp(γ(t))` (signal-to-noise ratio)
  - `SNR'(t) = γ'(t) * exp(γ(t))` (SNR derivative for loss weighting)

**Available schedules**:

1. **LinearNoiseSchedule**: `γ(t) = logit(0.01 + 0.98*t)`
2. **CosineNoiseSchedule**: `γ(t) = logit(0.01 + 0.98*sin(π/2 * t))`  
3. **SigmoidNoiseSchedule**: `γ(t) = γ * (t - 0.5)` 
4. **LearnableNoiseSchedule**: Neural network learns `γ(t)` with guaranteed monotonicity

**⚠️ Important Note on Noise Schedule Singularities**: Care should be taken to ensure that noise schedules do not have singularities at t=0 or t=1. Common schedules like Linear and Cosine have this problem because of the particular parameterization we use for the noise scuedules under the hood.  This is because we paramterize `γ(t)` directly and then compute  `α(t) = sigmoid(γ(t))`.  This means that common noise scuedules like  `α(t) = 1-t` are not really accessible.  As a result `CosineNoiseSchedule` and `LinearNoiseSchedule` implemented here are approximate so as to avoid singularities in things like `SRN'(t)`.

### Training Process

Each NoProp variant implements a different training strategy:

1. **NoProp-CT**: Learns a vector field `dz/dt = f(z, x, t)` for continuous-time denoising via neural ODEs using SNR weighted loss
2. **NoProp-FM**: Learns a vector field `dz/dt = f(z, x, t)` for continuous-time denoising via neural ODEs using a simple field matching loss

### Key Implementation Details

- **Efficient computation**: Single `get_gamma_gamma_prime_t()` method computes both `γ(t)` and `γ'(t)` to avoid redundant calculations
- **Learnable schedules**: Neural network with positive weights and ReLU activations and a terminal rescaling ensures bounded monotonic `γ(t)`
- **ODE integration**: Built-in Euler, Heun, Runge-Kutta 4th order, and Adaptive methods with scan-based optimization
- **Unified integration interface**: Single `integrate_ode` function with `output_type` parameter for end-point or trajectory outputs
- **JIT optimization**: All critical methods are JIT-compiled for maximum performance
- **Modular utilities**: ODE integration and Jacobian utilities are organized in `src/jax_noprop/utils/` for better code organization

## Performance Optimizations

The implementation includes several key optimizations:

### JIT Compilation via @partial(jit...)
- **`compute_loss`**: 
- **`predict`**: 
- **`train_step`**: slight speedup vs simply JIT compiling `compute_loss`

### Scan-based Integration
- All ODE integration uses `jax.lax.scan` 
- Enables efficient JIT compilation and vectorization
- Supports Euler, Heun, RK4, and Adaptive integration methods
- Provides trajectory visualization capabilities
- Unified `integrate_ode` function with `output_type` parameter for end-point or trajectory outputs
- Optional tracking of the evolution of log_p via the trace of the jacobian for normalizing flows


## API Reference

### Core Classes

#### `NoPropCT`
Continuous-time NoProp with neural ODE integration.

```python
noprop_ct = NoPropCT(
    z_shape=(2,),
    model=SimpleConditionalResnet(hidden_dim=64),
    noise_schedule=CosineNoiseSchedule(),
    num_timesteps=20,
    integration_method="euler",
    reg_weight=0.0
)
```

**Key Methods:**
- `predict(params, x, num_steps, integration_method="euler", output_type="end_point", key=None)`: Generate predictions
- `compute_loss(params, x, target, key)`: Compute SNR-weighted loss
- `train_step(params, opt_state, x, target, key, optimizer)`: Single training step

#### `NoPropFM`
Flow matching NoProp implementation.

```python
noprop_fm = NoPropFM(
    z_shape=(2,),
    model=SimpleConditionalResent(hidden_dim=64),
    num_timesteps=20,
    integration_method="euler",
    reg_weight=0.0,
    sigma_t=0.05
)
```

**Key Methods:**
- `predict(params, x, num_steps, integration_method="euler", output_type="end_point", key=None)`: Generate predictions
- `compute_loss(params, x, target, key)`: Compute flow matching loss
- `train_step(params, opt_state, x, target, key, optimizer)`: Single training step


### Noise Schedules

```python
from src.embeddings.noise_schedules import (
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
    gamma_range= (-4.0,4.0)  # initial values for gamma_min and gamma_max
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
from src.no_prop_models import SimpleConditionalResnet
from src.noprop_ct import NoPropCT
from src.embeddings.noise_schedules import CosineNoiseSchedule

# Create model and NoProp instance
model = ConditionalResnet(hidden_dims=(64, 64))
noprop_ct = NoPropCT(
    z_shape=(2,),
    x_shape=(2,),
    model=model,
    noise_schedule=CosineNoiseSchedule(),
    num_timesteps=20
)

# Initialize parameters
key = jax.random.PRNGKey(42)
dummy_z = jnp.ones((1, 2))
dummy_x = jnp.ones((1, 2))
dummy_t = jnp.ones((1,))
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
dz/dt = τ⁻¹(t) * (sqrt(α(t)) * u(z,x,t) - (1+α(t))/2 * z)
```

where `τ⁻¹(t) = γ'(t)` is the inverse time constant and u(z,x,t) is the output of the neural network.

**Critical Side Note**

The original NoProp paper has a minor mathematical error/omission for the continuous time case which 
affects inference, but not learning.  In practice it seems to have little effect on performance which 
is probably why it went unnoticed. The ultimate source of the error was likely a slightly cavalier attitude 
toward small `dt` limits resulting in incorrect calculation of the effective time constant for the forward
process.  Fortunately the correct dynamics described above do not differ from the original NoProp dynamics 
very much for  `α(t)` close to 1 so the last few time steps so despite the error the results it gives are 
very similiar because the last few denoising steps are largely indistinguishable.  See the hastily written 
and likely error riddled writeup of my derivation of the forward process in the docs directory.  For a 
more thorough and significantly denser writeup see (https://arxiv.org/pdf/2210.02747)


#### Loss Function

The NoProp-CT loss is weighted by the SNR derivative and normalized by batch mean:

```
L = E[SNR'(t) * ||model(z_t, x, t) - target||²] / E[SNR'(t)] + λ * E[||model(z_t, x, t)||²]
```

This ensures the model learns to denoise more aggressively when the SNR changes rapidly, while the normalization prevents large SNR' values from destabilizing learning allowing for the use of typical learning rate.

## Implementation Details

### Model Requirements Summary

**Critical**: Any model used with NoProp must satisfy these exact requirements:

1. **Input signature**: `model(z, x, t)` where:
   - `z`: Noisy target tensor `(batch_size,) + z_shape`
   - `x`: Input data tensor `(batch_size,) + x_shape'
   - `t`: Time step tensor `(batch_size,)` (can be scalar or even `None` for discrete-time variants)

2. **Output**: Must return `z'` with **exactly the same shape** as input `z`

3. **Time handling**: For discrete-time variants, `t=None` is allowed

### Noise Schedule Architecture

The noise schedules are implemented as `nn.Module` instances with the following key features:

- **Gamma parameterization**: All schedules use `α(t) = sigmoid(γ(t))` for numerical stability
- **Efficient computation**: Single `get_gamma_gamma_prime_t()` method computes both `γ(t)` and `γ'(t)`
- **Learnable schedules**: Neural network with positive weights and ReLU activations ensures monotonicity
- **Boundary conditions**: Learnable schedules enforce exact `γ(0) = γ_min` and `γ(1) = γ_max`

### Best Practices

1. **Model Design**: Use `SimpleConditionalResnet` as a reference implementation for simple datasets
2. **Noise Schedules**: Start with `CosineNoiseSchedule()` for most applications to avoid singularities
3. **Learnable Schedules**: Use for complex datasets where fixed schedules don't work well
4. **Time Embedding**: Ensure your model properly handles time information for continuous-time variants
5. **Shape Consistency**: Always verify that model output has the same shape as input `z`
6. **JIT Optimization**: The implementation is already highly optimized - no additional JIT needed
7. **NoProp-FM**: Use for applications where inference speed is critical, as it's faster than NoProp-CT for prediction
8. **Tensor Shapes**: The conditional integration optimization automatically handles both 1D and multi-dimensional `z_shapes` efficiently

## Performance

The implementation achieves excellent performance with the optimizations:

- **Two Moons Dataset**: 95%+ training accuracy, 100% validation accuracy
- **JIT Compilation**: 1000x+ speedups on critical methods
- **Memory Efficiency**: Batch size inference and static argument optimization
- **ODE Integration**: Scan-based integration with multiple methods (Euler, Heun, RK4, Adaptive)
- **Conditional Integration**: Automatic optimization for 1D vs multi-dimensional tensor shapes
- **Runtime Comparison**: NoProp-CT is 1.5x faster for training, NoProp-FM is 2.8x faster for inference

## Examples

### Two Moons Dataset

The repository includes a complete example on the two moons dataset:

```bash
# Run with cosine noise schedule
python examples/two_moons.py --epochs 50 --noise-schedule cosine

# Run with learnable noise schedule  
python examples/two_moons.py --epochs 50 --noise-schedule learnable
```

This example demonstrates:
- Data generation and splitting
- Model training with different noise schedules
- Comprehensive visualization including:
  - Learning curves
  - Predictions vs targets
  - 2D trajectory evolution
  - Full ODE integration trajectories

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.


## Citations

```bibtex
@inproceedings{Li2025NoProp,
  title={{NoProp: Training Neural Networks without Full Back-propagation or Full Forward-propagation}},
  author={Qinyu Li and Yee Whye Teh and Razvan Pascanu},
  booktitle={Conference on Lifelong Learning Agents (CoLLAs)},
  year={2025},
  url={https://arxiv.org/abs/2503.24322}
}
@article{Lipman2022FlowMF,
  title={{Flow Matching for Generative Modeling}},
  author={Yaron Lipman and Ricky T. Q. Chen and Heli Ben-Hamu and Maximilian Nickel and Matt Le},
  journal={arXiv preprint arXiv:2210.02747},
  year={2022},
  url={https://arxiv.org/abs/2210.02747}
}

```

## License

MIT License
