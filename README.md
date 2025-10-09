# jax-noprop

A Jax/Flax Implementation of the NoProp Algorithm

## Overview

This package provides JAX implementations of three variations of the NoProp (No Propagation) algorithm for training neural networks:

- **NoPropDT**: Discrete time version for networks with layers that take `(z, x)` and output `z'`
- **NoPropCT**: Continuous time version for networks that take `(z, x, t)` and output `dz/dt`
- **NoPropFM**: Flow matching version for networks that learn velocity fields `v(z, x, t)`

The NoProp algorithm is a gradient-free training approach that uses noise injection and synthetic gradient estimation instead of traditional backpropagation.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Discrete Time (NoPropDT)

```python
import jax
import jax.numpy as jnp
from jax_noprop import NoPropDT
from jax_noprop.models import ConditionalResNetDT

# Create model
model = ConditionalResNetDT(hidden_dims=(64, 64), output_dim=32)

# Initialize parameters
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 32)), jnp.ones((1, 16)))

# Create NoProp wrapper
wrapper = NoPropDT(
    model=lambda p, z, x: model.apply(p, z, x),
    noise_scale=0.01,
    learning_rate=0.001,
)

# Training step
def loss_fn(output):
    return jnp.mean(output ** 2)

z = jax.random.normal(rng, (32, 32))
x = jax.random.normal(rng, (32, 16))
params, metrics = wrapper.train_step(params, (z, x), loss_fn, rng)
```

### Continuous Time (NoPropCT)

```python
from jax_noprop import NoPropCT
from jax_noprop.models import ConditionalResNetCT

# Create model for ODE dynamics
model = ConditionalResNetCT(hidden_dims=(64, 64), output_dim=32)

# Initialize
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 32)), jnp.ones((1, 16)), jnp.array([[0.5]]))

# Create wrapper
wrapper = NoPropCT(
    model=lambda p, z, x, t: model.apply(p, z, x, t),
    time_steps=10,
)

# Integrate trajectory
z0 = jax.random.normal(rng, (32, 32))
x = jax.random.normal(rng, (32, 16))
z_final = wrapper.integrate_trajectory(params, z0, x, t0=0.0, t1=1.0)
```

### Flow Matching (NoPropFM)

```python
from jax_noprop import NoPropFM
from jax_noprop.models import ConditionalResNetFM

# Create velocity field model
model = ConditionalResNetFM(hidden_dims=(64, 64), output_dim=32)

# Initialize
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, 32)), jnp.ones((1, 32)), jnp.array([[0.5]]))

# Create wrapper
wrapper = NoPropFM(
    model=lambda p, z, x, t: model.apply(p, z, x, t),
)

# Training with flow matching loss
z0 = jax.random.normal(rng, (32, 32))  # Base distribution
x = jax.random.normal(rng, (32, 32))   # Target distribution
t = jax.random.uniform(rng)
params, metrics = wrapper.train_step(params, (z0, x, t), None, rng)
```

## Architecture

The package is organized as follows:

- `jax_noprop/base.py`: Base class for all NoProp wrappers
- `jax_noprop/noprop_dt.py`: Discrete time implementation
- `jax_noprop/noprop_ct.py`: Continuous time implementation
- `jax_noprop/noprop_fm.py`: Flow matching implementation
- `jax_noprop/models.py`: Example conditional ResNet models
- `examples/`: Usage examples for each variation
- `jax_noprop/tests/`: Test suite

## Key Features

- **Gradient-free training**: Uses noise injection and synthetic gradients instead of backpropagation
- **Three variations**: Support for discrete time, continuous time, and flow matching scenarios
- **JAX/Flax compatible**: Built on top of JAX for performance and Flax for neural network modules
- **Flexible interface**: Works with any callable model that matches the expected signature
- **Easy to use**: Simple wrapper interface around existing models

## Testing

Run tests with pytest:

```bash
pytest jax_noprop/tests/
```

## Examples

See the `examples/` directory for complete working examples:

- `example_dt.py`: Discrete time training example
- `example_ct.py`: Continuous time ODE training example
- `example_fm.py`: Flow matching training example

Run an example:

```bash
python examples/example_dt.py
```

## References

- Original NoProp PyTorch implementation: https://github.com/yhgon/NoProp

## License

MIT