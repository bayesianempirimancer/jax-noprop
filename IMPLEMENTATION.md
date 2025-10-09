# NoProp Algorithm Implementation Reference

## Overview

This implementation provides three variations of the NoProp (No Propagation) algorithm for JAX/Flax:

1. **NoPropDT** - Discrete Time version
2. **NoPropCT** - Continuous Time version  
3. **NoPropFM** - Flow Matching version

## Implementation Details

### Base Architecture

All implementations inherit from `NoPropBase` which provides:
- Noise injection mechanism
- Parameter update logic
- Abstract methods for forward pass and gradient computation

### NoPropDT (Discrete Time)

**Use case**: Networks with layers that take `(z, x)` and output `z'`

**Key features**:
- Layer-by-layer noise injection
- Finite difference gradient estimation
- Suitable for discrete state transitions

**Example usage**:
```python
from jax_noprop import NoPropDT
from jax_noprop.models import ConditionalResNetDT

model = ConditionalResNetDT(hidden_dims=(64, 64), output_dim=32)
wrapper = NoPropDT(
    model=lambda p, z, x: model.apply(p, z, x),
    noise_scale=0.01,
    learning_rate=0.001,
)
```

### NoPropCT (Continuous Time)

**Use case**: Networks that take `(z, x, t)` and output `dz/dt` (ODE dynamics)

**Key features**:
- Trajectory integration with Euler method
- Time-dependent noise injection
- Configurable number of integration steps

**Example usage**:
```python
from jax_noprop import NoPropCT
from jax_noprop.models import ConditionalResNetCT

model = ConditionalResNetCT(hidden_dims=(64, 64), output_dim=32)
wrapper = NoPropCT(
    model=lambda p, z, x, t: model.apply(p, z, x, t),
    time_steps=10,
)
```

### NoPropFM (Flow Matching)

**Use case**: Networks that learn velocity fields `v(z, x, t)` for continuous normalizing flows

**Key features**:
- Linear interpolation between base and target distributions
- Target velocity computation: `v = (x - z) / (1 - t)`
- Sinusoidal time embeddings
- Built-in flow matching loss

**Example usage**:
```python
from jax_noprop import NoPropFM
from jax_noprop.models import ConditionalResNetFM

model = ConditionalResNetFM(hidden_dims=(64, 64), output_dim=32)
wrapper = NoPropFM(
    model=lambda p, z, x, t: model.apply(p, z, x, t),
)
```

## Model Architectures

Three example conditional ResNet models are provided:

### ConditionalResNetDT
- Input: `(z, x)` where z is hidden state, x is conditioning
- Output: `z'` (next state)
- Architecture: ResNet blocks with skip connections

### ConditionalResNetCT
- Input: `(z, x, t)` where z is state, x is conditioning, t is time
- Output: `dz/dt` (time derivative)
- Architecture: ResNet blocks with time concatenation

### ConditionalResNetFM
- Input: `(z, x, t)` where z is current state, x is target, t is time
- Output: `v` (velocity field)
- Architecture: ResNet blocks with sinusoidal time embeddings

## Training API

All wrappers provide a consistent training API:

```python
# Training step
new_params, metrics = wrapper.train_step(
    params=params,
    batch=(z, x) or (z0, x, t),  # depending on variant
    loss_fn=loss_function,
    rng=random_key,
)
```

## Testing

Run the test suite:
```bash
pytest jax_noprop/tests/
```

All tests check:
- Initialization
- Forward pass with noise
- Gradient computation
- Complete training steps
- Model-specific features (trajectory integration, flow sampling, etc.)

## Examples

Three complete examples are provided in the `examples/` directory:

- `example_dt.py` - Discrete time training
- `example_ct.py` - Continuous time ODE integration
- `example_fm.py` - Flow matching for generative modeling

## Key Implementation Choices

1. **JAX Compatibility**: Uses `jax.tree.map` (JAX 0.6.0+) instead of deprecated `jax.tree_map`

2. **Scalar Time Handling**: All models properly handle scalar time values by converting to JAX arrays

3. **Gradient Estimation**: Uses `jax.vmap` for efficient batched gradient computation

4. **Noise Injection**: Configurable noise scale with multiple samples for robust gradient estimates

5. **Modular Design**: Easy to extend with custom models by following the callable interface

## Performance Considerations

- Use `num_noise_samples=2` by default (good balance of accuracy/speed)
- Increase `time_steps` for CT/FM if higher accuracy is needed
- Batch operations are JIT-compiled for efficiency
- All operations are compatible with JAX transformations (grad, vmap, jit)
