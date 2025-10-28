# Flow Factory Guide

This guide shows you how to create flow models with any combination of flow wrapper type and CRN backbone type.

## Quick Start

### Basic Usage

```python
from src.factories.flow_factory import create_flow_model

# Create any flow type with any CRN backbone
model = create_flow_model(
    flow_type="potential",    # Flow wrapper type
    crn_type="convex",        # CRN backbone type  
    z_dim=8,                  # Latent dimension
    x_dim=4                   # Conditional input dimension
)
```

### Convenience Functions

```python
from src.factories.flow_factory import (
    create_potential_flow,
    create_natural_flow, 
    create_geometric_flow,
    create_hamiltonian_flow,
    create_convex_potential_flow
)

# Quick creation with sensible defaults
potential_flow = create_potential_flow("convex", z_dim=8, x_dim=4)
natural_flow = create_natural_flow("mlp", z_dim=8)  # x_dim auto-set to z_dim
geometric_flow = create_geometric_flow("bilinear", z_dim=8)  # x_dim auto-set to z_dim
```

## Available Flow Types

### 1. **Potential Flow** (`"potential"`)
- **Use case**: General purpose flow modeling
- **Dimensions**: `x_dim` can be different from `z_dim`
- **Example**: `create_flow_model("potential", "convex", z_dim=8, x_dim=4)`

### 2. **Natural Flow** (`"natural"`)
- **Use case**: Natural gradient flows
- **Dimensions**: Requires `x_dim = z_dim`
- **Example**: `create_flow_model("natural", "mlp", z_dim=8, x_dim=8)`

### 3. **Geometric Flow** (`"geometric"`)
- **Use case**: Geometric flow matching
- **Dimensions**: Requires `x_dim = z_dim`
- **Example**: `create_flow_model("geometric", "convex", z_dim=8, x_dim=8)`

### 4. **Hamiltonian Flow** (`"hamiltonian"`)
- **Use case**: Hamiltonian dynamics
- **Dimensions**: `x_dim` can be different from `z_dim`
- **Example**: `create_flow_model("hamiltonian", "bilinear", z_dim=8, x_dim=4)`


## Available CRN Types

### 1. **MLP** (`"mlp"`)
- **Architecture**: Multi-layer perceptron
- **Parameters**: ~93K for z_dim=8, x_dim=4
- **Use case**: Simple, fast, good baseline

### 2. **Convex** (`"convex"`)
- **Architecture**: Convex ResNet with convex activations
- **Parameters**: ~227K for z_dim=8, x_dim=4
- **Use case**: When you need convex potential functions

### 3. **Bilinear** (`"bilinear"`)
- **Architecture**: Bilinear ResNet
- **Parameters**: ~78K for z_dim=8, x_dim=4
- **Use case**: When you need bilinear interactions

### 4. **ResNet** (`"resnet"`)
- **Architecture**: Standard ResNet (currently same as MLP)
- **Parameters**: ~93K for z_dim=8, x_dim=4
- **Use case**: When you want ResNet architecture

## Complete Examples

### Example 1: Potential Flow with Convex ResNet

```python
import jax.numpy as jnp
import jax.random as jr
from src.factories.flow_factory import create_flow_model

# Create model
model = create_flow_model("potential", "convex", z_dim=8, x_dim=4)

# Create sample data
z = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])  # [batch, z_dim]
x = jnp.array([[0.1, 0.2, 0.3, 0.4]])  # [batch, x_dim]
t = jnp.array([0.5])  # [batch]

# Initialize and use
rng = jr.PRNGKey(42)
rng, init_key = jr.split(rng)
params = model.init(init_key, z, x, t, training=True)
dz_dt = model.apply(params, z, x, t, training=True, rngs={'dropout': init_key})

print(f"Model: {type(model).__name__}")
print(f"ResNet: {model.cond_resnet}")
print(f"Parameters: {sum(x.size for x in jax.tree.leaves(params)):,}")
print(f"Output shape: {dz_dt.shape}")
```

### Example 2: Natural Flow with MLP Backbone

```python
# Natural flows require x_dim = z_dim
model = create_flow_model("natural", "mlp", z_dim=8, x_dim=8)

# Sample data with matching dimensions
z = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
x = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])  # Same dim as z
t = jnp.array([0.5])

# Initialize and use
params = model.init(init_key, z, x, t, training=True)
output = model.apply(params, z, x, t, training=True, rngs={'dropout': init_key})
```

### Example 3: All Combinations

```python
from src.factories.flow_factory import create_all_flow_combinations

# Create all possible combinations
models = create_all_flow_combinations(z_dim=8, x_dim=4)

print("Available models:")
for key, model in models.items():
    print(f"  {key}: {type(model).__name__} with {model.cond_resnet}")
```

## Advanced Configuration

### Custom Hidden Dimensions

```python
model = create_flow_model(
    "potential", 
    "convex", 
    z_dim=8, 
    x_dim=4,
    hidden_dims=(256, 128, 64)  # Custom architecture
)
```

### Custom Activation Functions

```python
model = create_flow_model(
    "potential",
    "convex", 
    z_dim=8,
    x_dim=4,
    activation_fn="softplus"  # Convex activations only for convex CRN
)
```

### Custom Time Embedding

```python
model = create_flow_model(
    "potential",
    "mlp",
    z_dim=8,
    x_dim=4,
    time_embed_dim=128,
    time_embed_method="sinusoidal"
)
```

## Parameter Counts by CRN Type

| CRN Type | z_dim=8, x_dim=4 | z_dim=8, x_dim=8 | Notes |
|----------|------------------|------------------|-------|
| MLP      | ~93K            | ~101K           | Fastest, simplest |
| Bilinear | ~78K            | ~78K            | Efficient bilinear interactions |
| Convex   | ~227K           | ~228K           | Most parameters, convex guarantees |
| ResNet   | ~93K            | ~101K           | Same as MLP currently |

## Dimension Requirements

| Flow Type | x_dim Requirement | Example |
|-----------|-------------------|---------|
| Potential | Any | `z_dim=8, x_dim=4` ✓ |
| Natural   | Must equal z_dim | `z_dim=8, x_dim=8` ✓ |
| Geometric | Must equal z_dim | `z_dim=8, x_dim=8` ✓ |
| Hamiltonian | Any | `z_dim=8, x_dim=4` ✓ |

## Error Handling

The factory automatically handles dimension requirements:

```python
# This will raise an error
try:
    model = create_flow_model("natural", "mlp", z_dim=8, x_dim=4)
except ValueError as e:
    print(f"Error: {e}")  # "natural flow requires x_dim (4) to equal z_dim (8)"
```

## Performance Comparison

| Combination | Speed | Memory | Convexity | Use Case |
|-------------|-------|--------|-----------|----------|
| potential + mlp | Fast | Low | No | Baseline, prototyping |
| potential + convex | Slow | High | Yes | Probability distributions |
| natural + convex | Slow | High | Yes | Natural gradients |
| geometric + bilinear | Medium | Medium | No | Geometric flows |
| hamiltonian + mlp | Fast | Low | No | Hamiltonian dynamics |

## Best Practices

1. **Start Simple**: Use `potential + mlp` for initial experiments
2. **Add Complexity**: Switch to `convex` CRN when you need convexity guarantees
3. **Match Dimensions**: Use `natural`/`geometric` flows when `x_dim = z_dim`
4. **Profile Performance**: Test different combinations for your specific use case
5. **Use Convenience Functions**: They provide sensible defaults

## Troubleshooting

### Common Issues

1. **"x_dim must equal z_dim"**: Use matching dimensions for natural/geometric flows
2. **"Dropout needs PRNG"**: Add `rngs={'dropout': key}` to `model.apply()`
3. **"Unknown flow_type"**: Check spelling, use one of the supported types
4. **"Unknown crn_type"**: Check spelling, use one of the supported types

### Debug Tips

```python
# Check what you created
print(f"Model: {type(model).__name__}")
print(f"ResNet: {model.cond_resnet}")

# Check dimensions
print(f"z_dim: {z.shape[-1]}, x_dim: {x.shape[-1]}")

# Check output
print(f"Output shape: {output.shape}")
print(f"Output range: [{jnp.min(output):.3f}, {jnp.max(output):.3f}]")
```

---

**Ready to create any flow model combination you need!** 🎉
