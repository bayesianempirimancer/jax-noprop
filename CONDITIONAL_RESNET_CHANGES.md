# ConditionalResNet Changes

## Overview

I've successfully renamed `ResNetWrapper` to `ConditionalResNet` and ensured that the output shape matches the input z shape, which is crucial for proper NoProp functionality.

## Changes Made

### 1. **Class Rename**
- **Before**: `ResNetWrapper`
- **After**: `ConditionalResNet`

This better reflects the conditional nature of the model, which takes both the noisy target `z` and input `x` (and optionally time `t`) to produce a denoised output.

### 2. **Output Shape Guarantee**
The model now ensures that the output shape exactly matches the input `z` shape:

```python
# Before: Output was always [batch_size, num_classes]
# After: Output is [batch_size, z_dim] where z_dim matches input z

def __call__(self, z, x, t=None):
    z_dim = self.z_dim if self.z_dim is not None else self.num_classes
    
    # Process x through backbone and project to z_dim
    x_features = backbone(x)  # [batch_size, num_classes]
    x_features_proj = nn.Dense(z_dim)(x_features)  # [batch_size, z_dim]
    
    # Process z and time, project to z_dim
    z_features = nn.Dense(self.width)(z)
    if t is not None:
        t_embed = sinusoidal_time_embedding(t, self.time_embed_dim)
        t_proj = nn.Dense(self.width)(t_embed)
        combined_features = z_features + t_proj
    else:
        combined_features = z_features
    
    combined_features_proj = nn.Dense(z_dim)(combined_features)  # [batch_size, z_dim]
    
    # Fuse and output
    fused_features = x_features_proj + combined_features_proj
    output = nn.Dense(z_dim)(fused_features)  # [batch_size, z_dim]
    
    return output
```

### 3. **Updated Architecture Flow**

```
Input z [batch_size, z_dim] ──┐
                              ├──→ Output [batch_size, z_dim]
Input x [batch_size, H, W, C] ──┤
                              │
Input t [batch_size] (optional) ──┘
```

### 4. **Key Benefits**

#### **Shape Consistency**
- Output always matches input `z` shape
- Enables proper denoising in NoProp algorithms
- Supports different `z_dim` values independent of `num_classes`

#### **Flexible Dimensions**
```python
# Example 1: z_dim = num_classes (default)
model = ConditionalResNet(num_classes=10, z_dim=None)
# Input z: [batch_size, 10] → Output: [batch_size, 10]

# Example 2: z_dim different from num_classes
model = ConditionalResNet(num_classes=10, z_dim=20)
# Input z: [batch_size, 20] → Output: [batch_size, 20]
```

#### **Proper NoProp Integration**
- NoProp-DT: `model(z, x)` → denoised z
- NoProp-CT: `model(z, x, t)` → denoised z  
- NoProp-FM: `model(z, x, t)` → transformed z

### 5. **Updated Files**

#### **Core Implementation**
- `src/jax_noprop/models.py`: Renamed class and updated forward pass
- `src/jax_noprop/__init__.py`: Updated imports and exports

#### **Tests and Examples**
- `test_implementation.py`: Updated to use `ConditionalResNet`
- `examples/train_mnist.py`: Updated model creation
- `examples/train_cifar.py`: Updated model creation

### 6. **Testing Results**

All tests pass and shape matching is verified:

```python
# Test 1: z_dim = 20, num_classes = 10
Input z shape: (2, 20)
Output shape: (2, 20)
Shapes match: True

# Test 2: z_dim = None (uses num_classes = 10)
Input z2 shape: (2, 10)
Output2 shape: (2, 10)
Shapes match: True
```

### 7. **Usage Examples**

#### **Basic Usage**
```python
from jax_noprop import ConditionalResNet

# Create model
model = ConditionalResNet(
    num_classes=10,
    z_dim=20,  # Output will be [batch_size, 20]
    depth=18,
    time_embed_dim=128
)

# Forward pass
output = model.apply(params, z, x, t)  # output.shape == z.shape
```

#### **With NoProp**
```python
from jax_noprop import NoPropCT, ConditionalResNet

# Create conditional model
model = ConditionalResNet(num_classes=10, z_dim=10)

# Create NoProp-CT
noprop_ct = NoPropCT(model=model, num_timesteps=1000)

# Training: model learns to denoise z conditioned on x and t
loss, metrics = noprop_ct.compute_loss(params, z_t, x, target, t, key)
```

## Conclusion

The `ConditionalResNet` now properly ensures that:
1. **Output shape matches input z shape** - crucial for NoProp algorithms
2. **Flexible dimension handling** - supports different z_dim values
3. **Proper conditional modeling** - takes z, x, and optionally t as inputs
4. **Maintains all functionality** - all tests pass and examples work

This change makes the model more suitable for NoProp algorithms where the output must have the same shape as the input for proper denoising and transformation operations.
