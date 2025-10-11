# API Reference

This document provides detailed API documentation for the JAX NoProp implementation.

## Core Classes

### NoPropCT

Continuous-time NoProp implementation with neural ODE integration.

```python
class NoPropCT(nn.Module):
    def __init__(
        self,
        target_dim: int,
        model: nn.Module,
        noise_schedule: NoiseSchedule = CosineNoiseSchedule(),
        num_timesteps: int = 20,
        integration_method: str = "euler",
        reg_weight: float = 0.0
    )
```

**Parameters:**
- `target_dim`: Dimension of target z (output dimension)
- `model`: The neural network model (must take `(z, x, t)` inputs and output same shape as `z`)
- `noise_schedule`: Noise scheduling strategy (default: CosineNoiseSchedule)
- `num_timesteps`: Number of timesteps for continuous time (default: 20)
- `integration_method`: ODE integration method ("euler", "heun", or "rk4")
- `reg_weight`: Regularization hyperparameter (default: 0.0)

**Key Methods:**

#### `compute_loss(params, z_t, x, target, t, key)`
Compute the NoProp-CT training loss with SNR weighting.

```python
def compute_loss(
    self,
    params: Dict[str, Any],
    z_t: jnp.ndarray,
    x: jnp.ndarray,
    target: jnp.ndarray,
    t: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]
```

**Returns:**
- `loss`: Total loss (SNR-weighted MSE + regularization)
- `metrics`: Dictionary with loss components and metrics

#### `train_step(params, x, target, key)`
Single training step with gradient computation and optimizer update.

```python
def train_step(
    self,
    params: Dict[str, Any],
    x: jnp.ndarray,
    target: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[Dict[str, Any], jnp.ndarray, Dict[str, jnp.ndarray]]
```

**Returns:**
- `updated_params`: Updated model parameters
- `loss`: Training loss
- `metrics`: Training metrics

#### `predict(params, x, integration_method, output_dim, num_steps)`
Generate predictions by integrating the learned vector field.

```python
@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def predict(
    self,
    params: Dict[str, Any],
    x: jnp.ndarray,
    integration_method: str,
    output_dim: int,
    num_steps: int
) -> jnp.ndarray
```

**Returns:**
- `predictions`: Final predictions `[batch_size, output_dim]`

#### `predict_trajectory(params, x, integration_method, output_dim, num_steps)`
Generate full trajectory by integrating the learned vector field.

```python
@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def predict_trajectory(
    self,
    params: Dict[str, Any],
    x: jnp.ndarray,
    integration_method: str,
    output_dim: int,
    num_steps: int
) -> jnp.ndarray
```

**Returns:**
- `trajectory`: Full trajectory `[batch_size, num_steps + 1, output_dim]`

#### `sample_zt(key, target, timesteps)`
Sample noisy targets and timesteps for training.

```python
def sample_zt(
    self,
    key: jax.random.PRNGKey,
    target: jnp.ndarray,
    timesteps: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]
```

**Returns:**
- `z_t`: Noisy targets `[batch_size, target_dim]`
- `t`: Sampled timesteps `[batch_size]`

#### `dz_dt(params, z, x, t)`
Compute the vector field dz/dt for the neural ODE.

```python
def dz_dt(
    self,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray
```

**Returns:**
- `dz_dt`: Vector field `[batch_size, target_dim]`

### NoPropDT

Discrete-time NoProp implementation.

```python
class NoPropDT(nn.Module):
    def __init__(
        self,
        target_dim: int,
        model: nn.Module,
        num_timesteps: int = 10,
        noise_schedule: NoiseSchedule = LinearNoiseSchedule(),
        eta: float = 0.1
    )
```

**Parameters:**
- `target_dim`: Dimension of target z
- `model`: The neural network model
- `num_timesteps`: Number of discrete timesteps (default: 10)
- `noise_schedule`: Noise scheduling strategy
- `eta`: Regularization hyperparameter (default: 0.1)

### NoPropFM

Flow matching NoProp implementation with JIT optimization.

```python
class NoPropFM(nn.Module):
    def __init__(
        self,
        target_dim: int,
        model: nn.Module,
        num_timesteps: int = 20,
        integration_method: str = "euler",
        reg_weight: float = 0.0,
        sigma_t: float = 0.05
    )
```

**Parameters:**
- `target_dim`: Dimension of target z
- `model`: The neural network model (must take `(z, x, t)` inputs and output same shape as `z`)
- `num_timesteps`: Number of timesteps for continuous time (default: 20)
- `integration_method`: Flow integration method ("euler", "heun", or "rk4")
- `reg_weight`: Regularization hyperparameter (default: 0.0)
- `sigma_t`: Standard deviation of noise added to z_t (default: 0.05)

**Key Methods:**

#### `compute_loss(params, x, target, key)`
Compute the NoProp-FM training loss with MSE and regularization.

```python
@partial(jax.jit, static_argnums=(0,))
def compute_loss(
    self,
    params: Dict[str, Any],
    x: jnp.ndarray,
    target: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]
```

**Returns:**
- `loss`: Total loss (MSE + regularization)
- `metrics`: Dictionary with loss components and metrics

#### `train_step(params, opt_state, x, target, key, optimizer)`
Single training step with gradient computation and optimizer update.

```python
@partial(jax.jit, static_argnums=(0, 6))
def train_step(
    self,
    params: Dict[str, Any],
    opt_state: optax.OptState,
    x: jnp.ndarray,
    target: jnp.ndarray,
    key: jax.random.PRNGKey,
    optimizer: optax.GradientTransformation
) -> Tuple[Dict[str, Any], optax.OptState, jnp.ndarray, Dict[str, jnp.ndarray]]
```

**Returns:**
- `updated_params`: Updated model parameters
- `updated_opt_state`: Updated optimizer state
- `loss`: Training loss
- `metrics`: Training metrics

#### `predict(params, x, integration_method, output_dim, num_steps)`
Generate predictions by integrating the learned flow field.

```python
@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def predict(
    self,
    params: Dict[str, Any],
    x: jnp.ndarray,
    integration_method: str,
    output_dim: int,
    num_steps: int
) -> jnp.ndarray
```

**Returns:**
- `predictions`: Final predictions `[batch_size, output_dim]`

#### `predict_trajectory(params, x, integration_method, output_dim, num_steps)`
Generate full trajectory by integrating the learned flow field.

```python
@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def predict_trajectory(
    self,
    params: Dict[str, Any],
    x: jnp.ndarray,
    integration_method: str,
    output_dim: int,
    num_steps: int
) -> jnp.ndarray
```

**Returns:**
- `trajectory`: Full trajectory `[batch_size, num_steps + 1, output_dim]`

#### `dz_dt(params, z, x, t)`
Compute the vector field dz/dt for the neural ODE.

```python
def dz_dt(
    self,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray
```

**Returns:**
- `dz_dt`: Vector field `[batch_size, target_dim]`

## Model Architectures

### SimpleMLP

Lightweight MLP for NoProp-CT, designed for simple datasets like two moons.

```python
class SimpleMLP(nn.Module):
    def __init__(self, hidden_dim: int = 64)
```

**Parameters:**
- `hidden_dim`: Hidden layer dimension (default: 64)

**Forward Pass:**
```python
def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray
```

**Key Features:**
- Dynamically infers output dimension from input `z`
- Concatenates `z`, `x`, and time embedding
- 3-layer architecture with ReLU activations
- Output matches input `z` shape exactly

### ConditionalResNet

Wrapper for ResNet backbones that handles NoProp-specific inputs.

```python
class ConditionalResNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        depth: int = 18,
        width: int = 64,
        z_dim: Optional[int] = None,
        time_embed_dim: int = 128
    )
```

**Parameters:**
- `num_classes`: Number of output classes
- `depth`: ResNet depth (18, 50, or 152)
- `width`: Base width of the network
- `z_dim`: Dimension of noisy target (default: num_classes)
- `time_embed_dim`: Time embedding dimension

**Forward Pass:**
```python
def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> jnp.ndarray
```

## Noise Schedules

### NoiseSchedule (Abstract Base Class)

```python
class NoiseSchedule(ABC):
    @abstractmethod
    def get_gamma_gamma_prime_t(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]
```

**Key Methods:**
- `get_gamma_gamma_prime_t(t)`: Returns both `γ(t)` and `γ'(t)` for efficiency
- `get_alpha_t(t)`: Computes `α(t) = sigmoid(γ(t))`
- `get_sigma_t(t)`: Computes `σ(t) = sqrt(1 - α(t))`
- `get_snr_t(t)`: Computes `SNR(t) = α(t)/(1-α(t)) = exp(γ(t))`

### LinearNoiseSchedule

Linear noise schedule: `γ(t) = logit(t)`, `γ'(t) = 1/(t*(1-t))`

### CosineNoiseSchedule

Cosine noise schedule: `γ(t) = logit(sin(π/2 * t))`, smooth transitions

### SigmoidNoiseSchedule

Sigmoid noise schedule: `γ(t) = γ * (t - 0.5)`, `γ'(t) = γ` (constant)

**Parameters:**
- `gamma`: Controls steepness of sigmoid (default: 1.0)

### LearnableNoiseSchedule

Neural network learns `γ(t)` with guaranteed monotonicity.

```python
class LearnableNoiseSchedule(nn.Module):
    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 64),
        gamma_min: float = -5.0,
        gamma_max: float = 5.0
    )
```

**Parameters:**
- `hidden_dims`: Network architecture (default: (64, 64))
- `gamma_min`: γ(0) boundary condition (default: -5.0)
- `gamma_max`: γ(1) boundary condition (default: 5.0)

**Key Features:**
- Uses positive weights and ReLU activations for monotonicity
- Enforces exact boundary conditions `γ(0) = gamma_min`, `γ(1) = gamma_max`
- Learns optimal noise schedule for specific datasets

**⚠️ Important Note on Noise Schedule Singularities**: Care should be taken to ensure that noise schedules do not have singularities at t=0 or t=1. Common schedules like `LinearNoiseSchedule` have this problem where `γ'(t) = 1/(t*(1-t))` becomes infinite at the boundaries. The `CosineNoiseSchedule` and `LearnableNoiseSchedule` are designed to avoid these singularities and are generally recommended for stable training.

## ODE Integration

### Integration Methods

The implementation provides three integration methods:

#### Euler Method
```python
_integrate_ode_euler_scan(vector_field, params, z0, x, time_span, num_steps)
```

#### Heun Method (2nd order)
```python
_integrate_ode_heun_scan(vector_field, params, z0, x, time_span, num_steps)
```

#### Runge-Kutta 4th Order
```python
_integrate_ode_rk4_scan(vector_field, params, z0, x, time_span, num_steps)
```

### Trajectory Integration

For full trajectory visualization:

```python
_integrate_ode_euler_scan_trajectory(vector_field, params, z0, x, time_span, num_steps)
_integrate_ode_heun_scan_trajectory(vector_field, params, z0, x, time_span, num_steps)
_integrate_ode_rk4_scan_trajectory(vector_field, params, z0, x, time_span, num_steps)
```

**Returns:**
- `trajectory`: Full trajectory `[batch_size, num_steps + 1, output_dim]`

## Performance Optimizations

### JIT Compilation

The implementation uses JAX JIT compilation for optimal performance:

**NoProp-CT:**
- **`compute_loss`**: JIT-compiled with `static_argnums=(0,)` for `self`
- **`predict`**: JIT-compiled with `static_argnums=(0, 3, 4, 5)` for `self`, `integration_method`, `output_dim`, `num_steps`
- **`predict_trajectory`**: JIT-compiled with same static arguments as `predict`
- **`train_step`**: Not JIT-compiled (calls already-optimized `compute_loss`)

**NoProp-FM:**
- **`compute_loss`**: JIT-compiled with `static_argnums=(0,)` for `self`
- **`train_step`**: JIT-compiled with `static_argnums=(0, 6)` for `self` and `optimizer`
- **`predict`**: JIT-compiled with `static_argnums=(0, 3, 4, 5)` for `self`, `integration_method`, `output_dim`, `num_steps`
- **`predict_trajectory`**: JIT-compiled with same static arguments as `predict`

### Scan-based Integration

All ODE integration uses `jax.lax.scan` for efficient compilation:
- Avoids Python loops in JIT-compiled code
- Enables vectorization across batch dimensions
- Provides significant speedup over naive implementations

## Examples

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax_noprop import NoPropCT
from jax_noprop.models import SimpleMLP
from jax_noprop.noise_schedules import CosineNoiseSchedule

# Create model and NoProp instance
model = SimpleMLP(hidden_dim=64)
noprop_ct = NoPropCT(
    target_dim=2,
    model=model,
    noise_schedule=CosineNoiseSchedule(),
    num_timesteps=20,
    integration_method="euler"
)

# Initialize parameters
key = jax.random.PRNGKey(42)
dummy_z = jnp.ones((1, 2))
dummy_x = jnp.ones((1, 2))
dummy_t = jnp.ones((1,))
params = noprop_ct.init(key, dummy_z, dummy_x, dummy_t)

# Training step
x = jax.random.normal(key, (32, 2))
y = jax.nn.one_hot(jax.random.randint(key, (32,), 0, 2), 2)
params, loss, metrics = noprop_ct.train_step(params, x, y, key)

# Generate predictions
predictions = noprop_ct.predict(params, x, "euler", 2, 20)
```

### Custom Noise Schedule

```python
from jax_noprop.noise_schedules import LearnableNoiseSchedule

# Create learnable noise schedule
schedule = LearnableNoiseSchedule(
    hidden_dims=(64, 64),
    gamma_min=-5.0,
    gamma_max=5.0
)

# Use with NoProp-CT
noprop_ct = NoPropCT(
    target_dim=2,
    model=model,
    noise_schedule=schedule,
    num_timesteps=20
)
```

### NoProp-FM Usage

```python
import jax
import jax.numpy as jnp
import optax
from jax_noprop import NoPropFM
from jax_noprop.models import SimpleMLP

# Create model and NoProp-FM instance
model = SimpleMLP(hidden_dim=64)
noprop_fm = NoPropFM(
    target_dim=2,
    model=model,
    num_timesteps=20,
    integration_method="euler",
    reg_weight=0.0,
    sigma_t=0.05
)

# Initialize parameters
key = jax.random.PRNGKey(42)
dummy_z = jnp.ones((1, 2))
dummy_x = jnp.ones((1, 2))
dummy_t = jnp.ones((1,))
params = noprop_fm.init(key, dummy_z, dummy_x, dummy_t)

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# Training step
x = jax.random.normal(key, (32, 2))
y = jax.nn.one_hot(jax.random.randint(key, (32,), 0, 2), 2)
params, opt_state, loss, metrics = noprop_fm.train_step(params, opt_state, x, y, key, optimizer)

# Generate predictions
predictions = noprop_fm.predict(params, x, "euler", 2, 20)
```

### Trajectory Visualization

```python
# Generate full trajectory
trajectory = noprop_ct.predict_trajectory(params, x, "euler", 2, 20)
# trajectory shape: [batch_size, 21, 2] (includes initial state)

# Plot trajectory evolution
import matplotlib.pyplot as plt
for i in range(min(5, trajectory.shape[0])):
    plt.plot(trajectory[i, :, 0], trajectory[i, :, 1], 'o-', label=f'Sample {i}')
plt.legend()
plt.show()
```

### Model Evaluation

```python
# Generate predictions
predictions = noprop_ct.predict(params, x, "euler", 2, 20)

# Compute accuracy
accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == jnp.argmax(y, axis=-1))
print(f"Accuracy: {accuracy:.4f}")
```

## Mathematical Background

### Gamma Parameterization

The implementation uses a **gamma parameterization** for numerical stability:

- **Core relationship**: `α(t) = sigmoid(γ(t))` where `γ(t)` is monotonically increasing
- **Backward process**: `z_t = sqrt(α(t)) * z_1 + sqrt(1-α(t)) * ε`
- **Signal-to-Noise Ratio**: `SNR(t) = α(t)/(1-α(t)) = exp(γ(t))`
- **SNR derivative**: `SNR'(t) = γ'(t) * exp(γ(t))` (used for loss weighting)

### NoProp-CT Vector Field

The continuous-time variant learns a vector field:

```
dz/dt = τ⁻¹(t) * (sqrt(α(t)) * target - (1+α(t))/2 * z)
```

where `τ⁻¹(t) = γ'(t)` is the inverse time constant.

### Loss Function

The NoProp-CT loss is weighted by the SNR derivative and normalized by batch mean:

```
L = E[SNR'(t) * ||model(z_t, x, t) - target||²] / E[SNR'(t)] + λ * E[||model(z_t, x, t)||²]
```

This ensures the model learns to denoise more aggressively when the SNR changes rapidly, while the normalization prevents large SNR' values from dominating the learning rate.