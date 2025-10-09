# API Reference

This document provides detailed API documentation for the JAX NoProp implementation.

## Core Classes

### NoPropDT

Discrete-time NoProp implementation.

```python
class NoPropDT:
    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 10,
        noise_schedule: NoiseSchedule = LinearNoiseSchedule(),
        eta: float = 0.1
    )
```

**Parameters:**
- `model`: The neural network model (ResNetWrapper or SimpleCNN)
- `num_timesteps`: Number of discrete timesteps (default: 10)
- `noise_schedule`: Noise scheduling strategy (default: LinearNoiseSchedule)
- `eta`: Regularization hyperparameter (default: 0.1)

**Methods:**
- `sample_timestep(key, batch_size)`: Sample random timesteps for training
- `add_noise_to_target(target, key, t)`: Add noise to clean targets
- `compute_loss(params, z_t, x, target, t, key)`: Compute training loss
- `train_step(params, x, target, key)`: Single training step
- `generate(params, x, key, num_steps)`: Generate predictions
- `evaluate(params, x, target, key, num_steps)`: Evaluate model

### NoPropCT

Continuous-time NoProp with neural ODE integration.

```python
class NoPropCT:
    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        noise_schedule: NoiseSchedule = LinearNoiseSchedule(),
        integration_method: str = "euler",
        eta: float = 1.0
    )
```

**Parameters:**
- `model`: The neural network model
- `num_timesteps`: Number of timesteps for continuous time (default: 1000)
- `noise_schedule`: Noise scheduling strategy
- `integration_method`: ODE integration method ("euler" or "heun")
- `eta`: Regularization hyperparameter (default: 1.0)

**Methods:**
- `vector_field(params, z, x, t)`: Compute the vector field dz/dt
- `integrate_ode(params, z0, x, t_span, num_steps)`: Integrate neural ODE
- `compute_loss(params, z_t, x, target, t, key)`: Compute training loss
- `train_step(params, x, target, key)`: Single training step
- `generate(params, x, key, num_steps)`: Generate predictions
- `evaluate(params, x, target, key, num_steps)`: Evaluate model

### NoPropFM

Flow matching NoProp implementation.

```python
class NoPropFM:
    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        noise_schedule: NoiseSchedule = LinearNoiseSchedule(),
        integration_method: str = "euler",
        eta: float = 1.0
    )
```

**Parameters:**
- `model`: The neural network model
- `num_timesteps`: Number of timesteps for continuous time (default: 1000)
- `noise_schedule`: Noise scheduling strategy
- `integration_method`: Flow integration method ("euler" or "heun")
- `eta`: Regularization hyperparameter (default: 1.0)

**Methods:**
- `sample_base_distribution(key, shape)`: Sample from base distribution
- `interpolate_path(z0, z1, t)`: Interpolate between base and target
- `compute_flow_matching_loss(params, z0, z1, x, t, key)`: Compute flow matching loss
- `integrate_flow(params, z0, x, num_steps)`: Integrate the flow
- `train_step(params, x, target, key)`: Single training step
- `generate(params, x, key, num_steps)`: Generate predictions
- `evaluate(params, x, target, key, num_steps)`: Evaluate model

## Model Architectures

### ResNetWrapper

Wrapper for ResNet backbones that handles NoProp-specific inputs.

```python
class ResNetWrapper(nn.Module):
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

### SimpleCNN

Lightweight CNN for smaller datasets.

```python
class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        z_dim: Optional[int] = None,
        time_embed_dim: int = 64
    )
```

**Parameters:**
- `num_classes`: Number of output classes
- `z_dim`: Dimension of noisy target (default: num_classes)
- `time_embed_dim`: Time embedding dimension

## Noise Schedules

### NoiseSchedule (Abstract Base Class)

```python
class NoiseSchedule(ABC):
    @abstractmethod
    def get_alpha_t(self, t: jnp.ndarray) -> jnp.ndarray
    
    @abstractmethod
    def get_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray
    
    def get_noise_params(self, t: jnp.ndarray) -> Dict[str, jnp.ndarray]
```

### LinearNoiseSchedule

Linear noise schedule: `alpha_t = 1 - t`, `sigma_t = sqrt(t)`

### CosineNoiseSchedule

Cosine noise schedule: `alpha_t = cos(π/2 * t)`, `sigma_t = sin(π/2 * t)`

### SigmoidNoiseSchedule

Sigmoid noise schedule: `alpha_t = σ(-γt)`, `sigma_t = σ(γt)`

**Parameters:**
- `gamma`: Controls steepness of sigmoid (default: 1.0)

## Utility Functions

### Training Utilities

```python
def create_train_state(
    model: Any,
    params: Dict[str, Any],
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
    weight_decay: float = 1e-4
) -> TrainState

def train_step(
    state: TrainState,
    x: jnp.ndarray,
    target: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[TrainState, jnp.ndarray, Dict[str, jnp.ndarray]]

def eval_step(
    state: TrainState,
    x: jnp.ndarray,
    target: jnp.ndarray,
    key: jax.random.PRNGKey,
    num_steps: Optional[int] = None
) -> Dict[str, jnp.ndarray]
```

### Data Utilities

```python
def one_hot_encode(labels: jnp.ndarray, num_classes: int) -> jnp.ndarray

def normalize_images(images: jnp.ndarray) -> jnp.ndarray

def denormalize_images(images: jnp.ndarray) -> jnp.ndarray

def compute_accuracy(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray

def create_data_iterators(
    train_data: Tuple[jnp.ndarray, jnp.ndarray],
    test_data: Tuple[jnp.ndarray, jnp.ndarray],
    batch_size: int,
    key: jax.random.PRNGKey
) -> Tuple[Any, Any]
```

### Noise Utilities

```python
def add_noise(
    clean: jnp.ndarray,
    noise: jnp.ndarray,
    alpha_t: jnp.ndarray,
    sigma_t: jnp.ndarray
) -> jnp.ndarray

def sample_noise(key: jax.random.PRNGKey, shape: tuple) -> jnp.ndarray

def create_noise_schedule(
    schedule_type: str = "linear",
    **kwargs: Any
) -> NoiseSchedule
```

## Data Structures

### TrainState

Training state for NoProp models.

```python
@struct.dataclass
class TrainState:
    step: int
    apply_fn: Callable
    params: Dict[str, Any]
    tx: optax.GradientTransformation
    opt_state: optax.OptState
```

## Examples

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax_noprop import NoPropDT, ResNetWrapper
from jax_noprop.utils import create_train_state, train_step

# Create model
model = ResNetWrapper(num_classes=10, depth=18)
noprop = NoPropDT(model, num_timesteps=10)

# Initialize parameters
key = jax.random.PRNGKey(42)
dummy_z = jnp.ones((1, 10))
dummy_x = jnp.ones((1, 28, 28, 1))
params = model.init(key, dummy_z, dummy_x)

# Create training state
state = create_train_state(noprop, params, learning_rate=1e-3)

# Training step
x = jax.random.normal(key, (32, 28, 28, 1))
y = jax.nn.one_hot(jax.random.randint(key, (32,), 0, 10), 10)
state, loss, metrics = train_step(state, x, y, key)
```

### Custom Noise Schedule

```python
from jax_noprop.noise_schedules import SigmoidNoiseSchedule

# Create custom noise schedule
schedule = SigmoidNoiseSchedule(gamma=2.0)

# Use with NoProp
noprop = NoPropDT(model, noise_schedule=schedule)
```

### Model Evaluation

```python
# Generate predictions
predictions = noprop.generate(state.params, x, key, num_steps=40)

# Evaluate model
metrics = noprop.evaluate(state.params, x, y, key, num_steps=40)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```
