# API Reference

This document provides detailed API documentation for the JAX NoProp implementation with the new unified flow models architecture.

## Import Structure

The codebase uses a clean, modular import structure with unified flow models:

```python
# Core NoProp implementations
from src.flow_models.ct import NoPropCT
from src.flow_models.fm import NoPropFM
from src.flow_models.df import NoPropDF

# Model architectures
from src.flow_models.crn import (
    ConditionalResnet_MLP,
    ConvexConditionalResnet,
    BilinearConditionalResnet
)

# Noise schedules and embeddings
from src.embeddings.noise_schedules import (
    LinearNoiseSchedule, 
    CosineNoiseSchedule, 
    SigmoidNoiseSchedule,
    LearnableNoiseSchedule
)
from src.embeddings.embeddings import (
    sinusoidal_time_embedding, 
    fourier_time_embedding,
    get_time_embedding
)

# Utilities
from src.utils.ode_integration import euler_step, heun_step, rk4_step
from src.utils.jacobian_utils import trace_jacobian, divergence
from src.utils.plotting.plot_learning_curves import create_enhanced_learning_plot
from src.utils.plotting.plot_trajectories import (
    create_trajectory_diagnostic_plot,
    create_model_output_trajectory_plot,
    create_dzdt_trajectory_plot
)

# Training
from src.flow_models.trainer import NoPropTrainer
from src.flow_models.train import main as train_main
```

## Core Classes

### NoPropCT

Continuous-time NoProp implementation with neural ODE integration.

```python
class NoPropCT(nn.Module):
    def __init__(
        self,
        config: Config,
        z_shape: Tuple[int, ...],
        model: nn.Module,
        noise_schedule: NoiseSchedule = LinearNoiseSchedule(),
        num_timesteps: int = 20,
        integration_method: str = "euler"
    )
```

**Parameters:**
- `config`: Configuration object containing model parameters
- `z_shape`: Shape of target z (excluding batch dimensions), e.g., `(2,)` for 1D, `(10, 5)` for 2D
- `model`: The neural network model (must take `(z, x, t)` inputs and output same shape as `z`)
- `noise_schedule`: Noise scheduling strategy (default: LinearNoiseSchedule)
- `num_timesteps`: Number of timesteps for continuous time (default: 20)
- `integration_method`: ODE integration method ("euler", "heun", or "rk4")

**Note:** The `z_shape` parameter enables automatic optimization - 1D shapes use efficient `integrate_ode`, while multi-dimensional shapes use `integrate_tensor_ode`.

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

#### `predict(params, x, num_steps, integration_method, output_type, key)`
Generate predictions by integrating the learned vector field.

```python
@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def predict(
    self,
    params: Dict[str, Any],
    x: jnp.ndarray,
    num_steps: int,
    integration_method: str = "euler",
    output_type: str = "end_point",
    key: jr.PRNGKey = None
) -> jnp.ndarray
```

**Parameters:**
- `params`: Model parameters
- `x`: Input data `[batch_size, ...]`
- `num_steps`: Number of integration steps
- `integration_method`: Integration method ("euler", "heun", "rk4", "adaptive")
- `output_type`: Output type ("end_point" or "trajectory")
- `key`: Random key for initialization (default: None for deterministic zeros)

**Returns:**
- `predictions`: Final predictions `[batch_size, z_shape]` or trajectory `[batch_size, num_steps+1, z_shape]`

#### `predict_trajectory(params, x, num_steps, integration_method, key)`
Generate full trajectory by integrating the learned vector field.

```python
def predict_trajectory(
    self,
    params: Dict[str, Any],
    x: jnp.ndarray,
    num_steps: int,
    integration_method: str = "euler",
    key: jr.PRNGKey = None
) -> jnp.ndarray
```

**Parameters:**
- `params`: Model parameters
- `x`: Input data `[batch_size, ...]`
- `num_steps`: Number of integration steps
- `integration_method`: Integration method ("euler", "heun", "rk4", "adaptive")
- `key`: Random key for initialization (default: None for deterministic zeros)

**Note:** This is a wrapper around the `predict` method with `output_type="trajectory"`.

**Returns:**
- `trajectory`: Full trajectory `[batch_size, num_steps + 1, z_shape]`

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

### NoPropFM

Flow matching NoProp implementation with JIT optimization.

```python
class NoPropFM(nn.Module):
    def __init__(
        self,
        z_shape: Tuple[int, ...],
        x_shape: Tuple[int, ...],
        model: nn.Module,
        num_timesteps: int = 20,
        integration_method: str = "euler",
        reg_weight: float = 0.0,
        sigma_t: float = 0.05
    )
```

**Parameters:**
- `z_shape`: Shape of target z (excluding batch dimensions), e.g., `(2,)` for 1D, `(10, 5)` for 2D
- `x_shape`: Shape of input x (excluding batch dimensions), e.g., `(2,)` for 1D, `(28, 28, 1)` for images
- `model`: The neural network model (must take `(z, x, t)` inputs and output same shape as `z`)
- `num_timesteps`: Number of timesteps for continuous time (default: 20)
- `integration_method`: Flow integration method ("euler", "heun", or "rk4")
- `reg_weight`: Regularization hyperparameter (default: 0.0)
- `sigma_t`: Standard deviation of noise added to z_t (default: 0.05)

**Note:** The `z_shape` parameter enables automatic optimization - 1D shapes use efficient `integrate_ode`, while multi-dimensional shapes use `integrate_tensor_ode`.

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

#### `predict(params, x, num_steps, integration_method, output_type, key)`
Generate predictions by integrating the learned flow field.

```python
@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def predict(
    self,
    params: Dict[str, Any],
    x: jnp.ndarray,
    num_steps: int,
    integration_method: str = "euler",
    output_type: str = "end_point",
    key: jr.PRNGKey = None
) -> jnp.ndarray
```

**Parameters:**
- `params`: Model parameters
- `x`: Input data `[batch_size, ...]`
- `num_steps`: Number of integration steps
- `integration_method`: Integration method ("euler", "heun", "rk4", "adaptive")
- `output_type`: Output type ("end_point" or "trajectory")
- `key`: Random key for initialization (default: None for deterministic zeros)

**Returns:**
- `predictions`: Final predictions `[batch_size, z_shape]` or trajectory `[batch_size, num_steps+1, z_shape]`

#### `predict_trajectory(params, x, num_steps, integration_method, key)`
Generate full trajectory by integrating the learned flow field.

```python
def predict_trajectory(
    self,
    params: Dict[str, Any],
    x: jnp.ndarray,
    num_steps: int,
    integration_method: str = "euler",
    key: jr.PRNGKey = None
) -> jnp.ndarray
```

**Parameters:**
- `params`: Model parameters
- `x`: Input data `[batch_size, ...]`
- `num_steps`: Number of integration steps
- `integration_method`: Integration method ("euler", "heun", "rk4", "adaptive")
- `key`: Random key for initialization (default: None for deterministic zeros)

**Note:** This is a wrapper around the `predict` method with `output_type="trajectory"`.

**Returns:**
- `trajectory`: Full trajectory `[batch_size, num_steps + 1, z_shape]`

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

### ConditionalResnet_MLP

Multi-layer perceptron Conditional ResNet for NoProp models.

```python
class ConditionalResnet_MLP(nn.Module):
    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 64),
        output_dim: int = 2,
        dropout_rate: float = 0.1,
        activation: str = "relu"
    )
```

**Parameters:**
- `hidden_dims`: Hidden layer dimensions (default: (64, 64))
- `output_dim`: Output dimension (default: 2)
- `dropout_rate`: Dropout rate (default: 0.1)
- `activation`: Activation function (default: "relu")

**Forward Pass:**
```python
def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray
```

**Key Features:**
- Concatenates `z`, `x`, and time embedding
- Configurable hidden layers with dropout
- Output matches input `z` shape exactly

### ConvexConditionalResnet

Convex Conditional ResNet for NoProp models.

```python
class ConvexConditionalResnet(nn.Module):
    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 64),
        output_dim: int = 2,
        dropout_rate: float = 0.1
    )
```

### BilinearConditionalResnet

Bilinear Conditional ResNet for NoProp models.

```python
class BilinearConditionalResnet(nn.Module):
    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (64, 64),
        output_dim: int = 2,
        dropout_rate: float = 0.1
    )
```

## Unified Training Interface

### NoPropTrainer

Unified trainer for all NoProp variants (CT, FM, DF).

```python
class NoPropTrainer:
    def __init__(self, model: Union[NoPropCT, NoPropFM, NoPropDF])
    
    def train(
        self,
        train_x: jnp.ndarray,
        train_y: jnp.ndarray,
        val_x: jnp.ndarray,
        val_y: jnp.ndarray,
        num_epochs: int,
        test_x: Optional[jnp.ndarray] = None,
        test_y: Optional[jnp.ndarray] = None,
        dropout_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        eval_steps: int = 1,
        save_steps: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]
```

**Parameters:**
- `model`: NoProp model instance (CT, FM, or DF)
- `train_x`: Training input data [N_train, x_dim]
- `train_y`: Training target data [N_train, y_dim]
- `val_x`: Validation input data [N_val, x_dim]
- `val_y`: Validation target data [N_val, y_dim]
- `num_epochs`: Number of training epochs
- `test_x`: Test input data [N_test, x_dim] (optional)
- `test_y`: Test target data [N_test, y_dim] (optional)
- `dropout_epochs`: Number of epochs with dropout (if None, uses all epochs)
- `learning_rate`: Learning rate (if None, uses config default)
- `batch_size`: Batch size (if None, uses config default)
- `eval_steps`: Steps between detailed evaluation prints
- `save_steps`: Steps between model saves (if None, saves only at end)
- `output_dir`: Directory to save results

**Returns:**
- Dictionary containing training results, metrics, and generated plots

**Key Features:**
- Works with any NoProp variant (CT, FM, DF)
- Automatic model type detection
- Integrated plotting and visualization
- Comprehensive result saving
- Uses model-specific `train_step` methods

## Plotting Utilities

### Learning Curve Visualization

```python
from src.utils.plotting.plot_learning_curves import create_enhanced_learning_plot

create_enhanced_learning_plot(
    results: Dict[str, Any],
    train_pred: jnp.ndarray,
    val_pred: jnp.ndarray,
    test_pred: jnp.ndarray,
    train_y: jnp.ndarray,
    val_y: jnp.ndarray,
    test_y: jnp.ndarray,
    output_path: str,
    model_name: str = "NoProp Model",
    skip_epochs: int = 4
)
```

**Parameters:**
- `results`: Training results dictionary
- `train_pred`: Training predictions
- `val_pred`: Validation predictions
- `test_pred`: Test predictions
- `train_y`: Training targets
- `val_y`: Validation targets
- `test_y`: Test targets
- `output_path`: Path to save the plot
- `model_name`: Name for the model in the plot
- `skip_epochs`: Number of initial epochs to skip in visualization

### Trajectory Visualization

```python
from src.utils.plotting.plot_trajectories import (
    create_trajectory_diagnostic_plot,
    create_model_output_trajectory_plot,
    create_dzdt_trajectory_plot
)

# Trajectory diagnostic plot
create_trajectory_diagnostic_plot(
    results: Dict[str, Any],
    output_path: str,
    model_name: str = "NoProp Model"
)

# Model output trajectory plot
create_model_output_trajectory_plot(
    results: Dict[str, Any],
    x_sample: jnp.ndarray,
    output_path: str,
    model_name: str = "NoProp Model"
)

# dz/dt trajectory plot
create_dzdt_trajectory_plot(
    results: Dict[str, Any],
    x_sample: jnp.ndarray,
    output_path: str,
    model_name: str = "NoProp Model"
)
```

**Key Features:**
- Comprehensive trajectory visualization
- Model output evolution tracking
- dz/dt field visualization
- Automatic plot generation and saving

## Command Line Interface

### Training Script

```bash
python src/flow_models/train.py [OPTIONS]
```

**Required Arguments:**
- `--data`: Path to data file (pickle format with 'x' and 'y' keys)
- `--training-protocol`: Training protocol ('ct', 'fm', or 'df')
- `--model`: Model architecture ('conditional_resnet_mlp', 'convex_conditional_resnet', 'bilinear_conditional_resnet')

**Optional Arguments:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-3)
- `--dropout-epochs`: Number of epochs with dropout (default: 0)
- `--eval-steps`: Steps between evaluation prints (default: 1)
- `--loss-type`: Loss function type (default: 'mse')

**Example Usage:**

```bash
# Train a CT model
python src/flow_models/train.py \
    --data data/your_data.pkl \
    --training-protocol ct \
    --model conditional_resnet_mlp \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3

# Train a FM model
python src/flow_models/train.py \
    --data data/your_data.pkl \
    --training-protocol fm \
    --model conditional_resnet_mlp \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3

# Train a DF model
python src/flow_models/train.py \
    --data data/your_data.pkl \
    --training-protocol df \
    --model conditional_resnet_mlp \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3
```

**Data Format:**
The data file should be a pickle file containing a dictionary with the following structure:
```python
{
    'train': {'x': train_x, 'y': train_y},
    'val': {'x': val_x, 'y': val_y},
    'test': {'x': test_x, 'y': test_y}
}
```

**Output:**
The training script generates:
- Model checkpoints
- Training metrics
- Learning curve plots
- Trajectory visualizations
- Results saved to `artifacts/` directory

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

### Unified Integration Function

The implementation provides a unified `integrate_ode` function that handles all integration methods and output types:

```python
def integrate_ode(
    vector_field: Callable,
    params: Dict[str, Any],
    z0: jnp.ndarray,
    x: jnp.ndarray,
    time_span: Tuple[float, float],
    num_steps: int,
    method: str = "euler",
    output_type: str = "end_point"
) -> jnp.ndarray
```

**Parameters:**
- `vector_field`: Function that computes dz/dt = f(z, x, t)
- `params`: Model parameters
- `z0`: Initial state `[batch_size, state_dim]`
- `x`: Input data `[batch_size, ...]`
- `time_span`: Tuple of (start_time, end_time)
- `num_steps`: Number of integration steps
- `method`: Integration method ("euler", "heun", "rk4", "adaptive")
- `output_type`: Output type ("end_point" or "trajectory")

**Returns:**
- If `output_type="end_point"`: Final state `[batch_size, state_dim]`
- If `output_type="trajectory"`: Full trajectory `[batch_size, num_steps+1, state_dim]`

### Integration Methods

The implementation provides four integration methods:

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

#### Adaptive Method
```python
_integrate_ode_adaptive_scan(vector_field, params, z0, x, time_span, max_steps)
```

### Trajectory Integration

For full trajectory visualization:

```python
_integrate_ode_euler_scan_trajectory(vector_field, params, z0, x, time_span, num_steps)
_integrate_ode_heun_scan_trajectory(vector_field, params, z0, x, time_span, num_steps)
_integrate_ode_rk4_scan_trajectory(vector_field, params, z0, x, time_span, num_steps)
_integrate_ode_adaptive_scan_trajectory(vector_field, params, z0, x, time_span, max_steps)
```

**Returns:**
- `trajectory`: Full trajectory `[batch_size, num_steps + 1, output_dim]`

### Utility Functions

#### Individual Step Functions
```python
euler_step(vector_field, params, z, x, t, dt)
heun_step(vector_field, params, z, x, t, dt)
rk4_step(vector_field, params, z, x, t, dt)
adaptive_step(vector_field, params, z, x, t, dt)
```

**Note:** All ODE integration utilities are located in `src/jax_noprop/utils/ode_integration.py` for better code organization.

## Jacobian Utilities

### Optimized Jacobian Computation

The implementation provides optimized Jacobian computation utilities in `src/jax_noprop/utils/jacobian_utils.py`:

#### `trace_jacobian(apply_fn, params, z, x, t)`
Compute the trace of the Jacobian matrix efficiently using forward-mode automatic differentiation.

```python
@partial(jax.jit, static_argnums=(0,))
def trace_jacobian(
    apply_fn: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray
```

**Parameters:**
- `apply_fn`: Function that computes the vector field
- `params`: Model parameters
- `z`: State tensor `[batch_size, z_dim]`
- `x`: Input data `[batch_size, ...]`
- `t`: Time tensor `[batch_size]`

**Returns:**
- `trace`: Jacobian trace `[batch_size]`

**Key Features:**
- Uses `jax.vmap` to process batch elements individually
- Uses `jax.jacfwd` for forward-mode automatic differentiation
- Avoids computing the full Jacobian matrix for memory efficiency
- Returns `batch_shape + (target_dim,)` instead of `batch_shape + (target_dim,) + batch_shape + (target_dim,)`

#### `divergence(apply_fn, params, z, x, t)`
Compute the divergence of the vector field (alias for `trace_jacobian`).

```python
def divergence(
    apply_fn: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray
```

#### `jacobian_diagonal(apply_fn, params, z, x, t)`
Compute the diagonal elements of the Jacobian matrix.

```python
def jacobian_diagonal(
    apply_fn: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray
```

#### `compute_log_det_jacobian(apply_fn, params, z, x, t)`
Compute the log determinant of the Jacobian matrix.

```python
def compute_log_det_jacobian(
    apply_fn: Callable,
    params: Dict[str, Any],
    z: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray
```

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

### Conditional Integration Optimization

Both NoProp-CT and NoProp-FM automatically select the most efficient integration method based on `z_shape` complexity:

**1D z_shapes** (e.g., `(10,)`, `(2,)`):
- Uses optimized `integrate_ode` function
- Maximum performance for common vector outputs
- Direct integration without tensor reshaping overhead

**Multi-dimensional z_shapes** (e.g., `(10, 5)`, `(8, 4, 2)`):
- Uses `integrate_tensor_ode` function
- Proper handling of arbitrary tensor shapes
- Automatic flattening and reshaping for ODE integration

**Implementation Details:**
```python
# Automatic selection in predict() method
if len(self.z_shape) > 1:
    # Use tensor field integrator for multi-dimensional z_shapes
    result = integrate_tensor_ode(...)
else:
    # Use regular integrator for 1D z_shapes (more efficient)
    result = integrate_ode(...)
```

**Performance Benefits:**
- Reduces computational overhead for 1D cases
- Maintains full compatibility with arbitrary tensor shapes
- Transparent to users - no API changes required

## Examples

### Basic Usage

```python
import jax
import jax.numpy as jnp
from jax_noprop import NoPropCT
from jax_noprop.models import SimpleConditionalResnet
from jax_noprop.noise_schedules import CosineNoiseSchedule

# Create model and NoProp instance
model = SimpleConditionalResnet(hidden_dims=(64,))
noprop_ct = NoPropCT(
    z_shape=(2,),
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
predictions = noprop_ct.predict(params, x, num_steps=20)
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
    z_shape=(2,),
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
from jax_noprop.models import SimpleConditionalResnet

# Create model and NoProp-FM instance
model = SimpleConditionalResnet(hidden_dims=(64,))
noprop_fm = NoPropFM(
    z_shape=(2,),
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
predictions = noprop_fm.predict(params, x, num_steps=20)
```

### Trajectory Visualization

```python
# Generate full trajectory
trajectory = noprop_ct.predict_trajectory(params, x, num_steps=20)
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
predictions = noprop_ct.predict(params, x, num_steps=20)

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