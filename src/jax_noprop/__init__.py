"""
JAX/Flax implementation of NoProp algorithm.

This package provides implementations of the three NoProp variants:
- NoProp-DT: Discrete-time
- NoProp-CT: Continuous-time with neural ODEs  
- NoProp-FM: Flow matching
"""

from .noprop_dt import NoPropDT
from .noprop_ct import NoPropCT
from .noprop_fm import NoPropFM
from .models import (
    SimpleMLP,
    ResNetBlock,
    ResNet,
    ConditionalResNet,
    SimpleCNN,
)
from .noise_schedules import (
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    SigmoidNoiseSchedule,
    LearnableNoiseSchedule,
)
from .embeddings import (
    sinusoidal_time_embedding,
    fourier_time_embedding,
    linear_time_embedding,
    get_time_embedding,
)
from .utils.ode_integration import (
    euler_step,
    heun_step,
    rk4_step,
    adaptive_step,
    integrate_ode,
)
# Training utilities are available in the utils module
# from ...utils.training_utils import create_train_state, train_step, eval_step
from .utils import (
    trace_jacobian,
    compute_jacobian_diagonal,
    compute_divergence,
    compute_log_det_jacobian,
)

__version__ = "0.1.0"
__all__ = [
    # NoProp implementations
    "NoPropDT",
    "NoPropCT", 
    "NoPropFM",
    # Model architectures
    "SimpleMLP",
    "ResNetBlock",
    "ResNet",
    "ConditionalResNet",
    "SimpleCNN",
    # Noise schedules
    "LinearNoiseSchedule",
    "CosineNoiseSchedule", 
    "SigmoidNoiseSchedule",
    "LearnableNoiseSchedule",
    # Embeddings
    "sinusoidal_time_embedding",
    "fourier_time_embedding",
    "linear_time_embedding",
    "get_time_embedding",
    # ODE integration
    "euler_step",
    "heun_step",
    "rk4_step",
    "adaptive_step",
    "integrate_ode",
    # Training utilities (available in utils module)
    # "create_train_state",
    # "train_step", 
    # "eval_step",
    # Jacobian utilities
    "trace_jacobian",
    "compute_jacobian_diagonal",
    "compute_divergence",
    "compute_log_det_jacobian",
]
