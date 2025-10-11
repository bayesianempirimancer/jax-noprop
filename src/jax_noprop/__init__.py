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
from .models import ConditionalResNet, SimpleCNN
from .noise_schedules import (
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    SigmoidNoiseSchedule,
    LearnableNoiseSchedule,
)
from .embeddings import (
    sinusoidal_time_embedding,
    fourier_features,
    positional_encoding,
    get_time_embedding,
)
from .ode_integration import (
    euler_step,
    heun_step,
    rk4_step,
    adaptive_step,
    integrate_ode,
)
from .utils import create_train_state, train_step, eval_step

__version__ = "0.1.0"
__all__ = [
    "NoPropDT",
    "NoPropCT", 
    "NoPropFM",
    "ConditionalResNet",
    "SimpleCNN",
    "LinearNoiseSchedule",
    "CosineNoiseSchedule", 
    "SigmoidNoiseSchedule",
    "LearnableNoiseSchedule",
    "sinusoidal_time_embedding",
    "fourier_features",
    "positional_encoding",
    "get_time_embedding",
    "euler_step",
    "heun_step",
    "rk4_step",
    "adaptive_step",
    "integrate_ode",
    "create_train_state",
    "train_step",
    "eval_step",
]
