"""
JAX/Flax implementation of NoProp algorithm.

This package provides implementations of the NoProp variants:
- NoProp-CT: Continuous-time with neural ODEs  
- NoProp-FM: Flow matching
"""

# from .flow_models.archive.ct import NoPropCT
# from .flow_models.archive.fm import NoPropFM
# Note: archive.no_prop_models module not found - commenting out imports
# from .archive.no_prop_models import (
#     ConditionalResnet_MLP,
#     ConditionalResNet_CNNx,
# )
from .embeddings.noise_schedules import (
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    SigmoidNoiseSchedule,
    LearnableNoiseSchedule,
)
from .embeddings.embeddings import (
    sinusoidal_time_embedding,
    fourier_time_embedding,
    linear_time_embedding,
    gaussian_time_embedding,
    get_time_embedding,
)
from .embeddings.positional_encoding import (
    positional_encoding,
    relative_positional_encoding,
    rotary_positional_encoding,
    get_positional_encoding,
)
from .utils.ode_integration import (
    euler_step,
    heun_step,
    rk4_step,
    adaptive_step,
    integrate_ode,
)
# Training utilities are handled by the built-in train_step methods in the models
from .utils.jacobian_utils import (
    trace_jacobian,
    jacobian_diagonal,
    divergence,
    grad_potential,
    compute_log_det_jacobian,
)
# Note: VitSmallFeatureExtractor is not available in current structure
# from .models.pretrained.vit_small_feature_extractor import (
#     VitSmallFeatureExtractor,
#     create_vit_gradient_stop_fn,
# )
from .models.vit_crn import (
    ViTCRN,
    create_vit_crn_model,
)

__version__ = "0.1.0"
__all__ = [
    # NoProp implementations
    # "NoPropCT", 
    # "NoPropFM",
    # Model architectures (commented out - archive module not found)
    # "ConditionalResnet_MLP",
    # "ConditionalResNet_CNNx",
    # Noise schedules
    "LinearNoiseSchedule",
    "CosineNoiseSchedule", 
    "SigmoidNoiseSchedule",
    "LearnableNoiseSchedule",
    # Embeddings
    "sinusoidal_time_embedding",
    "fourier_time_embedding",
    "linear_time_embedding",
    # "learnable_time_embedding",  # Not exported from embeddings module
    "gaussian_time_embedding",
    "get_time_embedding",
    # Positional encodings
    "positional_encoding",
    "relative_positional_encoding",
    "rotary_positional_encoding",
    "get_positional_encoding",
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
    "jacobian_diagonal",
    "divergence",
    "grad_potential",
    "compute_log_det_jacobian",
    # ViT feature extractor and utilities (not available in current structure)
    # "VitSmallFeatureExtractor",
    # "create_vit_gradient_stop_fn",
    # ViT-based Conditional ResNet
    "ViTCRN",
    "create_vit_crn_model",
]
