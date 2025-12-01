"""
Unified configuration for Two Moons example.
"""

from dataclasses import dataclass, field
from flax.core import FrozenDict
from src.flow_models.config import Config as BaseConfig

@dataclass(frozen=True)
class Config(BaseConfig):
    """Configuration for Two Moons example."""
    
    main: FrozenDict = field(default_factory=lambda: FrozenDict({
        "input_shape": (2,), # One hot encoded labels (2 classes)
        "output_shape": (2,),
        "latent_shape": (2,),
        "loss_type": "flow_loss", # Set default loss type
        "use_snr_weight": False,
        "use_recon_snr_weight": False,
        "normalize_snr_weight": False,
        "recon_loss_type": "mse",
        "recon_weight": 0.0,
        "reg_weight": 0.0,
        "vae_weight": 0.0,
        "integration_method": "midpoint",
        "num_steps": 20,
        "encode_x": False,
    }))
    
    flow_schedule: FrozenDict = field(default_factory=lambda: FrozenDict({
        "schedule_type": "softplus",
        "learnable": False,
        "latent_shape": (2,),
        "hidden_dims": (64, 64),
        "alpha_min": 0.0,
        "alpha_max": 1.0,
        "sigma_min": 0.0,
        "sigma_max": 1.0,
        "k": 10.0,
        "beta": 2.0,
        "softplus_beta": 50.0,
        "loc": 0.5,
        "log_scale": -1.0,
        "log_power": 0.69,
        "eps": 1e-4,
    }))
    
    crn: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "vanilla",
        "network_type": "mlp",
        "hidden_dims": (32, 32, 32, 32, 32, 32),
        "time_embed_dim": 32,
        "time_embed_method": "sinusoidal",
        "activation_fn": "swish",
        "use_batch_norm": False,
        "dropout_rate": 0.1,
    }))
    
    encoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "identity",
        "encoder_type": "deterministic",
        "input_shape": (2,),
        "latent_shape": (2,),
        "hidden_dims": (16, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.0,
        "rescale": False,
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "identity",
        "decoder_type": "mse",
        "latent_shape": (2,),
        "output_shape": (2,),
        "hidden_dims": (16, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.0,
        "rescale": False,
    }))
