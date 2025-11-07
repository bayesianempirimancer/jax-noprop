"""
Configuration for flow models on two moons dataset.

This config works with all flow models: flow_matching, diffusion, and ct.

Two moons data structure:
- x: (N, 2) - 2D coordinates
- y: (N,) - 1D labels (0 or 1)

Config class tailored to two moons dataset.
"""

from dataclasses import dataclass, field
from flax.core import FrozenDict
from src.configs.base_config import BaseConfig


@dataclass(frozen=True)
class TwoMoonsFlowConfig(BaseConfig):
    """Configuration for VAE with flow model on two moons dataset."""
    # BaseConfig fields
    model_name: str = "two_moons_vae_flow"

    main: FrozenDict = field(default_factory=lambda: FrozenDict({
        "input_shape": (2,),  # x coordinates are 2D
        "output_shape": (2,),  # y labels are one-hot encoded [n_samples, 2]
        "latent_shape": (2,),
        "recon_loss_type": "mse",  # For one-hot encoded labels
        "recon_weight": 1.0,
        "reg_weight": 0.0,
        "use_snr_weight": True,  # False for flow_matching, True for diffusion/ct
        "integration_method": "midpoint",  # "euler" for flow_matching, "midpoint" for diffusion/ct
        "sigma": 0.02,
        "encode_x": False,  # Not using sequence encoding for two moons
    }))
    
    noise_schedule: FrozenDict = field(default_factory=lambda: FrozenDict({
        "schedule_type": "exponential",
        "learnable": True,
        "hidden_dims": (64, 64),
        "default_params": FrozenDict({
            "alpha_bar_min": 0.01,
            "alpha_bar_max": 0.99,
            "s": 0.008,
            "k": 10.0,
            "t_mid": 0.5,
            "beta": 2.0,
            "loc": 0.5,
            "scale": 0.1,
            "power": 2.0,
            "gamma_range": (-4.0, 4.0),
        }),
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
        "model_type": "identity",  # Identity encoder for latent_dim=2
        "encoder_type": "deterministic",
        "input_shape": (2,),  # y labels are one-hot encoded [n_samples, 2]
        "latent_shape": (2,),  # latent is 2D
        "hidden_dims": (16, 32, 16),  # not used for linear encoder
        "activation": "swish",
        "dropout_rate": 0.0,
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "mlp",
        "decoder_type": "identity",  # For one-hot output
        "latent_shape": (2,),
        "output_shape": (2,),  # One-hot encoded labels [n_samples, 2]
        "hidden_dims": (16, 32, 16), # not used for linear decdoer
        "activation": "swish",
        "dropout_rate": 0.0,
    }))


def get_two_moons_config():
    """Get default TwoMoonsFlowConfig instance."""
    return TwoMoonsFlowConfig()

