"""
Configuration for flow models on two moons dataset.

This config works with all flow models: flow_matching, diffusion, and ct.

Two moons data structure:
- x: (N, 2) - 2D coordinates
- y: (N,) - 1D labels (0 or 1)

Config class tailored to two moons dataset.

USAGE:
You can use this config file in two ways:
1. Command line arguments: Override specific parameters using --flags (see README)
2. Direct editing: Modify values in this file for finer-grained control over all options

Note: Command line arguments take precedence over config file values.
"""

from dataclasses import dataclass, field
from flax.core import FrozenDict
from src.flow_models.config_mix import Config 


@dataclass(frozen=True)
class Config(Config):
    """
    Configuration for VAE with flow model on two moons dataset.
    
    This config provides default values for all model parameters. Many of these
    can be overridden via command line arguments, but editing this file directly
    gives you access to all configuration options.
    """
    # BaseConfig fields
    model_name: str = "two_moons_vae_flow"  # Name identifier for the model

    main: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Data shapes
        "input_shape": (2,),  # Input dimension: 2D coordinates (x)
        "output_shape": (2,),  # Output dimension: one-hot encoded labels (y) [n_samples, 2]
        "latent_shape": (2,),  # Latent space dimension (2D for two moons)
        
        # Loss configuration
        "recon_loss_type": "mse",  # Reconstruction loss type: "mse", "cross_entropy", or "none"
        "recon_weight": 1.0,  # Weight for reconstruction loss (can override with --recon_weight)
        "vae_weight": 0.0,  # Weight for VAE loss (can override with --vae_weight)
        "reg_weight": 0.0,  # Weight for regularization loss (can override with --reg_weight)
        # Flow model settings
        "no_noise_schedule": True,  # Set to True to disable noise schedule (use_noise_schedule=False)
        "normalize_snr_weight": False,  # Apply signal-to-noise ratio weighting to loss
                                  # False for flow_matching, True for diffusion/ct
        "integration_method": "midpoint",  # ODE integration method: "euler" or "midpoint"
                                           # "euler" for flow_matching, "midpoint" for diffusion/ct
        "encode_x": False,  # Whether to use sequence encoding (False for two moons dataset)
    }))
    
    noise_schedule: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Noise schedule configuration (for diffusion and CT models)
        "schedule_type": "linear",  # Schedule type: "linear", "exponential", "cosine", etc.
                                         # (can override with --noise_schedule)
        "learnable": False,  # Whether noise schedule parameters are learnable
                           # (can override with --noise_schedule_learnable)
        "hidden_dims": (64, 64),  # Hidden dimensions for learnable noise schedule network
        
        # Default parameters for different schedule types
        "default_params": FrozenDict({
            "alpha_bar_min": 0.01,  # Minimum value for alpha_bar (not applied to Laplace shedule)
            "alpha_bar_max": 0.99,  # Maximum value for alpha_bar (not applied to Laplace schedule)
            "beta": 0.3,  # Beta parameter for exponential schedule
            "loc": 0.5,  # Location parameter for Laplace schedule only
            "log_scale": 0.0,  # Scale parameter for Cauchy, Laplace schedules only
            "log_power": 0.0,  # Power parameter for polynomial schedules
            "gamma_range": (-4.0, 4.0),  # Range for gamma parameter for neural network
            "gamma_prime_max": 1000.0,  # Maximum value for clipping gamma_prime_t (not applied to neural network schedule)
        }),
    }))

    crn: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Continuous-time ResNet (CRN) configuration
        "model_type": "vanilla",  # CRN type: "vanilla", "geometric", "potential", "natural", "hamiltonian"
                                  # (can override with --crn_type)
        "network_type": "mlp",  # Network backbone: "mlp", "bilinear", or "convex"
                                # (can override with --network_type)
        "hidden_dims": (32, 32, 32, 32, 32, 32),  # Hidden layer dimensions for the network
                                                   # (can override with --hidden_dims)
        "time_embed_dim": 32,  # Dimension of time embedding
        "time_embed_method": "sinusoidal",  # Time embedding method: "sinusoidal" or other
        "activation_fn": "swish",  # Activation function: "swish", "relu", "tanh", etc.
        "use_batch_norm": False,  # Whether to use batch normalization
        "dropout_rate": 0.1,  # Dropout rate (0.0 to 1.0)
    }))
    
    encoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Encoder configuration (maps input y to latent z)
        "model_type": "identity",  # Encoder type: "identity", "linear", or "mlp"
                                   # Identity for latent_dim=2, linear/mlp for latent_dim>2
                                   # (can override with --encoder_model_type)
        "encoder_type": "deterministic",  # Encoder type: "deterministic" or "stochastic"
        "input_shape": (2,),  # Input shape: one-hot encoded labels (y) [n_samples, 2]   can be empty since its value is inherited from main
        "latent_shape": (2,),  # Latent space shape: 2D for two moons
        "hidden_dims": (16, 32, 16),  # Hidden dimensions for MLP encoder (not used for identity/linear)
        "activation": "swish",  # Activation function for encoder
        "dropout_rate": 0.0,  # Dropout rate for encoder
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Decoder configuration (maps latent z to output y)
        "model_type": "identity",  # Decoder model type: "identity", "linear", or "mlp"
                              # (can override with --decoder_model_type)
        "decoder_type": "none",  # Decoder type: "linear", "softmax", "none", or "identity"
                                     # "identity" is same as 'none')
                                     # (can override with --decoder_type)
        "latent_shape": (2,),  # Latent space shape: 2D for two moons.  can be empty since its value is inherited from main
        "output_shape": (2,),  # Output shape: one-hot encoded labels [n_samples, 2]
        "hidden_dims": (16, 32, 16),  # Hidden dimensions for MLP decoder (not used for identity/linear)
        "activation": "swish",  # Activation function for decoder
        "dropout_rate": 0.0,  # Dropout rate for decoder
    }))


def get_two_moons_config():
    """Get default TwoMoonsFlowConfig instance."""
    return Config()

