"""
Configuration for flow models on MNIST dataset.

This config works with all flow models: flow_matching, diffusion, and ct.

MNIST data structure:
- x: (N, 784) - Input flattened MNIST images (28x28 = 784 pixels)
- y: (N, 10) - Output continuous predictions (e.g., digit class probabilities or other continuous outputs)

Config class tailored to MNIST regression tasks.

USAGE:
You can use this config file in two ways:
1. Command line arguments: Override specific parameters using --flags (see README)
2. Direct editing: Modify values in this file for finer-grained control over all options

Note: Command line arguments take precedence over config file values.
"""

from dataclasses import dataclass, field
from flax.core import FrozenDict
from src.flow_models.config import Config as BaseConfig 

@dataclass(frozen=True)
class Config(BaseConfig):
    """
    Configuration for flow models on MNIST regression dataset.
    
    This config provides default values for all model parameters. Many of these
    can be overridden via command line arguments, but editing this file directly
    gives you access to all configuration options.
    """
    # BaseConfig fields
    model_name: str = "mnist_regression"  # Name identifier for the model

    main: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Data shapes
        "input_shape": (784,),  # Input dimension: flattened MNIST images (28x28 = 784)
        "output_shape": (10,),  # Output dimension: continuous predictions (e.g., digit class probabilities)
        "latent_shape": (10,),  # Latent space shape: same as output for regression
        
        # Loss configuration
        "recon_loss_type": "mse",  # Reconstruction loss type: "mse", "cross_entropy", or "none"
        "recon_weight": 1.0,  # Weight for reconstruction loss (can override with --recon_weight)
        "vae_weight": 0.0,  # Weight for VAE loss (can override with --vae_weight)
        "reg_weight": 0.0,  # Weight for regularization loss (can override with --reg_weight)
        # Flow model settings
        "normalize_snr_weight": False,  # Apply signal-to-noise ratio weighting to loss
                                  # False for flow_matching, True for diffusion/ct
        "integration_method": "midpoint",  # ODE integration method: "euler" or "midpoint"
                                        # "euler" for flow_matching, "midpoint" for diffusion/ct
        "encode_x": False,  # Whether to encode x (False for MLP regression)
    }))
    
    noise_schedule: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Noise schedule configuration (for diffusion and CT models)
        "schedule_type": "linear",  # Schedule type: "linear", "exponential", "cosine", etc.
                                         # (can override with --noise_schedule)
        "learnable": True,  # Whether noise schedule parameters are learnable
                           # (can override with --noise_schedule_learnable)
        "hidden_dims": (64, 64),  # Hidden dimensions for learnable noise schedule network
        
        # Default parameters for different schedule types
        "default_params": FrozenDict({
            "alpha_bar_min": 0.01,  # Minimum value for alpha_bar
            "alpha_bar_max": 0.99,  # Maximum value for alpha_bar
            "beta": 0.3,  # Beta parameter for exponential schedule
            "loc": 0.5,  # Location parameter for Laplace schedule only
            "log_scale": 0.0,  # Scale parameter for Cauchy, Laplace schedules only
            "log_power": 0.0,  # Power parameter for polynomial schedules
            "gamma_range": (-4.0, 4.0),  # Range for gamma parameter for neural network
            "gamma_prime_max": 100.0,  # Maximum value for clipping gamma_prime_t
        }),
    }))

    crn: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Continuous-time ResNet (CRN) configuration
        "model_type": "vanilla",  # CRN type: "vanilla", "geometric", "potential", "natural", "hamiltonian"
                                  # (can override with --crn_type)
        "network_type": "mlp",  # Network backbone: "mlp", "bilinear", or "convex"
                                # Use "mlp" for regression (sequences will be flattened)
        "hidden_dims": (128, 128, 128, 128),  # Hidden layer dimensions for the network
        "time_embed_dim": 32,  # Dimension of time embedding
        "time_embed_method": "sinusoidal",  # Time embedding method: "sinusoidal" or other
        "activation_fn": "swish",  # Activation function: "swish", "relu", "tanh", etc.
        "use_batch_norm": False,  # Whether to use batch normalization
        "dropout_rate": 0.1,  # Dropout rate (0.0 to 1.0)
    }))
    
    encoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Encoder configuration (maps input y to latent z)
        "model_type": "identity",  # Encoder type: "identity", "linear", or "mlp"
                                   # Identity for regression (no encoding needed)
        "encoder_type": "deterministic",  # Encoder type: "deterministic" or "stochastic"
        "input_shape": "NA",  # Input shape: will use main.output_shape
        "latent_shape": "NA",  # Latent space shape: will use main.latent_shape
        "hidden_dims": (128, 128),  # Hidden dimensions for MLP encoder (not used for identity)
        "activation": "swish",  # Activation function for encoder
        "dropout_rate": 0.0,  # Dropout rate for encoder
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Decoder configuration (maps latent z to output y)
        "model_type": "identity",  # Decoder model type: "identity", "linear", or "mlp"
                                   # Identity for regression (no decoding needed)
        "decoder_type": "none",  # Decoder type: "linear", "softmax", "none", or "identity"
        "latent_shape": "NA",  # Latent shape: will use main.latent_shape
        "output_shape": "NA",  # Output shape: will use main.output_shape
        "hidden_dims": (128, 128),  # Hidden dimensions for MLP decoder (not used for identity)
        "activation": "swish",  # Activation function for decoder
        "dropout_rate": 0.0,  # Dropout rate for decoder
    }))

