"""
Configuration for flow models on Lorenz system dataset.

This config works with all flow models: flow_matching, diffusion, and ct.

Lorenz system data structure:
- x: (N, input_seq_len, 3) - Input sequences (past states)
- y: (N, output_seq_len, 3) - Output sequences (future states)

Config class tailored to Lorenz system sequence modeling.

USAGE:
You can use this config file in two ways:
1. Command line arguments: Override specific parameters using --flags (see README)
2. Direct editing: Modify values in this file for finer-grained control over all options

Note: Command line arguments take precedence over config file values.
"""

from dataclasses import dataclass, field
from flax.core import FrozenDict
from src.flow_models.config import Config 


@dataclass(frozen=True)
class Config(Config):
    """
    Configuration for VAE with flow model on Lorenz system sequences.
    
    This config provides default values for all model parameters. Many of these
    can be overridden via command line arguments, but editing this file directly
    gives you access to all configuration options.
    """
    # BaseConfig fields
    model_name: str = "lorenz_vae_flow"  # Name identifier for the model

    main: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Data shapes (sequences)
        "input_shape": (20, 3),  # Input sequence shape: (seq_len, embed_dim) - past states
        "output_shape": (20, 3),  # Output sequence shape: (seq_len, embed_dim) - future states
        "latent_shape": (20, 3),  # Latent space shape: (seq_len, embed_dim)
        
        # Loss configuration
        "recon_loss_type": "mse",  # Reconstruction loss type: "mse", "cross_entropy", or "none"
        "recon_weight": 1.0,  # Weight for reconstruction loss (can override with --recon_weight)
        "reg_weight": 0.0,  # Weight for regularization loss (can override with --reg_weight)
        
        # Flow model settings
        "use_snr_weight": True,  # Apply signal-to-noise ratio weighting to loss
                                  # False for flow_matching, True for diffusion/ct
        "integration_method": "midpoint",  # ODE integration method: "euler" or "midpoint"
                                           # "euler" for flow_matching, "midpoint" for diffusion/ct
        "encode_x": False,  # Whether to use sequence encoding (False for Lorenz, sequences handled by CRN)
    }))
    
    noise_schedule: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Noise schedule configuration (for diffusion and CT models)
        "schedule_type": "exponential",  # Schedule type: "linear", "exponential", "cosine", etc.
                                         # (can override with --noise_schedule)
        "learnable": False,  # Whether noise schedule parameters are learnable
                             # (can override with --noise_schedule_learnable)
        "hidden_dims": (64, 64),  # Hidden dimensions for learnable noise schedule network
        "default_params": FrozenDict({
            # Default parameters for different schedule types
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
    
    encoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Encoder configuration
        "model_type": "mlp",  # Encoder type: "identity", "linear", "mlp", "mlp_normal", "resnet", "resnet_normal"
                              # (can override with --encoder_model_type)
        "encoder_type": "deterministic",  # "deterministic" or "stochastic"
        "input_shape": (20, 3),  # Input shape (matches main.input_shape)
        "latent_shape": (20, 3),  # Latent shape (matches main.latent_shape)
        "hidden_dims": (64, 128, 64),  # Hidden dimensions for MLP encoder
        "activation": "swish",  # Activation function
        "dropout_rate": 0.0,  # Dropout rate
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Decoder configuration
        "model_type": "mlp",  # Decoder type: "identity", "mlp", "resnet"
                              # (can override with --decoder_model_type)
        "decoder_type": "none",  # Output type: "linear", "softmax", or "none"
                                 # (can override with --decoder_type)
        "latent_shape": (20, 3),  # Latent shape (matches main.latent_shape)
        "output_shape": (20, 3),  # Output shape (matches main.output_shape)
        "hidden_dims": (64, 128, 64),  # Hidden dimensions for MLP decoder
        "activation": "swish",  # Activation function
        "dropout_rate": 0.0,  # Dropout rate
    }))
    
    crn: FrozenDict = field(default_factory=lambda: FrozenDict({
        # CRN (Conditional Residual Network) configuration
        "model_type": "transformer_seq2seq",  # CRN type: "vanilla", "geometric", "potential", "transformer_seq2seq"
                                              # (can override with --crn_type)
        "network_type": "transformer_seq2seq",  # Network backbone: "mlp", "bilinear", "convex", "transformer_seq2seq"
                                                # (can override with --network_type)
        "hidden_dims": (128, 128),  # Hidden dimensions for MLP-based CRNs
                                    # (can override with --hidden_dims)
        "time_embed_dim": 64,  # Time embedding dimension
        "time_embed_method": "sinusoidal",  # Time embedding method: "sinusoidal" or "learnable"
        "activation_fn": "swish",  # Activation function
        "use_batch_norm": False,  # Whether to use batch normalization
        "dropout_rate": 0.1,  # Dropout rate
        # Transformer-specific parameters
        "num_layers": 4,  # Number of transformer layers (can override with --num_layers)
        "num_heads": 8,  # Number of attention heads (can override with --num_heads)
        "mlp_ratio": 4.0,  # MLP ratio for transformer (can override with --mlp_ratio)
    }))

