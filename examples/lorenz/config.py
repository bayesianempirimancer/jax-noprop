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
        "input_shape": (1, 3),  # minimum input sequence length
        "output_shape": (4, 3),  # Output sequence shape: (seq_len, embed_dim) - future states
        "latent_shape": (4, 16),  # Latent space shape: (seq_len, latent_dim) - 16-dimensional latent space
        
        # Loss configuration
        "recon_loss_type": "mse",  # Reconstruction loss type: "mse", "cross_entropy", or "none"
        "recon_weight": 1.0,  # Weight for reconstruction loss (can override with --recon_weight)
        "vae_weight": 1.0,  # Weight for reconstruction loss (can override with --recon_weight)
        "reg_weight": 0.0,  # Weight for regularization loss (can override with --reg_weight)
        
        # Flow model settings
        "use_snr_weight": True,  # Apply signal-to-noise ratio weighting to loss
                                  # False for flow_matching, True for diffusion/ct
        "integration_method": "midpoint",  # ODE integration method: "euler" or "midpoint"
                                           # "euler" for flow_matching, "midpoint" for diffusion/ct
        "encode_x": True,  # Whether to encode x sequences (True to encode x to latent space before CRN)
    }))
    
    noise_schedule: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Noise schedule configuration (for diffusion and CT models)
        "schedule_type": "linear",  # Schedule type: "linear", "exponential", "cosine", etc.
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
        # For sequences: encoder treats (batch, seq_len) as batch dimension
        # Only encodes the observation dimension (last dim of input_shape)
        "model_type": "mlp",  # Encoder type: "identity", "linear", "mlp", "mlp_normal", "resnet", "resnet_normal"
        "encoder_type": "deterministic",  # "deterministic" or "stochastic"
        "input_shape": (3,),  # Input shape: only observation dimension (encoder handles batch+seq_len automatically)
        "latent_shape": (16,),  # Latent shape: 16-dimensional latent space
        "hidden_dims": (64, 64),  # Hidden dimensions for MLP encoder
        "activation": "swish",  # Activation function
        "dropout_rate": 0.0,  # Dropout rate
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        # Decoder configuration
        # For sequences: decoder treats (batch, seq_len) as batch dimension
        # Only decodes the observation dimension (last dim of output_shape)
        "model_type": "mlp",  # Decoder type: "identity", "mlp", "resnet"
        "decoder_type": "linear",  # Output type: "linear", "softmax", "none", or "identity"
        "latent_shape": (16,),  # Latent shape: 16-dimensional latent space
        "output_shape": (3,),  # Output shape: only observation dimension (decoder handles batch+seq_len automatically)
        "hidden_dims": (64, 64),  # Hidden dimensions for MLP decoder
        "activation": "swish",  # Activation function
        "dropout_rate": 0.0,  # Dropout rate
    }))
    
    crn: FrozenDict = field(default_factory=lambda: FrozenDict({
        # CRN (Conditional Residual Network) configuration
        "model_type": "vanilla",  # CRN type: "vanilla", "geometric", "potential", "natural", "hamiltonian"
                                  # Use "vanilla" for transformer_seq2seq (no gradient wrapper needed)
                                  # (can override with --crn_type)
        "network_type": "transformer_seq2seq",  # Network backbone: "mlp", "bilinear", "convex", "transformer_seq2seq"
                                                # (can override with --network_type)
        "hidden_dims": (64, 64),  # Hidden dimensions for MLP-based CRNs
                                    # (can override with --hidden_dims)
        "time_embed_dim": 64,  # Time embedding dimension
        "time_embed_method": "sinusoidal",  # Time embedding method: "sinusoidal" or "learnable"
        "x_embed_method": "fourier_features_3d",  # X embedding method: "fourier_features_3d", "positional_encoding", "none"
        "z_embed_method": "fourier_features_3d",  # Z embedding method: "fourier_features_3d", "positional_encoding", "none"
        "activation_fn": "swish",  # Activation function
        "use_batch_norm": False,  # Whether to use batch normalization
        "dropout_rate": 0.1,  # Dropout rate
        # Transformer-specific parameters
        "transformer_config": FrozenDict({
            "embed_dim": 16,  # Embedding dimension for transformer
            "num_heads": 4,  # Number of attention heads
            "num_layers": 3,  # Number of transformer layers
            "mlp_ratio": 4.0,  # Ratio for MLP hidden dimension relative to embed_dim
            "qkv_bias": True,  # Whether to use bias in QKV projections
            "rope_base": 10000.0,  # Base for RoPE frequency calculation
            "lora_rank": 8,  # Rank for LoRA decomposition in TwistedAttention (default: 8)
            "attention_dropout": 0.1,  # Dropout rate for attention (optional, uses top-level dropout_rate if not specified)
            "mlp_dropout": 0.1,  # Dropout rate for MLP (optional, uses top-level dropout_rate if not specified)
            "activation": "swish",  # Activation function (can also use "activation_fn")
            "x_static_dim": 0,  # Dimension of static features x_static (0 means no static features)
            "projection_seed": 42,  # Random seed for projection matrices (if needed)
        }),
    }))

