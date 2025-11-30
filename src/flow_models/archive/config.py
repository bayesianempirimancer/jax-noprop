"""
Unified configuration for flow models (Flow Matching, Diffusion, and CT).

This config class works for all three model types: flow_matching, diffusion, and ct.
Model-specific defaults can be set via the config file or command-line arguments.
"""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING
from flax.core import FrozenDict
from src.configs.base_config import BaseConfig

if TYPE_CHECKING:
    import argparse


@dataclass(frozen=True)
class Config(BaseConfig):
    """Unified configuration for VAE with flow models (FM, DF, CT)."""
    # BaseConfig fields
    model_name: str = "vae_flow_network"

    main: FrozenDict = field(default_factory=lambda: FrozenDict({
        "input_shape": "NA",  # Will be set based on z_dim
        "output_shape": "NA",  # Will be set based on z_dim or z_dim**2
        "latent_shape": "NA",  # Will be set based on x_dim
        "recon_loss_type": "mse",  # Options: "cross_entropy", "mse", "none". Should be consistent with decoder type
        "recon_weight": 0.0,  # Weight for reconstruction loss in total loss
        "reg_weight": 0.0,  # Weight for regularization loss in total loss
        "vae_weight": 1.0,  # Weight for VAE loss in total loss
        "use_snr_weight": True,
        "normalize_snr_weight": False,  # Normalize SNR weights by their mean (False for flow_matching, True for diffusion/ct)
        "integration_method": "midpoint",  # Options: "euler", "heun", "rk4", "adaptive", "midpoint"
                                        # "euler" for flow_matching, "midpoint" for diffusion/ct
        "encode_x": False,  # Whether to encode x before passing to CRN (True for sequences, False for backward compatibility)
    }))
    
    noise_schedule: FrozenDict = field(default_factory=lambda: FrozenDict({
        "schedule_type": "linear",  # Type of schedule (linear, exponential, cosine, sigmoid, cauchy, laplace, logistic, quadratic, polynomial, monotonic_nn, learnable, network)
        "learnable": True,  # Whether schedule parameters are learnable (False uses stop_gradient)
        "hidden_dims": (64, 64),  # Hidden dimensions for NoiseScheduleNetwork schedule
        # Comprehensive default parameters for all schedules (common naming convention)
        "default_params": FrozenDict({
            "alpha_bar_min": 0.05,  # Minimum value for alpha_bar (not applied to Laplace shedule)
            "alpha_bar_max": 0.95,  # Maximum value for alpha_bar (not applied to Laplace schedule)
            "beta": 0.3,  # Beta parameter for exponential schedule
            "loc": 0.5,  # Location parameter for Laplace schedule only
            "log_scale": 0.0,  # Scale parameter for Cauchy, Laplace schedules only
            "log_power": 0.0,  # Power parameter for polynomial schedules
            "gamma_range": (-4.0, 4.0),  # Range for gamma parameter for neural network
            "gamma_prime_max": 100.0,  # Maximum value for clipping gamma_prime_t (not applied to neural network schedule)
        }),
    }))

    crn: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "vanilla",  # Options: "vanilla", "geometric", "potential", "natural", "hamiltonian"
        "network_type": "mlp",  # Options: "mlp", "bilinear", "convex"
        "hidden_dims": (32, 32, 32, 32, 32, 32),
        "time_embed_dim": 32,
        "time_embed_method": "sinusoidal",
        "activation_fn": "swish",
        "use_batch_norm": False,
        "dropout_rate": 0.1,
        "transformer_config": FrozenDict({
            "embed_dim": 32,  # Embedding dimension for transformer
            "num_heads": 4,  # Number of attention heads
            "num_layers": 2,  # Number of transformer layers
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
    
    encoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "identity",  # Options: "mlp", "mlp_normal", "resnet", "resnet_normal", "identity", "linear"
        "encoder_type": "deterministic",  # Options: "deterministic", "normal"
        "input_shape": "NA",  # Will be set from main config if not specified
        "latent_shape": "NA",
        "hidden_dims": (16, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.0,
        "rescale": False,
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "identity",  # Options: "mlp", "resnet", "identity", "linear"
        "decoder_type": "none",  # Options: "linear", "softmax", "none"
        "latent_shape": "NA",  # Will be set from main config if not specified
        "output_shape": "NA",
        "hidden_dims": (16, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.0,
        "rescale": False,
    }))
    
    def override_from_args(self, args: "argparse.Namespace", model_type: str, 
                          unconditional: bool) -> "Config":
        """
        Override config values with command-line arguments and set up for generation task.
        
        For generation: inputs are y (labels), outputs are x (coordinates).
        If config file has forward direction (x->y), we reverse the shapes.
        
        Args:
            args: Parsed command-line arguments
            model_type: Model type
            unconditional: Whether unconditional generation
            
        Returns:
            Updated config instance with overrides applied, configured for generation
        """
        main_dict = dict(self.main)
        
        # Determine shapes for generation:
        # - output_shape: shape of data being generated (from config or args)
        # - latent_shape: latent space shape (from config or args, unchanged)
        # - input_shape: conditional input shape (empty () for unconditional, from config/args otherwise)
        # Note: For generation, config file typically has x->y format, but we're doing y->x generation
        #       So we reverse: config's input_shape becomes our output_shape, config's output_shape becomes our input_shape
        #       BUT for unconditional: input_shape is always (), output_shape comes from config's input_shape
        
        if unconditional:
            # Unconditional generation: input_shape is empty, output_shape and latent_shape from config/args
            if args.output_shape is not None:
                output_shape = tuple(args.output_shape)
            elif args.input_shape is not None:
                # If input_shape provided, it's actually the output shape (data being generated)
                output_shape = tuple(args.input_shape)
            else:
                # For unconditional: config's input_shape might be empty, so use output_shape from config
                # If config's input_shape is empty, use output_shape; otherwise use input_shape (reversed for generation)
                config_input = main_dict.get('input_shape', (2,))
                config_output = main_dict.get('output_shape', (2,))
                # If config input_shape is empty, it's already set up for unconditional, so use output_shape
                if isinstance(config_input, (list, tuple)) and len(config_input) == 0:
                    output_shape = tuple(config_output) if isinstance(config_output, (list, tuple)) else (config_output,)
                else:
                    # Config is x->y format, we generate x, so use input_shape
                    output_shape = tuple(config_input) if isinstance(config_input, (list, tuple)) else (config_input,)
            
            input_shape = ()  # Always empty for unconditional
        else:
            # Conditional generation: reverse config values (y->x generation)
            # BUT: if args.input_shape/args.output_shape are None, use config values directly (don't reverse)
            # This allows config files to specify shapes directly without being reversed
            if args.input_shape is not None or args.output_shape is not None:
                # Args provided: input_shape is x shape, output_shape is y shape
                # For generation: input is y, output is x
                input_shape = tuple(args.output_shape) if args.output_shape is not None else tuple(main_dict.get('output_shape', (2,)))
                output_shape = tuple(args.input_shape) if args.input_shape is not None else tuple(main_dict.get('input_shape', (2,)))
            else:
                # No args: use config values directly (don't reverse) - config file should have correct shapes
                # This is important for sequences where config already has x->y format
                input_shape = tuple(main_dict.get('input_shape', (2,))) if isinstance(main_dict.get('input_shape'), (list, tuple)) else (main_dict.get('input_shape', 2),)
                output_shape = tuple(main_dict.get('output_shape', (2,))) if isinstance(main_dict.get('output_shape'), (list, tuple)) else (main_dict.get('output_shape', 2),)
        
        # Latent shape is always from config or args, unchanged
        latent_shape = tuple(args.latent_shape) if args.latent_shape is not None else tuple(main_dict.get('latent_shape', (2,)))
        
        # Build updates for main config (filter None values)
        # Check if vae_weight is in args (for sequences)
        main_updates_dict = {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'latent_shape': latent_shape,
            'recon_weight': args.recon_weight,
            'reg_weight': args.reg_weight,
            'recon_loss_type': args.recon_loss_type,
            'normalize_snr_weight': args.normalize_snr_weight if hasattr(args, 'normalize_snr_weight') and args.normalize_snr_weight is not None else main_dict.get('normalize_snr_weight', None),
            'integration_method': getattr(args, 'integration_method', None) if getattr(args, 'integration_method', None) is not None else main_dict.get('integration_method', None),
        }
        # Add vae_weight if it exists in args (for sequences)
        if hasattr(args, 'vae_weight') and args.vae_weight is not None:
            main_updates_dict['vae_weight'] = args.vae_weight
        main_updates = BaseConfig.filter_none(main_updates_dict)
        
        # Build updates for CRN config
        crn_updates = BaseConfig.filter_none({
            'model_type': args.crn_type,
            'network_type': args.network_type,
            'hidden_dims': tuple(args.hidden_dims) if args.hidden_dims is not None else None,
        })
        
        # Build updates for encoder config (encoder encodes coordinates x)
        # Command-line arguments always override config values
        encoder_updates = BaseConfig.filter_none({
            'model_type': args.encoder_model_type,
            'input_shape': output_shape,  # Encoder encodes x (coordinates) - always override from main config
            'latent_shape': latent_shape,  # Must match main config - always override
        })
        
        # Build updates for decoder config
        # Command-line arguments always override config values
        decoder_updates = BaseConfig.filter_none({
            'model_type': args.decoder_model_type,
            'decoder_type': args.decoder_type,
            'output_shape': output_shape,  # Always override from main config
            'latent_shape': latent_shape,  # Must match main config - always override
        })
        
        # Build updates for noise schedule config
        noise_schedule_updates = BaseConfig.filter_none({
            'schedule_type': args.noise_schedule,
            'learnable': args.noise_schedule_learnable,
            'latent_shape': latent_shape,  # Propagate latent_shape to noise schedule
        })
        
        # Apply updates using merge_frozen_dict and replace
        updated_config = self
        if main_updates:
            updated_main = updated_config.merge_frozen_dict('main', main_updates)
            updated_config = replace(updated_config, main=updated_main)
        if crn_updates:
            updated_crn = updated_config.merge_frozen_dict('crn', crn_updates)
            updated_config = replace(updated_config, crn=updated_crn)
        if encoder_updates:
            updated_encoder = updated_config.merge_frozen_dict('encoder', encoder_updates)
            updated_config = replace(updated_config, encoder=updated_encoder)
        if decoder_updates:
            updated_decoder = updated_config.merge_frozen_dict('decoder', decoder_updates)
            updated_config = replace(updated_config, decoder=updated_decoder)
        if noise_schedule_updates:
            updated_noise_schedule = updated_config.merge_frozen_dict('noise_schedule', noise_schedule_updates)
            updated_config = replace(updated_config, noise_schedule=updated_noise_schedule)
        
        return updated_config
    
    def override_from_args_regression(self, args: "argparse.Namespace", model_type: str) -> "Config":
        """
        Override config values with command-line arguments for regression task (x -> y).
        
        For regression: inputs are x, outputs are y (no shape reversal needed).
        
        Rules:
        - "NA" values in main config must be provided via command-line arguments (error if not)
        - "NA" values in encoder/decoder config will use values from main config
        - No fallback defaults - strict validation
        
        Args:
            args: Parsed command-line arguments
            model_type: Model type
            
        Returns:
            Updated config instance with overrides applied, configured for regression
            
        Raises:
            ValueError: If main config has "NA" values but args are not provided
        """
        main_dict = dict(self.main)
        
        # Determine shapes for regression (x -> y):
        # - If args provided, use them directly
        # - If config has "NA", require args (throw error)
        # - Otherwise, use config file values (no reversal needed for regression)
        input_shape_val = main_dict.get('input_shape')
        if args.input_shape is not None:
            input_shape = tuple(args.input_shape)
        elif isinstance(input_shape_val, str) and input_shape_val == "NA":
            raise ValueError(
                "main.input_shape is 'NA' but --input_shape or --input_dim was not provided. "
                "You must specify input shape via command-line arguments."
            )
        else:
            input_shape = tuple(input_shape_val) if not isinstance(input_shape_val, tuple) else input_shape_val
        
        output_shape_val = main_dict.get('output_shape')
        if args.output_shape is not None:
            output_shape = tuple(args.output_shape)
        elif isinstance(output_shape_val, str) and output_shape_val == "NA":
            raise ValueError(
                "main.output_shape is 'NA' but --output_shape or --output_dim was not provided. "
                "You must specify output shape via command-line arguments."
            )
        else:
            output_shape = tuple(output_shape_val) if not isinstance(output_shape_val, tuple) else output_shape_val
        
        latent_shape_val = main_dict.get('latent_shape')
        if args.latent_shape is not None:
            latent_shape = tuple(args.latent_shape)
        elif isinstance(latent_shape_val, str) and latent_shape_val == "NA":
            raise ValueError(
                "main.latent_shape is 'NA' but --latent_shape or --latent_dim was not provided. "
                "You must specify latent shape via command-line arguments."
            )
        else:
            latent_shape = tuple(latent_shape_val) if not isinstance(latent_shape_val, tuple) else latent_shape_val
        
        # Build updates for main config (filter None values)
        # Check if vae_weight is in args (for sequences)
        main_updates_dict = {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'latent_shape': latent_shape,
            'recon_weight': args.recon_weight,
            'reg_weight': args.reg_weight,
            'recon_loss_type': args.recon_loss_type,
            'normalize_snr_weight': args.normalize_snr_weight if hasattr(args, 'normalize_snr_weight') and args.normalize_snr_weight is not None else main_dict.get('normalize_snr_weight', (model_type != 'flow_matching')),
            'integration_method': getattr(args, 'integration_method', None) if getattr(args, 'integration_method', None) is not None else main_dict.get('integration_method', ('midpoint' if model_type in ('ct', 'diffusion') else 'euler')),
        }
        # Add vae_weight if it exists in args (for sequences)
        if hasattr(args, 'vae_weight') and args.vae_weight is not None:
            main_updates_dict['vae_weight'] = args.vae_weight
        main_updates = BaseConfig.filter_none(main_updates_dict)
        
        # Build updates for CRN config
        crn_updates = BaseConfig.filter_none({
            'model_type': args.crn_type,
            'network_type': args.network_type,
            'hidden_dims': tuple(args.hidden_dims) if args.hidden_dims is not None else None,
        })
        
        # Build updates for encoder config (encoder encodes y for regression)
        # Preserve existing encoder shapes if they are not "NA" or None
        encoder_dict = dict(self.encoder)
        encoder_updates_dict = {
            'model_type': args.encoder_model_type,
        }
        # Only override shapes if they are "NA" or None in the config
        if encoder_dict.get('input_shape') in ("NA", None):
            encoder_updates_dict['input_shape'] = output_shape  # Encoder encodes y (output) for regression
        if encoder_dict.get('latent_shape') in ("NA", None):
            encoder_updates_dict['latent_shape'] = latent_shape
        encoder_updates = BaseConfig.filter_none(encoder_updates_dict)
        
        # Build updates for decoder config
        # Preserve existing decoder shapes if they are not "NA" or None
        decoder_dict = dict(self.decoder)
        decoder_updates_dict = {
            'model_type': args.decoder_model_type,
            'decoder_type': args.decoder_type,
        }
        # Only override shapes if they are "NA" or None in the config
        if decoder_dict.get('output_shape') in ("NA", None):
            decoder_updates_dict['output_shape'] = output_shape
        if decoder_dict.get('latent_shape') in ("NA", None):
            decoder_updates_dict['latent_shape'] = latent_shape
        decoder_updates = BaseConfig.filter_none(decoder_updates_dict)
        
        # Build updates for noise schedule config
        noise_schedule_updates = BaseConfig.filter_none({
            'schedule_type': args.noise_schedule,
            'learnable': args.noise_schedule_learnable,
            'latent_shape': latent_shape,  # Propagate latent_shape to noise schedule
        })
        
        # Apply updates using merge_frozen_dict and replace
        updated_config = self
        if main_updates:
            updated_main = updated_config.merge_frozen_dict('main', main_updates)
            updated_config = replace(updated_config, main=updated_main)
        if crn_updates:
            updated_crn = updated_config.merge_frozen_dict('crn', crn_updates)
            updated_config = replace(updated_config, crn=updated_crn)
        if encoder_updates:
            updated_encoder = updated_config.merge_frozen_dict('encoder', encoder_updates)
            updated_config = replace(updated_config, encoder=updated_encoder)
        if decoder_updates:
            updated_decoder = updated_config.merge_frozen_dict('decoder', decoder_updates)
            updated_config = replace(updated_config, decoder=updated_decoder)
        if noise_schedule_updates:
            updated_noise_schedule = updated_config.merge_frozen_dict('noise_schedule', noise_schedule_updates)
            updated_config = replace(updated_config, noise_schedule=updated_noise_schedule)
        
        return updated_config

