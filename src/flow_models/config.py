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
        "use_snr_weight": False,  # Apply signal-to-noise ratio weighting to loss (False for flow_matching, True for diffusion/ct)
        "integration_method": "euler",  # Options: "euler", "heun", "rk4", "adaptive", "midpoint"
                                        # "euler" for flow_matching, "midpoint" for diffusion/ct
        "encode_x": False,  # Whether to encode x before passing to CRN (True for sequences, False for backward compatibility)
    }))
    
    noise_schedule: FrozenDict = field(default_factory=lambda: FrozenDict({
        "schedule_type": "linear",  # Type of schedule (linear, exponential, cosine, sigmoid, cauchy, laplace, logistic, quadratic, polynomial, monotonic_nn, learnable, network)
        "learnable": True,  # Whether schedule parameters are learnable (False uses stop_gradient)
        "hidden_dims": (64, 64),  # Hidden dimensions for NoiseScheduleNetwork schedule
        # Comprehensive default parameters for all schedules (common naming convention)
        "default_params": FrozenDict({
            # Linear, Exponential, Cauchy, Laplace schedules
            "alpha_bar_min": 0.01,
            "alpha_bar_max": 0.99,
            # Cosine schedule
            "s": 0.008,
            # Sigmoid, Logistic schedules
            "k": 10.0,
            "t_mid": 0.5,
            # Exponential schedule
            "beta": 2.0,
            # Cauchy, Laplace schedules
            "loc": 0.5,
            "scale": 0.1,
            # Polynomial schedule
            "power": 2.0,
            # NoiseScheduleNetwork schedule
            "gamma_range": (-4.0, 4.0),
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
    }))
    
    encoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "identity",  # Options: "mlp", "mlp_normal", "resnet", "resnet_normal", "identity", "linear"
        "encoder_type": "deterministic",  # Options: "deterministic", "normal"
        "input_shape": "NA",  # Will be set from main config if not specified
        "latent_shape": "NA",
        "hidden_dims": (16, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.0,
    }))
    
    decoder: FrozenDict = field(default_factory=lambda: FrozenDict({
        "model_type": "identity",  # Options: "mlp", "resnet", "identity", "linear"
        "decoder_type": "none",  # Options: "linear", "softmax", "none"
        "latent_shape": "NA",  # Will be set from main config if not specified
        "output_shape": "NA",
        "hidden_dims": (16, 32, 16),
        "activation": "swish",
        "dropout_rate": 0.0,
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
            if args.input_shape is not None or args.output_shape is not None:
                # Args provided: input_shape is x shape, output_shape is y shape
                # For generation: input is y, output is x
                input_shape = tuple(args.output_shape) if args.output_shape is not None else tuple(main_dict.get('output_shape', (2,)))
                output_shape = tuple(args.input_shape) if args.input_shape is not None else tuple(main_dict.get('input_shape', (2,)))
            else:
                # No args: reverse config file values (config assumed to be x->y, we need y->x)
                input_shape = tuple(main_dict.get('output_shape', (2,)))
                output_shape = tuple(main_dict.get('input_shape', (2,)))
        
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
            'use_snr_weight': args.use_snr_weight if args.use_snr_weight is not None else main_dict.get('use_snr_weight', None),
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
        encoder_updates = BaseConfig.filter_none({
            'model_type': args.encoder_model_type,
            'input_shape': output_shape,  # Encoder encodes x (coordinates)
            'latent_shape': latent_shape,  # Must match main config
        })
        
        # Build updates for decoder config
        decoder_updates = BaseConfig.filter_none({
            'model_type': args.decoder_model_type,
            'decoder_type': args.decoder_type,
            'output_shape': output_shape,
            'latent_shape': latent_shape,  # Must match main config
        })
        
        # Build updates for noise schedule config
        noise_schedule_updates = BaseConfig.filter_none({
            'schedule_type': args.noise_schedule,
            'learnable': args.noise_schedule_learnable,
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
        
        Args:
            args: Parsed command-line arguments
            model_type: Model type
            
        Returns:
            Updated config instance with overrides applied, configured for regression
        """
        main_dict = dict(self.main)
        
        # Determine shapes for regression (x -> y):
        # - If args provided, use them directly
        # - Otherwise, use config file values (no reversal needed for regression)
        if args.input_shape is not None or args.output_shape is not None:
            # Args provided: use them directly
            input_shape = tuple(args.input_shape) if args.input_shape is not None else tuple(main_dict.get('input_shape', (2,)))
            output_shape = tuple(args.output_shape) if args.output_shape is not None else tuple(main_dict.get('output_shape', (2,)))
        else:
            # No args: use config file values as-is (no reversal for regression)
            input_shape = tuple(main_dict.get('input_shape', (2,)))
            output_shape = tuple(main_dict.get('output_shape', (2,)))
        
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
            'use_snr_weight': args.use_snr_weight if args.use_snr_weight is not None else main_dict.get('use_snr_weight', (model_type != 'flow_matching')),
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
        
        # Build updates for encoder config (encoder encodes x)
        encoder_updates = BaseConfig.filter_none({
            'model_type': args.encoder_model_type,
            'input_shape': input_shape,  # Encoder encodes x (inputs)
            'latent_shape': latent_shape,  # Must match main config
        })
        
        # Build updates for decoder config
        decoder_updates = BaseConfig.filter_none({
            'model_type': args.decoder_model_type,
            'decoder_type': args.decoder_type,
            'output_shape': output_shape,
            'latent_shape': latent_shape,  # Must match main config
        })
        
        # Build updates for noise schedule config
        noise_schedule_updates = BaseConfig.filter_none({
            'schedule_type': args.noise_schedule,
            'learnable': args.noise_schedule_learnable,
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

