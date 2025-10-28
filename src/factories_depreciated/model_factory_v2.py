"""
Clean model factory functions with simplified interface.

This version eliminates the clunky z_dim/x_dim requirement and uses
a clean interface: create_model(model_name, config_dict, **kwargs)
"""

from typing import Dict, Any, Optional, Tuple, Union
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict

from src.flow_models_wip.crn_wip import Config as CRNConfig, create_crn


# ============================================================================
# CLEAN FACTORY FUNCTIONS
# ============================================================================

def create_model(model_name: str, config_dict: Union[Dict[str, Any], FrozenDict], **kwargs) -> nn.Module:
    """
    Create any model with a clean, simple interface.
    
    Args:
        model_name: Name of the model to create
        config_dict: Configuration dictionary for the model
        **kwargs: Additional parameters specific to the model type
        
    Returns:
        Instantiated model
        
    Examples:
        # Create a conditional ResNet
        model = create_model("conditional_resnet", crn_config, z_dim=8, x_dim=4)
        
        # Create an encoder
        model = create_model("encoder", encoder_config, input_dim=4, latent_shape=(8,))
        
        # Create a decoder  
        model = create_model("decoder", decoder_config, input_dim=8, output_dim=10)
        
        # Create a VAE flow
        model = create_model("vae_flow", main_config, z_dim=8, x_dim=4, output_dim=2)
    """
    if model_name == "conditional_resnet":
        return _create_conditional_resnet(config_dict, **kwargs)
    
    elif model_name == "encoder":
        return _create_encoder_model(config_dict, **kwargs)
    
    elif model_name == "decoder":
        return _create_decoder_model(config_dict, **kwargs)
    
    elif model_name == "vae_flow":
        return _create_vae_flow_model(config_dict, **kwargs)
    
    elif model_name in ["potential_flow", "geometric_flow", "natural_flow"]:
        return _create_flow_model(model_name, config_dict, **kwargs)
    
    else:
        raise ValueError(f"Unknown model_name: {model_name}. "
                        f"Supported models: conditional_resnet, encoder, decoder, "
                        f"vae_flow, potential_flow, geometric_flow, natural_flow")


def _create_conditional_resnet(config_dict: Union[Dict[str, Any], FrozenDict], **kwargs) -> nn.Module:
    """Create a conditional ResNet model using the homogenized approach."""
    from src.flow_models_wip.crn_wip import create_cond_resnet
    
    z_dim = kwargs.get('z_dim')
    x_dim = kwargs.get('x_dim')
    
    if z_dim is None or x_dim is None:
        raise ValueError("conditional_resnet requires z_dim and x_dim parameters")
    
    # Convert config_dict to regular dict if needed
    if hasattr(config_dict, 'unfreeze'):
        final_config = config_dict.unfreeze()
    else:
        final_config = dict(config_dict)
    
    # Update config with proper shapes
    final_config.update({
        "input_shape": (z_dim,),
        "output_shape": (z_dim,),
        "x_shape": (x_dim,),
    })
    
    # Filter out model_type and network_type from config as they're not needed by the ResNet classes
    resnet_config = {k: v for k, v in final_config.items() if k not in ["model_type", "network_type"]}
    
    # Get model_type and network_type
    model_type = final_config.get("model_type", "vanilla")
    network_type = final_config.get("network_type", "mlp")
    
    # Create the model using the existing factory
    return create_cond_resnet(model_type, network_type, resnet_config)


def _create_encoder_model(config_dict: Union[Dict[str, Any], FrozenDict], **kwargs) -> nn.Module:
    """Create an encoder model using the homogenized approach."""
    from src.models.vae.encoders import create_encoder_model
    
    input_shape = kwargs.get('input_shape')
    latent_shape = kwargs.get('latent_shape')
    
    if input_shape is None or latent_shape is None:
        raise ValueError("encoder requires input_shape and latent_shape parameters")
    
    return create_encoder_model(config_dict, input_shape=input_shape, latent_shape=latent_shape)


def _create_decoder_model(config_dict: Union[Dict[str, Any], FrozenDict], **kwargs) -> nn.Module:
    """Create a decoder model using the homogenized approach."""
    from src.models.vae.decoders import create_decoder_model
    
    latent_shape = kwargs.get('latent_shape')
    output_shape = kwargs.get('output_shape')
    
    if latent_shape is None or output_shape is None:
        raise ValueError("decoder requires latent_shape and output_shape parameters")
    
    return create_decoder_model(config_dict, latent_shape=latent_shape, output_shape=output_shape)


def _create_vae_flow_model(config_dict: Union[Dict[str, Any], FrozenDict], **kwargs) -> nn.Module:
    """Create a VAE flow model."""
    from src.flow_models_wip.fm_wip import VAE_flow, VAEFlowConfig
    
    z_dim = kwargs.get('z_dim')
    x_dim = kwargs.get('x_dim')
    output_dim = kwargs.get('output_dim', 2)
    
    if z_dim is None or x_dim is None:
        raise ValueError("vae_flow requires z_dim and x_dim parameters")
    
    # Get submodel configs from kwargs or use defaults
    crn_config = kwargs.get('crn_config', CRNConfig().config)
    encoder_config = kwargs.get('encoder_config', get_default_config("encoder"))
    decoder_config = kwargs.get('decoder_config', get_default_config("decoder"))
    
    # Update main config with proper shapes
    main_config_updated = dict(config_dict)
    main_config_updated.update({
        "input_shape": (x_dim,),
        "output_shape": (output_dim,),
        "latent_shape": (z_dim,),
    })
    
    # Update CRN config with proper shapes
    crn_config_updated = dict(crn_config)
    crn_config_updated.update({
        "input_shape": (z_dim,),
        "output_shape": (z_dim,),
        "x_shape": (x_dim,),
    })
    
    # Create the VAE flow config
    vae_config = VAEFlowConfig(
        model_name=main_config_updated.get("model_name", "vae_flow_network"),
        config=FrozenDict(main_config_updated),
        crn_config=FrozenDict(crn_config_updated),
        encoder_config=FrozenDict(encoder_config),
        decoder_config=FrozenDict(decoder_config)
    )
    
    return VAE_flow(config=vae_config)


def _create_flow_model(flow_type: str, config_dict: Union[Dict[str, Any], FrozenDict], **kwargs) -> nn.Module:
    """Create flow models (potential, geometric, natural) with a backbone."""
    z_dim = kwargs.get('z_dim')
    x_dim = kwargs.get('x_dim')
    
    if z_dim is None or x_dim is None:
        raise ValueError(f"{flow_type} requires z_dim and x_dim parameters")
    
    # Create the backbone model
    backbone = create_crn(config_dict, z_dim=z_dim, x_dim=x_dim)
    
    # Import flow wrappers
    from src.flow_models.crn import (
        PotentialFlowWrapper, 
        GeometricFlowWrapper, 
        NaturalFlowWrapper
    )
    
    # Create appropriate wrapper
    if flow_type == "potential_flow":
        return PotentialFlowWrapper(
            resnet_config=CRNConfig(),
            cond_resnet="conditional_resnet_mlp"
        )
    elif flow_type == "geometric_flow":
        return GeometricFlowWrapper(
            resnet_config=CRNConfig(),
            cond_resnet="conditional_resnet_mlp"
        )
    elif flow_type == "natural_flow":
        return NaturalFlowWrapper(
            resnet_config=CRNConfig(),
            cond_resnet="conditional_resnet_mlp"
        )
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_vae_flow_with_defaults(
    z_dim: int,
    x_dim: int, 
    output_dim: int = 2,
    **kwargs
) -> nn.Module:
    """
    Create a VAE flow model with sensible defaults.
    
    Args:
        z_dim: Latent space dimension
        x_dim: Conditional input dimension
        output_dim: Output classes (default: 2)
        **kwargs: Additional parameters
        
    Returns:
        Instantiated VAE flow model
    """
    main_config = {
        "model_name": "vae_flow_network",
        "loss_type": kwargs.get("loss_type", "cross_entropy"),
        "flow_loss_weight": kwargs.get("flow_loss_weight", 0.01),
        "reg_weight": kwargs.get("reg_weight", 0.0),
    }
    
    return create_model(
        "vae_flow",
        main_config,
        z_dim=z_dim,
        x_dim=x_dim,
        output_dim=output_dim,
        crn_config=CRNConfig().config,
        encoder_config=get_default_config("encoder"),
        decoder_config=get_default_config("decoder"),
        **kwargs
    )


def create_conditional_resnet_with_defaults(
    z_dim: int,
    x_dim: int,
    **kwargs
) -> nn.Module:
    """
    Create a conditional ResNet with sensible defaults.
    
    Args:
        z_dim: Latent space dimension
        x_dim: Conditional input dimension
        **kwargs: Additional parameters
        
    Returns:
        Instantiated conditional ResNet model
    """
    config = CRNConfig().config
    return create_model("conditional_resnet", config, z_dim=z_dim, x_dim=x_dim, **kwargs)


def create_encoder_with_defaults(
    input_dim: int,
    latent_shape: Tuple[int, ...],
    **kwargs
) -> nn.Module:
    """
    Create an encoder with sensible defaults.
    
    Args:
        input_dim: Data input dimension
        latent_shape: Latent output shape
        **kwargs: Additional parameters
        
    Returns:
        Instantiated encoder model
    """
    config = get_default_config("encoder")
    return create_model("encoder", config, input_dim=input_dim, latent_shape=latent_shape, **kwargs)


def create_decoder_with_defaults(
    input_dim: int,
    output_dim: int,
    **kwargs
) -> nn.Module:
    """
    Create a decoder with sensible defaults.
    
    Args:
        input_dim: Latent input dimension
        output_dim: Data output dimension
        **kwargs: Additional parameters
        
    Returns:
        Instantiated decoder model
    """
    config = get_default_config("decoder")
    return create_model("decoder", config, input_dim=input_dim, output_dim=output_dim, **kwargs)


# ============================================================================
# CONFIG HELPERS
# ============================================================================

def get_default_config(model_name: str) -> Dict[str, Any]:
    """Get default configuration for a model type."""
    defaults = {
        "conditional_resnet": {
            "model_type": "vanilla",
            "network_type": "mlp",
            "hidden_dims": (128, 128, 128),
            "time_embed_dim": 64,
            "time_embed_method": "sinusoidal",
            "dropout_rate": 0.1,
            "activation_fn": "swish",
            "use_batch_norm": False,
        },
        "encoder": {
            "model_type": "mlp_normal",
            "encoder_type": "normal",
            "hidden_dims": (64, 32, 16),
            "dropout_rate": 0.1,
            "activation": "swish",
        },
        "decoder": {
            "model_type": "mlp",
            "decoder_type": "logits",
            "hidden_dims": (64, 32, 16),
            "dropout_rate": 0.1,
            "activation": "swish",
        }
    }
    
    if model_name not in defaults:
        raise ValueError(f"No default config for model_name: {model_name}")
    
    return defaults[model_name].copy()
