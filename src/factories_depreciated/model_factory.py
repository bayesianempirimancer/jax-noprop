"""
Unified model factory functions for creating models with consistent interfaces.

This module provides simple factory functions that take config dictionaries
and dimension parameters to create model instances. The pattern follows:

    model = create_model(config_dict, z_dim=8, x_dim=4, **kwargs)

All factory functions follow this consistent interface for easy use.
"""

from typing import Dict, Any, Optional, Tuple, Union
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict

from src.flow_models_wip.crn_wip import Config as CRNConfig, create_crn
from src.models.vae.encoders import create_encoder
from src.models.vae.decoders import create_decoder


# ============================================================================
# CORE FACTORY FUNCTIONS
# ============================================================================

def create_conditional_resnet(
    config_dict: Union[Dict[str, Any], FrozenDict], 
    z_dim: int, 
    x_dim: int, 
    **kwargs
) -> nn.Module:
    """
    Create a conditional ResNet model.
    
    Args:
        config_dict: Configuration dictionary for the model
        z_dim: Dimension of the state space z
        x_dim: Dimension of the conditional input x
        **kwargs: Additional parameters passed to the model
        
    Returns:
        Instantiated conditional ResNet model
    """
    return create_crn(config_dict, z_dim=z_dim, x_dim=x_dim, **kwargs)


def create_encoder_model(
    config_dict: Union[Dict[str, Any], FrozenDict],
    input_dim: int,
    latent_shape: Tuple[int, ...],
    **kwargs
) -> nn.Module:
    """
    Create an encoder model.
    
    Args:
        config_dict: Configuration dictionary for the encoder
        input_dim: Dimension of the input data
        latent_shape: Shape of the latent representation
        **kwargs: Additional parameters passed to the model
        
    Returns:
        Instantiated encoder model
    """
    # Remove input_dim from kwargs to avoid conflict
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'input_dim'}
    return create_encoder(config_dict, input_dim=input_dim, latent_shape=latent_shape, **kwargs_clean)


def create_decoder_model(
    config_dict: Union[Dict[str, Any], FrozenDict],
    input_dim: int,
    output_dim: int,
    input_shape: Optional[Tuple[int, ...]] = None,
    **kwargs
) -> nn.Module:
    """
    Create a decoder model.
    
    Args:
        config_dict: Configuration dictionary for the decoder
        input_dim: Dimension of the input data
        output_dim: Dimension of the output data
        input_shape: Shape of the input (for structured inputs)
        **kwargs: Additional parameters passed to the model
        
    Returns:
        Instantiated decoder model
    """
    return create_decoder(
        config_dict, 
        input_dim=input_dim, 
        output_dim=output_dim, 
        input_shape=input_shape, 
        **kwargs
    )


# ============================================================================
# UNIFIED MODEL FACTORY
# ============================================================================

def create_model(
    model_name: str,
    config_dict: Union[Dict[str, Any], FrozenDict],
    z_dim: int,
    x_dim: int,
    **kwargs
) -> nn.Module:
    """
    Unified factory function for creating any model type.
    
    Args:
        model_name: Name of the model to create
        config_dict: Configuration dictionary for the model
        z_dim: Dimension of the state space z
        x_dim: Dimension of the conditional input x
        **kwargs: Additional parameters (e.g., latent_shape, output_dim)
        
    Returns:
        Instantiated model
        
    Examples:
        # Create a conditional ResNet
        model = create_model("conditional_resnet", crn_config, z_dim=8, x_dim=4)
        
        # Create an encoder
        model = create_model("encoder", encoder_config, z_dim=8, x_dim=4, 
                           latent_shape=(8,), input_dim=4)
        
        # Create a decoder  
        model = create_model("decoder", decoder_config, z_dim=8, x_dim=4,
                           input_dim=8, output_dim=10)
    """
    if model_name == "conditional_resnet":
        return create_conditional_resnet(config_dict, z_dim=z_dim, x_dim=x_dim, **kwargs)
    
    elif model_name == "encoder":
        # For encoders, we need input_dim and latent_shape
        input_dim = kwargs.get('input_dim', x_dim)
        latent_shape = kwargs.get('latent_shape', (z_dim,))
        # Remove input_dim from kwargs to avoid conflict
        kwargs_clean = {k: v for k, v in kwargs.items() if k not in ['input_dim', 'latent_shape']}
        return create_encoder_model(config_dict, input_dim=input_dim, latent_shape=latent_shape, **kwargs_clean)
    
    elif model_name == "decoder":
        # For decoders, we need input_dim and output_dim
        input_dim = kwargs.get('input_dim', z_dim)
        output_dim = kwargs.get('output_dim', x_dim)
        input_shape = kwargs.get('input_shape', (z_dim,))
        # Remove conflicting parameters from kwargs
        kwargs_clean = {k: v for k, v in kwargs.items() if k not in ['input_dim', 'output_dim', 'input_shape']}
        return create_decoder_model(
            config_dict, 
            input_dim=input_dim, 
            output_dim=output_dim, 
            input_shape=input_shape, 
            **kwargs_clean
        )
    
    else:
        raise ValueError(f"Unknown model_name: {model_name}. "
                        f"Supported models: conditional_resnet, encoder, decoder")


# ============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON PATTERNS
# ============================================================================

def create_flow_model(
    flow_type: str,
    backbone_config: Union[Dict[str, Any], FrozenDict],
    z_dim: int,
    x_dim: int,
    **kwargs
) -> nn.Module:
    """
    Create flow models (potential, geometric, natural) with a backbone.
    
    Args:
        flow_type: Type of flow ("potential", "geometric", "natural")
        backbone_config: Configuration for the backbone model
        z_dim: Dimension of the state space z
        x_dim: Dimension of the conditional input x
        **kwargs: Additional parameters
        
    Returns:
        Instantiated flow model
    """
    # Create the backbone model
    backbone = create_conditional_resnet(backbone_config, z_dim=z_dim, x_dim=x_dim, **kwargs)
    
    # Import flow wrappers
    from src.flow_models.crn import (
        PotentialFlowWrapper, 
        GeometricFlowWrapper, 
        NaturalFlowWrapper
    )
    
    # Create appropriate wrapper
    if flow_type == "potential":
        return PotentialFlowWrapper(
            resnet_config=CRNConfig(),
            cond_resnet="conditional_resnet_mlp"
        )
    elif flow_type == "geometric":
        return GeometricFlowWrapper(
            resnet_config=CRNConfig(),
            cond_resnet="conditional_resnet_mlp"
        )
    elif flow_type == "natural":
        # Natural flow needs z_dim**2 output
        return NaturalFlowWrapper(
            resnet_config=CRNConfig(),
            cond_resnet="conditional_resnet_mlp"
        )
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}. "
                        f"Supported types: potential, geometric, natural")


def create_vae_flow_model(
    main_config: Union[Dict[str, Any], FrozenDict],
    crn_config: Union[Dict[str, Any], FrozenDict],
    encoder_config: Union[Dict[str, Any], FrozenDict],
    decoder_config: Union[Dict[str, Any], FrozenDict],
    z_dim: int,
    x_dim: int,
    **kwargs
) -> nn.Module:
    """
    Create a VAE with flow model using separate configs.
    
    Args:
        main_config: Main configuration for the VAE flow
        crn_config: Configuration for the CRN backbone
        encoder_config: Configuration for the encoder
        decoder_config: Configuration for the decoder
        z_dim: Dimension of the state space z
        x_dim: Dimension of the conditional input x
        **kwargs: Additional parameters
        
    Returns:
        Instantiated VAE flow model
    """
    from src.flow_models_wip.fm_wip import VAE_flow, VAEFlowConfig
    
    # Update configs with proper shapes
    main_config_updated = dict(main_config)
    main_config_updated.update({
        "input_shape": (x_dim,),
        "output_shape": (kwargs.get('output_dim', 2),),  # Default to 2 classes
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


# ============================================================================
# CONFIG VALIDATION HELPERS
# ============================================================================

def validate_model_config(
    config_dict: Union[Dict[str, Any], FrozenDict],
    required_fields: list,
    model_name: str = "model"
) -> None:
    """
    Validate that a config dictionary has all required fields.
    
    Args:
        config_dict: Configuration dictionary to validate
        required_fields: List of required field names
        model_name: Name of the model for error messages
        
    Raises:
        ValueError: If required fields are missing
    """
    missing_fields = []
    for field in required_fields:
        if field not in config_dict:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Missing required fields for {model_name}: {missing_fields}")


def get_default_config(model_name: str) -> Dict[str, Any]:
    """
    Get default configuration for a model type.
    
    Args:
        model_name: Name of the model type
        
    Returns:
        Default configuration dictionary
    """
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
