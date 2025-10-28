"""
Factory functions specifically for trainer model creation.

This module provides simple factory functions that replace the complex
model creation logic in trainer.py with a clean, consistent interface.
"""

from typing import Dict, Any, Tuple, Union
import flax.linen as nn
from flax.core import FrozenDict

from src.factories.model_factory import create_model, create_flow_model, get_default_config
from src.flow_models_wip.crn_wip import Config as CRNConfig
from src.flow_models.fm import NoPropFM, Config as FMConfig
from src.flow_models.ct import NoPropCT, Config as CTConfig
from src.flow_models.df import NoPropDF, Config as DFConfig


def create_noprop_model(
    training_protocol: str,
    model_name: str,
    z_shape: Tuple[int, ...],
    x_ndims: int = 1,
    **kwargs
) -> nn.Module:
    """
    Create a NoProp model with unified interface.
    
    This replaces the complex create_model function in trainer.py with
    a simple, consistent interface.
    
    Args:
        training_protocol: Training protocol ("fm", "ct", "df")
        model_name: Model architecture name
        z_shape: Shape of the target z (excluding batch dimensions)
        x_ndims: Number of dimensions in input x
        **kwargs: Additional parameters
        
    Returns:
        Instantiated NoProp model
        
    Examples:
        # Create FM model with conditional ResNet
        model = create_noprop_model("fm", "conditional_resnet", z_shape=(8,))
        
        # Create CT model with potential flow
        model = create_noprop_model("ct", "potential_flow", z_shape=(8,))
        
        # Create DF model with VAE flow
        model = create_noprop_model("df", "vae_flow", z_shape=(8,))
    """
    # Extract dimensions
    z_dim = z_shape[0] if len(z_shape) == 1 else z_shape
    x_dim = kwargs.get('x_dim', z_dim)  # Default x_dim to z_dim if not provided
    
    # Create the appropriate config based on training protocol
    if training_protocol == "fm":
        config = FMConfig()
    elif training_protocol == "ct":
        config = CTConfig()
    elif training_protocol == "df":
        config = DFConfig()
    else:
        raise ValueError(f"Unsupported training protocol: {training_protocol}")
    
    # For wrapper models, ensure the config has the right model type for the CRN backbone
    if model_name in ["potential_flow", "geometric_flow", "natural_flow"]:
        config.config_dict['model'] = "conditional_resnet_mlp"
        print(f"Using CRN MLP backbone for {model_name} wrapper")
    else:
        print(f"Using direct {model_name} architecture")
    
    # Create the appropriate model based on training protocol
    if training_protocol == "fm":
        return NoPropFM(config=config, z_shape=z_shape, x_ndims=x_ndims, model=model_name)
    elif training_protocol == "ct":
        return NoPropCT(config=config, z_shape=z_shape, x_ndims=x_ndims, model=model_name)
    elif training_protocol == "df":
        return NoPropDF(config=config, z_shape=z_shape, x_ndims=x_ndims, model=model_name)
    else:
        raise ValueError(f"Unsupported training protocol: {training_protocol}")


def create_simple_model(
    model_name: str,
    z_dim: int,
    x_dim: int,
    **kwargs
) -> nn.Module:
    """
    Create a simple model using the unified factory.
    
    This is a convenience function for creating individual model components
    without the NoProp wrapper.
    
    Args:
        model_name: Name of the model to create
        z_dim: Dimension of the state space z
        x_dim: Dimension of the conditional input x
        **kwargs: Additional parameters
        
    Returns:
        Instantiated model
    """
    if model_name == "conditional_resnet":
        config = CRNConfig()
        return create_model(
            model_name="conditional_resnet",
            config_dict=config.config,
            z_dim=z_dim,
            x_dim=x_dim,
            **kwargs
        )
    
    elif model_name in ["potential_flow", "geometric_flow", "natural_flow"]:
        config = CRNConfig()
        return create_flow_model(
            flow_type=model_name.replace("_flow", ""),
            backbone_config=config.config,
            z_dim=z_dim,
            x_dim=x_dim,
            **kwargs
        )
    
    elif model_name == "vae_flow":
        # Use default configs for VAE flow
        main_config = {
            "model_name": "vae_flow_network",
            "loss_type": kwargs.get("loss_type", "cross_entropy"),
            "flow_loss_weight": kwargs.get("flow_loss_weight", 0.01),
            "reg_weight": kwargs.get("reg_weight", 0.0),
        }
        
        crn_config = CRNConfig().config
        encoder_config = get_default_config("encoder")
        decoder_config = get_default_config("decoder")
        
        return create_vae_flow_model(
            main_config=main_config,
            crn_config=crn_config,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            z_dim=z_dim,
            x_dim=x_dim,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model_name: {model_name}. "
                        f"Supported models: conditional_resnet, potential_flow, "
                        f"geometric_flow, natural_flow, vae_flow")


def create_model_from_config(
    config: Union[Dict[str, Any], FrozenDict],
    z_dim: int,
    x_dim: int,
    **kwargs
) -> nn.Module:
    """
    Create a model from a configuration dictionary.
    
    This function automatically determines the model type from the config
    and creates the appropriate model.
    
    Args:
        config: Configuration dictionary
        z_dim: Dimension of the state space z
        x_dim: Dimension of the conditional input x
        **kwargs: Additional parameters
        
    Returns:
        Instantiated model
    """
    # Extract model name from config
    model_name = config.get("model_name", "conditional_resnet")
    
    # Create model using the simple factory
    return create_simple_model(
        model_name=model_name,
        z_dim=z_dim,
        x_dim=x_dim,
        **kwargs
    )


# ============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON TRAINER PATTERNS
# ============================================================================

def create_all_model_types(z_dim: int, x_dim: int) -> Dict[str, nn.Module]:
    """
    Create all available model types for comparison.
    
    Args:
        z_dim: Dimension of the state space z
        x_dim: Dimension of the conditional input x
        
    Returns:
        Dictionary mapping model names to instantiated models
    """
    models = {}
    
    # Basic models
    models["conditional_resnet"] = create_simple_model("conditional_resnet", z_dim, x_dim)
    
    # Flow models
    for flow_type in ["potential_flow", "geometric_flow", "natural_flow"]:
        models[flow_type] = create_simple_model(flow_type, z_dim, x_dim)
    
    # VAE flow
    models["vae_flow"] = create_simple_model("vae_flow", z_dim, x_dim)
    
    return models


def create_training_models(
    model_names: list,
    z_dim: int,
    x_dim: int,
    **kwargs
) -> Dict[str, nn.Module]:
    """
    Create multiple models for training comparison.
    
    Args:
        model_names: List of model names to create
        z_dim: Dimension of the state space z
        x_dim: Dimension of the conditional input x
        **kwargs: Additional parameters
        
    Returns:
        Dictionary mapping model names to instantiated models
    """
    models = {}
    for model_name in model_names:
        try:
            models[model_name] = create_simple_model(model_name, z_dim, x_dim, **kwargs)
            print(f"✓ Created {model_name}")
        except Exception as e:
            print(f"✗ Failed to create {model_name}: {e}")
    
    return models
