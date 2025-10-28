"""
Factory functions for creating flow models with convex ResNet backbones.

This module provides specialized factory functions for creating potential,
geometric, and natural flow models that use convex ResNets as their backbone.
"""

from typing import Dict, Any, Optional, Union
import flax.linen as nn
from flax.core import FrozenDict

from src.flow_models.crn import (
    PotentialFlowWrapper, 
    GeometricFlowWrapper, 
    NaturalFlowWrapper
)
from src.flow_models_wip.crn_wip import Config as CRNConfig


def create_convex_flow_model(
    flow_type: str,
    z_dim: int,
    x_dim: int,
    hidden_dims: tuple = (128, 128, 128),
    activation_fn: str = "softplus",
    **kwargs
) -> nn.Module:
    """
    Create a flow model with convex ResNet backbone.
    
    Args:
        flow_type: Type of flow ("potential", "geometric", "natural")
        z_dim: Latent space dimension
        x_dim: Conditional input dimension
        hidden_dims: Hidden layer dimensions for the ResNet
        activation_fn: Activation function (must be convex, e.g., "softplus")
        **kwargs: Additional parameters
        
    Returns:
        Instantiated flow model with convex ResNet
        
    Examples:
        # Create convex potential flow
        model = create_convex_flow_model("potential", z_dim=8, x_dim=4)
        
        # Create convex geometric flow with custom config
        model = create_convex_flow_model("geometric", z_dim=8, x_dim=4, 
                                       hidden_dims=(256, 128, 64))
    """
    # Create convex ResNet config
    convex_config = _create_convex_resnet_config(
        z_dim=z_dim,
        x_dim=x_dim,
        hidden_dims=hidden_dims,
        activation_fn=activation_fn,
        **kwargs
    )
    
    # Create appropriate flow wrapper
    if flow_type == "potential":
        return PotentialFlowWrapper(
            resnet_config=convex_config,
            cond_resnet="convex_conditional_resnet"
        )
    elif flow_type == "geometric":
        return GeometricFlowWrapper(
            resnet_config=convex_config,
            cond_resnet="convex_conditional_resnet"
        )
    elif flow_type == "natural":
        return NaturalFlowWrapper(
            resnet_config=convex_config,
            cond_resnet="convex_conditional_resnet"
        )
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}. "
                        f"Supported types: potential, geometric, natural")


def create_convex_potential_flow(
    z_dim: int,
    x_dim: int,
    hidden_dims: tuple = (128, 128, 128),
    **kwargs
) -> nn.Module:
    """
    Create a potential flow model with convex ResNet backbone.
    
    Args:
        z_dim: Latent space dimension
        x_dim: Conditional input dimension
        hidden_dims: Hidden layer dimensions for the ResNet
        **kwargs: Additional parameters
        
    Returns:
        Instantiated potential flow model with convex ResNet
    """
    return create_convex_flow_model(
        "potential", z_dim, x_dim, hidden_dims, **kwargs
    )


def create_convex_geometric_flow(
    z_dim: int,
    x_dim: int,
    hidden_dims: tuple = (128, 128, 128),
    **kwargs
) -> nn.Module:
    """
    Create a geometric flow model with convex ResNet backbone.
    
    Args:
        z_dim: Latent space dimension
        x_dim: Conditional input dimension
        hidden_dims: Hidden layer dimensions for the ResNet
        **kwargs: Additional parameters
        
    Returns:
        Instantiated geometric flow model with convex ResNet
    """
    return create_convex_flow_model(
        "geometric", z_dim, x_dim, hidden_dims, **kwargs
    )


def create_convex_natural_flow(
    z_dim: int,
    x_dim: int,
    hidden_dims: tuple = (128, 128, 128),
    **kwargs
) -> nn.Module:
    """
    Create a natural flow model with convex ResNet backbone.
    
    Args:
        z_dim: Latent space dimension
        x_dim: Conditional input dimension
        hidden_dims: Hidden layer dimensions for the ResNet
        **kwargs: Additional parameters
        
    Returns:
        Instantiated natural flow model with convex ResNet
    """
    return create_convex_flow_model(
        "natural", z_dim, x_dim, hidden_dims, **kwargs
    )


def _create_convex_resnet_config(
    z_dim: int,
    x_dim: int,
    hidden_dims: tuple = (128, 128, 128),
    activation_fn: str = "softplus",
    **kwargs
) -> CRNConfig:
    """
    Create a CRN config for convex ResNet.
    
    Args:
        z_dim: Latent space dimension
        x_dim: Conditional input dimension
        hidden_dims: Hidden layer dimensions
        activation_fn: Activation function (must be convex)
        **kwargs: Additional parameters
        
    Returns:
        CRNConfig configured for convex ResNet
    """
    # Convex ResNet only accepts these specific parameters
    convex_params = {
        "output_dim": z_dim,  # Convex ResNet uses output_dim, not output_shape
        "hidden_dims": hidden_dims,
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "activation_fn": activation_fn,
        "use_batch_norm": False,
        "dropout_rate": 0.1,
        "use_bias": True,
        "use_projection": True,
        "block_type": "simple"
    }
    
    # Update with any additional parameters
    convex_params.update(kwargs)
    
    # Create and return config
    config = CRNConfig()
    config.__dict__['config_dict'] = convex_params
    return config


def create_all_convex_flows(
    z_dim: int,
    x_dim: int,
    hidden_dims: tuple = (128, 128, 128),
    **kwargs
) -> Dict[str, nn.Module]:
    """
    Create all types of convex flow models for comparison.
    
    Args:
        z_dim: Latent space dimension
        x_dim: Conditional input dimension
        hidden_dims: Hidden layer dimensions
        **kwargs: Additional parameters
        
    Returns:
        Dictionary mapping flow types to models
    """
    models = {}
    
    for flow_type in ["potential", "geometric", "natural"]:
        try:
            models[flow_type] = create_convex_flow_model(
                flow_type, z_dim, x_dim, hidden_dims, **kwargs
            )
            print(f"✓ Created convex {flow_type} flow")
        except Exception as e:
            print(f"✗ Failed to create convex {flow_type} flow: {e}")
    
    return models
