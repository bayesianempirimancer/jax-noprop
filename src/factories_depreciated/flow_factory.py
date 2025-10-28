"""
Comprehensive factory function for creating flow models with any combination of
flow wrapper type and CRN backbone type.

This factory allows you to specify:
- Flow wrapper: potential, natural, geometric, hamiltonian, convex_potential, vanilla
- CRN backbone: mlp, convex, bilinear, resnet
"""

from typing import Dict, Any, Optional, Union, Tuple
import flax.linen as nn
from flax.core import FrozenDict

from src.flow_models.crn import (
    PotentialFlowWrapper,
    NaturalFlowWrapper, 
    GeometricFlowWrapper,
    HamiltonianFlowWrapper,
    create_cond_resnet
)
from src.flow_models_wip.crn_wip import Config as CRNConfig


def create_flow_model(
    flow_type: str,
    crn_type: str,
    z_dim: int,
    x_dim: int,
    hidden_dims: Tuple[int, ...] = (128, 128, 128),
    **kwargs
) -> nn.Module:
    """
    Create a flow model with specified flow wrapper and CRN backbone.
    
    Args:
        flow_type: Type of flow wrapper ("potential", "natural", "geometric", 
                  "hamiltonian", "convex_potential", "vanilla")
        crn_type: Type of CRN backbone ("mlp", "convex", "bilinear", "resnet")
        z_dim: Latent space dimension
        x_dim: Conditional input dimension
        hidden_dims: Hidden layer dimensions for the CRN
        **kwargs: Additional parameters for the CRN
        
    Returns:
        Instantiated flow model with specified wrapper and backbone
        
    Examples:
        # Potential flow with convex ResNet
        model = create_flow_model("potential", "convex", z_dim=8, x_dim=4)
        
        # Geometric flow with MLP backbone
        model = create_flow_model("geometric", "mlp", z_dim=8, x_dim=8)
        
        # Natural flow with bilinear ResNet
        model = create_flow_model("natural", "bilinear", z_dim=8, x_dim=8)
    """
    
    # Map flow types to wrapper classes
    flow_wrappers = {
        "potential": PotentialFlowWrapper,
        "natural": NaturalFlowWrapper,
        "geometric": GeometricFlowWrapper,
        "hamiltonian": HamiltonianFlowWrapper,
        "vanilla": PotentialFlowWrapper  # Default to potential for vanilla
    }
    
    # Map CRN types to their identifiers
    crn_types = {
        "mlp": "conditional_resnet_mlp",
        "convex": "convex_conditional_resnet", 
        "bilinear": "bilinear_conditional_resnet",
        "resnet": "conditional_resnet_mlp"  # Default ResNet is MLP
    }
    
    # Validate inputs
    if flow_type not in flow_wrappers:
        raise ValueError(f"Unknown flow_type: {flow_type}. "
                        f"Supported types: {list(flow_wrappers.keys())}")
    
    if crn_type not in crn_types:
        raise ValueError(f"Unknown crn_type: {crn_type}. "
                        f"Supported types: {list(crn_types.keys())}")
    
    # Check dimension requirements
    if flow_type in ["natural", "geometric"] and x_dim != z_dim:
        raise ValueError(f"{flow_type} flow requires x_dim ({x_dim}) to equal z_dim ({z_dim})")
    
    # Create CRN config
    crn_config = _create_crn_config(
        crn_type=crn_type,
        z_dim=z_dim,
        x_dim=x_dim,
        hidden_dims=hidden_dims,
        **kwargs
    )
    
    # Get the wrapper class
    wrapper_class = flow_wrappers[flow_type]
    
    # Create the flow model
    return wrapper_class(
        resnet_config=crn_config,
        cond_resnet=crn_types[crn_type]
    )


def create_potential_flow(
    crn_type: str = "mlp",
    z_dim: int = 8,
    x_dim: int = 4,
    **kwargs
) -> nn.Module:
    """Create a potential flow with specified CRN backbone."""
    return create_flow_model("potential", crn_type, z_dim, x_dim, **kwargs)


def create_natural_flow(
    crn_type: str = "mlp", 
    z_dim: int = 8,
    x_dim: int = None,
    **kwargs
) -> nn.Module:
    """Create a natural flow with specified CRN backbone."""
    if x_dim is None:
        x_dim = z_dim  # Natural flows require x_dim = z_dim
    return create_flow_model("natural", crn_type, z_dim, x_dim, **kwargs)


def create_geometric_flow(
    crn_type: str = "mlp",
    z_dim: int = 8, 
    x_dim: int = None,
    **kwargs
) -> nn.Module:
    """Create a geometric flow with specified CRN backbone."""
    if x_dim is None:
        x_dim = z_dim  # Geometric flows require x_dim = z_dim
    return create_flow_model("geometric", crn_type, z_dim, x_dim, **kwargs)


def create_hamiltonian_flow(
    crn_type: str = "mlp",
    z_dim: int = 8,
    x_dim: int = 4,
    **kwargs
) -> nn.Module:
    """Create a hamiltonian flow with specified CRN backbone."""
    return create_flow_model("hamiltonian", crn_type, z_dim, x_dim, **kwargs)


def create_convex_potential_flow(
    crn_type: str = "convex",
    z_dim: int = 8,
    x_dim: int = 4,
    **kwargs
) -> nn.Module:
    """Create a convex potential flow with specified CRN backbone.
    
    Note: This is equivalent to create_flow_model("potential", "convex", ...).
    This function is kept for backward compatibility but is deprecated.
    Use create_flow_model("potential", "convex", ...) instead.
    """
    return create_flow_model("potential", "convex", z_dim, x_dim, **kwargs)


def create_all_flow_combinations(
    z_dim: int = 8,
    x_dim: int = 4,
    **kwargs
) -> Dict[str, nn.Module]:
    """
    Create all possible combinations of flow types and CRN types.
    
    Args:
        z_dim: Latent space dimension
        x_dim: Conditional input dimension (will be set to z_dim for natural/geometric)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary mapping "flow_type_crn_type" to models
    """
    flow_types = ["potential", "natural", "geometric", "hamiltonian"]
    crn_types = ["mlp", "convex", "bilinear"]
    
    models = {}
    
    for flow_type in flow_types:
        for crn_type in crn_types:
            try:
                # Handle dimension requirements
                test_x_dim = x_dim
                if flow_type in ["natural", "geometric"]:
                    test_x_dim = z_dim
                
                model = create_flow_model(flow_type, crn_type, z_dim, test_x_dim, **kwargs)
                key = f"{flow_type}_{crn_type}"
                models[key] = model
                print(f"✓ Created {key} flow")
                
            except Exception as e:
                print(f"✗ Failed to create {flow_type}_{crn_type}: {e}")
    
    return models


def _create_crn_config(
    crn_type: str,
    z_dim: int,
    x_dim: int,
    hidden_dims: Tuple[int, ...] = (128, 128, 128),
    **kwargs
) -> CRNConfig:
    """
    Create a CRN config for the specified CRN type.
    
    Args:
        crn_type: Type of CRN ("mlp", "convex", "bilinear", "resnet")
        z_dim: Latent space dimension
        x_dim: Conditional input dimension
        hidden_dims: Hidden layer dimensions
        **kwargs: Additional parameters
        
    Returns:
        CRNConfig configured for the specified CRN type
    """
    
    # Base parameters for all CRN types
    base_params = {
        "hidden_dims": hidden_dims,
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "dropout_rate": 0.1,
        "use_batch_norm": False,
    }
    
    # CRN-specific parameters - only include parameters that each class actually accepts
    if crn_type == "mlp":
        # ConditionalResnet_MLP parameters
        params = {
            **base_params,
            "output_dim": z_dim,
            "activation_fn": "swish",
        }
    elif crn_type == "convex":
        # ConvexConditionalResnet parameters
        params = {
            **base_params,
            "output_dim": z_dim,
            "activation_fn": "softplus",  # Convex activations only
            "use_bias": True,
            "use_projection": True,
            "block_type": "simple",
        }
    elif crn_type == "bilinear":
        # BilinearConditionalResnet parameters
        params = {
            **base_params,
            "output_dim": z_dim,
            "activation_fn": "relu",  # Default for bilinear
            "use_bias": True,
            "use_projection": True,
        }
    elif crn_type == "resnet":
        # For now, treat resnet same as mlp since we don't have a separate ResNet class
        params = {
            **base_params,
            "output_dim": z_dim,
            "activation_fn": "swish",
        }
    else:
        raise ValueError(f"Unknown crn_type: {crn_type}")
    
    # Update with any additional parameters
    params.update(kwargs)
    
    # Create and return config
    config = CRNConfig()
    config.__dict__['config_dict'] = params
    return config


def list_available_combinations():
    """List all available flow type and CRN type combinations."""
    flow_types = ["potential", "natural", "geometric", "hamiltonian"]
    crn_types = ["mlp", "convex", "bilinear", "resnet"]
    
    print("Available Flow Type + CRN Type Combinations:")
    print("=" * 50)
    
    for flow_type in flow_types:
        print(f"\n{flow_type.upper()} FLOW:")
        for crn_type in crn_types:
            dim_req = ""
            if flow_type in ["natural", "geometric"]:
                dim_req = " (requires x_dim = z_dim)"
            print(f"  - {crn_type}: create_flow_model('{flow_type}', '{crn_type}', z_dim, x_dim{dim_req})")
    
    print(f"\nConvenience functions:")
    print(f"  - create_potential_flow(crn_type, z_dim, x_dim)")
    print(f"  - create_natural_flow(crn_type, z_dim, x_dim)")
    print(f"  - create_geometric_flow(crn_type, z_dim, x_dim)")
    print(f"  - create_hamiltonian_flow(crn_type, z_dim, x_dim)")
    print(f"  - create_convex_potential_flow(crn_type, z_dim, x_dim)  # DEPRECATED: use create_flow_model('potential', 'convex', ...)")
