"""
Vector-based Conditional ResNet architectures for NoProp implementations.

This module provides MLP and flow-based ResNet wrappers that can be used with the NoProp algorithm
for vector inputs (e.g., natural parameters, expected sufficient statistics). The wrappers handle 
the specific input/output requirements for each NoProp variant.

For image-based models, see crn_image.py.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Dict, Any
from functools import cached_property

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict

from src.configs.base_config import BaseConfig
from src.embeddings.time_embeddings import create_time_embedding
from src.layers.concatsquash import ConcatSquash
from src.layers.convex import ConvexResNetBivariate

from src.utils.activation_utils import get_activation_function



# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass(frozen=True)
class Config(BaseConfig):
    """Configuration for Conditional ResNet."""
    
    # Set model_name from config_dict
    model_name: str = "conditional_resnet"
    
    # Hierarchical configuration structure
    config: dict = field(default_factory=lambda: {
        "model_type": "vanilla",  # Options: "vanilla", "geometric", "potential", "natural"
        "network_type": "mlp",  # Options: "mlp", "bilinear", "convex"
        "latent_shape": "NA",  # Will be set based on z_dim
        "input_shape": "NA",  # Will be set based on z_dim
        "output_shape": "NA",  # Will be set based on z_dim or z_dim**2
        "hidden_dims": (128, 128, 128),
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "dropout_rate": 0.1,
        "activation_fn": "swish",
        "use_batch_norm": False,
    })


########  CRN CLASSES AVAILABLE   ###########

def get_crn_class(crn_type: str):
    """Get CRN class by type string."""
    CRN_CLASSES = {
        'mlp': ConditionalResnet_MLP,
        'bilinear': BilinearConditionalResnet,
        'convex': ConvexConditionalResnet,
    }

    if crn_type not in CRN_CLASSES:
        raise ValueError(f"Unknown CRN type: {crn_type}. Available: {list(CRN_CLASSES.keys())}")
    
    return CRN_CLASSES[crn_type]


########  FACTORY FUNCTION   ###########
def create_conditional_resnet(config_dict: Union[Dict[str, Any], FrozenDict], latent_shape: Tuple[int, ...], input_shape: Tuple[int, ...], output_shape: Optional[Tuple[int, ...]] = None) -> nn.Module:
    """Create a conditional ResNet model using the homogenized approach.
    
    Args:
        config_dict: Configuration dictionary for the model
        latent_shape: Shape of the latent state (z)
        input_shape: Shape of the conditional input (x)
        output_shape: Shape of the output (optional, defaults to latent_shape)
        
    Returns:
        Instantiated CRN model
    """
    # Convert config_dict to regular dict if needed
    if hasattr(config_dict, 'unfreeze'):
        final_config = config_dict.unfreeze()
    else:
        final_config = dict(config_dict)
    
    # Get model_type and network_type
    model_type = final_config.get("model_type", "vanilla")
    network_type = final_config.get("network_type", "mlp")
    
    # Use provided output_shape or default to latent_shape
    if output_shape is None:
        output_shape = latent_shape
    
    # Create config with proper shapes and all necessary fields
    config = Config.with_shapes(
        latent_shape=latent_shape,
        output_shape=output_shape,
        input_shape=input_shape
    )
    
    # Add missing fields that CRN classes need using append
    if network_type == "convex":
        additional_fields = {
            "use_bias": True,
            "use_projection": False
        }
        config = config.append(additional_fields)
    
    # Update with additional parameters from config_dict (excluding model_type and network_type)
    resnet_config = {k: v for k, v in final_config.items() if k not in ["model_type", "network_type"]}
    final_config_obj = config.update_config(resnet_config)
    
    # Filter out model_type and network_type from the final config before passing to CRN classes
    crn_config = {k: v for k, v in final_config_obj.config.items() if k not in ["model_type", "network_type"]}
    
    # Create the base ResNet
    if network_type == "mlp":
        base_resnet = ConditionalResnet_MLP(**crn_config)
    elif network_type == "convex":
        base_resnet = ConvexConditionalResnet(**crn_config)
    elif network_type == "bilinear":
        base_resnet = BilinearConditionalResnet(**crn_config)
    else:
        raise ValueError(f"Unknown network_type: {network_type}. Supported types: mlp, bilinear, convex")
    
    # Apply wrapper if specified
    if model_type == "vanilla":
        return base_resnet
    else:
        # Use the unified gradient wrapper factory
        from src.flow_models.crn_grads import create_gradient_wrapper
        
        # Convert the config back to a dictionary for the wrapper factory
        wrapper_config = {
            "model_type": "vanilla",  # The wrapper will create its own CRN
            "network_type": network_type,
            **crn_config  # All the CRN-specific parameters
        }
        
        return create_gradient_wrapper(model_type, wrapper_config, latent_shape, input_shape, output_shape)


########  CRN CLASS DEFINITIONS   ###########

class ConditionalResnet_MLP(nn.Module):
    """Simple MLP for NoProp with structured inputs.
    
    Args:
        latent_shape: Latent shape tuple (e.g., (8,))
        output_shape: Output shape tuple (e.g., (8,))
        input_shape: Conditional input shape tuple (e.g., (4,))
        hidden_dims: Tuple of hidden layer dimensions
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation_fn: Activation function to use (string)
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
    """
    latent_shape: Tuple[int,...]
    input_shape: Tuple[int,...]
    output_shape: Tuple[int,...]
    hidden_dims: Tuple[int, ...] = (128, 128, 128)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    activation_fn: str = "swish"
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
        

    @cached_property
    def latent_dim(self) -> int:
        """Latent dimension of the conditional ResNet."""
        dim = 1
        for shape in self.latent_shape:
            dim *= shape
        return dim

    @cached_property
    def input_dim(self) -> int:
        """Input dimension of the conditional ResNet."""
        dim = 1
        for shape in self.input_shape:
            dim *= shape
        return dim

    @cached_property
    def output_dim(self) -> int:
        """Output dimension of the conditional ResNet."""
        dim = 1
        for shape in self.output_shape:
            dim *= shape
        return dim

        
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """Forward pass through simple MLP.
        
        Args:
            z: Current state [batch_size, latent_shape[0]]
            x: Conditional input [batch_size, input_shape[0]]
            t: Time values [batch_size] or scalar (optional, defaults to 0.0)
            
        Returns:
            Updated state [batch_size, output_shape[0]]
        """        
        # Convert string activation function to callable
        activation_fn = get_activation_function(self.activation_fn)

        # Flatten inputs to work with dense layers
        t = jnp.asarray(t)
        batch_shape_z = z.shape[:-len(self.latent_shape)]
        batch_shape_x = x.shape[:-len(self.input_shape)]
        batch_shape = jnp.broadcast_shapes(batch_shape_z, batch_shape_x, t.shape)

        z = jnp.broadcast_to(z, batch_shape + z.shape[-len(self.latent_shape):])
        x = jnp.broadcast_to(x, batch_shape + x.shape[-len(self.input_shape):])

        z_flat = z.reshape(-1, self.latent_dim)
        x_flat = x.reshape(-1, self.input_dim)
        t = jnp.broadcast_to(t, batch_shape)

        # Compute output dimension
        
        # 1. input (x) preprocessing
        for hidden_dim in self.hidden_dims:
            x_flat = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.xavier_normal())(x_flat)
            if self.use_batch_norm:
                x_flat = nn.BatchNorm(use_running_average=True)(x_flat)
            x_flat = activation_fn(x_flat)
            if self.dropout_rate > 0:
                x_flat = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_flat)
        
        # 2. time embeddings - handle t=None case
        if t is None:
            t_embedding = jnp.zeros((1,))  # Default to t=0 when None
            x_flat = ConcatSquash(self.hidden_dims[0])(z_flat, x_flat)
        else:
            # Create time embedding directly on the flattened time tensor
            t_flat = t.reshape(-1)
            t_embedding = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)(t_flat)
            x_flat = ConcatSquash(self.hidden_dims[0])(z_flat, x_flat, t_embedding)
        
        # 3. processing layers
        for hidden_dim in self.hidden_dims[1:]:
            x_flat = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.xavier_normal())(x_flat)
            if self.use_batch_norm:
                x_flat = nn.BatchNorm(use_running_average=True)(x_flat)
            x_flat = activation_fn(x_flat)            
            if self.dropout_rate > 0:
                x_flat = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_flat)
                
        # 5. output projection
        output = nn.Dense(self.output_dim, kernel_init=jax.nn.initializers.xavier_normal())(x_flat)
        
        return output.reshape(batch_shape + self.output_shape)


class BilinearConditionalResnet(nn.Module):
    """
    Bilinear Conditional ResNet that processes x through MLP, combines with t via ConcatSquash,
    and then uses a bilinear ResNet to combine with z.
    
    This architecture is designed for cases where we want to process the conditional input x
    through a standard MLP before combining it with time information and the state z.
    
    Args:
        latent_shape: Latent shape tuple (e.g., (8,))
        output_shape: Output shape tuple (e.g., (8,))
        input_shape: Conditional input shape tuple (e.g., (4,))
        hidden_dims: Tuple of hidden layer dimensions (used for both MLP and bilinear ResNet)
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation_fn: Activation function to use (string)
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
    """
    latent_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    hidden_dims: Tuple[int, ...] = (128, 128, 128)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    activation_fn: str = "relu"
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
    
    @cached_property
    def latent_dim(self) -> int:
        """Latent dimension of the conditional ResNet."""
        dim = 1
        for shape in self.latent_shape:
            dim *= shape
        return dim

    @cached_property
    def input_dim(self) -> int:
        """Input dimension of the conditional ResNet."""
        dim = 1
        for shape in self.input_shape:
            dim *= shape
        return dim

    @cached_property
    def output_dim(self) -> int:
        """Output dimension of the conditional ResNet."""
        dim = 1
        for shape in self.output_shape:
            dim *= shape
        return dim
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """Forward pass through bilinear conditional ResNet.
        
        Args:
            z: Current state [batch_size, latent_shape[0]]
            x: Conditional input [batch_size, input_shape[0]]
            t: Time values [batch_size] or scalar (optional, defaults to 0.0)
            training: Whether in training mode
            
        Returns:
            Updated state [batch_size, output_shape[0]]
        """
        # Convert string activation function to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        # Handle broadcasting and flattening
        t = jnp.asarray(t)
        batch_shape_z = z.shape[:-len(self.latent_shape)]
        batch_shape_x = x.shape[:-len(self.input_shape)]
        batch_shape = jnp.broadcast_shapes(batch_shape_z, batch_shape_x, t.shape)

        z = jnp.broadcast_to(z, batch_shape + z.shape[-len(self.latent_shape):])
        x = jnp.broadcast_to(x, batch_shape + x.shape[-len(self.input_shape):])
        t = jnp.broadcast_to(t, batch_shape)

        z_flat = z.reshape(-1, self.latent_dim)
        x_flat = x.reshape(-1, self.input_dim)
        
        # 1. x preprocessing
        # 2. Process x through standard MLP (number of layers specified by hidden_dims)
        x_processed = x_flat
        for i, hidden_dim in enumerate(self.hidden_dims):
            x_processed = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.xavier_normal())(x_processed)
            if self.use_batch_norm:
                x_processed = nn.BatchNorm(use_running_average=True)(x_processed)
            x_processed = activation_fn(x_processed)
            if self.dropout_rate > 0:
                x_processed = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_processed)
        
        # 3. Time embeddings and ConcatSquash
        if t is None:
            # Default to t=0 when None
            x_processed = ConcatSquash(self.hidden_dims[-1])(z_flat, x_processed)
        else:
            # Create time embedding directly on the flattened time tensor
            t_flat = t.reshape(-1)
            t_embedding = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)(t_flat)
            x_processed = ConcatSquash(self.hidden_dims[-1])(z_flat, x_processed, t_embedding)
        
        # 4. Simple bilinear-like processing that combines processed x with z
        # Use Dense layers to create a bilinear-like transformation
        # Project z and x_processed to the same feature space
        z_proj = nn.Dense(self.hidden_dims[-1], kernel_init=jax.nn.initializers.xavier_normal())(z_flat)
        x_proj = nn.Dense(self.hidden_dims[-1], kernel_init=jax.nn.initializers.xavier_normal())(x_processed)
        
        # Element-wise multiplication (bilinear-like interaction)
        z_updated = z_proj * x_proj
        
        # Apply activation
        z_updated = activation_fn(z_updated)
        # Apply batch norm if enabled
        if self.use_batch_norm:
            z_updated = nn.BatchNorm(use_running_average=True)(z_updated)
        
        # 5. Output projection to match desired output dimension
        output = nn.Dense(self.output_dim, kernel_init=jax.nn.initializers.xavier_normal())(z_updated)
        
        return output.reshape(batch_shape + self.output_shape)

class ConvexConditionalResnet(nn.Module):
    """
    Conditional ResNet with convex blocks for learning convex potentials.
    
    This ResNet uses convex layers to ensure the learned potential function is convex,
    which is essential for valid probability distributions in exponential families.
    
    Args:
        latent_shape: Latent shape tuple (e.g., (8,))
        output_shape: Output shape tuple (e.g., (8,))
        input_shape: Conditional input shape tuple (e.g., (4,))
        hidden_dims: Tuple of hidden layer dimensions
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation_fn: Activation function to use (string)
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
        use_bias: Whether to use bias terms
        use_projection: Whether to use projection layers
    """
    latent_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    hidden_dims: Tuple[int, ...] = (128, 128, 128)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    activation_fn: str = "softplus"  # Only smooth convex activations allowed
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
    use_bias: bool = True
    use_projection: bool = True
    
    @cached_property
    def latent_dim(self) -> int:
        """Latent dimension of the conditional ResNet."""
        dim = 1
        for shape in self.latent_shape:
            dim *= shape
        return dim

    @cached_property
    def input_dim(self) -> int:
        """Input dimension of the conditional ResNet."""
        dim = 1
        for shape in self.input_shape:
            dim *= shape
        return dim

    @cached_property
    def output_dim(self) -> int:
        """Output dimension of the conditional ResNet."""
        dim = 1
        for shape in self.output_shape:
            dim *= shape
        return dim
        
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """Forward pass through convex conditional ResNet.
        
        Args:
            z: Current state [batch_size, latent_shape[0]]
            x: Conditional input [batch_size, input_shape[0]]
            t: Time values [batch_size] or scalar (optional, defaults to 0.0)
            training: Whether in training mode
            
        Returns:
            Updated state [batch_size, output_shape[0]]
        """
        # Convert string activation function to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        # Handle broadcasting and flattening
        jnp.asarray(t)
        batch_shape_z = z.shape[:-len(self.latent_shape)]
        batch_shape_x = x.shape[:-len(self.input_shape)]
        batch_shape = jnp.broadcast_shapes(batch_shape_z, batch_shape_x, t.shape)

        z = jnp.broadcast_to(z, batch_shape + z.shape[-len(self.latent_shape):])
        x = jnp.broadcast_to(x, batch_shape + x.shape[-len(self.input_shape):])

        z_flat = z.reshape(-1, self.latent_dim)
        x_flat = x.reshape(-1, self.input_dim)
        t = jnp.broadcast_to(t, batch_shape)
        
        # 1. x preprocessing
        # Preprocess x through standard layers
        for hidden_dim in self.hidden_dims:
            x_flat = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.xavier_normal())(x_flat)
            if self.use_batch_norm:
                x_flat = nn.BatchNorm(use_running_average=True)(x_flat)
            x_flat = activation_fn(x_flat)
            if self.dropout_rate > 0:
                x_flat = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_flat)
        
        # 2. time embeddings - handle t=None case
        if t is None:
            # Default to t=0 when None
            x_flat = ConcatSquash(self.hidden_dims[0])(z_flat, x_flat)
        else:
            # Create time embedding directly on the flattened time tensor
            t_flat = t.reshape(-1)
            t_embedding = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)(t_flat)
            x_flat = ConcatSquash(self.hidden_dims[0])(z_flat, x_flat, t_embedding)
        
        # 3. processing layers via convex bilinear layers 
        convex_resnet_bivariate = ConvexResNetBivariate(
            features=self.latent_dim,  # Output should match z_flat dimension
            hidden_sizes=self.hidden_dims,
            activation=self.activation_fn,
            use_bias=self.use_bias,
            use_projection=self.use_projection
        )
        z = convex_resnet_bivariate(z_flat, x_flat, training=training)
        
        # 4. output projection to match desired output dimension
        output = nn.Dense(self.output_dim, kernel_init=jax.nn.initializers.xavier_normal())(z)
        return output.reshape(batch_shape + self.output_shape)



