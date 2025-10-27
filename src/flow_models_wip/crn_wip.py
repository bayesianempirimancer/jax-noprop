"""
Vector-based Conditional ResNet architectures for NoProp implementations.

This module provides MLP and flow-based ResNet wrappers that can be used with the NoProp algorithm
for vector inputs (e.g., natural parameters, expected sufficient statistics). The wrappers handle 
the specific input/output requirements for each NoProp variant.

For image-based models, see crn_image.py.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from src.embeddings.time_embeddings import create_time_embedding
from src.layers.concatsquash import ConcatSquash
from src.layers.bilinear import BilinearLayer
from src.layers.resnet_wrapper import ResNetWrapperBivariate
from src.layers.convex import ConvexResNetBivariate
from src.layers.gradnet_utils import handle_broadcasting
from src.layers.gradnet_utils import GradNetUtility, GradAndHessNetUtility, handle_broadcasting
from src.utils.activation_utils import get_activation_function
from src.configs.base_config import BaseConfig


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
        "input_shape": "NA",  # Will be set based on z_dim
        "output_shape": "NA",  # Will be set based on z_dim or z_dim**2
        "x_shape": "NA",  # Will be set based on x_dim
        "hidden_dims": (128, 128, 128),
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "dropout_rate": 0.1,
        "activation_fn": "swish",
        "use_batch_norm": False,
    })


class ConditionalResnet_MLP(nn.Module):
    """Simple MLP for NoProp with structured inputs.
    
    Args:
        input_shape: Input shape tuple (e.g., (8,))
        output_shape: Output shape tuple (e.g., (8,))
        x_shape: Conditional input shape tuple (e.g., (4,))
        hidden_dims: Tuple of hidden layer dimensions
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation_fn: Activation function to use (string)
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
    """
    input_shape: Tuple[int]
    output_shape: Tuple[int]
    x_shape: Tuple[int]
    hidden_dims: Tuple[int, ...] = (128, 128, 128)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    activation_fn: str = "swish"
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
        
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """Forward pass through simple MLP.
        
        Args:
            z: Current state [batch_size, input_shape[0]]
            x: Conditional input [batch_size, x_shape[0]]
            t: Time values [batch_size] or scalar (optional, defaults to 0.0)
            
        Returns:
            Updated state [batch_size, output_shape[0]]
        """        
        # Convert string activation function to callable
        activation_fn = get_activation_function(self.activation_fn)

        # Assert shapes have length 1
        assert len(self.input_shape) == 1, f"input_shape must have length 1, got {self.input_shape}"
        assert len(self.output_shape) == 1, f"output_shape must have length 1, got {self.output_shape}"
        assert len(self.x_shape) == 1, f"x_shape must have length 1, got {self.x_shape}"

        # Flatten inputs to work with dense layers
        z_flat = z  # Already 1D
        x_flat = x  # Already 1D

        # Compute output dimension
        output_dim = self.output_shape[0]
        
        # 1. x preprocessing
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
            t = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)(t)
            t = jnp.broadcast_to(t, z_flat.shape[:-1] + t.shape[-1:])
            x_flat = ConcatSquash(self.hidden_dims[0])(z_flat, x_flat, t)
        
        # 3. processing layers
        for hidden_dim in self.hidden_dims[1:]:
            x_flat = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.xavier_normal())(x_flat)
            if self.use_batch_norm:
                x_flat = nn.BatchNorm(use_running_average=True)(x_flat)
            x_flat = activation_fn(x_flat)            
            if self.dropout_rate > 0:
                x_flat = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_flat)
                
        # 5. output projection
        output = nn.Dense(output_dim, kernel_init=jax.nn.initializers.xavier_normal())(x_flat)
        
        return output


class BilinearConditionalResnet(nn.Module):
    """
    Bilinear Conditional ResNet that processes x through MLP, combines with t via ConcatSquash,
    and then uses a bilinear ResNet to combine with z.
    
    This architecture is designed for cases where we want to process the conditional input x
    through a standard MLP before combining it with time information and the state z.
    
    Args:
        input_shape: Input shape tuple (e.g., (8,))
        output_shape: Output shape tuple (e.g., (8,))
        x_shape: Conditional input shape tuple (e.g., (4,))
        hidden_dims: Tuple of hidden layer dimensions (used for both MLP and bilinear ResNet)
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation_fn: Activation function to use (string)
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
        use_bias: Whether to use bias terms
        use_projection: Whether to use projection layers
    """
    input_shape: Tuple[int]
    output_shape: Tuple[int]
    x_shape: Tuple[int]
    hidden_dims: Tuple[int, ...] = (128, 128, 128)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    activation_fn: str = "relu"
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
    use_bias: bool = True
    use_projection: bool = True
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """Forward pass through bilinear conditional ResNet.
        
        Args:
            z: Current state [batch_size, input_shape[0]]
            x: Conditional input [batch_size, x_shape[0]]
            t: Time values [batch_size] or scalar (optional, defaults to 0.0)
            training: Whether in training mode
            
        Returns:
            Updated state [batch_size, output_shape[0]]
        """
        # Convert string activation function to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        # Assert shapes have length 1
        assert len(self.input_shape) == 1, f"input_shape must have length 1, got {self.input_shape}"
        assert len(self.output_shape) == 1, f"output_shape must have length 1, got {self.output_shape}"
        assert len(self.x_shape) == 1, f"x_shape must have length 1, got {self.x_shape}"

        # Flatten inputs to work with dense layers
        z_flat = z  # Already 1D
        x_flat = x  # Already 1D

        # Compute output dimension
        output_dim = self.output_shape[0]
        
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
            t_embedding = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)(t)
            t_embedding = jnp.broadcast_to(t_embedding, z_flat.shape[:-1] + t_embedding.shape[-1:])
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
        output = nn.Dense(output_dim, kernel_init=jax.nn.initializers.xavier_normal())(z_updated)
        
        return output

class ConvexConditionalResnet(nn.Module):
    """
    Conditional ResNet with convex blocks for learning convex potentials.
    
    This ResNet uses convex layers to ensure the learned potential function is convex,
    which is essential for valid probability distributions in exponential families.
    
    Args:
        input_shape: Input shape tuple (e.g., (8,))
        output_shape: Output shape tuple (e.g., (8,))
        x_shape: Conditional input shape tuple (e.g., (4,))
        hidden_dims: Tuple of hidden layer dimensions
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation_fn: Activation function to use (string)
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
        use_bias: Whether to use bias terms
        use_projection: Whether to use projection layers
        block_type: Type of convex block ("simple" or "icnn")
    """
    input_shape: Tuple[int]
    output_shape: Tuple[int]
    x_shape: Tuple[int]
    hidden_dims: Tuple[int, ...] = (128, 128, 128)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    activation_fn: str = "softplus"  # Only smooth convex activations allowed
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
    use_bias: bool = True
    use_projection: bool = True
    block_type: str = "simple"
        
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """Forward pass through convex conditional ResNet.
        
        Args:
            z: Current state [batch_size, input_shape[0]]
            x: Conditional input [batch_size, x_shape[0]]
            t: Time values [batch_size] or scalar (optional, defaults to 0.0)
            training: Whether in training mode
            
        Returns:
            Updated state [batch_size, output_shape[0]]
        """
        # Convert string activation function to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        # Assert shapes have length 1
        assert len(self.input_shape) == 1, f"input_shape must have length 1, got {self.input_shape}"
        assert len(self.output_shape) == 1, f"output_shape must have length 1, got {self.output_shape}"
        assert len(self.x_shape) == 1, f"x_shape must have length 1, got {self.x_shape}"

        # Flatten inputs to work with dense layers
        z_flat = z  # Already 1D
        x_flat = x  # Already 1D
        
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
            t_embedding = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)(t)
            t_embedding = jnp.broadcast_to(t_embedding, z_flat.shape[:-1] + t_embedding.shape[-1:])
            x_flat = ConcatSquash(self.hidden_dims[0])(z_flat, x_flat, t_embedding)
        
        # 3. processing layers via convex bilinear layers 
        convex_resnet_bivariate = ConvexResNetBivariate(
            features=self.hidden_dims[-1],
            hidden_sizes=self.hidden_dims[:-1],
            activation=self.activation_fn,
            use_bias=self.use_bias,
            use_projection=self.use_projection
        )
        z = convex_resnet_bivariate(z_flat, x_flat, training=training)
        
        # 4. output projection to match desired output dimension
        output = nn.Dense(self.output_shape[0], kernel_init=jax.nn.initializers.xavier_normal())(z)
        return output


class PotentialFlow(nn.Module):
    """
    Wrapper that converts any conditional ResNet into a potential flow.
    
    Takes a conditional ResNet (function of z,x,t) and uses it to define a potential function.
    The flow is then computed as the negative gradient of this potential.
    
    Args:
        resnet_config: Configuration for the ResNet
        cond_resnet: String specifying the ResNet type ("conditional_resnet_mlp", "geometric_flow", "potential_flow")
    """
    resnet_config: Config
    cond_resnet: str = "conditional_resnet_mlp"
    
    @classmethod
    def create(cls, z_dim: int, x_dim: int, cond_resnet: str = "conditional_resnet_mlp", **config_kwargs):
        """
        Create a PotentialFlow with the proper configuration for the given dimensions.
        
        Args:
            z_dim: Dimension of the state space
            x_dim: Dimension of the conditional input
            cond_resnet: Type of ResNet to use
            **config_kwargs: Additional configuration parameters
            
        Returns:
            PotentialFlow instance with properly configured ResNet
        """
        # Create config with proper shapes for potential flow
        config = Config.with_shapes(
            input_shape=(z_dim,),
            output_shape=(z_dim,),  # Potential flow outputs same dimension as input
            x_shape=(x_dim,),
            **config_kwargs
        )
        
        return cls(resnet_config=config, cond_resnet=cond_resnet)
    
    def setup(self):
        """Initialize the ResNet and gradient utility once for efficiency."""
        # Create the ResNet instance once
        # Filter out model_type and network_type from resnet_config
        filtered_config = {k: v for k, v in self.resnet_config.config.items() if k not in ["model_type", "network_type"]}
        self.resnet_instance = create_cond_resnet(
            model_type="vanilla",
            network_type=self.cond_resnet,
            resnet_config=filtered_config
        )
        
        # Create ResNet factory function that uses the wrapper's parameters
        # NOTE: This factory function is ESSENTIAL - the gradient utilities expect a callable
        # function, not a Flax module. The factory provides the correct interface while
        # maintaining the parameter scope from the wrapper's self.variables.
        def resnet_factory(z_input, x_input, t_input, training=True):
            return self.resnet_instance(z_input, x_input, t_input, training=training)
        
        # Create gradient utility once for efficiency
        # NOTE: We pass resnet_factory (not resnet_instance) because GradNetUtility expects
        # a callable function, not a Flax module. The factory maintains proper parameter scoping.
        self.grad_utility = GradNetUtility(resnet_factory, reduction_method="sum")
    
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True, rngs=None) -> jnp.ndarray:
        """Compute the potential flow dz/dt = -∇_z V(z,x,t), where V is the potential.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim] 
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Flow dz/dt [batch_size, z_dim]
        """
        
        # Compute gradients using the pre-created utility class (handles broadcasting automatically)
        # Pass the wrapper's own parameters to the utility
        gradients = self.grad_utility(self.variables, z, x, t, training=training, rngs=rngs)
        
        # Return negative gradient (potential flow)
        return -gradients

class HamiltonianFlow(nn.Module):
    """
    Wrapper that converts any conditional ResNet into a Hamiltonian flow.
    
    Takes a conditional ResNet (function of z,x,t) and uses it to define a potential function.
    The flow is then computed under the assumption that z = jnp.concatenate([p, q], axis=-1).
    
    Args:
        resnet_config: Configuration for the ResNet
        cond_resnet: String specifying the ResNet type ("conditional_resnet_mlp", "geometric_flow", "potential_flow")
    """
    resnet_config: Config
    cond_resnet: str = "conditional_resnet_mlp"
    
    def setup(self):
        """Initialize the ResNet and gradient utility once for efficiency."""
        # Create the ResNet instance once
        # Filter out model_type and network_type from resnet_config
        filtered_config = {k: v for k, v in self.resnet_config.config.items() if k not in ["model_type", "network_type"]}
        self.resnet_instance = create_cond_resnet(
            model_type="vanilla",
            network_type=self.cond_resnet,
            resnet_config=filtered_config
        )
        
        # Create ResNet factory function that uses the wrapper's parameters
        # NOTE: This factory function is ESSENTIAL - the gradient utilities expect a callable
        # function, not a Flax module. The factory provides the correct interface while
        # maintaining the parameter scope from the wrapper's self.variables.
        def resnet_factory(z_input, x_input, t_input, training=True):
            return self.resnet_instance(z_input, x_input, t_input, training=training)
        
        # Create gradient utility once for efficiency
        # NOTE: We pass resnet_factory (not resnet_instance) because GradNetUtility expects
        # a callable function, not a Flax module. The factory maintains proper parameter scoping.
        self.grad_utility = GradNetUtility(resnet_factory, reduction_method="sum")
    
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True, rngs=None) -> jnp.ndarray:
        """Compute the potential flow dz/dt = -∇_z V(z,x,t), where V is the potential.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim] 
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Flow dz/dt [batch_size, z_dim]
        """
        
        # Compute gradients using the pre-created utility class (handles broadcasting automatically)
        # Pass the wrapper's own parameters to the utility
        gradients = self.grad_utility(self.variables, z, x, t, training=training, rngs=rngs)

        dHdp, dHdq = jnp.split(gradients, 2, axis=-1)

        # Return Hamiltonian flow: dq/dt = dH/dp, dp/dt = -dH/dq
        return jnp.concatenate([-dHdq, dHdp], axis=-1)

class NaturalFlow(nn.Module):
    """
    Wrapper that converts any conditional ResNet into a natural parameter flow. 
    
    Takes a conditional ResNet (function of z,x,t) with output_dim = z_dim**2 and:
    1. Reshapes ResNet output to get Sigma matrix
    2. Computes dz/dt = Sigma @ Sigma.T @ x_input
    
    Args:
        resnet_config: Configuration for the ResNet
        cond_resnet: String specifying the ResNet type ("conditional_resnet_mlp", "geometric_flow", "potential_flow")
    """
    resnet_config: Config
    cond_resnet: str = "conditional_resnet_mlp"
    
    @classmethod
    def create(cls, z_dim: int, x_dim: int, cond_resnet: str = "conditional_resnet_mlp", **config_kwargs):
        """
        Create a NaturalFlow with the proper configuration for the given dimensions.
        
        Args:
            z_dim: Dimension of the state space
            x_dim: Dimension of the conditional input (must equal z_dim for natural flow)
            cond_resnet: Type of ResNet to use
            **config_kwargs: Additional configuration parameters
            
        Returns:
            NaturalFlow instance with properly configured ResNet
        """
        # Create config with proper shapes for natural flow
        config = Config.with_shapes(
            input_shape=(z_dim,),
            output_shape=(z_dim ** 2,),  # Natural flow needs z_dim^2 output
            x_shape=(x_dim,),
            **config_kwargs
        )
        
        # Set activation to swish for natural flow (convenience method - prefer create_flow_model)
        config = config.update_config({"activation_fn": "swish"})
        
        return cls(resnet_config=config, cond_resnet=cond_resnet)
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True, rngs=None) -> jnp.ndarray:
        """Compute the natural flow dz/dt = Sigma @ Sigma.T @ x.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim] 
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Flow dz/dt [batch_size, z_dim]
        """
        
        # Check terminal dimension consistency - x_dim must match z_dim for the ResNet
        assert x.shape[-1] == z.shape[-1], f"x x_dim ({x.shape[-1]}) must match z_dim ({z.shape[-1]}) for natural flow"
        
        # Create the ResNet instance with correct output_shape
        # Get the model config and ensure output_shape is set correctly
        model_config = self.resnet_config.config.copy()
        model_config['output_shape'] = (z.shape[-1] ** 2,)
        # Filter out model_type and network_type from resnet_config
        filtered_config = {k: v for k, v in model_config.items() if k not in ["model_type", "network_type"]}
        
        resnet_instance = create_cond_resnet(
            model_type="vanilla",
            network_type=self.cond_resnet,
            resnet_config=filtered_config
        )
        
        # Create ResNet factory function that uses the wrapper's parameters
        # NOTE: This factory function is ESSENTIAL - the gradient utilities expect a callable
        # function, not a Flax module. The factory provides the correct interface while
        # maintaining the parameter scope from the wrapper's self.variables.
        def resnet_factory(z_input, x_input, t_input, training=training):
            return resnet_instance(z_input, x_input, t_input, training=training)
        
        # Handle broadcasting
        z_broadcasted, x_broadcasted, t_broadcasted = handle_broadcasting(z, x, t)
        
        # Apply the ResNet to get the matrix elements
        # The rngs parameter is handled at the Flax module level by the parent module
        resnet_output = resnet_factory(z_broadcasted, x_broadcasted, t_broadcasted)
        
        # Reshape to get Sigma matrix [batch_size, z_dim, z_dim] or [z_dim, z_dim]
        z_dim = z.shape[-1]
        if resnet_output.ndim == 1:
            # Single sample case
            Sigma = resnet_output.reshape(z_dim, z_dim)
            dz_dt = jnp.einsum("ij, j -> i", Sigma @ Sigma.T, x_broadcasted) / z_dim
        else:
            # Batch case
            Sigma = resnet_output.reshape(-1, z_dim, z_dim)
            dz_dt = jnp.einsum("...ij, ...jk, ...k -> ...i", Sigma, Sigma.transpose(0, 2, 1), x_broadcasted) / z_dim
        
        return dz_dt
        

class GeometricFlow(nn.Module):
    """
    Wrapper that converts any conditional ResNet into a geometric flow using Hessian.
    
    This wrapper uses the Hessian of a potential function as the Sigma matrix 
    to compute dz/dt = Sigma @ x, where Sigma is the Hessian of some potential.
    
    Args:
        resnet_config: Configuration for the ResNet
        cond_resnet: String specifying the ResNet type ("conditional_resnet_mlp", "geometric_flow", "potential_flow")
    """
    resnet_config: Config
    cond_resnet: str = "conditional_resnet_mlp"
    
    @classmethod
    def create(cls, z_dim: int, x_dim: int, cond_resnet: str = "conditional_resnet_mlp", **config_kwargs):
        """
        Create a GeometricFlow with the proper configuration for the given dimensions.
        
        Args:
            z_dim: Dimension of the state space
            x_dim: Dimension of the conditional input (must equal z_dim for geometric flow)
            cond_resnet: Type of ResNet to use
            **config_kwargs: Additional configuration parameters
            
        Returns:
            GeometricFlow instance with properly configured ResNet
        """
        # Create config with proper shapes for geometric flow
        config = Config.with_shapes(
            input_shape=(z_dim,),
            output_shape=(z_dim,),  # Geometric flow outputs same dimension as input
            x_shape=(x_dim,),
            **config_kwargs
        )
        
        return cls(resnet_config=config, cond_resnet=cond_resnet)
    
    def setup(self):
        """Initialize the ResNet and Hessian utility once for efficiency."""
        # Create the ResNet instance once
        # Filter out model_type and network_type from resnet_config
        filtered_config = {k: v for k, v in self.resnet_config.config.items() if k not in ["model_type", "network_type"]}
        self.resnet_instance = create_cond_resnet(
            model_type="vanilla",
            network_type=self.cond_resnet,
            resnet_config=filtered_config
        )
        
        # Create ResNet factory function that uses the wrapper's parameters
        # NOTE: This factory function is ESSENTIAL - the gradient utilities expect a callable
        # function, not a Flax module. The factory provides the correct interface while
        # maintaining the parameter scope from the wrapper's self.variables.
        def resnet_factory(z_input, x_input, t_input, training=True):
            return self.resnet_instance(z_input, x_input, t_input, training=training)
        
        # Create Hessian utility once for efficiency
        # NOTE: We pass resnet_factory (not resnet_instance) because GradAndHessNetUtility expects
        # a callable function, not a Flax module. The factory maintains proper parameter scoping.
        self.hess_utility = GradAndHessNetUtility(resnet_factory, reduction_method="sum")
    
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True, rngs=None) -> jnp.ndarray:
        """Compute the geometric flow dz/dt = Sigma @ x, where Sigma is the Hessian of some potential.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim] 
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Flow dz/dt [batch_size, z_dim]
        """
        
        # Check terminal dimension consistency - x_dim must match z_dim for the ResNet
        assert x.shape[-1] == z.shape[-1], f"x x_dim ({x.shape[-1]}) must match z_dim ({z.shape[-1]}) for geometric flow"
        
        # Compute Hessians using the pre-created utility class (handles broadcasting automatically)
        # Pass the wrapper's own parameters to the utility
        _, hessians = self.hess_utility(self.variables, z, x, t, training=training, rngs=rngs)
        
        # Compute geometric flow: dz/dt = Hessian @ x
            
        # Compute dz/dt = Hessian @ x for each sample in the batch
        dz_dt = jnp.einsum("...ij, ...j -> ...i", hessians, x)
        
        return dz_dt



# ============================================================================
# MODEL FACTORY
# ============================================================================


def create_crn(config_dict: dict, z_dim: int = None, x_dim: int = None) -> nn.Module:
    """
    Convenience function to create CRN models with proper configuration.
    
    Args:
        config_dict: Dictionary containing model configuration parameters
        z_dim: Dimension of the state space (optional, uses config if not provided)
        x_dim: Dimension of the conditional input (optional, uses config if not provided)
        
    Returns:
        Instantiated CRN model with proper configuration
    """
    # Build final config dict with provided parameters
    # Convert FrozenDict to regular dict if needed
    if hasattr(config_dict, 'unfreeze'):
        final_config = config_dict.unfreeze()
    else:
        final_config = dict(config_dict)
    
    # Validate required fields are present
    if final_config.get("model_type") == "NA":
        raise ValueError("model_type must be provided in config_dict")
    if final_config.get("network_type") == "NA":
        raise ValueError("network_type must be provided in config_dict")
    
    # Handle z_dim
    if z_dim is not None:
        final_config["z_dim"] = z_dim
    elif final_config.get("z_dim") == "NA":
        raise ValueError("z_dim must be provided either as parameter or in config_dict")
    
    # Handle x_dim
    if x_dim is not None:
        final_config["x_dim"] = x_dim
    elif final_config.get("x_dim") == "NA":
        raise ValueError("x_dim must be provided either as parameter or in config_dict")
    
    # Determine output shape based on model type
    if final_config["model_type"] == "natural":
        output_shape = (final_config["z_dim"] ** 2,)  # Natural flow needs z_dim^2 output
    else:
        output_shape = (final_config["z_dim"],)  # Other flows output same dimension as input
    
    # Create config with proper shapes and all parameters
    config = Config.with_shapes(
        input_shape=(final_config["z_dim"],),
        output_shape=output_shape,
        x_shape=(final_config["x_dim"],)
    )
    
    # Update config with additional parameters from final_config
    config = config.update_config({
        "hidden_dims": final_config["hidden_dims"],
        "time_embed_dim": final_config["time_embed_dim"],
        "time_embed_method": final_config["time_embed_method"],
        "dropout_rate": final_config["dropout_rate"],
        "activation_fn": final_config["activation_fn"],
        "use_batch_norm": final_config["use_batch_norm"],
    })
    
    # Set activation for natural flow if not already specified
    if final_config["model_type"] == "natural" and final_config.get("activation_fn") is None:
        config = config.update_config({"activation_fn": "swish"})
    
    # Create the model
    # model_type is the flow wrapper type, network_type is the base network type
    # Filter out model_type and network_type from resnet_config as they're not needed by the ResNet classes
    resnet_config = {k: v for k, v in config.config.items() if k not in ["model_type", "network_type"]}
    return create_cond_resnet(final_config["model_type"], final_config["network_type"], resnet_config)

def create_cond_resnet(model_type: str, network_type: str, resnet_config: dict) -> nn.Module:
    """
    Factory function to create model instances with optional wrappers.
    
    Args:
        model_type: Type of wrapper to use ("vanilla", "geometric", "potential", "natural")
        network_type: Type of base network to create ("mlp", "bilinear", "convex")
        resnet_config: Dictionary containing ResNet configuration parameters
        
    Returns:
        Instantiated model (wrapped or unwrapped)
    """
    # Create the base ResNet
    if network_type == "mlp":
        base_resnet = ConditionalResnet_MLP(**resnet_config)
    elif network_type == "convex":
        base_resnet = ConvexConditionalResnet(**resnet_config)
    elif network_type == "bilinear":
        base_resnet = BilinearConditionalResnet(**resnet_config)
    else:
        raise ValueError(f"Unknown network_type: {network_type}. Supported types: mlp, bilinear, convex")
    
    # Apply wrapper if specified
    if model_type == "vanilla":
        return base_resnet
    elif model_type == "geometric":
        # Create a Config object for the wrapper (resnet_config should already be complete)
        config_obj = Config()
        config_obj = config_obj.update_config(resnet_config)
        return GeometricFlow(resnet_config=config_obj, cond_resnet=network_type)
    elif model_type == "potential":
        # Create a Config object for the wrapper (resnet_config should already be complete)
        config_obj = Config()
        config_obj = config_obj.update_config(resnet_config)
        return PotentialFlow(resnet_config=config_obj, cond_resnet=network_type)
    elif model_type == "natural":
        # Create a Config object for the wrapper (resnet_config should already be complete)
        config_obj = Config()
        config_obj = config_obj.update_config(resnet_config)
        return NaturalFlow(resnet_config=config_obj, cond_resnet=network_type)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported types: vanilla, geometric, potential, natural")
