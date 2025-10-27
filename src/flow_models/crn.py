"""
Vector-based Conditional ResNet architectures for NoProp implementations.

This module provides MLP and flow-based ResNet wrappers that can be used with the NoProp algorithm
for vector inputs (e.g., natural parameters, expected sufficient statistics). The wrappers handle 
the specific input/output requirements for each NoProp variant.

For image-based models, see crn_image.py.
"""

from dataclasses import dataclass
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
    config_dict = {
        "output_dim": None,
        "hidden_dims": (128, 128, 128),
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "dropout_rate": 0.1,
        "activation_fn": "swish",
        "use_batch_norm": False,
    }


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_cond_resnet(model_type: str, model_config: dict) -> nn.Module:
    """
    Factory function to create model instances.
    
    Args:
        model_type: Type of model to create ("conditional_resnet_mlp", "convex_conditional_resnet", "geometric_flow", "potential_flow", "potential_flow_wrapper", "geometric_flow_wrapper")
        model_config: Dictionary containing model configuration parameters
        
    Returns:
        Instantiated model
    """
    if model_type == "conditional_resnet_mlp":
        return ConditionalResnet_MLP(**model_config)
    elif model_type == "convex_conditional_resnet":
        return ConvexConditionalResnet(**model_config)
    elif model_type == "bilinear_conditional_resnet":
        return BilinearConditionalResnet(**model_config)
    # Note: GeometricFlow and PotentialFlow classes have been commented out
    # as they are replaced by the wrapper classes (GeometricFlowWrapper, PotentialFlowWrapper)
    elif model_type == "potential_flow_wrapper":
        # For wrapper classes, we need to create a Config object and pass it properly
        from .fm import Config as FMConfig
        config = FMConfig()
        return PotentialFlowWrapper(resnet_config=config, cond_resnet=model_config.get('cond_resnet', 'conditional_resnet_mlp'))
    elif model_type == "geometric_flow_wrapper":
        # For wrapper classes, we need to create a Config object and pass it properly
        from .fm import Config as FMConfig
        config = FMConfig()
        return GeometricFlowWrapper(resnet_config=config, cond_resnet=model_config.get('cond_resnet', 'conditional_resnet_mlp'))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_flow_model(model_name: str, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> jnp.ndarray:
    """
    Convenience function to create and apply flow models.
    
    This function handles all the logic for creating flow wrappers and direct CRN models,
    eliminating code duplication across fm.py, ct.py, and df.py.
    
    Args:
        model_name: Name of the model ("potential_flow", "geometric_flow", 
                   "natural_flow", "conditional_resnet_mlp")
        z: Current state [batch_size, z_dim]
        x: Conditional input [batch_size, x_dim] 
        t: Time values [batch_size] or scalar (optional)
        training: Whether in training mode
        
    Returns:
        Model output [batch_size, z_dim] or [batch_size, output_dim]
    """
    # Create fresh CRN config for each call to avoid shared state issues
    crn_config = Config()
    
    # Handle flow wrapper models
    if model_name in ["potential_flow", "geometric_flow", "natural_flow", "convex_potential_flow"]:
        if model_name == "potential_flow":
            wrapper = PotentialFlowWrapper(
                resnet_config=crn_config,
                cond_resnet="conditional_resnet_mlp"
            )
        elif model_name == "geometric_flow":
            wrapper = GeometricFlowWrapper(
                resnet_config=crn_config,
                cond_resnet="conditional_resnet_mlp"
            )
        elif model_name == "natural_flow":
            # For natural flow, we need to create a ConditionalResnet_MLP with output_dim = z_dim**2
            # Create a new config with the correct output_dim
            model_config = crn_config.config_dict.copy()
            model_config['output_dim'] = z.shape[-1] ** 2
            
            # Create a new Config instance with the updated model_config
            updated_crn_config = Config()
            # Replace the config_dict for this instance
            updated_crn_config.__dict__['config_dict'] = model_config
            
            wrapper = NaturalFlowWrapper(
                resnet_config=updated_crn_config,
                cond_resnet="conditional_resnet_mlp"
            )
        elif model_name == "convex_potential_flow":
            wrapper = ConvexPotentialFlowWrapper(
                resnet_config=crn_config,
                cond_resnet="convex_conditional_resnet"
            )
        
        return wrapper(z, x, t, training=training)
    
    # Handle direct CRN models
    elif model_name in ["conditional_resnet_mlp", "convex_conditional_resnet", "bilinear_conditional_resnet"]:
        # Use factory function to create model instance
        model_instance = create_cond_resnet(
            model_type=model_name,
            model_config=crn_config.config_dict
        )
        # For models that need compact context, we need to use apply
        # But create_flow_model is called from within a compact context, so direct call should work
        return model_instance(z, x, t, training=training)
    
    else:
        raise ValueError(f"Unsupported model type: {model_name}. Supported models: potential_flow, geometric_flow, natural_flow, convex_potential_flow, conditional_resnet_mlp, convex_conditional_resnet, bilinear_conditional_resnet")


class ConditionalResnet_MLP(nn.Module):
    """Simple MLP for NoProp with vector inputs.
    
    Args:
        hidden_dims: Tuple of hidden layer dimensions
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation_fn: Activation function to use (string)
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
    """
    output_dim: Optional[int] = None
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
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim]
            t: Time values [batch_size] or scalar (optional, defaults to 0.0)
            
        Returns:
            Updated state [batch_size, z_dim] (same shape as input z)
        """        
        # Convert string activation function to callable
        activation_fn = get_activation_function(self.activation_fn)

        if self.output_dim is None:
            output_dim = z.shape[-1]
        else:
            output_dim = self.output_dim
        
        # 1. x preprocessing
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.xavier_normal())(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=True)(x)
            x = activation_fn(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # 2. time embeddings - handle t=None case
        if t is None:
            t_embedding = jnp.zeros((1,))  # Default to t=0 when None
            x = ConcatSquash(self.hidden_dims[0])(z, x)
        else:
            t = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)(t)
            t = jnp.broadcast_to(t, z.shape[:-1] + t.shape[-1:])
            x = ConcatSquash(self.hidden_dims[0])(z, x, t)
        
        # 3. processing layers
        for hidden_dim in self.hidden_dims[1:]:
            x = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.xavier_normal())(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=True)(x)
            x = activation_fn(x)            
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
                
        # 5. output projection to match input
        return nn.Dense(output_dim, kernel_init=jax.nn.initializers.xavier_normal())(x)


class ConvexConditionalResnet(nn.Module):
    """
    Conditional ResNet with convex blocks for learning convex potentials.
    
    This ResNet uses convex layers to ensure the learned potential function is convex,
    which is essential for valid probability distributions in exponential families.
    
    Args:
        output_dim: Output dimension (defaults to z_dim if None)
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
    output_dim: Optional[int] = None
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
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim]
            t: Time values [batch_size] or scalar (optional, defaults to 0.0)
            training: Whether in training mode
            
        Returns:
            Updated state [batch_size, output_dim] or [batch_size, z_dim] if output_dim is None
        """
        # Convert string activation function to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        if self.output_dim is None:
            output_dim = z.shape[-1]
        else:
            output_dim = self.output_dim
        
        # 1. x preprocessing
        # Preprocess x through standard layers
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.xavier_normal())(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=True)(x)
            x = activation_fn(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # 2. time embeddings - handle t=None case
        if t is None:
            # Default to t=0 when None
            x = ConcatSquash(self.hidden_dims[0])(z, x)
        else:
            t_embedding = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)(t)
            t_embedding = jnp.broadcast_to(t_embedding, z.shape[:-1] + t_embedding.shape[-1:])
            x = ConcatSquash(self.hidden_dims[0])(z, x, t_embedding)
        
        # 3. processing layers via convex bilinear layers 
        convex_resnet_bivariate = ConvexResNetBivariate(
            features=self.hidden_dims[-1],
            hidden_sizes=self.hidden_dims[:-1],
            activation=self.activation_fn,
            use_bias=self.use_bias,
            use_projection=self.use_projection
        )
        z = convex_resnet_bivariate(z, x, training=training)
        
        # 4. output projection to match desired output dimension
        return z


class PotentialFlowWrapper(nn.Module):
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
    
    def setup(self):
        """Initialize the ResNet and gradient utility once for efficiency."""
        # Create the ResNet instance once
        self.resnet_instance = create_cond_resnet(
            model_type=self.cond_resnet,
            model_config=self.resnet_config.config_dict
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

class HamiltonianFlowWrapper(nn.Module):
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
        self.resnet_instance = create_cond_resnet(
            model_type=self.cond_resnet,
            model_config=self.resnet_config.config_dict
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

        dHdq, dHdp = jnp.split(gradients, 2, axis=-1)

        # Return Hamiltonian flow: dq/dt = dH/dp, dp/dt = -dH/dq
        return jnp.concatenate([dHdp, -dHdq], axis=-1)

class NaturalFlowWrapper(nn.Module):
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
        
        # Create the ResNet instance with correct output_dim
        # Get the model config and ensure output_dim is set correctly
        model_config = self.resnet_config.config_dict.copy()
        model_config['output_dim'] = z.shape[-1] ** 2
        
        resnet_instance = create_cond_resnet(
            model_type=self.cond_resnet,
            model_config=model_config
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
        

class GeometricFlowWrapper(nn.Module):
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
    
    def setup(self):
        """Initialize the ResNet and Hessian utility once for efficiency."""
        # Create the ResNet instance once
        self.resnet_instance = create_cond_resnet(
            model_type=self.cond_resnet,
            model_config=self.resnet_config.config_dict
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


class ConvexPotentialFlowWrapper(nn.Module):
    """
    Wrapper that converts a convex conditional ResNet into a potential flow.
    
    This wrapper uses a ConvexConditionalResNet to learn a convex potential function,
    ensuring the learned potential is convex (essential for valid probability distributions).
    The flow is computed as the negative gradient of this convex potential.
    
    Args:
        resnet_config: Configuration for the ResNet
        cond_resnet: String specifying the ResNet type (should be "convex_conditional_resnet")
    """
    resnet_config: Config
    cond_resnet: str = "convex_conditional_resnet"
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True, rngs=None) -> jnp.ndarray:
        """Compute the convex potential flow dz/dt = -∇_z V(z,x,t), where V is a convex potential.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim] 
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Flow dz/dt [batch_size, z_dim]
        """
        
        # Create the ResNet instance with correct output_dim
        # Get the model config and ensure output_dim is set correctly
        model_config = self.resnet_config.config_dict.copy()
        model_config['output_dim'] = z.shape[-1]
        
        resnet_instance = create_cond_resnet(
            model_type=self.cond_resnet,
            model_config=model_config
        )
        
        # Create ResNet factory function that uses the wrapper's parameters
        # NOTE: This factory function is ESSENTIAL - the gradient utilities expect a callable
        # function, not a Flax module. The factory provides the correct interface while
        # maintaining the parameter scope from the wrapper's self.variables.
        def resnet_factory(z_input, x_input, t_input, training=training):
            return resnet_instance(z_input, x_input, t_input, training=training)
        
        # Create gradient utility
        # NOTE: We pass resnet_factory (not resnet_instance) because GradNetUtility expects
        # a callable function, not a Flax module. The factory maintains proper parameter scoping.
        grad_utility = GradNetUtility(resnet_factory, reduction_method="sum")
        
        # Compute gradients using the utility class (handles broadcasting automatically)
        # Pass the wrapper's own parameters to the utility
        gradients = grad_utility(self.variables, z, x, t, training=training, rngs=rngs)
        
        # Return negative gradient (potential flow)
        return -gradients


class BilinearConditionalResnet(nn.Module):
    """
    Bilinear Conditional ResNet that processes x through MLP, combines with t via ConcatSquash,
    and then uses a bilinear ResNet to combine with z.
    
    This architecture is designed for cases where we want to process the conditional input x
    through a standard MLP before combining it with time information and the state z.
    
    Args:
        output_dim: Output dimension (defaults to z_dim if None)
        hidden_dims: Tuple of hidden layer dimensions (used for both MLP and bilinear ResNet)
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation_fn: Activation function to use (string)
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
        use_bias: Whether to use bias terms
        use_projection: Whether to use projection layers
    """
    output_dim: Optional[int] = None
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
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim]
            t: Time values [batch_size] or scalar (optional, defaults to 0.0)
            training: Whether in training mode
            
        Returns:
            Updated state [batch_size, output_dim] or [batch_size, z_dim] if output_dim is None
        """
        # Convert string activation function to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        if self.output_dim is None:
            output_dim = z.shape[-1]
        else:
            output_dim = self.output_dim
        
        # 1. x preprocessing
        # 2. Process x through standard MLP (number of layers specified by hidden_dims)
        x_processed = x
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
            x_processed = ConcatSquash(self.hidden_dims[-1])(z, x_processed)
        else:
            t_embedding = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)(t)
            t_embedding = jnp.broadcast_to(t_embedding, z.shape[:-1] + t_embedding.shape[-1:])
            x_processed = ConcatSquash(self.hidden_dims[-1])(z, x_processed, t_embedding)
        
        # 4. Simple bilinear-like processing that combines processed x with z
        # Use Dense layers to create a bilinear-like transformation
        # Project z and x_processed to the same feature space
        z_proj = nn.Dense(self.hidden_dims[-1], kernel_init=jax.nn.initializers.xavier_normal())(z)
        x_proj = nn.Dense(self.hidden_dims[-1], kernel_init=jax.nn.initializers.xavier_normal())(x_processed)
        
        # Element-wise multiplication (bilinear-like interaction)
        z_updated = z_proj * x_proj
        
        # Apply activation
        z_updated = activation_fn(z_updated)
        # Apply batch norm if enabled
        if self.use_batch_norm:
            z_updated = nn.BatchNorm(use_running_average=True)(z_updated)
        
        # 5. Output projection to match desired output dimension
        return nn.Dense(output_dim, kernel_init=jax.nn.initializers.xavier_normal())(z_updated)
