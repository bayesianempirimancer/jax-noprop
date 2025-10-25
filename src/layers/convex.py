"""
Convex layers for Input Convex Neural Networks (ICNN).

This module provides convex layer implementations that maintain convexity properties
essential for learning log normalizers in exponential family distributions.
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Union, Callable


class ConvexHiddenLayer(nn.Module):
    """
    Input Convex Neural Network (ICNN) layer that maintains convexity.
    
    Maintains convexity by:
    1. Non-negative weights from previous layer
    2. Skip connections from input with unrestricted weights
    3. Smooth convex activation functions (only softplus and linear allowed with relu ok since second derivative is continuous, i.e. 0)
    
    This is essential for learning log normalizers A(η) in exponential families
    where A(η) must be convex to ensure valid probability distributions.
    
    Note: Only smooth convex activations are allowed to avoid numerical issues
    with discontinuous second derivatives. Allowed activations: "softplus", "linear", "relu"
    """
    
    features: int  # Output dimension
    use_bias: bool = True
    activation: str = "softplus"  # Only smooth convex activations allowed
    weight_scale: float = 1.0  # Scale factor for weight initialization (default: 1.0)
    
    @nn.compact
    def __call__(self, z_prev: Optional[jnp.ndarray], x: jnp.ndarray, 
                 training: bool = True) -> jnp.ndarray:
        """
        Forward pass through convex layer.
        
        Args:
            z_prev: Output from previous layer [batch_size, prev_features] or None
            x: Original input (skip connection) [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            Layer output [batch_size, features]
        """
        # Handle previous layer connection (if exists)
        if z_prev is not None:
            # Compute proper scaling factor: 1/sqrt(z_prev.shape[-1])
#            scale_factor = self.weight_scale/jnp.sqrt(z_prev.shape[-1])
            scale_factor = self.weight_scale/z_prev.shape[-1]
            
            # Non-negative weights for convexity
            W_z = self.param('W_z', 
                           nn.initializers.uniform(scale=1.0), 
                           (z_prev.shape[-1], self.features))
            W_z = W_z + 0.5  # Positive bias
            W_z_processed = nn.softplus(W_z) * scale_factor  # Ensure non-negative and scale
            z_term = z_prev @ W_z_processed
        else:
            z_term = 0.0
        
        # Skip connection from input (unrestricted weights)
        W_x = self.param('W_x',
                        nn.initializers.lecun_normal(),
                        (x.shape[-1], self.features))
        x_term = x @ W_x
        
        # Bias term
        if self.use_bias:
            b = self.param('b', nn.initializers.zeros, (self.features,))
            output = z_term + x_term + b
        else:
            output = z_term + x_term

        # Apply activation function
        return self._apply_activation(output)


    def _apply_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply activation function."""
        if self.activation == "softplus":
            return nn.softplus(x)
        elif self.activation == "relu":
            return nn.relu(x)  # No activation
        elif self.activation == "linear":
            return x  # No activation
        else:
            # Default to softplus (smooth convex activation)
            return nn.softplus(x)



class ICNNBlock(nn.Module):
    """
    Input Convex Neural Network block with multiple convex layers.
    
    This block implements a complete ICNN structure with proper convexity
    constraints for learning log normalizers in exponential families.
    """
    
    features: int  # Output dimension
    hidden_sizes: tuple[int, ...]  # Hidden layer sizes
    activation: str = "softplus"  # Only smooth convex activations allowed
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through ICNN block.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch_size, features]
        """
        z_prev = None
        # Apply convex layers
        for i, hidden_size in enumerate(self.hidden_sizes):
            z_prev = ConvexHiddenLayer(
                features=hidden_size,
                use_bias=self.use_bias,
                activation=self.activation,
                name=f'icnn_layer_{i}'
            )(z_prev, x, training=training)
                    
        # Final output layer - use simple mean to maintain convexity
        if z_prev.shape[-1] != self.features:
            # Instead of Dense layer, use mean to maintain convexity
            # This assumes we want to reduce dimension by averaging
            if z_prev.shape[-1] > self.features:
                # Mean groups of features to reduce dimension
                groups = z_prev.shape[-1] // self.features
                remainder = z_prev.shape[-1] % self.features
                
                if remainder == 0:
                    # Perfect division - reshape and mean
                    z_prev = z_prev.reshape(z_prev.shape[:-1] + (self.features, groups))
                    z_prev = jnp.mean(z_prev, axis=-1)
                else:
                    # Not perfect division - pad and then mean
                    pad_size = self.features - remainder
                    z_prev = jnp.pad(z_prev, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
                    z_prev = z_prev.reshape(z_prev.shape[:-1] + (self.features, groups + 1))
                    z_prev = jnp.mean(z_prev, axis=-1)
            else:
                # Expand dimension by repeating features
                repeat_factor = self.features // z_prev.shape[-1]
                remainder = self.features % z_prev.shape[-1]
                
                if remainder == 0:
                    # Perfect expansion
                    z_prev = jnp.repeat(z_prev, repeat_factor, axis=-1)
                else:
                    # Not perfect expansion - repeat and pad
                    z_prev = jnp.repeat(z_prev, repeat_factor, axis=-1)
                    z_prev = jnp.pad(z_prev, ((0, 0), (0, remainder)), mode='constant', constant_values=0)
        
        return z_prev


class ConvexResNet(nn.Module):
    """
    ResNet wrapper specifically designed for convex blocks (SimpleConvexBlock, ICNNBlock).
    
    This wrapper creates multiple convex blocks with residual connections, maintaining
    convexity properties throughout the network. Each block is independently initialized
    to avoid parameter sharing issues.
    
    Example:
        # Create a convex ResNet with 3 blocks
        convex_resnet = ConvexResNetWrapper(
            features=64,
            hidden_sizes=(32, 64, 32),
            num_blocks=3,
            activation='softplus'
        )
        
        # Forward pass
        output = convex_resnet.apply(params, x, training=True)
    """
    
    features: int  # Output dimension for each block
    hidden_sizes: tuple[int, ...]  # Hidden layer sizes for each block
    num_blocks: int = 3  # Number of convex blocks to create
    activation: str = "softplus"  # Only smooth convex activations allowed
    use_bias: bool = True  # Whether to use bias terms
    weight_scale: float = 1.0  # Weight scaling factor
    use_projection: bool = True  # Whether to use projection layers for dimension mismatch
    block_type: str = "simple"  # Type of block: "simple" or "icnn"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the convex ResNet wrapper.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch_size, features]
        """
        current_input = x
        
        # Apply each convex block
        for i in range(self.num_blocks):
            # Create convex block for each iteration (avoids parameter sharing)
            convex_block = ICNNBlock(
                features=self.features,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                use_bias=self.use_bias,
                name=f'icnn_block_{i}'
            )
            # Apply the convex block
            output = convex_block(current_input, training=training)
            
            # Handle residual connection
            if self.use_projection and output.shape[-1] != current_input.shape[-1]:
                # Project residual to match output dimension
                projected_residual = nn.Dense(output.shape[-1], name=f'residual_proj_{i}')(current_input)
                current_input = output + projected_residual
            else:
                # Direct residual connection (assumes same output dimension)
                current_input = output + current_input
        
        return current_input


class ConvexResNetBivariate(nn.Module):
    """
    ResNet wrapper for convex blocks that take two inputs (x, y).
    
    This wrapper creates multiple convex blocks with residual connections with respect to x,
    maintaining convexity properties throughout the network. Each block is independently
    initialized to avoid parameter sharing issues.
    
    The output shape will always match x.shape, making it residual with respect to x.
    
    Example:
        # Create a bivariate convex ResNet with 3 blocks
        convex_resnet = ConvexResNetWrapperBivariate(
            features=64,
            hidden_sizes=(32, 64, 32),
            num_blocks=3,
            activation='softplus'
        )
        
        # Forward pass
        output = convex_resnet.apply(params, x, y, training=True)  # output.shape == x.shape
    """
    
    features: int  # Output dimension for each block
    hidden_sizes: tuple[int, ...]  # Hidden layer sizes for each block
    num_blocks: int = 3  # Number of convex blocks to create
    activation: str = "softplus"  # Only smooth convex activations allowed
    use_bias: bool = True  # Whether to use bias terms
    weight_scale: float = 1.0  # Weight scaling factor
    use_projection: bool = True  # Whether to use projection layers for dimension mismatch
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the bivariate convex ResNet wrapper.
        
        Args:
            x: First input tensor [batch_size, input_dim] (residual connection applied to this)
            y: Second input tensor [batch_size, y_dim]
            training: Whether in training mode
            
        Returns:
            Output tensor with shape matching x.shape
        """
        current_x = x
        
        # Apply each convex block
        for i in range(self.num_blocks):
            # Create convex block for each iteration (avoids parameter sharing)

            convex_block = ICNNBlock(
                features=self.features,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                use_bias=self.use_bias,
                name=f'icnn_block_{i}'
            )

            # Apply the convex block (note: convex blocks only take x input, not (x, y))
            # For bivariate case, we need to handle this differently
            # We'll use a simple approach: apply the block to x and add a linear transformation of y
            output = convex_block(current_x, training=training)
            
            # Add contribution from y
            y_contribution = nn.Dense(self.features, name=f'y_contribution_{i}')(y)
            output = output + y_contribution
            
            # Handle residual connection with respect to x
            if self.use_projection and output.shape[-1] != current_x.shape[-1]:
                # Project residual to match output dimension
                projected_residual = nn.Dense(output.shape[-1], name=f'residual_proj_{i}')(current_x)
                current_x = output + projected_residual
            else:
                # Direct residual connection (assumes same output dimension as x)
                current_x = output + current_x
        
        return current_x


# Convenience functions for creating convex ResNet wrappers
def create_convex_resnet(features: int,
                                hidden_sizes: tuple[int, ...],
                                num_blocks: int = 3,
                                activation: str = "softplus",
                                use_bias: bool = True,
                                weight_scale: float = 1.0,
                                use_projection: bool = True,
                                block_type: str = "simple") -> ConvexResNet:
    """
    Create a convex ResNet wrapper.
    
    Args:
        features: Output dimension for each block
        hidden_sizes: Hidden layer sizes for each block
        num_blocks: Number of convex blocks to create
        activation: Activation function
        use_bias: Whether to use bias terms
        weight_scale: Weight scaling factor
        use_projection: Whether to use projection layers for dimension mismatch
        block_type: Type of block ("simple" or "icnn")
        
    Returns:
        ConvexResNetWrapper instance
    """
    return ConvexResNet(
        features=features,
        hidden_sizes=hidden_sizes,
        num_blocks=num_blocks,
        activation=activation,
        use_bias=use_bias,
        weight_scale=weight_scale,
        use_projection=use_projection,
        block_type=block_type
    )


def create_convex_resnet_bivariate(features: int,
                                          hidden_sizes: tuple[int, ...],
                                          num_blocks: int = 3,
                                          activation: str = "softplus",
                                          use_bias: bool = True,
                                          weight_scale: float = 1.0,
                                          use_projection: bool = True,
                                          block_type: str = "simple") -> ConvexResNetBivariate:
    """
    Create a bivariate convex ResNet wrapper.
    
    Args:
        features: Output dimension for each block
        hidden_sizes: Hidden layer sizes for each block
        num_blocks: Number of convex blocks to create
        activation: Activation function
        use_bias: Whether to use bias terms
        weight_scale: Weight scaling factor
        use_projection: Whether to use projection layers for dimension mismatch
        block_type: Type of block ("simple" or "icnn")
        
    Returns:
        ConvexResNetWrapperBivariate instance
    """
    return ConvexResNetBivariate(
        features=features,
        hidden_sizes=hidden_sizes,
        num_blocks=num_blocks,
        activation=activation,
        use_bias=use_bias,
        weight_scale=weight_scale,
        use_projection=use_projection,
        block_type=block_type
    )
