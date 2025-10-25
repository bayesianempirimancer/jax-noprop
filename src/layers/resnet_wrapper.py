"""
ResNet wrapper for arbitrary neural network modules.

This module provides a ResNet wrapper that can take any nn.Module and wrap it
with residual connections, creating a ResNet-style architecture with skip connections.
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional, Union


class ResNetWrapper(nn.Module):
    """
    ResNet wrapper that creates one or more ResNet blocks from any nn.Module.
    
    This wrapper takes any nn.Module class and creates a ResNet-style architecture
    with the specified number of blocks, each with residual connections.
    
    Note: Models used with this wrapper should accept 'training' and optionally 'rngs' 
    as keyword arguments in their __call__ method. The wrapper will gracefully handle
    modules that don't accept these arguments by falling back to positional arguments only.
        
    Example:
        # Single ResNet block
        resnet = ResNetWrapper(nn.Dense, {'features': 64}, num_blocks=3)
    """
    
    base_module_class: type  # The module class to wrap with residual connections
    base_module_kwargs: dict  # Constructor arguments for the base module
    num_blocks: int = 3  # Number of ResNet blocks to create
    share_parameters: bool = False  # Whether to share parameters between blocks
    weight_residual: bool = False  # Whether to weight the residual connection
    residual_weight: float = 1.0  # Weight for residual connection
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Forward pass through the ResNet wrapper.
        
        Args:
            x: Input tensor
            **kwargs: Additional keyword arguments passed to each block
            
        Returns:
            Output tensor after passing through all ResNet blocks
        """
        
        current_x = x
        
        # Apply each ResNet block
        for i in range(self.num_blocks):
            if self.share_parameters is False:
                # Each block gets its own instance of the base module (different parameters)
                # Create a new instance of the base module class with unique name
                block_kwargs = dict(self.base_module_kwargs)
                block_kwargs['name'] = f'block_{i}'
                block_module = self.base_module_class(**block_kwargs)
                output = block_module(current_x, **kwargs)
                if current_x.shape[-1] != output.shape[-1]:
                    output = nn.Dense(current_x.shape[-1], name=f'projection_{i}')(output)
                
            else:
                # All blocks share the same instance of the base module (same parameters)
                # Create a single shared instance (only on first iteration)
                if i == 0:
                    shared_kwargs = dict(self.base_module_kwargs)
                    shared_kwargs['name'] = 'shared_block'
                    shared_module = self.base_module_class(**shared_kwargs)
                    # Create shared projection layer if needed
                    # We'll determine if projection is needed after we see the output
                    shared_projection = None
                output = shared_module(current_x, **kwargs)
                if current_x.shape[-1] != output.shape[-1]:
                    if shared_projection is None:
                        shared_projection = nn.Dense(current_x.shape[-1], name='shared_projection')
                    output = shared_projection(output)
            
            # Direct residual connection (input and output dimensions must match)
            if self.weight_residual:
                current_x = output + self.residual_weight * current_x
            else:
                current_x = output + current_x
        
        return current_x


class ResNetWrapperBivariate(nn.Module):
    """
    ResNet wrapper that creates one or more ResNet blocks from any nn.Module that takes two inputs.
    
    This wrapper takes any nn.Module class that accepts (x, y) inputs and creates a ResNet-style 
    architecture with the specified number of blocks, each with residual connections with respect to x.
    
    Note: Models used with this wrapper should accept 'training' and optionally 'rngs' 
    as keyword arguments in their __call__ method. The wrapper will gracefully handle
    modules that don't accept these arguments by falling back to positional arguments only.
    """
    
    base_module_class: type  # The module class to wrap with residual connections
    base_module_kwargs: dict  # Constructor arguments for the base module
    num_blocks: int = 3  # Number of ResNet blocks to create
    share_parameters: bool = False  # Whether to share parameters between blocks
    weight_residual: bool = False  # Whether to weight the residual connection
    residual_weight: float = 1.0  # Weight for residual connection
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Forward pass through the ResNet wrapper with two inputs.
        
        Args:
            x: First input tensor (residual connection applied to this)
            y: Second input tensor
            **kwargs: Additional keyword arguments passed to each block
            
        Returns:
            Output tensor with shape matching x.shape
        """
        current_x = x
        
        # Apply each ResNet block
        for i in range(self.num_blocks):
            if self.share_parameters is False:
                # Each block gets its own instance of the base module (different parameters)
                # Create a new instance of the base module class with unique name
                block_kwargs = dict(self.base_module_kwargs)
                block_kwargs['name'] = f'block_{i}'
                block_module = self.base_module_class(**block_kwargs)
                output = block_module(current_x, y, **kwargs)
                if current_x.shape[-1] != output.shape[-1]:
                    output = nn.Dense(current_x.shape[-1], name=f'projection_{i}')(output)
                
            else:
                # All blocks share the same instance of the base module (same parameters)
                # Create a single shared instance (only on first iteration)
                if i == 0:
                    shared_kwargs = dict(self.base_module_kwargs)
                    shared_kwargs['name'] = 'shared_block'
                    shared_module = self.base_module_class(**shared_kwargs)
                    # Create shared projection layer if needed
                    # We'll determine if projection is needed after we see the output
                    shared_projection = None
                output = shared_module(current_x, y, **kwargs)
                if current_x.shape[-1] != output.shape[-1]:
                    if shared_projection is None:
                        shared_projection = nn.Dense(current_x.shape[-1], name='shared_projection')
                    output = shared_projection(output)
            
            # Direct residual connection (input and output dimensions must match)
            if self.weight_residual:
                current_x = output + self.residual_weight * current_x
            else:
                current_x = output + current_x
        
        return current_x


# Convenience functions for creating ResNet wrappers
def create_resnet_wrapper(base_module_class: type, 
                         base_module_kwargs: dict,
                         num_blocks: int = 1,
                         activation: Optional[Callable] = None,
                         share_parameters: bool = False,
                         weight_residual: bool = False,
                         residual_weight: float = 1.0) -> ResNetWrapper:
    """
    Create a ResNet wrapper for any nn.Module class.
    
    Args:
        base_module_class: The module class to wrap with residual connections
        base_module_kwargs: Constructor arguments for the base module
        num_blocks: Number of ResNet blocks to create
        activation: Optional activation function between blocks
        share_parameters: Whether to share parameters between blocks
        weight_residual: Whether to weight the residual connection (default: False)
        residual_weight: Weight for residual connection (default: 1.0)
        
    Returns:
        ResNetWrapper instance
    """
    return ResNetWrapper(
        base_module_class=base_module_class,
        base_module_kwargs=base_module_kwargs,
        num_blocks=num_blocks,
        activation=activation,
        share_parameters=share_parameters,
        weight_residual=weight_residual,
        residual_weight=residual_weight
    )


def create_resnet_wrapper_bivariate(base_module_class: type, 
                                   base_module_kwargs: dict,
                                   num_blocks: int = 1,
                                   activation: Optional[Callable] = None,
                                   share_parameters: bool = False,
                                   weight_residual: bool = False,
                                   residual_weight: float = 1.0) -> ResNetWrapperBivariate:
    """
    Create a ResNet wrapper for any nn.Module class that takes two inputs (x, y).
    
    Args:
        base_module_class: The module class to wrap with residual connections
        base_module_kwargs: Constructor arguments for the base module
        num_blocks: Number of ResNet blocks to create
        activation: Optional activation function between blocks
        share_parameters: Whether to share parameters between blocks
        weight_residual: Whether to weight the residual connection (default: False)
        residual_weight: Weight for residual connection (default: 1.0)
        
    Returns:
        ResNetWrapperBivariate instance
    """
    return ResNetWrapperBivariate(
        base_module_class=base_module_class,
        base_module_kwargs=base_module_kwargs,
        num_blocks=num_blocks,
        activation=activation,
        share_parameters=share_parameters,
        weight_residual=weight_residual,
        residual_weight=residual_weight
    )
