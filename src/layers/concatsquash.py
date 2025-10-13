"""
ConcatSquash layer implementation for Flax.

A ConcatSquash layer is a memory-efficient alternative to simple concatenation that:
1. Concatenates multiple inputs along the last dimension
2. Applies a "squash" operation (typically a linear transformation) to compress the result
3. Optionally applies activation and normalization

This is commonly used in neural ODEs and flow-based models where you need to combine
multiple inputs efficiently without creating large intermediate tensors.
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional

from numpy import False_


class ConcatSquash(nn.Module):
    """
    ConcatSquash layer that concatenates inputs and then compresses them.
    
    This layer takes multiple input tensors, concatenates them along the last dimension,
    and then applies a linear transformation to "squash" the result to a smaller dimension.
    
    Args:
        features: Output dimension after squashing
        activation: Activation function to apply after squashing (default: None)
        use_bias: Whether to use bias in the squash layer (default: True)
        use_input_layer_norm: Whether to apply layer normalization to each input before projection (default: False)
    """
    
    features: int
    use_bias: bool = False_
    use_input_layer_norm: bool = False
    
    @nn.compact
    def __call__(self, *inputs: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Apply ConcatSquash transformation to multiple inputs.
        
        Args:
            *inputs: Variable number of input tensors to concatenate and squash
            training: Whether in training mode (affects dropout)
            
        Returns:
            Squashed output tensor with shape (..., output_dim)
        """
        if not inputs:
            raise ValueError("At least one input tensor must be provided")
        
        # Check that all inputs have compatible batch shapes
        output = 0.0 
        for i, input in enumerate(inputs):
            if self.use_input_layer_norm:
                input = nn.LayerNorm()(input)
            output += nn.Dense(self.features, use_bias=False, name=f'input_proj_{i}')(input)

        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            output += bias
        return output


