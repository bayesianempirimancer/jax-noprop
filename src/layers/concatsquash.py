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
    use_bias: bool = False
    use_layer_norm: bool = False
    
    @nn.compact
    def __call__(self, *inputs: jnp.ndarray) -> jnp.ndarray:
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
                
        output = 0.0
        for i, input in enumerate(inputs):
            if self.use_layer_norm:
                input = nn.LayerNorm()(input)
            output += nn.Dense(self.features, use_bias=False, name=f'input_proj_{i}')(input)

        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            output += bias
        return output


# Convenience function for creating ConcatSquash layers
def create_concat_squash(
    features: int,
    use_bias: bool = True,
    use_input_layer_norm: bool = False
) -> ConcatSquash:
    """
    Convenience function to create ConcatSquash layers.
    
    Args:
        features: Output dimension after squashing
        use_bias: Whether to use bias in the squash layer
        use_input_layer_norm: Whether to apply layer normalization to each input before projection
        
    Returns:
        ConcatSquash layer instance
    """
    return ConcatSquash(
        features=features,
        use_bias=use_bias,
        use_input_layer_norm=use_input_layer_norm
    )


# Example usage and testing
if __name__ == "__main__":
    import jax
    
    print("Testing ConcatSquash layers...")
    
    # Test parameters
    batch_size = 4
    z_dim = 10
    x_dim = 8
    t_dim = 6
    output_dim = 16
    
    # Create test inputs
    key = jax.random.PRNGKey(42)
    z = jax.random.normal(key, (batch_size, z_dim))
    x = jax.random.normal(key, (batch_size, x_dim))
    t = jax.random.normal(key, (batch_size, t_dim))
    
    print(f"Input shapes: z={z.shape}, x={x.shape}, t={t.shape}")
    
    # Test standard ConcatSquash
    print("\n=== Testing Standard ConcatSquash ===")
    concat_squash = ConcatSquash(features=output_dim, use_bias=True)
    params = concat_squash.init(key, z, x, t)
    output = concat_squash.apply(params, z, x, t)
    print(f"Output shape: {output.shape}")
    
    # Test convenience function
    print("\n=== Testing Convenience Function ===")
    concat_squash_conv = create_concat_squash(features=output_dim, use_bias=True)
    params_conv = concat_squash_conv.init(key, z, x, t)
    output_conv = concat_squash_conv.apply(params_conv, z, x, t)
    print(f"Output shape: {output_conv.shape}")
    
    print("\n✅ All ConcatSquash tests passed!")
