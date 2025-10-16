import jax.numpy as jnp
import flax.linen as nn


class Identity(nn.Module):
    """An identity module that returns the input unchanged."""

    @nn.compact
    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        """Apply the identity operation.

        Args:
            x (jnp.ndarray): Input array.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            jnp.ndarray: The input array unchanged.
        """
        return x


class Downsample(nn.Module):
    """Downsampling block for reducing spatial dimensions of feature maps."""

    keep_dim: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, dim: int) -> jnp.ndarray:
        """Apply downsampling to the input.

        Args:
            x (jnp.ndarray): Input tensor of shape (B, H, W, C).
            dim (int): Number of input channels.

        Returns:
            jnp.ndarray: Downsampled tensor of shape (B, H/2, W/2, C) or (B, H/2, W/2, 2C).
        """
        dim_out = dim if self.keep_dim else 2 * dim
        return nn.Conv(
            features=dim_out,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
        )(x)
