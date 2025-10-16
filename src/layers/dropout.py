import jax
import jax.numpy as jnp
import flax.linen as nn


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    This module applies the DropPath regularization technique, which randomly drops
    entire paths (channels) in residual networks during training.

    Attributes:
        scale_by_keep (bool): Whether to scale the kept values.
        deterministic (bool): Whether to use deterministic behavior.
    """

    scale_by_keep: bool = True
    deterministic: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        drop_prob: float,
        *,
        deterministic: bool | None = None,
    ) -> jnp.ndarray:
        """
        Apply DropPath to the input.

        Args:
            x (jnp.ndarray): Input array of shape (B, ...).
            drop_prob (float): Probability of dropping a path.
            deterministic (bool | None, optional): Override for deterministic behavior.

        Returns:
            jnp.ndarray: Output after applying DropPath, same shape as input.
        """
        deterministic = deterministic if deterministic is not None else self.deterministic

        if (drop_prob == 0.0) or deterministic:
            return x

        # Prevent gradient NaNs in 1.0 edge-case.
        if drop_prob == 1.0:
            return jnp.zeros_like(x)

        keep_prob = 1.0 - drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = jax.random.bernoulli(self.make_rng('dropout'), p=keep_prob, shape=shape)

        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor /= keep_prob

        return x * random_tensor
