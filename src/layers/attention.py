import jax.numpy as jnp
from einops import rearrange, reduce
import flax.linen as nn

from .configs import AttentionConfig
from .norm import RMSNormGated
from .posemb import PosEmbMLPSwinv2D, RoPE


class Attention(nn.Module):
    """
    Multi-head Attention module.

    This module implements multi-head attention mechanism as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Args:
        config (AttentionConfig): Configuration object containing attention parameters.
    """

    config: AttentionConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the attention module.

        Args:
            x (jnp.ndarray): Input tensor of shape (B, N, C) where B is batch size,
                             N is sequence length, and C is input dimension.

        Returns:
            jnp.ndarray: Output tensor of shape (B, N, C).

        Raises:
            AssertionError: If input embedding dimension doesn't match layer embedding dimension.
        """
        B, N, C = x.shape
        if C != self.config.dim:
            raise AssertionError(
                f"Input embedding dimension ({C}) should match layer embedding dimension ({self.config.dim})."
            )

        qkv = nn.Dense(self.config.dim * 3, use_bias=self.config.qkv_bias, name='qkv')(x)
        qkv = jnp.reshape(
            qkv, (B, N, 3, self.config.num_heads, C // self.config.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tuple(qkv)
        if self.config.qk_norm:
            match self.config.norm_layer:
                case "layernorm":
                    q = nn.LayerNorm()(q)
                    k = nn.LayerNorm()(k)
                case "rmsnormgated":
                    q = RMSNormGated()(q)
                    k = RMSNormGated()(k)
                case "batchnorm":
                    q = nn.BatchNorm()(q)
                    k = nn.BatchNorm()(k)
                case _:
                    raise ValueError(f"Unknown norm `{self.config.norm_layer}`")

        # TODO: implement fused attention for better performance
        attn = q @ k.transpose((0, 1, 3, 2)) / jnp.sqrt(self.config.head_dim)
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.config.attn_drop)(attn, deterministic=False)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(self.config.dim, use_bias=self.config.proj_bias, name='proj')(x)
        x = nn.Dropout(self.config.proj_drop)(x, deterministic=False)

        return x


class LinearAttention(nn.Module):
    """Linear Attention from Mamba-like Linear Attention (MLLA) paper."""

    config: AttentionConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        b, n, c = x.shape
        h = int(n**0.5)
        w = int(n**0.5)
        num_heads = self.config.num_heads

        q, k = rearrange(nn.Dense(self.config.dim * 2)(x),
                         "b n (qk h d) -> qk b h n d",
                         qk=2,
                         h=num_heads)
        v = rearrange(x, "b n (h d) -> b h n d", h=num_heads)

        q = nn.elu(q) + 1.0
        k = nn.elu(k) + 1.0

        # TODO: Try to define rope here to avoid setting input_resolution a priori
        rope = RoPE(shape=(h, w, c))
        q_2d = rearrange(q, "b h (x y) d -> b x y (h d)", x=h, y=w)
        k_2d = rearrange(k, "b h (x y) d -> b x y (h d)", x=h, y=w)

        q_rope = rearrange(rope(q_2d),
                           "b x y (h d) -> b h (x y) d",
                           h=num_heads)
        k_rope = rearrange(rope(k_2d),
                           "b x y (h d) -> b h (x y) d",
                           h=num_heads)

        # Compute attention
        z = 1 / (jnp.einsum("bhnd,bhd->bhn", q,
                            reduce(k, "b h n d -> b h d", "mean")) + 1e-6)
        kv = jnp.einsum("bhnd,bhne->bhde", k_rope * (n**-0.5), v * (n**-0.5))
        x = jnp.einsum("bhnd,bhde->bhne", q_rope, kv) * z[..., None]

        # Reshape output
        x = rearrange(x, "b h n d -> b n (h d)")

        # Apply LePE
        v_2d = rearrange(v, "b h (x y) d -> b x y (h d)", x=h, y=w)
        lepe_out = nn.Conv(
            features=self.config.dim,
            kernel_size=(3, 3),
            padding=(1, 1),
            feature_group_count=self.config.dim,
        )(v_2d)

        lepe_out = rearrange(lepe_out,
                             "b x y (h d) -> b (x y) (h d)",
                             h=num_heads)

        # Combine attention output and LePE
        x = x + lepe_out

        return x


class WindowedAttention(nn.Module):
    """
    Windowed Attention module.

    This module implements Window attention based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention"

    Args:
        resolution (int): Resolution of the window.
        seq_len (int): Sequence length.
        config (AttentionConfig): Configuration object containing attention parameters.
    """

    resolution: int
    seq_len: int
    config: AttentionConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the attention module.

        Args:
            x (jnp.ndarray): Input tensor of shape (B, N, C) where B is batch size,
                             N is sequence length, and C is input dimension.

        Returns:
            jnp.ndarray: Output tensor of shape (B, N, C).

        Raises:
            AssertionError: If input embedding dimension doesn't match layer embedding dimension.
        """
        B, N, C = x.shape
        if C != self.config.dim:
            raise AssertionError(
                f"Input embedding dimension ({C}) should match layer embedding dimension ({self.config.dim})."
            )

        qkv = nn.Dense(self.config.dim * 3, use_bias=self.config.qkv_bias, name='qkv')(x)
        qkv = jnp.reshape(
            qkv, (B, N, 3, self.config.num_heads, C // self.config.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tuple(qkv)
        if self.config.qk_norm:
            match self.config.norm_layer:
                case "layernorm":
                    q = nn.LayerNorm()(q)
                    k = nn.LayerNorm()(k)
                case "rmsnormgated":
                    q = RMSNormGated()(q)
                    k = RMSNormGated()(k)
                case "batchnorm":
                    q = nn.BatchNorm()(q)
                    k = nn.BatchNorm()(k)
                case _:
                    raise ValueError(f"Unknown norm `{self.config.norm_layer}`")

        # TODO: implement fused attention for better performance
        attn = q @ k.transpose((0, 1, 3, 2)) / jnp.sqrt(self.config.head_dim)
        
        # Attention positional bias
        pos_emb_funct = PosEmbMLPSwinv2D(
            window_size=[self.resolution, self.resolution],
            pretrained_window_size=[self.resolution, self.resolution],
            num_heads=self.config.num_heads,
            seq_len=self.seq_len,
        )
        attn = pos_emb_funct(attn, self.resolution**2)
        
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.config.attn_drop)(attn, deterministic=False)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(self.config.dim, use_bias=self.config.proj_bias, name='proj')(x)
        x = nn.Dropout(self.config.proj_drop)(x, deterministic=False)

        return x
