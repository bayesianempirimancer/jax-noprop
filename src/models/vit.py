import math
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import field

from ..layers import Identity, PatchEmbed, ViTBlock
from ..layers.configs import ViTBlockConfig

# TODO: pos_drop
# TODO: compare to prepare_tokens_with_masks


class DinoV2(nn.Module):
    """
    Implementation of the DinoV2 (Vision Transformer) model.

    This class implements the DinoV2 architecture, which is a variant of the Vision Transformer
    designed for self-supervised learning tasks.

    Args:
        img_size (int): Size of the input image (assumed to be square).
        in_channels (int): Number of input channels.
        patch_size (int): Size of the patches to be extracted from the input image.
        embed_dim (int): Dimensionality of the token embeddings.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        qk_norm (bool): If True, normalize the query and key.
        ffn_bias (bool): If True, use bias in the feed-forward network.
        proj_bias (bool): If True, use bias in the projection layers.
        drop_path_rate (float):  Stochastic depth rate.
        drop_path_uniform (bool): If True, use a uniform drop rate across layers.
        class_token (bool): If True, add a class token.
        reg_tokens (int): Number of register tokens to use.
        pos_embed (str): Type of positional embedding to use.
        no_embed_class (bool): If True, don't add positional embedding to class token.
        pos_embed_reg_tokens (bool): If True, add positional embedding to register tokens.
        dynamic_img_size (bool): If True, allow dynamic image sizes.
        dynamic_img_pad (bool): If True, use dynamic padding for images.
        embed_layer (nn.Module): Module to use for patch embedding.
        act_layer (str): Activation function to use.
        block (nn.Module): Module to use for transformer blocks.
        attention (str): Module to use for attention mechanism.
        ffn_layer (str): Module to use for feed-forward network.
        init_values (float | None): Initial value for layer scale.
        interpolate_antialias (bool): If True, use antialiasing when interpolating.
    """

    # Model configuration
    img_size: int = 224
    in_channels: int = 3
    patch_size: int = 14
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_norm: bool = False
    ffn_bias: bool = True
    proj_bias: bool = True
    drop_path_rate: float = 0.0
    drop_path_uniform: bool = True
    class_token: bool = True
    reg_tokens: int = 1
    pos_embed: str = "learn"
    no_embed_class: bool = False
    pos_embed_reg_tokens: bool = False
    dynamic_img_size: bool = False
    dynamic_img_pad: bool = False
    embed_layer: nn.Module = PatchEmbed
    act_layer: str = "gelu"
    block: nn.Module = ViTBlock
    attention: str = "attention"
    ffn_layer: str = "mlp"
    init_values: float | None = None
    interpolate_antialias: bool = False


    def resample_pos_embed(
        self,
        pos_embed: jnp.ndarray,
        new_size: Tuple[int],
        old_size: Tuple[int] = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
        num_embedded_prefix_tokens: int = 0,
        num_prefix_tokens: int = 0,
        num_patches: int = 0,
        embed_dim: int = 768,
    ):
        """
        Resample the positional embeddings to a new size.

        Args:
            pos_embed (jnp.ndarray): The current positional embeddings.
            new_size (Tuple[int]): The new size to resample to.
            old_size (Tuple[int], optional): The old size of the positional embeddings.
            interpolation (str, optional): The interpolation method to use. Defaults to "bicubic".
            antialias (bool, optional): Whether to use antialiasing. Defaults to True.
            num_embedded_prefix_tokens (int): Number of embedded prefix tokens.
            num_prefix_tokens (int): Number of prefix tokens.
            num_patches (int): Number of patches.
            embed_dim (int): Embedding dimension.

        Returns:
            jnp.ndarray: The resampled positional embeddings.
        """
        previous_dtype = pos_embed.dtype

        num_new_tokens = new_size[0] * new_size[1] + num_embedded_prefix_tokens
        embed_len = num_patches + num_embedded_prefix_tokens

        if num_new_tokens == embed_len and new_size[0] == new_size[1]:
            return pos_embed

        if old_size is None:
            hw = int(math.sqrt(num_patches))
            old_size = hw, hw

        prefix_embed = (
            pos_embed[:, : num_prefix_tokens] if num_prefix_tokens else None
        )
        pos_embed = pos_embed[:, num_prefix_tokens :]

        pos_embed = pos_embed.astype("float32")
        pos_embed = jnp.reshape(
            pos_embed, (1, old_size[0], old_size[1], embed_dim)
        )

        pos_embed = jax.image.resize(
            pos_embed,
            (1, new_size[0], new_size[1], embed_dim),
            method=interpolation,
            antialias=antialias,
        )
        pos_embed = pos_embed.reshape(1, -1, embed_dim).astype(previous_dtype)

        if prefix_embed is not None:
            pos_embed = jnp.concatenate([prefix_embed, pos_embed], axis=1)

        return pos_embed

    def _pos_embed(self, x: jnp.ndarray, h: int, w: int, 
                   pos_embed: jnp.ndarray = None,
                   cls_token: jnp.ndarray = None,
                   register_tokens: jnp.ndarray = None,
                   num_register_tokens: int = 0,
                   no_embed_class: bool = False,
                   pos_embed_reg_tokens: bool = False,
                   dynamic_img_size: bool = False,
                   interpolate_antialias: bool = False,
                   num_embedded_prefix_tokens: int = 0,
                   num_prefix_tokens: int = 0,
                   num_patches: int = 0,
                   embed_dim: int = 768):
        """
        Apply positional embedding to the input.

        Args:
            x (jnp.ndarray): The input tensor.
            h (int): Height of the input.
            w (int): Width of the input.
            pos_embed (jnp.ndarray): Positional embeddings.
            cls_token (jnp.ndarray): Class token.
            register_tokens (jnp.ndarray): Register tokens.
            num_register_tokens (int): Number of register tokens.
            no_embed_class (bool): Whether to not embed class token.
            pos_embed_reg_tokens (bool): Whether to embed register tokens.
            dynamic_img_size (bool): Whether to use dynamic image size.
            interpolate_antialias (bool): Whether to use antialiasing.
            num_embedded_prefix_tokens (int): Number of embedded prefix tokens.
            num_prefix_tokens (int): Number of prefix tokens.
            num_patches (int): Number of patches.
            embed_dim (int): Embedding dimension.

        Returns:
            jnp.ndarray: The input with positional embeddings applied.
        """
        if pos_embed is None:
            return jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

        if dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = self.resample_pos_embed(
                pos_embed, new_size=(H, W), antialias=interpolate_antialias,
                num_embedded_prefix_tokens=num_embedded_prefix_tokens,
                num_prefix_tokens=num_prefix_tokens,
                num_patches=num_patches,
                embed_dim=embed_dim
            )
            x = jnp.reshape(x, (B, -1, C))

        to_cat = []
        if cls_token is not None:
            # Broadcast cls_token to match batch size
            expanded_cls_token = jnp.broadcast_to(
                cls_token, (x.shape[0], 1, cls_token.shape[-1])
            )
            to_cat.append(expanded_cls_token)

        if register_tokens is not None:
            expanded_register_tokens = jnp.broadcast_to(
                register_tokens,
                (x.shape[0], num_register_tokens, register_tokens.shape[-1]),
            )
            to_cat.append(expanded_register_tokens)

        if no_embed_class:
            x = x + pos_embed
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=1)
        elif pos_embed_reg_tokens:
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=1)
            x = x + pos_embed
        else:
            x = jnp.concatenate(to_cat[:1] + [x], axis=1)
            x = x + pos_embed
            if register_tokens is not None:
                x = jnp.concatenate([x[:, :1], to_cat[1], x[:, 1:]], axis=1)

        return x

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the DinoV2 model.

        Args:
            x (jnp.ndarray): The input tensor of shape (batch_size, height, width, channels).

        Returns:
            jnp.ndarray: The output of the model.
        """
        # Validate pos_embed parameter
        if self.pos_embed not in ("", "none", "learn"):
            raise AssertionError("pos_embed must be '', 'none', or 'learn'")

        # Calculate derived parameters
        num_prefix_tokens = (1 if self.class_token else 0) + self.reg_tokens
        num_embedded_prefix_tokens = 0
        num_register_tokens = self.reg_tokens
        num_patches = (self.img_size // self.patch_size) ** 2

        # Calculate embedding length
        if self.no_embed_class:
            embed_len = num_patches
        elif self.pos_embed_reg_tokens:
            embed_len = num_patches + num_prefix_tokens
            num_embedded_prefix_tokens += num_prefix_tokens
        else:
            num_embedded_prefix_tokens += 1
            embed_len = num_patches + 1

        # Create patch embedding
        patch_embed = self.embed_layer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            flatten=not self.dynamic_img_size,
            dynamic_img_size=self.dynamic_img_size,
            dynamic_img_pad=self.dynamic_img_pad,
        )
        x = patch_embed(x)

        # Create class token
        cls_token = None
        if self.class_token:
            cls_token = self.param('cls_token', nn.initializers.zeros, (1, 1, self.embed_dim))

        # Create register tokens
        register_tokens = None
        if self.reg_tokens:
            register_tokens = self.param('register_tokens', nn.initializers.zeros, 
                                       (1, self.reg_tokens, self.embed_dim))

        # Create positional embedding
        pos_embed = None
        if self.pos_embed and self.pos_embed != "none":
            pos_embed = self.param('pos_embed', nn.initializers.normal(0.02), 
                                 (1, embed_len, self.embed_dim))

        # Stochastic depth decay rule
        if self.drop_path_uniform:
            dpr = [self.drop_path_rate] * self.depth
        else:
            dpr = list(jnp.linspace(0, self.drop_path_rate, self.depth))

        # Create transformer blocks
        for i in range(self.depth):
            block = self.block(
                dim=self.embed_dim,
                config=ViTBlockConfig(
                    mlp_ratio=self.mlp_ratio,
                    drop_path=dpr[i],
                    act_layer=self.act_layer,
                    attention=self.attention,
                    ffn_layer=self.ffn_layer,
                    ffn_bias=self.ffn_bias,
                    init_values=self.init_values,
                ),
                attention_kwargs={
                    "num_heads": self.num_heads,
                    "qkv_bias": self.qkv_bias,
                    "qk_norm": self.qk_norm,
                    "proj_bias": self.proj_bias,
                },
            )
            x = block(x)

        # Apply positional embedding
        # After patch embedding, x is 3D: (batch, sequence_length, embed_dim)
        # We need to get the original spatial dimensions for positional embedding
        N, L, C = x.shape
        H = W = int((L - (1 if self.class_token else 0) - self.reg_tokens) ** 0.5)
        x = self._pos_embed(
            x, h=H, w=W,
            pos_embed=pos_embed,
            cls_token=cls_token,
            register_tokens=register_tokens,
            num_register_tokens=num_register_tokens,
            no_embed_class=self.no_embed_class,
            pos_embed_reg_tokens=self.pos_embed_reg_tokens,
            dynamic_img_size=self.dynamic_img_size,
            interpolate_antialias=self.interpolate_antialias,
            num_embedded_prefix_tokens=num_embedded_prefix_tokens,
            num_prefix_tokens=num_prefix_tokens,
            num_patches=num_patches,
            embed_dim=self.embed_dim
        )

        # Apply normalization
        norm = nn.LayerNorm()
        x = norm(x)

        # Apply head (Identity by default)
        # In flax.linen, we need to call the module directly within the compact context
        x = Identity()(x)

        return x
