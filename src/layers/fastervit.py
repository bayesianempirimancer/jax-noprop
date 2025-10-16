from typing import Callable
from dataclasses import field

import jax.numpy as jnp
from einops import rearrange
import flax.linen as nn
from flax.linen import avg_pool

from .blocks import HATBlock
from .configs import ViTBlockConfig
from .misc import Downsample


class TokenInitializer(nn.Module):
    """
    Carrier token Initializer based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    dim: int
    input_resolution: tuple
    window_size: int
    ct_size: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Args:
            x: input tensor
        """
        output_size = int((self.ct_size) * self.input_resolution / self.window_size)
        strides = int(self.input_resolution / output_size)
        kernel = self.input_resolution - (output_size - 1) * strides

        x = nn.Conv(
            features=self.dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            feature_group_count=self.dim,
        )(x)
        
        x = avg_pool(
            x,
            window_shape=(kernel, kernel),
            strides=(strides, strides),
        )

        ct = rearrange(
            x,
            "b (h h1) (w w1) c -> b (h h1 w w1) c",
            h1=self.ct_size,
            w1=self.ct_size,
        )
        return ct


# TODO: merge with GenericLayer?
class FasterViTLayer(nn.Module):
    """
    FasterViT layer for vision models.

    This class implements a FasterViT layer that implementing Hierarchical
    Attention (HAT).

    Attributes:
        dim (int): Number of input channels.
        depth (int): Number of blocks in the layer.
        input_resolution (int): Input resolution.
        window_size (int): Size of the window for windowed attention.
        block (nn.Module): Block module to use (e.g., ViTBlock or ConvBlock).
        block_config (Callable): Configuration for the block.
        msa_window_size (int): Size of the multi-head self-attention window.
        drop_path (float | list): Stochastic depth rate.
        downsample (bool): Whether to apply downsampling after the blocks.
        block_kwargs (dict): Additional keyword arguments for blocks.
        config_kwargs (dict): Additional keyword arguments for block configuration.
    """

    dim: int
    depth: int
    input_resolution: int
    window_size: int
    block: nn.Module = HATBlock
    block_config: Callable = ViTBlockConfig
    msa_window_size: int = -1
    drop_path: float | list = 0.0
    downsample: bool = True
    block_kwargs: dict = field(default_factory=dict)
    config_kwargs: dict = field(default_factory=dict)
    downsampler: nn.Module = Downsample
    only_local: bool = False
    hierarchy: bool = True
    ct_size: int = 1

    def window_partition(self, x: jnp.ndarray, window_size: int):
        return rearrange(x,
                         "b (h h1) (w w1) c -> (b h w) (h1 w1) c",
                         h1=window_size,
                         w1=window_size)

    def window_reverse(self, x: jnp.ndarray, window_size: int, h: int, w: int):
        return rearrange(
            x,
            "(b h w) (h1 w1) c -> b (h h1) (w w1) c",
            h=h // window_size,
            w=w // window_size,
            h1=window_size,
            w1=window_size,
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        is_hat = self.block is HATBlock
        b, h, w, c = x.shape

        # Create global tokenizer if needed
        do_gt = (not self.only_local
                and self.input_resolution // self.window_size > 1 
                and self.hierarchy
                and is_hat)
        
        if do_gt:
            global_tokenizer = TokenInitializer(
                dim=self.dim,
                input_resolution=self.input_resolution,
                window_size=self.window_size,
                ct_size=self.ct_size,
            )
            ct = global_tokenizer(x)
        else:
            ct = None

        if is_hat:
            x = self.window_partition(x, self.window_size)
            for i in range(self.depth):
                cfg = self.config_kwargs | {
                    "drop_path":
                    self.drop_path[i] if isinstance(self.drop_path, list) else self.drop_path,
                }
                cfg = {
                    k: v
                    for k, v in cfg.items() if k in self.block_config.__dict__.keys()
                }

                block_kwargs = self.block_kwargs | {
                    "sr_ratio": (self.input_resolution //
                                 self.window_size if not self.only_local else 1),
                    "window_size": self.window_size,
                    "last": i == self.depth - 1,
                    "ct_size": self.ct_size,
                }

                x, ct = self.block(
                    dim=self.dim,
                    config=self.block_config(**cfg),
                    **block_kwargs,
                )(x, ct)
            x = self.window_reverse(x, self.window_size, h, w)
        else:
            for i in range(self.depth):
                cfg = self.config_kwargs | {
                    "drop_path":
                    self.drop_path[i] if isinstance(self.drop_path, list) else self.drop_path,
                }
                cfg = {
                    k: v
                    for k, v in cfg.items() if k in self.block_config.__dict__.keys()
                }

                x = self.block(
                    dim=self.dim,
                    config=self.block_config(**cfg),
                    **self.block_kwargs,
                )(x)

        if self.downsample:
            x = self.downsampler()(x, self.dim)

        return x
