from typing import Callable
from dataclasses import field

import jax.numpy as jnp
import flax.linen as nn

from ..utils.image_utils import window_partition, window_reverse

from .blocks import ConvBlock, ViTBlock
from .configs import ViTBlockConfig
from .misc import Downsample


class GenericLayer(nn.Module):
    """
    Generic layer for vision models from jimmy.

    This class implements a flexible layer that can be used in various vision model
    architectures. It supports both convolutional and transformer-style blocks,
    with optional downsampling.

    Attributes:
        dim (int): Number of input channels.
        depth (int): Number of blocks in the layer.
        block (nn.Module): Block module to use (e.g., ViTBlock or ConvBlock).
        block_config (Callable): Configuration for the block.
        layer_window_size (int): Size of the layer window for attention.
        msa_window_size (int): Size of the multi-head self-attention window.
        drop_path (float | list): Stochastic depth rate.
        block_types (list): Types of blocks to use in the layer.
        downsample (bool): Whether to apply downsampling after the blocks.
        block_kwargs (dict): Additional keyword arguments for blocks.
        config_kwargs (dict): Additional keyword arguments for block configuration.
    """

    dim: int
    depth: int
    block: nn.Module = ViTBlock
    block_config: Callable = ViTBlockConfig
    layer_window_size: int = -1
    msa_window_size: int = -1
    drop_path: float | list = 0.0
    block_types: list = field(default_factory=list)
    downsample: bool = True
    block_kwargs: dict = field(default_factory=dict)
    config_kwargs: dict = field(default_factory=dict)
    downsampler: nn.Module = Downsample

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        shape = x.shape
        reshape = self.block is not ConvBlock

        if reshape:
            assert len(shape) == 4
            _, H, W, _ = x.shape
            ws = max(H, W) if self.layer_window_size == -1 else self.layer_window_size

            pad_b = (ws - H % ws) % ws
            pad_r = (ws - W % ws) % ws
            if pad_r > 0 or pad_b > 0:
                x = jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
                _, Hp, Wp, _ = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, ws)

        for i in range(self.depth):
            if len(self.block_types) != self.depth:
                if len(self.block_types) != 1:
                    raise ValueError(
                        "Length mismatch between `block_types` and `depth`.")
                block_type = self.block_types[0]
            else:
                block_type = self.block_types[i]

            cfg = self.config_kwargs | {
                "drop_path":
                self.drop_path[i] if isinstance(self.drop_path, list) else self.drop_path,
                "attention": block_type,
                "msa_window_size": self.msa_window_size,
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

        if reshape:
            x = window_reverse(x, ws, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]

        if self.downsample:
            x = self.downsampler()(x, self.dim)

        return x
