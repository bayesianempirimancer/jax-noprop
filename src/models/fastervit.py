from typing import List

import jax.numpy as jnp
from einops import reduce
import flax.linen as nn
from dataclasses import field

from ..layers.blocks import ConvBlock, HATBlock
from ..layers.builders import get_norm
from ..layers.configs import ConvBlockConfig, ViTBlockConfig
from ..layers.fastervit import FasterViTLayer
from ..layers.misc import Identity
from ..layers.patch import ConvPatchEmbed


class FasterViT(nn.Module):
    """
    FasterViT model architecture.

    Args:
        in_features (int): Number of input channels.
        dim (int): Base dimension of the model.
        in_dim (int): Input dimension for the patch embedding.
        depths (List[int]): Number of blocks in each stage.
        window_size (List[int]): Window sizes for each stage.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_heads (List[int]): Number of attention heads in each stage.
        drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
        qk_norm (bool, optional): If True, apply normalization to query, key, value. Defaults to True.
        ffn_bias (bool, optional): If True, use bias in the feed-forward network. Defaults to True.
        proj_bias (bool, optional): If True, use bias in the projection layers. Defaults to True.
        proj_drop (float, optional): Dropout rate for projection layers. Defaults to 0.0.
        attn_drop (float, optional): Dropout rate for attention. Defaults to 0.0.
        init_values (float | None, optional): Initial layer scale value. Defaults to None.
        init_values_conv (float | None, optional): Initial layer scale value for conv blocks. Defaults to None.
        transformer_attention (Callable, optional): Attention mechanism to use. Defaults to Attention.
        mamba_mixer (Callable, optional): Mamba mixer to use. Defaults to MambaVisionMixer.
        act_layer (Callable, optional): Activation function to use. Defaults to nnx.gelu.
        norm_layer (Callable, optional): Normalization layer to use. Defaults to nnx.LayerNorm.
        ffn_layer (Callable, optional): Feed-forward network layer to use. Defaults to Mlp.
        num_classes (int, optional): Number of classes for classification. Defaults to 1000.
    """

    block_config = {
        "mlp_ratio": 4.0,
        "ffn_layer": "mlp",
        "ffn_bias": True,
        "act_layer": "gelu",
        "init_values": 1e-5,
    }
    attention_config = {
        "qkv_bias": True,
        "qk_norm": True,
        "proj_bias": True,
        "proj_drop": 0.0,
        "attn_drop": 0.0,
        "norm_layer": "layernorm",
    }

    norm_layer = "batchnorm"
    hat: List[bool] = field(default_factory=lambda: [False, False, True, False])
    do_propagation: bool = False

    drop_path_rate: float = 0.2
    num_classes: int = 1000

    ls_convblock: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, 
                 in_features: int,
                 dim: int,
                 in_dim: int,
                 resolution: int,
                 depths: List[int],
                 num_heads: List[int],
                 window_sizes: List[int],
                 ct_size: int,
                 *,
                 block_kwargs: dict = {},
                 attention_kwargs: dict = {},
                 mamba_kwargs: dict = {},
                 **kwargs):
        """
        Forward pass of the FasterViT model.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, height, width, channels).
            in_features (int): Number of input channels.
            dim (int): Base dimension of the model.
            in_dim (int): Input dimension for the patch embedding.
            resolution (int): Input resolution.
            depths (List[int]): Number of blocks in each stage.
            num_heads (List[int]): Number of attention heads in each stage.
            window_sizes (List[int]): Window sizes for each stage.
            ct_size (int): Carrier token size.
            block_kwargs (dict): Additional keyword arguments for blocks.
            attention_kwargs (dict): Additional keyword arguments for attention.
            mamba_kwargs (dict): Additional keyword arguments for mamba.
            **kwargs: Additional keyword arguments.

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, num_classes).
        """
        # Update configs with kwargs
        attention_config = self.attention_config.copy()
        block_config = self.block_config.copy()
        attention_config.update(**attention_kwargs)
        block_config.update(**block_kwargs)

        num_features = int(dim * 2**(len(depths) - 1))

        # Create patch embedding
        patch_embed = ConvPatchEmbed(
            in_features=in_features,
            hidden_features=in_dim,
            out_features=dim,
        )
        x = patch_embed(x)

        # Create drop path rates
        dpr = list(jnp.linspace(0, self.drop_path_rate, sum(depths)))

        # Create levels
        for i, depth in enumerate(depths):
            conv = i < 2
            _config = {
                "init_values": None
            } if not self.ls_convblock and conv else {}

            level = FasterViTLayer(
                dim=int(dim * 2**i),
                depth=depth,
                input_resolution=int(2**(-2 - i) * resolution),
                window_size=window_sizes[i],
                ct_size=ct_size,
                block=ConvBlock if conv else HATBlock,
                block_config=ConvBlockConfig if conv else ViTBlockConfig,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                block_types=[None],
                downsample=i < len(depths) - 1,
                config_kwargs={
                    **(block_config | _config),
                },
                block_kwargs={
                    "attention_kwargs": {
                        "num_heads": num_heads[i],
                        **attention_config,
                    },
                    "do_propagation": self.do_propagation,
                },
            )
            x = level(x)

        # Create normalization and head
        norm = get_norm(self.norm_layer)(num_features=num_features)
        x = norm(x)
        x = reduce(x, "b h w c -> b c", "mean")

        if self.num_classes:
            head = nn.Dense(self.num_classes)
            x = head(x)

        return x
