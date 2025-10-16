import math
from typing import Callable, List, Optional, Tuple, Union

import jax.numpy as jnp
from einops import rearrange
import flax.linen as nn
from dataclasses import field


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding, inspired from Timm.

    This module converts an image into a sequence of embedded patches.

    Attributes:
        patch_size (List[int]): Size of the patches.
        img_size (Tuple[int, int] | None): Size of the input image (assumed square).
        grid_size (Tuple[int, int] | None): Size of the grid after patching.
        num_patches (int | None): Total number of patches.
        dynamic_img_size (bool): Whether to allow dynamic image sizes.
        dynamic_img_pad (bool): Whether to use dynamic padding.
        flatten (bool): Whether to flatten the output.
        embed_dim (int): Dimension of the embedded patches.
        norm_layer (Optional[nn.Module]): Normalization layer.

    Args:
        img_size (int | None, optional): Size of the input image (assumed square).
        patch_size (Union[List[int], int], optional): Size of the patches.
        in_channels (int, optional): Number of input channels.
        embed_dim (int, optional): Dimension of the embedded patches.
        norm_layer (Optional[nn.Module], optional): Normalization layer.
        flatten (bool, optional): Whether to flatten the output.
        dynamic_img_size (bool, optional): Whether to allow dynamic image sizes.
        dynamic_img_pad (bool, optional): Whether to use dynamic padding.
        use_bias (bool, optional): Whether to use bias in the projection.
    """

    img_size: int | None = None
    patch_size: Union[List[int], int] = 16
    in_channels: int = 3
    embed_dim: int = 768
    norm_layer: Optional[nn.Module] = None
    flatten: bool = True
    dynamic_img_size: bool = False
    dynamic_img_pad: bool = False
    use_bias: bool = True

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get grid (feature) size for given image size taking account of dynamic padding.
        Taken as is from timm

        Args:
            img_size (Tuple[int, int]): Size of the input image.

        Returns:
            Tuple[int, int]: Grid size after applying patches.
        """
        patch_size = (
            self.patch_size if isinstance(self.patch_size, list) else [self.patch_size, self.patch_size]
        )
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / patch_size[0]), math.ceil(
                img_size[1] / patch_size[1]
            )
        return img_size[0] // patch_size[0], img_size[1] // patch_size[1]

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the PatchEmbed module.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, height, width, channels).

        Returns:
            jnp.ndarray: Output tensor of embedded patches.

        Raises:
            AssertionError: If input dimensions don't match the expected dimensions.
        """
        _, H, W, C = x.shape
        patch_size = (
            self.patch_size if isinstance(self.patch_size, list) else [self.patch_size, self.patch_size]
        )

        if self.img_size is not None:
            img_size = (self.img_size, self.img_size)
            if not self.dynamic_img_size:
                if H != img_size[0]:
                    raise AssertionError(
                        f"Input height ({H}) doesn't match model ({img_size[0]})"
                    )
                if W != img_size[1]:
                    raise AssertionError(
                        f"Input width ({W}) doesn't match model ({img_size[1]})"
                    )
            elif not self.dynamic_img_pad:
                if H % patch_size[0] != 0:
                    raise AssertionError(
                        f"Input height ({H}) should be divisible by patch size ({patch_size[0]})"
                    )
                if W % patch_size[1] != 0:
                    raise AssertionError(
                        f"Input width ({W}) should be divisible by patch size ({patch_size[1]})"
                    )

        if self.dynamic_img_pad:
            pad_h = (patch_size[0] - H % patch_size[0]) % patch_size[0]
            pad_w = (patch_size[1] - W % patch_size[1]) % patch_size[1]
            x = jnp.pad(x, pad_width=((0, 0), (0, pad_h), (0, pad_w), (0, 0)))

        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(patch_size[0], patch_size[1]),
            strides=(patch_size[0], patch_size[1]),
            padding="VALID",
            use_bias=self.use_bias,
            name='proj',
        )(x)

        _, H, W, _ = x.shape
        x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))

        if self.norm_layer:
            x = self.norm_layer()(x)

        if not self.flatten:
            x = jnp.reshape(x, (-1, H, W, self.embed_dim))

        return x


class ConvPatchEmbed(nn.Module):
    """
    Convolutional Patch Embedding, used in MambaVision.

    This module applies a series of convolutional layers to embed patches.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        out_features (int): Number of output features.
        act_layer (Callable, optional): Activation function to use. Defaults to nn.relu.
        norm_layer (Callable, optional): Normalization layer to use. Defaults to nn.BatchNorm.
        norm_params (dict, optional): Parameters for the normalization layer. Defaults to {"epsilon": 1e-4}.
    """

    in_features: int
    hidden_features: int
    out_features: int
    act_layer: Callable = nn.relu
    norm_layer: Callable = nn.BatchNorm
    norm_params: dict = field(default_factory=lambda: {"epsilon": 1e-4})

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass of the ConvPatchEmbed module.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after applying convolutional patch embedding.
        """
        x = nn.Conv(
            features=self.hidden_features,
            kernel_size=(3, 3),
            strides=2,
            padding="SAME",
            use_bias=False,
        )(x)
        x = self.norm_layer(**self.norm_params)(x)
        x = self.act_layer(x)
        
        x = nn.Conv(
            features=self.out_features,
            kernel_size=(3, 3),
            strides=2,
            padding="SAME",
            use_bias=False,
        )(x)
        x = self.norm_layer(**self.norm_params)(x)
        x = self.act_layer(x)
        
        return x


class Conv(nn.Module):
    in_features: int
    out_features: int
    kernel_size: int = 3
    stride: int = 1
    padding: str = "SAME"
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    dropout: float = 0.0
    norm: nn.Module | None = nn.BatchNorm
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x, deterministic=False)

        x = nn.Conv(
            features=self.out_features,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding=self.padding,
            use_bias=self.bias,
            feature_group_count=self.groups,
        )(x)

        if self.norm is not None:
            x = self.norm()(x)
        if self.act is not None:
            x = self.act(x)

        return x


class SimpleConvStem(nn.Module):
    """Simple patch embed from Mlla paper"""

    patch_size: int
    in_features: int
    embed_dim: int
    flatten: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="SAME",
            use_bias=False,
        )(x)
        _, h, w, _ = x.shape

        x = nn.LayerNorm()(rearrange(x, "b h w c -> b (h w) c"))

        if not self.flatten:
            return rearrange(x, "b (h w) c -> b h w c", h=h, w=w)

        return x


class ConvStem(nn.Module):
    """Convolutional patch embed from Mlla paper"""

    patch_size: int
    in_features: int
    embed_dim: int
    flatten: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = Conv(
            in_features=self.in_features,
            out_features=self.embed_dim // 2,
            kernel_size=3,
            stride=2,
            padding="SAME",
            bias=False,
        )(x)
        
        conv2_out = Conv(
            in_features=self.embed_dim // 2,
            out_features=self.embed_dim // 2,
            kernel_size=3,
            stride=1,
            bias=False,
        )(x)
        conv2_out = Conv(
            in_features=self.embed_dim // 2,
            out_features=self.embed_dim // 2,
            kernel_size=3,
            stride=1,
            bias=False,
            act=None,
        )(conv2_out)
        x += conv2_out
        
        x = Conv(
            in_features=self.embed_dim // 2,
            out_features=self.embed_dim * 4,
            kernel_size=3,
            stride=2,
            bias=False,
        )(x)
        x = Conv(
            in_features=self.embed_dim * 4,
            out_features=self.embed_dim,
            kernel_size=1,
            bias=False,
            act=None,
        )(x)

        if self.flatten:
            return rearrange(x, "b h w c -> b (h w) c")

        return x


class SimplePatchMerging(nn.Module):
    """Simple patch merging from Mlla paper"""

    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = Conv(
            in_features=self.dim,
            out_features=2 * self.dim,
            kernel_size=3,
            stride=2,
            norm=None,
        )(x)
        _, h, w, c = x.shape

        x = nn.LayerNorm()(rearrange(x, "b h w c -> b (h w) c"))

        return rearrange(x, "b (h w) c -> b h w c", h=h, w=w)


class PatchMerging(nn.Module):
    """Patch merging from Mlla paper"""

    dim: int
    ratio: float = 4.0

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = Conv(
            in_features=self.dim,
            out_features=2 * self.dim * self.ratio,
            kernel_size=1,
            norm=None,
        )(x)
        x = Conv(
            in_features=2 * self.dim * self.ratio,
            out_features=2 * self.dim * self.ratio,
            kernel_size=3,
            stride=2,
            groups=int(2 * self.dim * self.ratio),
            norm=None,
        )(x)
        x = Conv(
            in_features=2 * self.dim * self.ratio,
            out_features=2 * self.dim,
            kernel_size=1,
            act=None,
        )(x)
        return x
