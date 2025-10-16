import jax
import jax.numpy as jnp
from einops import rearrange
import flax.linen as nn
from dataclasses import field

from ..utils.image_utils import get_defaults, window_partition, window_reverse

from .attention import WindowedAttention
from .builders import get_act, get_module, get_norm
from .configs import (AttentionConfig, ConvBlockConfig, MambaConfig,
                      ViTBlockConfig)
from .dropout import DropPath
from .misc import Identity
from .norm import LayerScale
from .posemb import PosEmbMLPSwinv1D


class ViTBlock(nn.Module):
    """Generic block for Vision Transformers and MambaVision."""

    dim: int
    config: ViTBlockConfig
    attention_kwargs: dict = field(default_factory=dict)
    mamba_kwargs: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the block to the input.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output after applying the block.
        """
        # Create attention and mamba configs
        attention_config = get_defaults(AttentionConfig)
        mamba_config = get_defaults(MambaConfig)
        attention_config.update(**self.attention_kwargs)
        mamba_config.update(**self.mamba_kwargs)

        # Create normalization layer
        norm_layer = get_norm(self.config.norm_layer)
        norm1 = norm_layer(name='norm1')

        is_mamba = "mamba" in self.config.attention
        use_windowed_attention = not is_mamba and self.config.msa_window_size != -1

        # Create attention module
        attention = get_module(self.config.attention)
        attn_cfg = (MambaConfig(d_model=self.dim, **mamba_config) if is_mamba
                    else AttentionConfig(self.dim, **attention_config))
        attn = attention(config=attn_cfg, name='attn')

        # Create layer scale and drop path modules
        ls1 = (LayerScale(init_values=self.config.init_values, name='ls1')
               if self.config.init_values else Identity(name='ls1'))
        drop_path1 = (DropPath()
                      if self.config.dr1 > 0.0 else Identity())

        norm2 = norm_layer(name='norm2')
        mlp = get_module(self.config.ffn_layer)(
            hidden_features=int(self.dim * self.config.mlp_ratio),
            act_layer=get_act(self.config.act_layer),
            dropout_rate=self.config.proj_drop,
            bias=self.config.ffn_bias,
            name='mlp'
        )
        ls2 = (LayerScale(init_values=self.config.init_values, name='ls2')
               if self.config.init_values else Identity(name='ls2'))
        drop_path2 = (DropPath()
                      if self.config.dr2 > 0.0 else Identity())

        if use_windowed_attention:
            # Apply windowed attention on Multi-Head Self Attention
            _, L, _ = x.shape
            H = W = int(L**0.5)
            x = rearrange(x, "b (h w) c -> b h w c", h=H, w=W)
            ws = self.config.msa_window_size

            pad_b = (ws - H % ws) % ws
            pad_r = (ws - W % ws) % ws
            if pad_r > 0 or pad_b > 0:
                x = jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
                _, Hp, Wp, _ = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, ws)

        x = x + drop_path1(ls1(attn(norm1(x))), self.config.dr1)
        x = x + drop_path2(ls2(mlp(norm2(x), self.dim)), self.config.dr2)

        if use_windowed_attention:
            x = window_reverse(x, ws, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]
            x = rearrange(x, "b h w c -> b (h w) c")

        return x


class ConvBlock(nn.Module):
    """Convolutional block with normalization, activation, and residual connection."""

    dim: int
    config: ConvBlockConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the ConvBlock to the input.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output after applying the ConvBlock.
        """
        norm_layer = get_norm(self.config.norm_layer)
        act_layer = get_act(self.config.act_layer)

        conv1 = nn.Conv(
            features=self.dim,
            kernel_size=self.config.kernel_size,
            strides=1,
            padding="SAME",
            use_bias=True,
        )
        norm1 = norm_layer()
        conv2 = nn.Conv(
            features=self.dim,
            kernel_size=self.config.kernel_size,
            strides=1,
            padding="SAME",
            use_bias=True,
        )
        norm2 = norm_layer()
        ls1 = (LayerScale(init_values=self.config.init_values)
               if self.config.init_values else Identity())
        drop_path1 = (DropPath()
                      if self.config.drop_path > 0.0 else Identity())

        x2 = act_layer(norm1(conv1(x)))
        x2 = ls1(norm2(conv2(x2)))
        x = x + drop_path1(x2, self.config.drop_path)

        return x


class MllaBlock(nn.Module):

    dim: int
    config: ViTBlockConfig
    attention_kwargs: dict = field(default_factory=dict)
    mamba_kwargs: dict = field(default_factory=dict)
    use_dwc: bool = True  # For Mlla but not for VMamba-2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the block to the input.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output after applying the block.
        """
        # Create attention and mamba configs
        attention_config = get_defaults(AttentionConfig)
        attention_config["num_heads"] = 12  # to match the original impl
        mamba_config = get_defaults(MambaConfig)
        attention_config.update(**self.attention_kwargs)
        mamba_config.update(**self.mamba_kwargs)

        norm_layer = get_norm(self.config.norm_layer)
        act = get_act(self.config.act_layer)

        cpe1 = nn.Conv(
            features=self.dim,
            kernel_size=3,
            strides=1,
            padding=1,
            feature_group_count=self.dim,
            use_bias=True,
        )
        norm1 = norm_layer()

        if self.use_dwc:
            in_proj = nn.Dense(self.dim)
            act_proj = nn.Dense(self.dim)
            dwc = nn.Conv(
                features=self.dim,
                kernel_size=3,
                strides=1,
                padding=1,
                feature_group_count=self.dim,
                use_bias=True,
            )
            out_proj = nn.Dense(self.dim)

        is_mamba = "mamba" in self.config.attention

        attention = get_module(self.config.attention)
        attn_cfg = (MambaConfig(d_model=self.dim, **mamba_config) if is_mamba
                    else AttentionConfig(self.dim, **attention_config))
        attn = attention(config=attn_cfg)

        drop_path1 = (DropPath()
                      if self.config.dr1 > 0.0 else Identity())

        cpe2 = nn.Conv(
            features=self.dim,
            kernel_size=3,
            strides=1,
            padding=1,
            feature_group_count=self.dim,
            use_bias=True,
        )

        norm2 = norm_layer()
        mlp = get_module(self.config.ffn_layer)(
            hidden_features=int(self.dim * self.config.mlp_ratio),
            act_layer=act,
            dropout_rate=self.config.proj_drop,
            bias=self.config.ffn_bias,
        )

        drop_path2 = (DropPath()
                      if self.config.dr2 > 0.0 else Identity())

        _, l, _ = x.shape
        # Let's assume a squared initial shape
        h = w = int(l**0.5)

        x1 = x + rearrange(
            cpe1(rearrange(x, "b (h w) c -> b h w c", h=h, w=w)),
            "b h w c -> b (h w) c",
        )
        x1 = norm1(x1)

        if self.use_dwc:
            act_res = act(act_proj(x1))

            x1 = rearrange(in_proj(x1), "b (h w) c -> b h w c", h=h, w=w)
            x1 = act(rearrange(dwc(x1), "b h w c -> b (h w) c"))

        x1 = attn(x1)

        if self.use_dwc:
            x1 = out_proj(x * act_res)

        x = x + drop_path1(x1, self.config.dr1)

        x += rearrange(
            cpe2(rearrange(x, "b (h w) c -> b h w c", h=h, w=w)),
            "b h w c -> b (h w) c",
        )

        x += drop_path2(mlp(norm2(x), self.dim), self.config.dr2)

        return x


# TODO: Merge in ViTBlock?
class HATBlock(nn.Module):
    """Generic block for Vision Transformers and MambaVision."""

    dim: int
    config: ViTBlockConfig
    attention_kwargs: dict = field(default_factory=dict)
    window_size: int = 7
    sr_ratio: float = 1.0
    ct_size: int = 1
    last: bool = False
    do_propagation: bool = False

    def ct_window(self, ct, W, H, window_size):
        bs, _, N = ct.shape
        ct = ct.reshape(bs, H // window_size, window_size, W // window_size,
                        window_size, N)
        ct = jnp.transpose(ct, (0, 1, 3, 2, 4, 5))
        return ct

    def ct_dewindow(self, ct, W, H, window_size):
        bs, _, N = ct.shape
        ct2 = ct.reshape(-1, W // window_size, H // window_size, window_size,
                         window_size, N)
        ct2 = jnp.transpose(ct2, (0, 5, 1, 3, 2, 4))
        ct2 = ct2.reshape(bs, N, W * H)
        ct2 = jnp.transpose(ct2, (0, 2, 1))
        return ct2

    @nn.compact
    def __call__(self, x: jnp.ndarray,
                 carrier_tokens: jnp.ndarray) -> jnp.ndarray:
        """Apply the block to the input.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output after applying the block.
        """
        # Create attention config
        attention_config = get_defaults(AttentionConfig)
        attention_config.update(**self.attention_kwargs)

        # positional encoding for windowed attention tokens
        pos_embed = PosEmbMLPSwinv1D(
            rank=2,
            seq_len=self.window_size**2,
        )

        norm_layer = get_norm(self.config.norm_layer)
        norm1 = norm_layer()

        # number of carrier tokens per every window
        cr_tokens_per_window = self.ct_size**2 if self.sr_ratio > 1 else 0
        cr_tokens_total = cr_tokens_per_window * self.sr_ratio * self.sr_ratio

        attn = WindowedAttention(
            resolution=self.window_size,
            seq_len=self.window_size**2 + cr_tokens_per_window,
            config=AttentionConfig(self.dim, **attention_config),
        )

        ls1 = (LayerScale(init_values=self.config.init_values)
               if self.config.init_values else Identity())
        drop_path1 = (DropPath()
                      if self.config.dr1 > 0.0 else Identity())

        norm2 = norm_layer()
        mlp = get_module(self.config.ffn_layer)(
            hidden_features=int(self.dim * self.config.mlp_ratio),
            act_layer=get_act(self.config.act_layer),
            dropout_rate=self.config.proj_drop,
            bias=self.config.ffn_bias,
        )
        ls2 = (LayerScale(init_values=self.config.init_values)
               if self.config.init_values else Identity())
        drop_path2 = (DropPath()
                      if self.config.dr2 > 0.0 else Identity())

        if self.sr_ratio > 1:
            # If hierarchical attention, this part is for carrier tokens
            hat_norm1 = norm_layer()
            hat_norm2 = norm_layer()
            hat_attn = WindowedAttention(
                resolution=int(cr_tokens_total**0.5),
                seq_len=cr_tokens_total,
                config=AttentionConfig(self.dim, **attention_config),
            )

            hat_mlp = get_module(self.config.ffn_layer)(
                hidden_features=int(self.dim * self.config.mlp_ratio),
                act_layer=get_act(self.config.act_layer),
                dropout_rate=self.config.proj_drop,
                bias=self.config.ffn_bias,
            )
            hat_drop_path = (DropPath()
                             if self.config.dr2 > 0.0 else Identity())
            hat_pos_embed = PosEmbMLPSwinv1D(
                rank=2,
                seq_len=cr_tokens_total,
            )
            hat_ls1 = (LayerScale(init_values=self.config.init_values)
                       if self.config.init_values else Identity())
            hat_ls2 = (LayerScale(init_values=self.config.init_values)
                       if self.config.init_values else Identity())
            hat_ls3 = (LayerScale(init_values=self.config.init_values)
                       if self.config.init_values else Identity())

        b, t, n = x.shape
        ct = carrier_tokens
        x = pos_embed(x, self.dim)

        if self.sr_ratio > 1:
            # do hierarchical attention via carrier tokens
            # first do attention for carrier tokens
            bg, ng, hg = ct.shape

            # ct are located quite differently
            ct = self.ct_dewindow(
                ct,
                self.ct_size * self.sr_ratio,
                self.ct_size * self.sr_ratio,
                self.ct_size,
            )

            # positional bias for carrier tokens
            ct = hat_pos_embed(ct, self.dim)

            # attention plus mlp
            ct = ct + hat_drop_path(
                hat_ls1(hat_attn(hat_norm1(ct))), self.config.dr2)
            ct = ct + hat_drop_path(
                hat_ls2(hat_mlp(hat_norm2(ct), self.dim)), self.config.dr2)

            # ct are put back to windows
            ct = self.ct_window(
                ct,
                self.ct_size * self.sr_ratio,
                self.ct_size * self.sr_ratio,
                self.ct_size,
            )

            ct = ct.reshape(x.shape[0], -1, n)

            # concatenate carrier_tokens to the windowed tokens
            x = jnp.concatenate((ct, x), axis=1)

        x = x + drop_path1(ls1(attn(norm1(x))), self.config.dr1)
        x = x + drop_path2(ls2(mlp(norm2(x), self.dim)), self.config.dr2)

        if self.sr_ratio > 1:
            # for hierarchical attention we need to split carrier tokens and window tokens back
            split_index = x.shape[1] - self.window_size * self.window_size
            ctr, x = jnp.split(x, [split_index], axis=1)

            ct = ctr.reshape(bg, ng, hg)  # reshape carrier tokens.
            if self.last and self.do_propagation:
                # propagate carrier token information into the image
                ctr_image_space = jnp.transpose(ctr, (0, 2, 1)).reshape(
                    b, n, self.ct_size, self.ct_size)
                upsampled = jax.image.resize(
                    ctr_image_space,
                    (b, n, self.window_size, self.window_size),
                    method="nearest",
                )
                upsampled = jnp.transpose(upsampled.reshape(b, n, -1),
                                          (0, 2, 1))

                x = x + hat_ls3(upsampled)

        return x, ct
