import math

import jax
import jax.numpy as jnp
from einops import rearrange, repeat
import flax.linen as nn
from jax import random

from ..utils.scan_utils import non_causal_linear_attn, selective_scan, ssd
from ..utils.image_utils import custom_uniform

from .configs import MambaConfig
from .norm import RMSNormGated

# TODO: Inference cache as in https://github.com/walln/scratch/blob/ab0b6b891830375b7aa64c8e46e77783b843f5ca/src/scratch/language_modeling/mamba/mamba.py
# TODO: Learnable init state (as in mamba.py)


class MambaVisionMixer(nn.Module):
    """MambaVision Mixer from Ali Hatamizadeh and Jan Kautz.

    This class implements the MambaVision Mixer, a novel architecture for vision tasks
    that combines the strengths of state space models and vision transformers.

    Args:
        config (MambaConfig): Configuration object for the MambaVisionMixer.

    Notes:
        - b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
        - l: sequence length                  (`L` in [1] Algorithm 2)
        - d or d_model: hidden dim
        - n or d_state: latent state dim      (`N` in [1] Algorithm 2)
        - expand: expansion factor            (`E` in [1] Section 3.4)
        - d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
        - A, B, C, D: state space parameters  (See any state space representation formula)
            (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
        - dt or delta: input-dependent step size
        - dt_rank: rank of dt                 (See [1] Section 3.6 "Parameterization of ∆")

    References:
        [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces
            (https://arxiv.org/abs/2312.00752)
    """

    config: MambaConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass of the MambaVisionMixer.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        _, L, _ = x.shape

        xz = nn.Dense(self.config.d_inner, use_bias=self.config.bias)(x)
        x, z = jnp.split(xz, 2, axis=-1)
        
        # Initialize A_log parameter
        A = jnp.arange(1, self.config.d_state + 1, dtype=jnp.float32)
        A = jnp.tile(A, (self.config.d_inner // 2, 1))
        A_log = jnp.log(A)
        A = -jnp.exp(A_log)
        
        x = nn.silu(nn.Conv(
            features=self.config.d_inner // 2,
            kernel_size=(self.config.d_conv,),
            feature_group_count=self.config.d_inner // 2,
            use_bias=self.config.conv_bias,
            padding="SAME",
        )(x))
        z = nn.silu(nn.Conv(
            features=self.config.d_inner // 2,
            kernel_size=(self.config.d_conv,),
            feature_group_count=self.config.d_inner // 2,
            use_bias=self.config.conv_bias,
            padding="SAME",
        )(z))
        
        x_dbl = nn.Dense(
            self.config.dt_rank + self.config.d_state * 2,
            use_bias=False,
        )(rearrange(x, "b l d -> (b l) d"))
        dt, B, C = jnp.split(
            x_dbl,
            [self.config.dt_rank, self.config.d_state + self.config.dt_rank],
            axis=-1,
        )

        # Initialize dt_proj with custom bias
        dt_init_std = self.config.dt_rank**-0.5 * self.config.dt_scale
        if self.config.dt_init == "constant":
            kernel_init = nn.initializers.constant(dt_init_std)
        elif self.config.dt_init == "random":
            kernel_init = custom_uniform(dt_init_std)
        else:
            raise NotImplementedError

        # Create bias for dt_proj
        rand_vals = random.uniform(self.make_rng('params'), (self.config.d_inner // 2,))
        dt_bias = jnp.exp(
            rand_vals * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        )
        dt_bias = jnp.clip(dt_bias, a_min=self.config.dt_init_floor)
        inv_dt = dt_bias + jnp.log(-jnp.expm1(-dt_bias))
        
        dt = rearrange(nn.Dense(
            self.config.d_inner // 2,
            use_bias=True,
            kernel_init=kernel_init,
            bias_init=lambda *_: inv_dt,
        )(dt), "(b l) d -> b d l", l=L)
        B = rearrange(B, "(b l) d -> b d l", l=L)
        C = rearrange(C, "(b l) d -> b d l", l=L)

        x = rearrange(x, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l")
        
        # Initialize D parameter
        D = self.param('D', nn.initializers.ones, (self.config.d_inner // 2,))
        
        y = selective_scan(
            x,
            dt,
            A,
            B,
            C,
            D,
            delta_bias=inv_dt,
            delta_softplus=True,
        )

        y = jnp.concatenate([y, z], axis=1)
        y = rearrange(y, "b d l -> b l d")

        out = nn.Dense(self.config.d_model, use_bias=True)(y)

        return out


class Mamba2Mixer(nn.Module):
    """Mamba2 Mixer.

    This class implements the a Mamba2 Mixer using State Space Duality (SSD),
    from Mamba2 [1]. Also supports implementation details from Visual State Space
    Duality (VSSD) [2].

    Args:
        config (MambaConfig): Configuration object for the Mamba2VisionMixer.

    Notes:
        This implementation is heavily based on wlln/scratch.

    References:
        [1] Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
            (https://arxiv.org/abs/2401.04054)
        [2] VSSD: Vision Mamba with Non-Causal State Space Duality
            (https://arxiv.org/abs/2407.18559)
    """

    config: MambaConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass of the VMamba2Mixer using non-causal attention duality.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        batch = x.shape[0]

        # Initialize parameters
        A_min, A_max = self.config.A_init_range
        A = random.uniform(self.make_rng('params'), (self.config.n_heads,), minval=A_min, maxval=A_max)
        A_log = jnp.log(A)
        A = -jnp.exp(A_log)

        # Initialize dt_bias
        rand_vals = random.uniform(self.make_rng('params'), (self.config.n_heads,))
        dt = jnp.exp(
            rand_vals * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        )
        dt = jnp.clip(dt, a_min=self.config.dt_init_floor)
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        dt_bias = self.param('dt_bias', lambda *_: inv_dt, ())

        # Initialize D parameter
        D = self.param('D', lambda *_: jnp.ones(self.config.n_heads), ())

        # Initialize init_states if needed
        if self.config.learnable_init_states:
            init_states = self.param('init_states', lambda *_: jnp.ones(self.config.n_heads), ())

        d_in_proj = (
            2 * self.config.d_inner + 2 * self.config.n_groups * self.config.d_state + self.config.n_heads
        )
        zxbcdt = nn.Dense(d_in_proj, use_bias=self.config.bias)(x)

        z, xbc, dt = jnp.split(
            zxbcdt,
            [self.config.d_inner, zxbcdt.shape[-1] - self.config.n_heads],
            axis=-1,
        )

        dt = jax.nn.softplus(dt + dt_bias)

        # apply 1d convolution and silu activation
        conv_dim = self.config.d_inner + 2 * self.config.n_groups * self.config.d_state
        xbc_conv = nn.Conv(
            features=conv_dim,
            kernel_size=(self.config.d_conv,),
            feature_group_count=conv_dim,
            padding=[((self.config.d_conv - 1) // 2, (self.config.d_conv - 1) // 2)],
            use_bias=self.config.conv_bias,
        )(xbc)
        xbc_silu = jax.nn.silu(xbc_conv[:, : x.shape[1], :])

        # split the conv state into the conv kernel and the conv state
        x, B, C = jnp.split(xbc_silu, self.config.indices_xBC, axis=-1)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.config.head_dim)

        if self.config.linear_attn_duality:
            y = non_causal_linear_attn(
                x, dt=dt, A=A, B=B, C=C, D=D, n_groups=self.config.n_groups
            )
        else:
            # apply ssd function
            # TODO: Bidirectional
            initial_states = (
                repeat(
                    init_states,
                    "h -> b c h p n",
                    b=batch,
                    c=x.shape[1] // self.config.chunk_size,
                    p=x.shape[-1],
                    n=B.shape[-1],
                )
                if self.config.learnable_init_states
                else None
            )
            y, ssm_state = ssd(
                x * jnp.expand_dims(dt, axis=-1),
                A * dt,
                rearrange(B, "b l (g n) -> b l g n", g=self.config.n_groups),
                rearrange(C, "b l (g n) -> b l g n", g=self.config.n_groups),
                self.config.chunk_size,
                initial_states=initial_states,
            )

            # Combine the output of the ssd function with the input and rearrange
            y = y + x * jnp.expand_dims(D, axis=-1)

        y = rearrange(y, "b l h p -> b l (h p)")

        # apply the output projection
        y = nn.LayerNorm()(y) * z

        y = nn.Dense(self.config.d_model, use_bias=self.config.bias)(y)

        return y


class Mamba2VisionMixer(nn.Module):
    """Mamba2Vision Mixer.

    This class implements the a Mamba2Vision Mixer using State Space Duality (SSD),
    from Mamba2 [1]. It extends the MambaVisionMixer by replacing SSM with SSD, leading to
    enhanced efficiency, and maybe accuracy.

    Args:
        config (MambaConfig): Configuration object for the Mamba2VisionMixer.

    Notes:
        This implementation is heavily based on wlln/scratch.

    References:
        [1] Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
            (https://arxiv.org/abs/2401.04054)
    """

    config: MambaConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass of the MambaVisionMixer.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        _, L, _ = x.shape

        # Initialize parameters
        A_min, A_max = self.config.A_init_range
        A = random.uniform(self.make_rng('params'), (self.config.n_heads,), minval=A_min, maxval=A_max)
        A_log = jnp.log(A)
        A = -jnp.exp(A_log)

        # Initialize dt_bias
        rand_vals = random.uniform(self.make_rng('params'), (self.config.n_heads,))
        dt = jnp.exp(
            rand_vals * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        )
        dt = jnp.clip(dt, a_min=self.config.dt_init_floor)
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        dt_bias = self.param('dt_bias', lambda *_: inv_dt, ())

        # Initialize D parameter
        D = self.param('D', lambda *_: jnp.ones(self.config.n_heads), ())

        xz = nn.Dense(2 * self.config.d_inner, use_bias=self.config.bias)(x)
        x, z = jnp.split(xz, 2, axis=-1)
        x = nn.silu(nn.Conv(
            features=self.config.d_inner,
            kernel_size=(self.config.d_conv,),
            feature_group_count=self.config.d_inner,
            use_bias=self.config.conv_bias,
            padding="SAME",
        )(x))
        z = nn.silu(nn.Conv(
            features=self.config.d_inner,
            kernel_size=(self.config.d_conv,),
            feature_group_count=self.config.d_inner,
            use_bias=self.config.conv_bias,
            padding="SAME",
        )(z))

        x_dbl = nn.Dense(
            self.config.n_heads + self.config.d_state * 2,
            use_bias=False,
        )(rearrange(x, "b l d -> (b l) d"))
        dt, B, C = jnp.split(
            x_dbl,
            [self.config.n_heads, self.config.d_state + self.config.n_heads],
            axis=-1,
        )
        dt = rearrange(dt, "(b l) n -> b l n", l=L)
        dt = jax.nn.softplus(dt + dt_bias)

        B = rearrange(B, "(b l) n -> b l n", l=L)
        C = rearrange(C, "(b l) n -> b l n", l=L)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.config.head_dim)

        # apply ssd function
        y, ssm_state = ssd(
            x * jnp.expand_dims(dt, axis=-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.config.chunk_size,
        )

        # Combine the output of the ssd function with the input and rearrange
        y = y + x * jnp.expand_dims(D, axis=-1)
        y = rearrange(y, "b l h p -> b l (h p)")

        # apply the output projection
        y = RMSNormGated()(y, z)
        y = nn.Dense(self.config.d_model, use_bias=self.config.bias)(y)

        return y
