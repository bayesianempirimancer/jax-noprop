"""
Sequence-to-Sequence Transformer Conditional ResNet architectures for NoProp implementations.

This module provides transformer-based sequence-to-sequence models with TwistedAttention,
a time-aware attention mechanism that perturbs QKV projection matrices based on time.
Models can be used with the NoProp algorithm for sequence inputs.
"""

from typing import Optional, Tuple, Callable
from functools import cached_property

import jax.numpy as jnp
import jax
import flax.linen as nn

from src.embeddings.time_embeddings import create_time_embedding
from src.embeddings.positional_encoding import rotary_positional_encoding
from src.layers.configs import AttentionConfig
from src.layers.mlp import Mlp

from src.utils.activation_utils import get_activation_function


class TransformerSeq2SeqConditionalResnet(nn.Module):
    """
    Sequence-to-Sequence Transformer Conditional ResNet with TwistedAttention.
    
    This architecture processes sequences x and z using transformer blocks:
    - x and z sequences are concatenated and processed through self-attention blocks
    - Uses TwistedAttention: time-dependent QKV matrices that are perturbed by time embeddings
    - Positional embeddings (RoPE) are applied to the concatenated sequence
    - Static features (x_static) are optionally embedded and prepended after RoPE
    - Time embedding is integrated into TwistedAttention for dynamical system modeling
    
    Args:
        latent_shape: Latent sequence shape tuple (e.g., (seq_len, 2)) - z is 2D (price, volume)
        output_shape: Output sequence shape tuple (e.g., (seq_len, embed_dim)) - output is in embed_dim
        input_shape: Conditional input sequence shape tuple (e.g., (seq_len, 2)) - x is 2D (price, volume)
        embed_dim: Embedding dimension for transformer (default: 20)
        hidden_dims: Tuple of hidden layer dimensions for embeddings and MLPs
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation_fn: Activation function to use (string)
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: Ratio for MLP hidden dimension relative to model dimension
        qkv_bias: Whether to use bias in QKV projections
        rope_base: Base for RoPE frequency calculation (default: 10000.0)
        projection_seed: Random seed for 2D->embed_dim projection matrix (default: 42)
        x_static_dim: Dimension of static features x_static (default: 0, meaning no static features)
    """
    latent_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    embed_dim: int = 20
    hidden_dims: Tuple[int, ...] = (256,)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    activation_fn: str = "swish"
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    rope_base: float = 10000.0
    projection_seed: int = 42
    x_static_dim: int = 0  # 0 means no static features
    
    @cached_property
    def latent_dim(self) -> int:
        """Latent dimension of the conditional ResNet."""
        dim = 1
        for shape in self.latent_shape:
            dim *= shape
        return dim

    @cached_property
    def input_dim(self) -> int:
        """Input dimension of the conditional ResNet."""
        dim = 1
        for shape in self.input_shape:
            dim *= shape
        return dim

    @cached_property
    def output_dim(self) -> int:
        """Output dimension of the conditional ResNet."""
        dim = 1
        for shape in self.output_shape:
            dim *= shape
        return dim
    
    def setup(self):
        """Initialize all components of the model."""
        # Convert activation function string to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        # No projection needed - x and z should already be in embed_dim space
        # A dimension mismatch will be caught by shape validation in __call__
        
        # Static feature embedding (will be created dynamically based on input dimension)
        # We'll create the embedding layer when x_static is first provided
        # This allows x_static to have any dimension and we'll embed it to embed_dim
        
        # Time embedding module (for TwistedAttention)
        self.time_embed = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)
        
        # No output projection - z should remain in embed_dim space as provided by encoder
        
        # Attention configuration
        self.attn_config = AttentionConfig(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.dropout_rate,
            proj_drop=self.dropout_rate,
        )
        
        # Create self-attention blocks for concatenated x,z sequences
        self.self_attention_blocks = [
            SelfAttentionBlock(
                embed_dim=self.embed_dim,
                mlp_ratio=self.mlp_ratio,
                activation_fn=self.activation_fn,  # Pass as string
                dropout_rate=self.dropout_rate,
                attn_config=self.attn_config,
                time_embed_dim=self.time_embed_dim,
                time_embed_method=self.time_embed_method,
                rope_base=self.rope_base,
                name=f'self_attention_block_{i}'
            )
            for i in range(self.num_layers)
        ]
        
        # MLP to process z after self-attention
        self.z_mlp = Mlp(
            hidden_features=int(self.embed_dim * self.mlp_ratio),
            out_features=self.embed_dim,
            act_layer=activation_fn,
            dropout_rate=self.dropout_rate,
            name='z_mlp'
        )
    
    @nn.compact
    def _embed_x_static(self, x_static: jnp.ndarray, x_static_dim: int) -> jnp.ndarray:
        """Embed x_static from x_static_dim to embed_dim using MLPBlock.
        
        Args:
            x_static: Static features [batch, x_static_dim]
            x_static_dim: Dimension of x_static input
            
        Returns:
            Embedded static features [batch, embed_dim]
        """
        # Use MLPBlock to embed x_static to embed_dim
        # Define features tuple: (input_dim, hidden_dim, hidden_dim, output_dim)
        hidden_dim = int(self.embed_dim * self.mlp_ratio)
        features = (x_static_dim, hidden_dim, hidden_dim, self.embed_dim)
        
        mlp_block = MLPBlock(
            features=features,
            activation_fn=self.activation_fn,  # Pass as string
            dropout_rate=0.0,  # No dropout for static embedding
            name='x_static_embed'
        )
        
        # Embed x_static: [batch, x_static_dim] -> [batch, embed_dim]
        x_static_emb = mlp_block(x_static)
        return x_static_emb
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: Optional[jnp.ndarray] = None, t: Optional[jnp.ndarray] = None, 
                 x_static: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """Forward pass through sequence-to-sequence transformer.
        
        Args:
            z: Current state sequence in embedding space [batch_size, z_seq_len, embed_dim]
            x: Conditional input sequence in embedding space [batch_size, x_seq_len, embed_dim] (optional)
               variable length allowed
            t: Time values [batch_size] or scalar (optional)
            x_static: Static features [batch_size, x_static_dim] or [batch_shape, x_static_dim] (optional, only used if x_static_dim > 0)
            training: Whether in training mode
            
        Returns:
            Updated state sequence [batch_size, z_seq_len, embed_dim] in embed_dim space
        """
        # Handle broadcasting and determine batch shape
        batch_shape_z = z.shape[:-len(self.latent_shape)]
        
        if x is not None:
            batch_shape_x = x.shape[:-len(self.input_shape)]
            batch_shape = jnp.broadcast_shapes(batch_shape_z, batch_shape_x)
        else:
            batch_shape = batch_shape_z
            
        if t is not None:
            t = jnp.asarray(t)
            batch_shape = jnp.broadcast_shapes(batch_shape, t.shape)
        
        if x_static is not None:
            x_static = jnp.asarray(x_static)
            # x_static shape should be (batch_shape, x_static_dim)
            # Extract batch shape from x_static (everything except last dimension)
            if x_static.ndim >= 1:
                x_static_batch_shape = x_static.shape[:-1]
                batch_shape = jnp.broadcast_shapes(batch_shape, x_static_batch_shape)
            else:
                raise ValueError(f"x_static must have shape (batch_shape, x_static_dim), got shape {x_static.shape}")

        z = jnp.broadcast_to(z, batch_shape + z.shape[-len(self.latent_shape):])
        
        # Validate that z has the correct embedding dimension (should match encoder output)
        z_flat_batch = z.reshape(-1, *self.latent_shape)
        z_seq_len = self.latent_shape[-2] if len(self.latent_shape) >= 2 else self.latent_shape[0]
        z_embed_dim = self.latent_shape[-1] if len(self.latent_shape) >= 2 else self.latent_shape[0]
        
        # Assert z matches encoder output shape: z should be in latent space (embed_dim) from encoder
        assert z_embed_dim == self.embed_dim, (
            f"z embedding dimension mismatch: expected {self.embed_dim} (from embed_dim, matching encoder latent_shape), "
            f"got {z_embed_dim} (from latent_shape={self.latent_shape}). "
            f"z should already be encoded by the encoder in fm.py to shape (..., {self.embed_dim})."
        )
        
        z_embed = z_flat_batch.reshape(-1, z_seq_len, self.embed_dim)  # [batch, seq_len, embed_dim]
        
        # Process x sequence if provided
        x_embed = None
        x_seq_len = 0
        if x is not None:
            # Get actual shape dimensions (excluding batch)
            x_trailing_dims = x.shape[-len(self.input_shape):]
            x = jnp.broadcast_to(x, batch_shape + x_trailing_dims)
            
            # Reshape to flatten batch for processing
            x_flat_batch = x.reshape(-1, *x_trailing_dims)
            
            # Get actual sequence length from input tensor (variable length)
            x_ndims = len(x_trailing_dims)
            if x_ndims >= 2:
                x_seq_len = x_trailing_dims[-2]  # Variable length
                x_embed_dim = x_trailing_dims[-1]
            else:
                x_seq_len = x_trailing_dims[-1]
                x_embed_dim = 1
            
            # Assert x matches encoder output shape: x should be encoded to latent space (embed_dim) by encoder in fm.py
            assert x_embed_dim == self.embed_dim, (
                f"x embedding dimension mismatch: expected {self.embed_dim} (from embed_dim, matching encoder latent_shape), "
                f"got {x_embed_dim} (from input_shape={self.input_shape}). "
                f"x should already be encoded by the encoder in fm.py to shape (..., {self.embed_dim}). "
                f"Set encode_x=True in fm.py config to enable x encoding."
            )
            
            x_embed = x_flat_batch.reshape(-1, x_seq_len, self.embed_dim)  # [batch, seq_len, embed_dim]
        
        # Concatenate x and z along sequence dimension: [batch, x_seq_len + z_seq_len, embed_dim]
        # x immediately precedes z temporally
        if x_embed is not None:
            xz_embed = jnp.concatenate([x_embed, z_embed], axis=-2)  # [batch, x_seq_len + z_seq_len, embed_dim]
        else:
            xz_embed = z_embed  # [batch, z_seq_len, embed_dim]
        
        # Apply static feature embeddings if x_static is provided
        # (RoPE is now applied inside TwistedAttention to Q and K)
        if x_static is not None:
            # Flatten batch for processing
            x_static_flat = x_static.reshape(-1, x_static.shape[-1])  # [batch, x_static_dim]
            x_static_dim = x_static.shape[-1]
            
            # Embed x_static using MLP: [batch, x_static_dim] -> [batch, embed_dim]
            # Create embedding layer dynamically if needed
            x_static_emb = self._embed_x_static(x_static_flat, x_static_dim)  # [batch, embed_dim]
            x_static_emb = x_static_emb[:, None, :]  # [batch, 1, embed_dim]
            xz_embed = jnp.concatenate([x_static_emb, xz_embed], axis=-2)  # [batch, 1 + x_seq_len + z_seq_len, embed_dim]
        
        # Process time embedding (for TwistedAttention)
        t_embed_vec = None
        if t is not None:
            t = jnp.broadcast_to(t, batch_shape)
            t_flat = t.reshape(-1)
            t_embed_vec = self.time_embed(t_flat)  # [batch, time_embed_dim]
        
        # Process concatenated x,z sequence with self-attention blocks
        for self_attn_block in self.self_attention_blocks:
            xz_embed = self_attn_block(xz_embed, t=t_embed_vec, training=training)
        
        # Extract z portion from concatenated sequence
        # Account for x_static (if present) and x_seq_len
        x_static_offset = 1 if (x_static is not None) else 0
        z_start_idx = x_static_offset + x_seq_len
        z_embed = xz_embed[:, z_start_idx:, :]  # [batch, z_seq_len, embed_dim]
        
        # Pass z through MLP (no additional encoding, just processing)
        z_embed = self.z_mlp(z_embed, self.embed_dim)
        
        # No output projection - z remains in embed_dim space as provided by encoder
        # z_embed shape: (batch, seq_len, embed_dim)
        
        # Reshape back to original batch shape and latent_shape
        # Output should match latent_shape (seq_len, embed_dim)
        output = z_embed.reshape((-1, *self.latent_shape))
        output = output.reshape(batch_shape + self.latent_shape)
        
        return output


# ============================================================================
# Network Blocks
# ============================================================================

class TwistedAttention(nn.Module):
    """Attention with time-dependent QKV matrices for dynamical systems.
    
    Time perturbs (twists) the Q, K, V projection matrices themselves, so that:
    q = (Q_base + Q_time(t)) @ x
    k = (K_base + K_time(t)) @ x
    v = (V_base + V_time(t)) @ x
    
    Also applies RoPE (Rotary Position Embedding) to Q and K vectors.
    """
    config: AttentionConfig
    time_embed_dim: int
    time_embed_method: str
    rope_base: float = 10000.0
    use_rope: bool = True
    
    def setup(self):
        # Base QKV projection (without time conditioning)
        self.qkv_base = nn.Dense(self.config.dim * 3, use_bias=self.config.qkv_bias, name='qkv_base')
    
    def _apply_rope_to_qk(self, q: jnp.ndarray, k: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply RoPE rotation to Q and K vectors.
        
        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            
        Returns:
            Rotated Q and K tensors with same shapes
        """
        B, H, N, head_dim = q.shape
        
        # RoPE requires even head_dim - if odd, use head_dim-1 pairs and keep last dimension unrotated
        head_dim_pairs = head_dim // 2
        head_dim_used = head_dim_pairs * 2
        
        # Generate RoPE encoding for sequence length (using head_dim_used, which is even)
        rope_encoding = rotary_positional_encoding(N, head_dim_used, base=self.rope_base)
        rope_encoding = rope_encoding.astype(q.dtype)
        
        # Reshape Q and K into pairs: [batch, num_heads, seq_len, head_dim_pairs, 2]
        q_reshaped = q[:, :, :, :head_dim_used].reshape(B, H, N, head_dim_pairs, 2)
        k_reshaped = k[:, :, :, :head_dim_used].reshape(B, H, N, head_dim_pairs, 2)
        rope_reshaped = rope_encoding.reshape(N, head_dim_pairs, 2)
        
        # Extract cos and sin components
        cos_vals = rope_reshaped[:, :, 1]  # [seq_len, head_dim_pairs] - cos components
        sin_vals = rope_reshaped[:, :, 0]  # [seq_len, head_dim_pairs] - sin components
        
        # Apply rotation matrix: [cos, -sin; sin, cos] to each pair using einsum
        # q_reshaped: [batch, num_heads, seq_len, head_dim_pairs, 2] -> indices: b, h, n, p, r
        # cos_vals: [seq_len, head_dim_pairs] -> indices: n, p
        # sin_vals: [seq_len, head_dim_pairs] -> indices: n, p
        
        # Rotation: [cos, -sin; sin, cos] @ [q_0, q_1]
        # First component: q_0 * cos - q_1 * sin
        q_rotated_0 = jnp.einsum('bhnp,np->bhnp', q_reshaped[:, :, :, :, 0], cos_vals) - \
                      jnp.einsum('bhnp,np->bhnp', q_reshaped[:, :, :, :, 1], sin_vals)
        
        # Second component: q_0 * sin + q_1 * cos
        q_rotated_1 = jnp.einsum('bhnp,np->bhnp', q_reshaped[:, :, :, :, 0], sin_vals) + \
                      jnp.einsum('bhnp,np->bhnp', q_reshaped[:, :, :, :, 1], cos_vals)
        
        q_rotated = jnp.stack([q_rotated_0, q_rotated_1], axis=-1)
        
        # Same for k
        k_rotated_0 = jnp.einsum('bhnp,np->bhnp', k_reshaped[:, :, :, :, 0], cos_vals) - \
                      jnp.einsum('bhnp,np->bhnp', k_reshaped[:, :, :, :, 1], sin_vals)
        k_rotated_1 = jnp.einsum('bhnp,np->bhnp', k_reshaped[:, :, :, :, 0], sin_vals) + \
                      jnp.einsum('bhnp,np->bhnp', k_reshaped[:, :, :, :, 1], cos_vals)
        
        k_rotated = jnp.stack([k_rotated_0, k_rotated_1], axis=-1)
        
        # Reshape back and concatenate with unrotated dimensions if head_dim was odd
        q_rotated = q_rotated.reshape(B, H, N, head_dim_used)
        k_rotated = k_rotated.reshape(B, H, N, head_dim_used)
        
        if head_dim_used < head_dim:
            # If head_dim was odd, concatenate the last unrotated dimension
            q_rotated = jnp.concatenate([q_rotated, q[:, :, :, head_dim_used:]], axis=-1)
            k_rotated = jnp.concatenate([k_rotated, k[:, :, :, head_dim_used:]], axis=-1)
        
        return q_rotated, k_rotated
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Forward pass of TwistedAttention.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            t: Time embedding [batch, time_embed_dim] or None
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        B, N, x_embed_dim = x.shape
        if x_embed_dim != self.config.dim:
            raise AssertionError(
                f"Input embedding dimension ({x_embed_dim}) should match layer embedding dimension ({self.config.dim})."
            )
        
        # Compute base QKV from x
        qkv_base = self.qkv_base(x)  # [batch, seq_len, dim * 3]
        
        # Apply time conditioning if time is provided
        if t is not None:
            # MLP that takes t and outputs QKV perturbation matrices
            # t: [batch, time_embed_dim] -> [batch, 3 * config.dim * x_embed_dim]
            t_qkv_mlp = nn.Dense(3 * self.config.dim * x_embed_dim, name='t_qkv_mlp')
            t_qkv_flat = t_qkv_mlp(t)  # [batch, 3 * config.dim * x_embed_dim]
            
            # Reshape to get QKV perturbation matrices: [batch, 3 * config_dim, x_embed_dim]
            # t_QKV contains the actual perturbation matrices for Q, K, V
            batch_shape = t_qkv_flat.shape[:-1]  # [batch]
            t_QKV = t_qkv_flat.reshape(batch_shape + (3 * self.config.dim, x_embed_dim))
            
            qkv_t = jnp.einsum('bkj,bnj->bnk', t_QKV, x)  # [batch, seq_len, 3*config.dim]
                        
            qkv = qkv_base + qkv_t  # [batch, seq_len, dim * 3]
        else:
            # No time conditioning - just use base QKV
            qkv = qkv_base
        
        # Reshape to [batch, seq_len, 3, num_heads, head_dim]
        qkv = qkv.reshape(B, N, 3, self.config.num_heads, x_embed_dim // self.config.num_heads)
        
        # Transpose to [3, batch, num_heads, seq_len, head_dim] and split
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = tuple(qkv)
        
        # Apply RoPE to Q and K (standard approach)
        q, k = self._apply_rope_to_qk(q, k)
        
        # Apply QK normalization if configured
        if self.config.qk_norm:
            match self.config.norm_layer:
                case "layernorm":
                    q = nn.LayerNorm()(q)
                    k = nn.LayerNorm()(k)
                case "rmsnormgated":
                    from src.layers.norm import RMSNormGated
                    q = RMSNormGated()(q)
                    k = RMSNormGated()(k)
                case "batchnorm":
                    q = nn.BatchNorm()(q)
                    k = nn.BatchNorm()(k)
                case _:
                    raise ValueError(f"Unknown norm `{self.config.norm_layer}`")
        
        # Compute attention
        attn = q @ k.transpose((0, 1, 3, 2)) / jnp.sqrt(self.config.head_dim)
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.config.attn_drop)(attn, deterministic=False)
        
        # Apply attention to values
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, x_embed_dim)
        
        return x


class MLPBlock(nn.Module):
    """Block of MLPs defined by a features tuple for embedding static features.
    
    The features tuple defines the dimensions: (input_dim, hidden_dim1, hidden_dim2, ..., output_dim)
    This creates len(features) - 1 MLP layers.
    """
    features: Tuple[int, ...]
    activation_fn: str
    dropout_rate: float
    
    def setup(self):
        # Convert activation function string to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        # Create MLP layers based on features tuple
        # features[i] -> features[i+1] for i in range(len(features)-1)
        self.mlps = []
        for i in range(len(self.features) - 1):
            mlp = Mlp(
                hidden_features=self.features[i+1],
                out_features=self.features[i+1],
                act_layer=activation_fn,
                dropout_rate=self.dropout_rate,
                name=f'mlp_{i}'
            )
            self.mlps.append(mlp)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through MLP layers.
        
        Args:
            x: Input features [batch, features[0]]
            
        Returns:
            Output features [batch, features[-1]]
        """
        # Process through each MLP layer
        for i, mlp in enumerate(self.mlps):
            x = mlp(x, self.features[i])
        
        return x


class SelfAttentionBlock(nn.Module):
    """Self-attention block for concatenated x,z sequences: self-attention + MLP."""
    embed_dim: int
    mlp_ratio: float
    activation_fn: str
    dropout_rate: float
    attn_config: AttentionConfig
    time_embed_dim: int
    time_embed_method: str
    rope_base: float = 10000.0
    
    def setup(self):
        # Convert activation function string to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        self.norm1 = nn.LayerNorm()
        # Use TwistedAttention instead of standard attention
        self.attn = TwistedAttention(
            config=self.attn_config,
            time_embed_dim=self.time_embed_dim,
            time_embed_method=self.time_embed_method,
            rope_base=self.rope_base
        )
        self.dropout1 = nn.Dropout(rate=self.dropout_rate)
        self.norm2 = nn.LayerNorm()
        self.mlp = Mlp(
            hidden_features=int(self.embed_dim * self.mlp_ratio),
            out_features=self.embed_dim,
            act_layer=activation_fn,
            dropout_rate=self.dropout_rate
        )
        self.dropout2 = nn.Dropout(rate=self.dropout_rate)
    
    def __call__(self, xz: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        # Self-attention block with time conditioning
        xz_norm = self.norm1(xz)
        xz_attn = self.attn(xz_norm, t=t)
        xz = xz + self.dropout1(xz_attn, deterministic=not training)
        
        # MLP block
        xz_norm2 = self.norm2(xz)
        xz_mlp_out = self.mlp(xz_norm2, self.embed_dim)
        xz = xz + self.dropout2(xz_mlp_out, deterministic=not training)
        
        return xz

