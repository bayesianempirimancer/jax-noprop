"""
Sequence-to-Sequence Transformer Conditional ResNet architectures for NoProp implementations.

This module provides transformer-based sequence-to-sequence models with cross-attention
that can be used with the NoProp algorithm for sequence inputs.
"""

from typing import Optional, Tuple
from functools import cached_property

import jax.numpy as jnp
import jax
import flax.linen as nn

from src.embeddings.time_embeddings import create_time_embedding
from src.layers.attention import Attention, CrossAttention
from src.layers.configs import AttentionConfig, CrossAttentionConfig
from src.layers.mlp import Mlp

from src.utils.activation_utils import get_activation_function


class TransformerSeq2SeqConditionalResnet(nn.Module):
    """
    Sequence-to-Sequence Transformer Conditional ResNet with cross-attention.
    
    This architecture processes sequences x and z using transformer blocks:
    - x sequence is processed through encoder blocks (self-attention)
    - z sequence is processed through decoder blocks (self-attention + cross-attention to x)
    - Time embedding is integrated into the processing
    - Positional embeddings (RoPE) are applied internally, with x positions relative to z
    - Static features (x_static) are optionally embedded and appended to x as an additional timestep after positional embeddings
    
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
    num_layers: int = 4
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
    
    @cached_property
    def input_feature_dim(self) -> int:
        """Input feature dimension (2D for price, volume)."""
        if len(self.latent_shape) >= 2:
            return self.latent_shape[-1]
        elif len(self.input_shape) >= 2:
            return self.input_shape[-1]
        else:
            return 2  # Default: price, volume
    
    def setup(self):
        """Initialize all components of the model."""
        # Convert activation function string to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        # Projection from 2D (price, volume) to embed_dim
        # Use fixed seed for reproducibility and later inversion
        # Note: This is a fixed (non-learned) matrix, so we store it as a regular attribute
        key = jax.random.PRNGKey(self.projection_seed)
        projection_matrix = jax.random.normal(key, (self.input_feature_dim, self.embed_dim))
        # Scale by 1/sqrt(embed_dim) to preserve variance
        projection_matrix = projection_matrix * (1.0 / jnp.sqrt(float(self.embed_dim)))
        self.projection_matrix = projection_matrix   # Could be learned
        
        # Static feature embedding (if enabled)
        if self.x_static_dim > 0:
            # MLP to embed x_static from x_static_dim to embed_dim
            activation_fn = get_activation_function(self.activation_fn)
            self.x_static_embed = Mlp(
                hidden_features=int(self.embed_dim * self.mlp_ratio),
                out_features=self.embed_dim,
                act_layer=activation_fn,
                dropout_rate=0.0,  # No dropout for static embedding
                name='x_static_embed'
            )
        
        # Time embedding module
        self.time_embed = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)
        
        # Time embedding projection
        self.t_proj = nn.Dense(self.embed_dim, name='t_proj')
        
        # Output projection: embed_dim -> 2D (to match latent_shape)
        # Use pseudo-inverse of projection matrix for inverse projection
        # W: [2, embed_dim], so W^T: [embed_dim, 2]
        # Pseudo-inverse: (W^T @ W)^(-1) @ W^T, or simpler: W^T scaled
        # For simplicity, use transpose and scale by 1/embed_dim to preserve scale
        # Actually, we can learn this projection or use a fixed inverse
        # For now, let's use a learned projection
        self.output_proj = nn.Dense(self.input_feature_dim, name='output_proj')
        
        # Attention configurations
        self.attn_config = AttentionConfig(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.dropout_rate,
            proj_drop=self.dropout_rate,
        )
        
        self.cross_attn_config = CrossAttentionConfig(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.dropout_rate,
            proj_drop=self.dropout_rate,
        )
        
        # Encoder layers: x self-attention blocks
        self.x_encoder_norms1 = [nn.LayerNorm(name=f'x_encoder_norm1_{i}') for i in range(self.num_layers)]
        self.x_encoder_attns = [Attention(config=self.attn_config, name=f'x_encoder_attn_{i}') for i in range(self.num_layers)]
        self.x_encoder_norms2 = [nn.LayerNorm(name=f'x_encoder_norm2_{i}') for i in range(self.num_layers)]
        self.x_encoder_mlps = [
            Mlp(
                hidden_features=int(self.embed_dim * self.mlp_ratio),
                out_features=self.embed_dim,
                act_layer=activation_fn,
                dropout_rate=self.dropout_rate,
                name=f'x_encoder_mlp_{i}'
            )
            for i in range(self.num_layers)
        ]
        
        # Interleaved cross-attention blocks
        # z cross-attention to x
        self.z_cross_norms = [nn.LayerNorm(name=f'z_cross_norm_{i}') for i in range(self.num_layers)]
        self.z_cross_attns = [CrossAttention(config=self.cross_attn_config, name=f'z_cross_attn_{i}') for i in range(self.num_layers)]
        
        # z self-attention
        self.z_self_norms = [nn.LayerNorm(name=f'z_self_norm_{i}') for i in range(self.num_layers)]
        self.z_self_attns = [Attention(config=self.attn_config, name=f'z_self_attn_{i}') for i in range(self.num_layers)]
        
        # z MLP
        self.z_mlp_norms = [nn.LayerNorm(name=f'z_mlp_norm_{i}') for i in range(self.num_layers)]
        self.z_mlps = [
            Mlp(
                hidden_features=int(self.embed_dim * self.mlp_ratio),
                out_features=self.embed_dim,
                act_layer=activation_fn,
                dropout_rate=self.dropout_rate,
                name=f'z_mlp_{i}'
            )
            for i in range(self.num_layers)
        ]
        
        # x cross-attention to z
        self.x_cross_norms = [nn.LayerNorm(name=f'x_cross_norm_{i}') for i in range(self.num_layers)]
        self.x_cross_attns = [CrossAttention(config=self.cross_attn_config, name=f'x_cross_attn_{i}') for i in range(self.num_layers)]
        
        # x MLP (in interleaved blocks)
        self.x_mlp_norms = [nn.LayerNorm(name=f'x_mlp_norm_{i}') for i in range(self.num_layers)]
        self.x_mlps = [
            Mlp(
                hidden_features=int(self.embed_dim * self.mlp_ratio),
                out_features=self.embed_dim,
                act_layer=activation_fn,
                dropout_rate=self.dropout_rate,
                name=f'x_mlp_{i}'
            )
            for i in range(self.num_layers)
        ]
    
    def _apply_rope_1d(self, x: jnp.ndarray, position_offset: int = 0) -> jnp.ndarray:
        """Apply 1D RoPE positional encoding to sequence data.
        
        Args:
            x: Input sequences [batch, seq_len, embed_dim]
            position_offset: Starting position offset (negative for x relative to z)
            
        Returns:
            Sequences with RoPE applied [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Create frequency tensor
        inv_freq = 1.0 / (self.rope_base ** (jnp.arange(0, embed_dim, 2, dtype=jnp.float32) / embed_dim))
        
        # Create position tensor with offset
        positions = jnp.arange(position_offset, position_offset + seq_len, dtype=jnp.float32)
        
        # Create angle tensor [seq_len, embed_dim//2]
        angle = positions[:, None] * inv_freq[None, :]
        
        # Create RoPE encoding [seq_len, embed_dim]
        rope_encoding = jnp.zeros((seq_len, embed_dim), dtype=x.dtype)
        rope_encoding = rope_encoding.at[:, 0::2].set(jnp.sin(angle))
        rope_encoding = rope_encoding.at[:, 1::2].set(jnp.cos(angle))
        
        # Normalize entire embedding vector to unit length
        norms = jnp.linalg.norm(rope_encoding, axis=1, keepdims=True)
        norms = jnp.maximum(norms, 1e-8)
        rope_encoding = rope_encoding / norms
        
        # Apply RoPE by rotating pairs of dimensions
        # Reshape to separate pairs
        x_reshaped = x.reshape(batch_size, seq_len, embed_dim // 2, 2)
        rope_reshaped = rope_encoding.reshape(seq_len, embed_dim // 2, 2)
        
        # Apply rotation: [cos, sin; -sin, cos] @ [x_i, x_{i+1}]
        cos_vals = rope_reshaped[:, :, 1]  # cos components
        sin_vals = rope_reshaped[:, :, 0]  # sin components
        
        # Rotation matrix: [cos, -sin; sin, cos]
        x_rotated = jnp.stack([
            x_reshaped[:, :, :, 0] * cos_vals[None, :, :] - x_reshaped[:, :, :, 1] * sin_vals[None, :, :],
            x_reshaped[:, :, :, 0] * sin_vals[None, :, :] + x_reshaped[:, :, :, 1] * cos_vals[None, :, :]
        ], axis=-1)
        
        return x_rotated.reshape(batch_size, seq_len, embed_dim)
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: Optional[jnp.ndarray] = None, t: Optional[jnp.ndarray] = None, 
                 x_static: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """Forward pass through sequence-to-sequence transformer.
        
        Args:
            z: Current state sequence in 2D format [batch_size, z_seq_len, 2] or [batch_size, *latent_shape]
               where last dimension is (price, volume)
            x: Conditional input sequence in 2D format [batch_size, x_seq_len, 2] or [batch_size, *input_shape] (optional)
               where last dimension is (price, volume), variable length allowed
            t: Time values [batch_size] or scalar (optional)
            x_static: Static features [batch_size, x_static_dim] or [batch_shape, x_static_dim] (optional, only used if x_static_dim > 0)
            training: Whether in training mode
            
        Returns:
            Updated state sequence [batch_size, *output_shape] in embed_dim
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
        
        # z and x are in 2D format (price, volume)
        # Reshape to flatten batch for processing
        z_flat_batch = z.reshape(-1, *self.latent_shape)
        z_seq_len = self.latent_shape[-2] if len(self.latent_shape) >= 2 else self.latent_shape[0]
        
        # Project z from 2D to embed_dim: [batch, seq_len, 2] -> [batch, seq_len, embed_dim]
        z_2d = z_flat_batch.reshape(-1, z_seq_len, self.input_feature_dim)
        z_embed = jnp.dot(z_2d, self.projection_matrix)  # [batch, seq_len, embed_dim]
        
        # Process x sequence if provided (2D format, variable length)
        x_embed = None
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
                x_2d = x_flat_batch.reshape(-1, x_seq_len, self.input_feature_dim)
            else:
                x_seq_len = x_trailing_dims[-1]
                x_2d = x_flat_batch.reshape(-1, x_seq_len, self.input_feature_dim)
            
            # Project x from 2D to embed_dim
            x_embed = jnp.dot(x_2d, self.projection_matrix)  # [batch, seq_len, embed_dim]
        
        # Apply RoPE positional encodings
        # z starts at position 0
        z_embed = self._apply_rope_1d(z_embed, position_offset=0)
        
        if x_embed is not None:
            # x positions are relative to z (negative offsets)
            # x_seq_len positions before z: positions [-x_seq_len, ..., -1]
            x_embed = self._apply_rope_1d(x_embed, position_offset=-x_seq_len)
        
        # Apply static feature embeddings if enabled
        if self.x_static_dim > 0 and x_static is not None:
            # Flatten batch for processing
            x_static_flat = x_static.reshape(-1, self.x_static_dim)  # [batch, x_static_dim]
            # Embed x_static using MLP: [batch, x_static_dim] -> [batch, embed_dim]
            x_static_emb = self.x_static_embed(x_static_flat, x_static_flat.shape[-1])  # [batch, embed_dim]
            # Expand to sequence dimension and append to x_embed as if it were a part of the sequence
            # x_static_emb: [batch, embed_dim] -> [batch, 1, embed_dim]
            x_static_emb_expanded = x_static_emb[:, None, :]  # [batch, 1, embed_dim]
            if x_embed is not None:
                # Concatenate along sequence dimension (axis=-2): [batch, seq_len, embed_dim] + [batch, 1, embed_dim] -> [batch, seq_len+1, embed_dim]
                x_embed = jnp.concatenate([x_embed, x_static_emb_expanded], axis=-2)  # [batch, seq_len+1, embed_dim]
        
        # Process time embedding
        if t is not None:
            t = jnp.broadcast_to(t, batch_shape)
            t_flat = t.reshape(-1)
            t_embed_vec = self.time_embed(t_flat)
            # Broadcast time embedding to sequence length and add to z_embed
            t_embed = t_embed_vec[:, None, :]  # (batch, 1, time_embed_dim)
            # Project and add to z embeddings
            t_proj = self.t_proj(t_embed)
            z_embed = z_embed + t_proj
        
        # Step 1: Process x with self-attention layers (encoder)
        if x_embed is not None:
            for i in range(self.num_layers):
                # Self-attention block for x
                x_norm = self.x_encoder_norms1[i](x_embed)
                x_attn = self.x_encoder_attns[i](x_norm)
                x_embed = x_embed + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_attn)
                
                # MLP block for x
                x_norm2 = self.x_encoder_norms2[i](x_embed)
                x_mlp_out = self.x_encoder_mlps[i](x_norm2, self.embed_dim)
                x_embed = x_embed + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_mlp_out)
        
        # Step 2: Repeated block with interleaved cross-attention and self-attention
        # Block pattern: z = cross_attention(z, x), z = self_attention(z), x = cross_attention(x, z)
        if x_embed is not None:
            for i in range(self.num_layers):
                # z = cross_attention(z, x)
                z_norm1 = self.z_cross_norms[i](z_embed)
                z_cross_attn = self.z_cross_attns[i](z_norm1, x_embed)
                z_embed = z_embed + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z_cross_attn)
                
                # z = self_attention(z)
                z_norm2 = self.z_self_norms[i](z_embed)
                z_attn = self.z_self_attns[i](z_norm2)
                z_embed = z_embed + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z_attn)
                
                # z MLP
                z_norm3 = self.z_mlp_norms[i](z_embed)
                z_mlp_out = self.z_mlps[i](z_norm3, self.embed_dim)
                z_embed = z_embed + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z_mlp_out)
                
                # x = cross_attention(x, z)
                x_norm_cross = self.x_cross_norms[i](x_embed)
                x_cross_attn = self.x_cross_attns[i](x_norm_cross, z_embed)
                x_embed = x_embed + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_cross_attn)
                
                # x MLP
                x_norm_mlp = self.x_mlp_norms[i](x_embed)
                x_mlp_block_out = self.x_mlps[i](x_norm_mlp, self.embed_dim)
                x_embed = x_embed + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_mlp_block_out)
        else:
            # If no x, just process z with self-attention
            for i in range(self.num_layers):
                # Self-attention block for z
                z_norm = self.z_self_norms[i](z_embed)
                z_attn = self.z_self_attns[i](z_norm)
                z_embed = z_embed + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z_attn)
                
                # MLP block for z
                z_norm_mlp = self.z_mlp_norms[i](z_embed)
                z_mlp_out = self.z_mlps[i](z_norm_mlp, self.embed_dim)
                z_embed = z_embed + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z_mlp_out)
        
        # Project back from embed_dim to 2D (to match latent_shape)
        # z_embed shape: (batch, seq_len, embed_dim)
        z_output = self.output_proj(z_embed)  # (batch, seq_len, 2)
        
        # Reshape back to original batch shape and output shape
        # Output shape should match latent_shape (seq_len, 2)
        output = z_output.reshape((-1, *self.output_shape))
        output = output.reshape(batch_shape + self.output_shape)
        
        return output

