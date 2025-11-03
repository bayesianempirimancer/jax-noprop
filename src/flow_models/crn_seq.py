"""
Sequence-to-Sequence Transformer Conditional ResNet architectures for NoProp implementations.

This module provides transformer-based sequence-to-sequence models with cross-attention
that can be used with the NoProp algorithm for sequence inputs.
"""

from typing import Optional, Tuple
from functools import cached_property

import jax.numpy as jnp
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
    
    Args:
        latent_shape: Latent sequence shape tuple (e.g., (seq_len, model_dim)) - z is already embedded
        output_shape: Output sequence shape tuple (e.g., (seq_len, model_dim)) - output is in model_dim
        input_shape: Conditional input sequence shape tuple (e.g., (seq_len, model_dim)) - x is already embedded
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
    """
    latent_shape: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
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
    def model_dim(self) -> int:
        """Model dimension (embedding dimension) for transformer."""
        return self.hidden_dims[0] if len(self.hidden_dims) > 0 else 256
    
    def setup(self):
        """Initialize all components of the model."""
        # Convert activation function string to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        # Time embedding module
        self.time_embed = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)
        
        # Time embedding projection
        self.t_proj = nn.Dense(self.model_dim, name='t_proj')
        
        # Attention configurations
        self.attn_config = AttentionConfig(
            dim=self.model_dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.dropout_rate,
            proj_drop=self.dropout_rate,
        )
        
        self.cross_attn_config = CrossAttentionConfig(
            dim=self.model_dim,
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
                hidden_features=int(self.model_dim * self.mlp_ratio),
                out_features=self.model_dim,
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
                hidden_features=int(self.model_dim * self.mlp_ratio),
                out_features=self.model_dim,
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
                hidden_features=int(self.model_dim * self.mlp_ratio),
                out_features=self.model_dim,
                act_layer=activation_fn,
                dropout_rate=self.dropout_rate,
                name=f'x_mlp_{i}'
            )
            for i in range(self.num_layers)
        ]
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: Optional[jnp.ndarray] = None, t: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """Forward pass through sequence-to-sequence transformer.
        
        Args:
            z: Current state sequence already embedded [batch_size, z_seq_len, model_dim] or [batch_size, *latent_shape]
            x: Conditional input sequence already embedded [batch_size, x_seq_len, model_dim] or [batch_size, *input_shape] (optional)
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            
        Returns:
            Updated state sequence [batch_size, *output_shape]
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

        z = jnp.broadcast_to(z, batch_shape + z.shape[-len(self.latent_shape):])
        
        # z and x are already embedded to model_dim outside the CRN
        # They should be in (batch, seq_len, model_dim) format or can be reshaped to that
        
        # Reshape z to (batch, seq_len, model_dim)
        # Assuming latent_shape is (seq_len, model_dim) or just (seq_len,) if model_dim is inferred
        z_ndims = len(self.latent_shape)
        z_seq_len = self.latent_shape[-2]
                
        # Process x sequence if provided (already embedded, variable length)
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
                # Actual sequence length from input tensor
                x_seq_len = x_trailing_dims[-2]  # Variable length
                # x is already embedded, last dim should be model_dim
                x_embed = x_flat_batch.reshape(-1, x_seq_len, self.model_dim)
            else:
                # Single dimension: treat as (seq_len,), assume model_dim
                x_seq_len = x_trailing_dims[-1]
                x_embed = x_flat_batch.reshape(-1, x_seq_len, self.model_dim)
        
        # Reshape z to (batch, seq_len, model_dim)
        z_ndims = len(self.latent_shape)
        z_seq_len = self.latent_shape[-2] if z_ndims >= 2 else self.latent_shape[0]
        z_flat_batch = z.reshape(-1, *self.latent_shape)
        z_embed = z_flat_batch.reshape(-1, z_seq_len, self.model_dim)
        
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
                x_mlp_out = self.x_encoder_mlps[i](x_norm2, self.model_dim)
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
                z_mlp_out = self.z_mlps[i](z_norm3, self.model_dim)
                z_embed = z_embed + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z_mlp_out)
                
                # x = cross_attention(x, z)
                x_norm_cross = self.x_cross_norms[i](x_embed)
                x_cross_attn = self.x_cross_attns[i](x_norm_cross, z_embed)
                x_embed = x_embed + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_cross_attn)
                
                # x MLP
                x_norm_mlp = self.x_mlp_norms[i](x_embed)
                x_mlp_block_out = self.x_mlps[i](x_norm_mlp, self.model_dim)
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
                z_mlp_out = self.z_mlps[i](z_norm_mlp, self.model_dim)
                z_embed = z_embed + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z_mlp_out)
        
        # Output is already in model_dim (no projection needed)
        # z_embed shape: (batch, seq_len, model_dim)
        # Reshape back to original batch shape and output shape
        # Output shape should match latent_shape (seq_len, model_dim)
        output = z_embed.reshape((-1, *self.output_shape))
        output = output.reshape(batch_shape + self.output_shape)
        
        return output

