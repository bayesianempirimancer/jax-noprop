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
from src.layers.configs import AttentionConfig
from src.layers.mlp import Mlp
from src.flow_models.crn_attention_blocks import TwistedAttentionBlock

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
    embed_dim: int = 32
    hidden_dims: Tuple[int, ...] = (64,64)
    time_embed_dim: int = 32
    time_embed_method: str = "sinusoidal"
    activation_fn: str = "swish"
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    rope_base: float = 10000.0
    lora_rank: int = 8  # Rank for LoRA decomposition in TwistedAttention
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
            TwistedAttentionBlock(
                embed_dim=self.embed_dim,
                mlp_ratio=self.mlp_ratio,
                activation_fn=self.activation_fn,  # Pass as string
                dropout_rate=self.dropout_rate,
                attn_config=self.attn_config,
                time_embed_dim=self.time_embed_dim,
                time_embed_method=self.time_embed_method,
                rope_base=self.rope_base,
                lora_rank=self.lora_rank,
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
                 x_static: Optional[jnp.ndarray] = None, x_mask: Optional[jnp.ndarray] = None, 
                 training: bool = True) -> jnp.ndarray:
        """Forward pass through sequence-to-sequence transformer.
        
        Args:
            z: Current state sequence in embedding space [batch_size, z_seq_len, embed_dim]
            x: Conditional input sequence in embedding space [batch_size, x_seq_len, embed_dim] (optional)
               variable length allowed
            t: Time values [batch_size] or scalar (optional)
            x_static: Static features [batch_size, x_static_dim] or [batch_shape, x_static_dim] (optional, only used if x_static_dim > 0)
            x_mask: Boolean mask for x sequence [batch_size, x_seq_len] where True=valid, False=masked (optional)
                   Only x positions are masked; z and x_static are never masked
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
        x_mask_processed = None
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
            
            # Process x_mask if provided
            if x_mask is not None:
                # Broadcast x_mask to match batch shape
                x_mask = jnp.broadcast_to(x_mask, batch_shape + (x_seq_len,))
                x_mask_processed = x_mask.reshape(-1, x_seq_len)  # [batch, x_seq_len]
        
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
        
        # Build combined mask for concatenated sequence (x_static + x + z)
        # x_static and z are never masked, only x positions use x_mask
        # Build mask after all concatenations so we know the final sequence structure
        combined_mask = None
        if x_mask_processed is not None:
            batch_size = xz_embed.shape[0]
            total_seq_len = xz_embed.shape[1]
            
            # Start with all True (unmasked)
            combined_mask = jnp.ones((batch_size, total_seq_len), dtype=bool)
            
            # Apply x_mask to x positions
            # Final sequence structure: [x_static (if present), x, z]
            # x positions are at indices [x_static_offset, x_static_offset + x_seq_len)
            x_static_offset = 1 if (x_static is not None) else 0
            x_start = x_static_offset
            x_end = x_start + x_seq_len
            combined_mask = combined_mask.at[:, x_start:x_end].set(x_mask_processed)
        
        # Process time embedding (for TwistedAttention)
        t_embed_vec = None
        if t is not None:
            t = jnp.broadcast_to(t, batch_shape)
            t_flat = t.reshape(-1)
            t_embed_vec = self.time_embed(t_flat)  # [batch, time_embed_dim]
        
        # Process concatenated x,z sequence with self-attention blocks
        for self_attn_block in self.self_attention_blocks:
            xz_embed = self_attn_block(xz_embed, t=t_embed_vec, mask=combined_mask, training=training)
        
        # Extract z portion from concatenated sequence
        # Account for x_static (if present) and x_seq_len
        x_static_offset = 1 if (x_static is not None) else 0
        z_start_idx = x_static_offset + x_seq_len
        z_embed = xz_embed[:, z_start_idx:, :]  # [batch, z_seq_len, embed_dim]
        
        # Pass z through MLP (no additional encoding, just processing)
        z_embed = self.z_mlp(z_embed, self.embed_dim, training=training)
        
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



