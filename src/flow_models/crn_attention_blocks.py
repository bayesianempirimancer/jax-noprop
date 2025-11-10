"""
Attention blocks for Conditional ResNet architectures.

This module provides attention mechanisms and attention blocks used in various CRN models:
- TwistedAttentionBlock: Complete transformer block with time-dependent QKV matrices
- PointCloudSelfAttentionBlock: Self-attention block for point clouds with time conditioning
"""

from typing import Optional, Tuple

import jax.numpy as jnp
import flax.linen as nn

from src.embeddings.positional_encoding import rotary_positional_encoding
from src.layers.configs import AttentionConfig
from src.layers.attention import Attention
from src.layers.mlp import Mlp
from src.utils.activation_utils import get_activation_function


class TwistedAttentionBlock(nn.Module):
    """Self-attention block with time-dependent QKV matrices for dynamical systems.
    
    This is a complete transformer block that includes:
    - LayerNorm before attention
    - TwistedAttention: Time perturbs (twists) the Q, K, V projection matrices themselves
    - Dropout and residual connection after attention
    - LayerNorm before MLP
    - MLP with dropout and residual connection
    
    Time perturbs the QKV matrices using LoRA (Low-Rank Adaptation) decomposition:
    Q_time(t) = B_q(t) @ A_q(t), where A_q: [rank, x_embed_dim], B_q: [dim, rank]
    Same for K and V.
    
    This reduces computational burden compared to full-rank perturbations:
    - Full-rank: 3 * dim * x_embed_dim parameters
    - LoRA: 3 * rank * (dim + x_embed_dim) parameters
    - With rank=8 and dim=64, this is ~75% reduction in parameters
    
    The perturbation is applied as:
    q = (Q_base + Q_time(t)) @ x
    k = (K_base + K_time(t)) @ x
    v = (V_base + V_time(t)) @ x
    
    Also applies RoPE (Rotary Position Embedding) to Q and K vectors.
    """
    embed_dim: int
    mlp_ratio: float
    activation_fn: str
    dropout_rate: float
    attn_config: AttentionConfig
    time_embed_dim: int
    time_embed_method: str
    rope_base: float = 10000.0
    use_rope: bool = True
    lora_rank: int = 8  # Rank for LoRA decomposition of time-dependent QKV perturbations
    
    def setup(self):
        # Convert activation function string to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        # LayerNorm before attention
        self.norm1 = nn.LayerNorm()
        
        # Base QKV projection (without time conditioning)
        self.qkv_base = nn.Dense(self.attn_config.dim * 3, use_bias=self.attn_config.qkv_bias, name='qkv_base')
        
        # Dropout after attention
        self.dropout1 = nn.Dropout(rate=self.dropout_rate)
        
        # LayerNorm before MLP
        self.norm2 = nn.LayerNorm()
        
        # MLP
        self.mlp = Mlp(
            hidden_features=int(self.embed_dim * self.mlp_ratio),
            out_features=self.embed_dim,
            act_layer=activation_fn,
            dropout_rate=self.dropout_rate
        )
        
        # Dropout after MLP
        self.dropout2 = nn.Dropout(rate=self.dropout_rate)
    
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
    def _twisted_attention(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """
        Forward pass of twisted attention mechanism (internal method) using LoRA-style low-rank adaptations.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            t: Time embedding [batch, time_embed_dim] or None
            mask: Boolean mask [batch, seq_len] where True=valid, False=masked (optional)
                 Prevents attention TO masked positions (columns of attention matrix)
            training: Whether in training mode
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        B, N, x_embed_dim = x.shape
        if x_embed_dim != self.attn_config.dim:
            raise AssertionError(
                f"Input embedding dimension ({x_embed_dim}) should match layer embedding dimension ({self.attn_config.dim})."
            )
        
        # Compute base QKV from x
        qkv_base = self.qkv_base(x)  # [batch, seq_len, dim * 3]
        
        # Apply time conditioning if time is provided (using LoRA decomposition)
        if t is not None:
            # LoRA decomposition: W_time = B @ A where:
            # - A: [rank, x_embed_dim] (down-projection)
            # - B: [config.dim, rank] (up-projection)
            # For Q, K, V we need 3 such decompositions
            # Total parameters: 3 * rank * (x_embed_dim + config.dim)
            # vs full-rank: 3 * config.dim * x_embed_dim
            
            rank = self.lora_rank
            dim = self.attn_config.dim
            
            # MLP that takes t and outputs LoRA parameters
            # Output: [batch, 3 * rank * (x_embed_dim + dim)]
            t_lora_mlp = nn.Dense(3 * rank * (x_embed_dim + dim), name='t_lora_mlp')
            t_lora_flat = t_lora_mlp(t)  # [batch, 3 * rank * (x_embed_dim + dim)]
            
            batch_size = t_lora_flat.shape[0]
            
            # Split into A and B parameters for Q, K, V
            # A parameters: [batch, 3 * rank * x_embed_dim]
            # B parameters: [batch, 3 * rank * dim]
            a_size = 3 * rank * x_embed_dim
            t_lora_a_flat = t_lora_flat[:, :a_size]  # [batch, 3 * rank * x_embed_dim]
            t_lora_b_flat = t_lora_flat[:, a_size:]  # [batch, 3 * rank * dim]
            
            # Reshape A matrices: [batch, 3, rank, x_embed_dim]
            t_lora_a = t_lora_a_flat.reshape(batch_size, 3, rank, x_embed_dim)
            # Reshape B matrices: [batch, 3, dim, rank]
            t_lora_b = t_lora_b_flat.reshape(batch_size, 3, dim, rank)
            
            # Apply LoRA: W_time = B @ A, so perturbation is x @ (B @ A).T
            # More efficient: compute B @ A first, then x @ (B @ A).T
            # B @ A: [batch, dim, rank] @ [batch, rank, x_embed_dim] -> [batch, dim, x_embed_dim]
            # Then x @ (B @ A).T: [batch, seq_len, x_embed_dim] @ [batch, x_embed_dim, dim] -> [batch, seq_len, dim]
            
            qkv_t_list = []
            for i in range(3):  # Q, K, V
                # B @ A: [batch, dim, rank] @ [batch, rank, x_embed_dim] -> [batch, dim, x_embed_dim]
                ba = jnp.einsum('bdr,brj->bdj', t_lora_b[:, i, :, :], t_lora_a[:, i, :, :])
                # x @ (B @ A).T: [batch, seq_len, x_embed_dim] @ [batch, x_embed_dim, dim] -> [batch, seq_len, dim]
                qkv_t_i = jnp.einsum('bnj,bdj->bnd', x, ba)
                qkv_t_list.append(qkv_t_i)
            
            # Concatenate Q, K, V: [batch, seq_len, 3 * dim]
            qkv_t = jnp.concatenate(qkv_t_list, axis=-1)  # [batch, seq_len, 3 * dim]
                        
            qkv = qkv_base + qkv_t  # [batch, seq_len, dim * 3]
        else:
            # No time conditioning - just use base QKV
            qkv = qkv_base
        
        # Reshape to [batch, seq_len, 3, num_heads, head_dim]
        qkv = qkv.reshape(B, N, 3, self.attn_config.num_heads, x_embed_dim // self.attn_config.num_heads)
        
        # Transpose to [3, batch, num_heads, seq_len, head_dim] and split
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = tuple(qkv)
        
        # Apply RoPE to Q and K (standard approach)
        if self.use_rope:
            q, k = self._apply_rope_to_qk(q, k)
        
        # Apply QK normalization if configured
        if self.attn_config.qk_norm:
            match self.attn_config.norm_layer:
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
                    raise ValueError(f"Unknown norm `{self.attn_config.norm_layer}`")
        
        # Compute attention
        attn = q @ k.transpose((0, 1, 3, 2)) / jnp.sqrt(self.attn_config.head_dim)
        
        # Apply mask if provided (prevent attention both TO and FROM masked positions)
        # mask shape: [batch, seq_len] where True=valid, False=masked
        # Standard approach: mask both columns (keys) and rows (queries) so masked positions:
        #   - Cannot be attended to (mask columns)
        #   - Cannot attend to anything (mask rows)
        # This prevents masked positions from contributing to or receiving attention
        if mask is not None:
            # Convert boolean mask to float: True -> 0.0 (keep), False -> -inf (mask out)
            mask_float = jnp.where(mask, 0.0, -1e9)  # [batch, seq_len]
            
            # Mask columns (keys): prevent attention TO masked positions
            # Broadcast to [batch, 1, 1, seq_len] to mask columns
            mask_cols = mask_float[:, None, None, :]  # [batch, 1, 1, seq_len]
            attn = attn + mask_cols  # Broadcast addition masks out columns
            
            # Mask rows (queries): prevent attention FROM masked positions
            # Broadcast to [batch, 1, seq_len, 1] to mask rows
            mask_rows = mask_float[:, None, :, None]  # [batch, 1, seq_len, 1]
            attn = attn + mask_rows  # Broadcast addition masks out rows
        
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.attn_config.attn_drop)(attn, deterministic=not training)
        
        # Apply attention to values
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, x_embed_dim)
        
        return x
    
    def __call__(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None, 
                 training: bool = True) -> jnp.ndarray:
        """Forward pass through twisted attention block.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            t: Time embedding [batch, time_embed_dim] (optional, used for time conditioning)
            mask: Boolean mask [batch, seq_len] where True=valid, False=masked (optional)
            training: Whether in training mode
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        # Self-attention block with time conditioning
        x_norm = self.norm1(x)
        x_attn = self._twisted_attention(x_norm, t=t, mask=mask, training=training)
        x = x + self.dropout1(x_attn, deterministic=not training)
        
        # MLP block
        x_norm2 = self.norm2(x)
        x_mlp_out = self.mlp(x_norm2, self.embed_dim, training=training)
        x = x + self.dropout2(x_mlp_out, deterministic=not training)
        
        return x


class PointCloudSelfAttentionBlock(nn.Module):
    """Self-attention block for point clouds with time-conditioned standard attention.
    
    Time conditioning can be applied via:
    - "adaln": Adaptive LayerNorm Zero (DiT approach)
    - "film": FiLM (Feature-wise Linear Modulation)
    - "none": No time conditioning
    """
    embed_dim: int
    mlp_ratio: float
    activation_fn: str
    dropout_rate: float
    attn_config: AttentionConfig
    time_conditioning_method: Optional[str] = None
    
    def setup(self):
        # Convert activation function string to callable
        activation_fn = get_activation_function(self.activation_fn)
        
        self.norm1 = nn.LayerNorm()
        
        # Standard attention
        self.attn = Attention(config=self.attn_config)
        
        # Set up time conditioning for standard attention
        if self.time_conditioning_method == "adaln":
            # Adaptive LayerNorm (adaLN-Zero): modulate LayerNorm parameters
            # Output: [batch, 2 * embed_dim] for scale and shift
            # Note: In DiT, this is initialized to zero (adaLN-Zero) for training stability
            self.t_adaln = nn.Dense(
                self.embed_dim * 2,
                kernel_init=nn.initializers.zeros,  # Zero initialization (adaLN-Zero)
                bias_init=nn.initializers.zeros,
                name='t_adaln'
            )
        elif self.time_conditioning_method == "film":
            # FiLM: Feature-wise Linear Modulation
            # Output: [batch, 2 * embed_dim] for scale and shift
            self.t_film = nn.Dense(
                self.embed_dim * 2,
                name='t_film'
            )
        elif self.time_conditioning_method == "none":
            # No time conditioning - no parameters needed
            pass
        else:
            raise ValueError(
                f"Unknown time_conditioning_method: {self.time_conditioning_method}. "
                f"Options: 'adaln', 'film', 'none'"
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
    
    def __call__(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None, 
                 training: bool = True) -> jnp.ndarray:
        """Forward pass through self-attention block.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            t: Time embedding [batch, time_embed_dim] (optional, used for time conditioning)
            mask: Boolean mask [batch, seq_len] where True=valid, False=masked (optional)
            training: Whether in training mode
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        # Standard attention with optional time conditioning
        method = self.time_conditioning_method or "none"
        
        if method == "adaln":
            # Adaptive LayerNorm (adaLN-Zero): modulate norm parameters with time
            # This matches DiT's approach: time modulates LayerNorm scale and shift
            if t is not None:
                t_params = self.t_adaln(t)  # [batch, 2 * embed_dim]
                scale, shift = jnp.split(t_params, 2, axis=-1)  # Each: [batch, embed_dim]
                # Apply adaptive normalization (DiT-style)
                # Standard LayerNorm computation
                x_mean = jnp.mean(x, axis=-1, keepdims=True)  # [batch, seq_len, 1]
                x_var = jnp.var(x, axis=-1, keepdims=True)  # [batch, seq_len, 1]
                x_norm = (x - x_mean) / jnp.sqrt(x_var + 1e-5)
                # Apply time-dependent scale and shift (adaLN-Zero formula)
                x_norm = (1.0 + scale[:, None, :]) * x_norm + shift[:, None, :]
            else:
                x_norm = self.norm1(x)
        elif method == "film":
            # FiLM: Feature-wise Linear Modulation
            x_norm = self.norm1(x)
            if t is not None:
                t_film = self.t_film(t)  # [batch, 2 * embed_dim]
                scale, shift = jnp.split(t_film, 2, axis=-1)  # Each: [batch, embed_dim]
                x_norm = (1.0 + scale[:, None, :]) * x_norm + shift[:, None, :]
        elif method == "none":
            # No time conditioning
            x_norm = self.norm1(x)
        else:
            raise ValueError(
                f"Unknown time_conditioning_method: {method}. "
                f"Options: 'adaln', 'film', 'none'"
            )
        
        # Apply attention
        x_attn = self.attn(x_norm)
        
        # Apply masking if provided (basic output masking)
        if mask is not None:
            mask_float = jnp.where(mask, 1.0, 0.0)[:, :, None]  # [batch, seq_len, 1]
            x_attn = x_attn * mask_float
        
        x = x + self.dropout1(x_attn, deterministic=not training)
        
        # MLP block
        x_norm2 = self.norm2(x)
        x_mlp_out = self.mlp(x_norm2, self.embed_dim, training=training)
        x = x + self.dropout2(x_mlp_out, deterministic=not training)
        
        return x

