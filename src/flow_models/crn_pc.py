"""
Point Cloud Conditional ResNet architectures for NoProp implementations.

This module provides transformer-based point cloud models that process point cloud data
where each point consists of a position (D-dimensional, e.g., 2D or 3D) and feature vector.
The model uses Fourier feature embeddings for positions and time-conditioned self-attention
to process the point clouds.

Key features:
- x: Variable number of points (point cloud with position + features)
- z: Fixed number of points (point cloud with position + features)
- Position encoding via Fourier features
- Feature combination via ConcatSquash
- Time-conditioned self-attention (adaln or film)
"""

from typing import Optional, Tuple
from dataclasses import dataclass, field

import jax.numpy as jnp
import jax
import flax.linen as nn
from flax.core import FrozenDict

from src.configs.base_config import BaseConfig
from src.embeddings.time_embeddings import create_time_embedding
from src.embeddings.point_cloud_positional_encoding import fourier_features_2d, fourier_features_3d
from src.layers.configs import AttentionConfig
from src.layers.mlp import Mlp
from src.layers.concatsquash import ConcatSquash
from src.flow_models.crn_attention_blocks import DitAttentionBlock
from src.utils.activation_utils import get_activation_function


@dataclass(frozen=True)
class Config(BaseConfig):
    """Configuration for Point Cloud Conditional ResNet."""
    
    # Set model_name from config_dict
    model_name: str = "point_cloud_conditional_resnet"
    
    # Hierarchical configuration structure
    config: FrozenDict = field(default_factory=lambda: FrozenDict({
        "point_dim": 3,  # Spatial dimension (2 for 2D, 3 for 3D)
        "z_num_points": 100,  # Fixed number of points in z
        "feature_dim": 16,  # Feature dimension for x and z (excluding position). Can be 0 for position-only points.
        "embed_dim": 64,
        "fourier_num_frequencies": 10,
        "fourier_include_original": True,
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "activation_fn": "swish",
        "dropout_rate": 0.1,
        "num_layers": 3,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "time_conditioning_method": "adaln",  # Options: "adaln", "film", "none"
        # "adaln": Adaptive LayerNorm Zero - modulate LayerNorm with time (DiT approach)
        # "film": FiLM (Feature-wise Linear Modulation) - modulate features with time
        # "none": No time conditioning
    }))

class PointCloudConditionalResnet(nn.Module):
    """
    Point Cloud Conditional ResNet with time-conditioned self-attention.
    
    This architecture processes point clouds x and z:
    - Each point consists of position (first D elements) and features (remaining elements)
    - Supports feature_dim = 0 for position-only point clouds (no features)
    - Positions are encoded using Fourier features
    - Fourier features are combined with point features via ConcatSquash
    - z and x embeddings are concatenated (z first, then x) and processed through self-attention
    - Uses time-conditioned standard attention (adaln or film)
    - Output maintains the same shape as z input
    
    Args:
        config: FrozenDict containing all model configuration parameters
    """
    config: FrozenDict
    
    def setup(self):
        """Initialize all components of the model."""
        # Convert activation function string to callable
        activation_fn = get_activation_function(self.config["activation_fn"])
        
        # Time embedding module (for time-conditioned standard attention)
        time_conditioning_method = self.config.get("time_conditioning_method", "film")
        
        if time_conditioning_method != "none":
            self.time_embed = create_time_embedding(
                embed_dim=self.config["time_embed_dim"], 
                method=self.config["time_embed_method"]
            )
        else:
            self.time_embed = None
        
        # ConcatSquash layers to combine Fourier features with point features
        # Input: Fourier features + point features -> Output: embed_dim
        self.z_concat_squash = ConcatSquash(
            features=self.config["embed_dim"],
            use_bias=True,
            use_layer_norm=False,
            name='z_concat_squash'
        )
        
        self.x_concat_squash = ConcatSquash(
            features=self.config["embed_dim"],
            use_bias=True,
            use_layer_norm=False,
            name='x_concat_squash'
        )
        
        # Attention configuration
        self.attn_config = AttentionConfig(
            dim=self.config["embed_dim"],
            num_heads=self.config["num_heads"],
            qkv_bias=self.config["qkv_bias"],
            attn_drop=self.config["dropout_rate"],
            proj_drop=self.config["dropout_rate"],
        )
        
        # Create self-attention blocks for concatenated z,x point clouds
        time_conditioning_method = self.config.get("time_conditioning_method", "film")
        self.self_attention_blocks = [
            DitAttentionBlock(
                embed_dim=self.config["embed_dim"],
                mlp_ratio=self.config["mlp_ratio"],
                activation_fn=self.config["activation_fn"],
                dropout_rate=self.config["dropout_rate"],
                attn_config=self.attn_config,
                time_conditioning_method=time_conditioning_method,
                name=f'self_attention_block_{i}'
            )
            for i in range(self.config["num_layers"])
        ]
        
        # MLP to process z after self-attention
        self.z_mlp = Mlp(
            hidden_features=int(self.config["embed_dim"] * self.config["mlp_ratio"]),
            out_features=self.config["embed_dim"],
            act_layer=activation_fn,
            dropout_rate=self.config["dropout_rate"],
            name='z_mlp'
        )
        
        # Output MLP: embed_dim -> z_total_dim (position + features)
        # This MLP removes any positional information that may have leaked through transformer processing
        # Keeps hidden dimension at embed_dim until final projection
        z_total_dim = self.config["point_dim"] + self.config["feature_dim"]
        self.output_mlp = Mlp(
            hidden_features=self.config["embed_dim"],  # Keep at embed_dim, not embed_dim * mlp_ratio
            out_features=z_total_dim,
            act_layer=activation_fn,
            dropout_rate=self.config["dropout_rate"],
            name='output_mlp'
        )
    
    def _extract_positions_and_features(
        self, 
        points: jnp.ndarray, 
        point_dim: int,
        feature_dim: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Extract positions and features from point cloud.
        
        Args:
            points: Point cloud [batch, num_points, point_dim + feature_dim]
            point_dim: Spatial dimension (D)
            feature_dim: Feature dimension (can be 0 for position-only points)
            
        Returns:
            positions: [batch, num_points, point_dim]
            features: [batch, num_points, feature_dim] (empty array if feature_dim = 0)
        """
        positions = points[:, :, :point_dim]
        if feature_dim > 0:
            features = points[:, :, point_dim:]
        else:
            # Return empty features array with correct shape when feature_dim = 0
            batch_size, num_points = points.shape[:2]
            features = jnp.empty((batch_size, num_points, 0), dtype=points.dtype)
        return positions, features
    
    def _apply_fourier_encoding(
        self, 
        positions: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply Fourier feature encoding to positions.
        
        Args:
            positions: Point positions [batch, num_points, point_dim]
            
        Returns:
            Fourier features [batch, num_points, fourier_feature_dim]
        """
        point_dim = self.config["point_dim"]
        if point_dim == 2:
            return fourier_features_2d(
                positions,
                num_frequencies=self.config["fourier_num_frequencies"],
                include_original=self.config["fourier_include_original"],
                normalize=True
            )
        elif point_dim == 3:
            return fourier_features_3d(
                positions,
                num_frequencies=self.config["fourier_num_frequencies"],
                include_original=self.config["fourier_include_original"],
                normalize=True
            )
        else:
            raise ValueError(
                f"Unsupported point_dim: {point_dim}. "
                f"Currently only 2D and 3D are supported."
            )
    
    def __call__(
        self, 
        z: jnp.ndarray, 
        x: Optional[jnp.ndarray] = None, 
        t: Optional[jnp.ndarray] = None,
        x_mask: Optional[jnp.ndarray] = None,
        training: bool = True
    ) -> jnp.ndarray:
        """Forward pass through point cloud conditional ResNet.
        
        Args:
            z: z point cloud [batch, z_num_points, z_total_dim]
               First point_dim elements are position, remaining are features
               z_total_dim = point_dim + feature_dim (feature_dim can be 0)
            x: x point cloud [batch, x_num_points, x_total_dim] (optional)
               First point_dim elements are position, remaining are features
               x_total_dim = point_dim + feature_dim (feature_dim can be 0)
               Can have variable number of points
            t: Time values [batch] or scalar (optional)
            x_mask: Boolean mask for x points [batch, x_num_points] where True=valid, False=masked (optional)
            training: Whether in training mode
            
        Returns:
            Updated z point cloud [batch, z_num_points, z_total_dim] with same shape as input z
        """
        # Handle broadcasting and determine batch shape
        batch_shape_z = z.shape[:-2]  # Everything except (num_points, total_dim)
        
        if x is not None:
            batch_shape_x = x.shape[:-2]
            batch_shape = jnp.broadcast_shapes(batch_shape_z, batch_shape_x)
        else:
            batch_shape = batch_shape_z
        
        if t is not None:
            t = jnp.asarray(t)
            batch_shape = jnp.broadcast_shapes(batch_shape, t.shape)
        
        # Broadcast z to batch shape
        z = jnp.broadcast_to(z, batch_shape + z.shape[-2:])
        
        # Validate z shape
        z_flat_batch = z.reshape(-1, *z.shape[-2:])  # [batch_flat, z_num_points, z_total_dim]
        z_num_points = self.config["z_num_points"]
        point_dim = self.config["point_dim"]
        feature_dim = self.config["feature_dim"]
        z_total_dim = point_dim + feature_dim
        
        assert z_flat_batch.shape[1] == z_num_points, (
            f"z must have {z_num_points} points, got {z_flat_batch.shape[1]}"
        )
        assert z_flat_batch.shape[2] == z_total_dim, (
            f"z total dimension mismatch: expected {z_total_dim} "
            f"(point_dim={point_dim} + feature_dim={feature_dim}), "
            f"got {z_flat_batch.shape[2]}"
        )
        
        # Extract positions and features from z
        z_positions, z_features = self._extract_positions_and_features(
            z_flat_batch, point_dim, feature_dim
        )
        
        # Apply Fourier encoding to z positions
        z_fourier = self._apply_fourier_encoding(z_positions)  # [batch_flat, z_num_points, fourier_feature_dim]
        
        # Combine Fourier features with z features via ConcatSquash
        # Handle case where feature_dim = 0 (no features, only positions)
        if feature_dim > 0:
            z_embed = self.z_concat_squash(z_fourier, z_features)  # [batch_flat, z_num_points, embed_dim]
        else:
            # Only Fourier features, no point features
            z_embed = self.z_concat_squash(z_fourier)  # [batch_flat, z_num_points, embed_dim]
        
        # Process x if provided
        x_embed = None
        x_num_points = 0
        x_mask_processed = None
        if x is not None:
            # Broadcast x to batch shape
            x = jnp.broadcast_to(x, batch_shape + x.shape[-2:])
            x_flat_batch = x.reshape(-1, *x.shape[-2:])  # [batch_flat, x_num_points, x_total_dim]
            x_num_points = x_flat_batch.shape[1]
            
            # Validate x shape
            x_total_dim = point_dim + feature_dim
            assert x_flat_batch.shape[2] == x_total_dim, (
                f"x total dimension mismatch: expected {x_total_dim} "
                f"(point_dim={point_dim} + feature_dim={feature_dim}), "
                f"got {x_flat_batch.shape[2]}"
            )
            
            # Extract positions and features from x
            x_positions, x_features = self._extract_positions_and_features(
                x_flat_batch, point_dim, feature_dim
            )
            
            # Apply Fourier encoding to x positions
            x_fourier = self._apply_fourier_encoding(x_positions)  # [batch_flat, x_num_points, fourier_feature_dim]
            
            # Combine Fourier features with x features via ConcatSquash
            # Handle case where feature_dim = 0 (no features, only positions)
            if feature_dim > 0:
                x_embed = self.x_concat_squash(x_fourier, x_features)  # [batch_flat, x_num_points, embed_dim]
            else:
                # Only Fourier features, no point features
                x_embed = self.x_concat_squash(x_fourier)  # [batch_flat, x_num_points, embed_dim]
            
            # Process x_mask if provided
            if x_mask is not None:
                x_mask = jnp.broadcast_to(x_mask, batch_shape + (x_num_points,))
                x_mask_processed = x_mask.reshape(-1, x_num_points)  # [batch_flat, x_num_points]
        
        # Concatenate z and x embeddings: z first, then x
        # [batch_flat, z_num_points + x_num_points, embed_dim]
        if x_embed is not None:
            zx_embed = jnp.concatenate([z_embed, x_embed], axis=1)
        else:
            zx_embed = z_embed  # [batch_flat, z_num_points, embed_dim]
        
        # Build combined mask for concatenated sequence (z + x)
        # z is never masked, only x positions use x_mask
        combined_mask = None
        if x_mask_processed is not None:
            batch_size_flat = zx_embed.shape[0]
            total_points = zx_embed.shape[1]
            
            # Start with all True (unmasked)
            combined_mask = jnp.ones((batch_size_flat, total_points), dtype=bool)
            
            # Apply x_mask to x positions
            # Final sequence structure: [z, x]
            # x positions are at indices [z_num_points, z_num_points + x_num_points)
            x_start = z_num_points
            x_end = x_start + x_num_points
            combined_mask = combined_mask.at[:, x_start:x_end].set(x_mask_processed)
        
        # Process time embedding (for time-conditioned standard attention)
        t_embed_vec = None
        time_conditioning_method = self.config.get("time_conditioning_method", "film")
        
        if t is not None and self.time_embed is not None and time_conditioning_method != "none":
            t = jnp.broadcast_to(t, batch_shape)
            t_flat = t.reshape(-1)
            t_embed_vec = self.time_embed(t_flat)  # [batch_flat, time_embed_dim]
        
        # Process concatenated z,x point clouds with self-attention blocks
        for self_attn_block in self.self_attention_blocks:
            zx_embed = self_attn_block(
                zx_embed, 
                t=t_embed_vec, 
                mask=combined_mask, 
                training=training
            )
        
        # Extract z portion from concatenated sequence
        z_embed = zx_embed[:, :z_num_points, :]  # [batch_flat, z_num_points, embed_dim]
        
        # Pass z through MLP
        embed_dim = self.config["embed_dim"]
        z_embed = self.z_mlp(z_embed, embed_dim, training=training)
        
        # Final output MLP: removes positional information and projects to z_total_dim (position + features)
        z_output = self.output_mlp(z_embed, embed_dim, training=training)  # [batch_flat, z_num_points, z_total_dim]
        
        # Reshape back to original batch shape
        output = z_output.reshape(batch_shape + (z_num_points, z_total_dim))
        
        return output

