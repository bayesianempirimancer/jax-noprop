"""
Weights loading and downloading utilities for pretrained models.
"""

from .loader import (
    load_pretrained_model,  # Generic loader
    load_pretrained_dinov2_vits14,
    load_pretrained_dinov2_vits14_reg,
    load_pretrained_dinov2_vitb14,
    load_pretrained_dinov2_vitb14_reg,
    load_pretrained_dinov2_vitl14,
    load_pretrained_dinov2_vitl14_reg,
    load_pretrained_dinov2_vitg14,
    load_pretrained_dinov2_vitg14_reg,
    load_pretrained_dinov2,  # Backward compatibility alias
    load_pretrained_dinov2_params_only,
)

from .downloader import (
    download_and_save_weights,
    download_dinov2_vits14,
    download_dinov2_vits14_reg,
    download_dinov2_vitb14,
    download_dinov2_vitb14_reg,
    download_dinov2_vitl14,
    download_dinov2_vitl14_reg,
    download_dinov2_vitg14,
    download_dinov2_vitg14_reg,
)

__all__ = [
    # Generic loading function
    "load_pretrained_model",
    # Specific loading functions
    "load_pretrained_dinov2_vits14",
    "load_pretrained_dinov2_vits14_reg",
    "load_pretrained_dinov2_vitb14",
    "load_pretrained_dinov2_vitb14_reg",
    "load_pretrained_dinov2_vitl14",
    "load_pretrained_dinov2_vitl14_reg",
    "load_pretrained_dinov2_vitg14",
    "load_pretrained_dinov2_vitg14_reg",
    "load_pretrained_dinov2",  # Backward compatibility alias
    "load_pretrained_dinov2_params_only",
    # Downloading functions
    "download_and_save_weights",
    "download_dinov2_vits14",
    "download_dinov2_vits14_reg",
    "download_dinov2_vitb14",
    "download_dinov2_vitb14_reg",
    "download_dinov2_vitl14",
    "download_dinov2_vitl14_reg",
    "download_dinov2_vitg14",
    "download_dinov2_vitg14_reg",
]
