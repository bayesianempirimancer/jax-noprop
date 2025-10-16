"""
Weights loading utilities for pretrained models.

This module provides functions to load pretrained model weights from pickle files.
It includes both generic and convenience functions for different model variants.

Example Usage:
    # Generic loader (works with any pickle file)
    from src.models.weights import load_pretrained_model
    model, params = load_pretrained_model("path/to/dinov2_vits14_local.pkl")
    output = model.apply(params, input_data)
    
    # Convenience functions (specific models)
    from src.models.weights import load_pretrained_dinov2_vits14
    model, params = load_pretrained_dinov2_vits14()  # Uses default path
    output = model.apply(params, input_data)
    
    # Download and load workflow
    from src.models.weights import download_dinov2_vits14, load_pretrained_dinov2_vits14
    result = download_dinov2_vits14()  # Download and save
    model, params = load_pretrained_dinov2_vits14()  # Load saved weights
    output = model.apply(params, input_data)
    
    # Handle missing pickle file by downloading first
    try:
        model, params = load_pretrained_model("missing_model.pkl")
    except FileNotFoundError:
        print("Model not found locally, downloading...")
        result = download_dinov2_vits14()  # Download the model
        model, params = load_pretrained_model(result['weights_path'])  # Load it
    output = model.apply(params, input_data)
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from ..vit import DinoV2
from ..configs import DINOV2_VITS14


def load_pretrained_model(weights_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Generic function to load any pretrained model from a pickle file.
    
    This function automatically determines the model class and creates the appropriate
    model instance based on the data stored in the pickle file.
    
    Args:
        weights_path: Path to the pickle file containing model data
        
    Returns:
        Tuple of (model, params) ready for use with model.apply(params, x)
        
    Raises:
        FileNotFoundError: If the weights file doesn't exist
        KeyError: If the pickle file doesn't contain required fields
        NotImplementedError: If the model class is not supported
        
    Example:
        model, params = load_pretrained_model("path/to/dinov2_vits14_local.pkl")
        output = model.apply(params, input_data)
    """
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    # Load the saved weights and configuration
    with open(weights_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract required fields
    try:
        model_class = data['class']
        model_config = data['config']
        params = data['params']
    except KeyError as e:
        raise KeyError(f"Missing required field in pickle file: {e}.  Maybe use downloader.py to download the weights?s")
    
    # Create the appropriate model based on class
    if model_class == "dinov2":
        model = DinoV2(**model_config)
    else:
        raise NotImplementedError(f"Model class '{model_class}' is not supported. Supported classes: ['dinov2']")
    
    return model, params


def load_pretrained_dinov2_vits14(weights_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load pretrained DinoV2-S model with converted flax.linen weights.
    
    Args:
        weights_path: Path to the weights file. If None, uses default VITS14 path.
        
    Returns:
        Tuple of (model, params) ready for use with model.apply(params, x)
    """
    if weights_path is None:
        weights_path = Path(__file__).parent / "dinov2_vits14_local.pkl"
    
    # Load the saved weights and configuration
    with open(weights_path, 'rb') as f:
        data = pickle.load(f)
    
    params = data['params']
    model_config = data['config']
    
    # Create the model with the saved configuration
    model = DinoV2(**model_config)
    
    return model, params


def load_pretrained_dinov2_vits14_reg(weights_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load pretrained DinoV2-S with register tokens model.
    
    Args:
        weights_path: Path to the weights file. If None, uses default VITS14_REG path.
        
    Returns:
        Tuple of (model, params) ready for use with model.apply(params, x)
    """
    if weights_path is None:
        weights_path = Path(__file__).parent / "dinov2_vits14_reg_local.pkl"
    
    with open(weights_path, 'rb') as f:
        data = pickle.load(f)
    
    params = data['params']
    model_config = data['config']
    model = DinoV2(**model_config)
    
    return model, params


def load_pretrained_dinov2_vitb14(weights_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load pretrained DinoV2-B model.
    
    Args:
        weights_path: Path to the weights file. If None, uses default VITB14 path.
        
    Returns:
        Tuple of (model, params) ready for use with model.apply(params, x)
    """
    if weights_path is None:
        weights_path = Path(__file__).parent / "dinov2_vitb14_local.pkl"
    
    with open(weights_path, 'rb') as f:
        data = pickle.load(f)
    
    params = data['params']
    model_config = data['config']
    model = DinoV2(**model_config)
    
    return model, params


def load_pretrained_dinov2_vitb14_reg(weights_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load pretrained DinoV2-B with register tokens model.
    
    Args:
        weights_path: Path to the weights file. If None, uses default VITB14_REG path.
        
    Returns:
        Tuple of (model, params) ready for use with model.apply(params, x)
    """
    if weights_path is None:
        weights_path = Path(__file__).parent / "dinov2_vitb14_reg_local.pkl"
    
    with open(weights_path, 'rb') as f:
        data = pickle.load(f)
    
    params = data['params']
    model_config = data['config']
    model = DinoV2(**model_config)
    
    return model, params


def load_pretrained_dinov2_vitl14(weights_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load pretrained DinoV2-L model.
    
    Args:
        weights_path: Path to the weights file. If None, uses default VITL14 path.
        
    Returns:
        Tuple of (model, params) ready for use with model.apply(params, x)
    """
    if weights_path is None:
        weights_path = Path(__file__).parent / "dinov2_vitl14_local.pkl"
    
    with open(weights_path, 'rb') as f:
        data = pickle.load(f)
    
    params = data['params']
    model_config = data['config']
    model = DinoV2(**model_config)
    
    return model, params


def load_pretrained_dinov2_vitl14_reg(weights_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load pretrained DinoV2-L with register tokens model.
    
    Args:
        weights_path: Path to the weights file. If None, uses default VITL14_REG path.
        
    Returns:
        Tuple of (model, params) ready for use with model.apply(params, x)
    """
    if weights_path is None:
        weights_path = Path(__file__).parent / "dinov2_vitl14_reg_local.pkl"
    
    with open(weights_path, 'rb') as f:
        data = pickle.load(f)
    
    params = data['params']
    model_config = data['config']
    model = DinoV2(**model_config)
    
    return model, params


def load_pretrained_dinov2_vitg14(weights_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load pretrained DinoV2-G model.
    
    Args:
        weights_path: Path to the weights file. If None, uses default VITG14 path.
        
    Returns:
        Tuple of (model, params) ready for use with model.apply(params, x)
    """
    if weights_path is None:
        weights_path = Path(__file__).parent / "dinov2_vitg14_local.pkl"
    
    with open(weights_path, 'rb') as f:
        data = pickle.load(f)
    
    params = data['params']
    model_config = data['config']
    model = DinoV2(**model_config)
    
    return model, params


def load_pretrained_dinov2_vitg14_reg(weights_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load pretrained DinoV2-G with register tokens model.
    
    Args:
        weights_path: Path to the weights file. If None, uses default VITG14_REG path.
        
    Returns:
        Tuple of (model, params) ready for use with model.apply(params, x)
    """
    if weights_path is None:
        weights_path = Path(__file__).parent / "dinov2_vitg14_reg_local.pkl"
    
    with open(weights_path, 'rb') as f:
        data = pickle.load(f)
    
    params = data['params']
    model_config = data['config']
    model = DinoV2(**model_config)
    
    return model, params


# Backward compatibility alias
def load_pretrained_dinov2(weights_path: str = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load pretrained DinoV2 model (backward compatibility alias for VITS14).
    
    Args:
        weights_path: Path to the weights file. If None, uses default VITS14 path.
        
    Returns:
        Tuple of (model, params) ready for use with model.apply(params, x)
    """
    return load_pretrained_dinov2_vits14(weights_path)


def load_pretrained_dinov2_params_only(weights_path: str = None) -> Dict[str, Any]:
    """
    Load only the pretrained parameters.
    
    Args:
        weights_path: Path to the parameters file. If None, uses default path.
        
    Returns:
        Parameters dictionary ready for use with model.apply(params, x)
    """
    if weights_path is None:
        weights_path = Path(__file__).parent / "dinov2_vits14_params.pkl"
    
    with open(weights_path, 'rb') as f:
        params = pickle.load(f)
    
    return params


