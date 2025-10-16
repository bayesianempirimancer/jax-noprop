"""
Weight Downloader for Pretrained Models

This module provides functionality to download, convert, and save pretrained model weights
from external sources (like Hugging Face) into a standardized local format.

Key Features:
- Downloads pretrained weights from URLs (e.g., Jimmy/flax.nnx format)
- Converts parameter structures from Jimmy format to flax.linen format
- Tests model functionality after conversion
- Saves weights in a standardized pickle format for easy loading
- Provides convenience functions for all DinoV2 model variants

Usage:
    # Download and save DinoV2-S weights (ultra-simplified - all defaults)
    from src.models.weights import download_dinov2_vits14
    
    result = download_dinov2_vits14()  # Uses default rngs and Hugging Face URL
    
    # Or specify custom parameters
    import jax
    rngs = jax.random.PRNGKey(123)
    result = download_dinov2_vits14(rngs, url="https://custom-url.com/model.jim")
    
    # Load the saved weights
    from src.models.weights import load_pretrained_dinov2
    model, params = load_pretrained_dinov2()

File Structure:
    weights/
    ├── dinov2_vits14_local.pkl    # Complete model data (name, class, config, params)
    ├── dinov2_vitb14_local.pkl    # Other model variants...
    └── ...

The saved pickle files contain:
    {
        "name": "dinov2_vits14",           # Model identifier
        "class": "dinov2",                 # Model type
        "config": {...},                   # Architecture parameters
        "params": {...}                    # Model weights
    }
"""

from typing import Optional, Dict, Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from pathlib import Path

from ...utils.io import load, save_pkl
from ...layers import Mlp, SwiGLU
from .. import vit
from ..configs import DINOV2_VITS14, DINOV2_VITS14_REG, DINOV2_VITB14, DINOV2_VITB14_REG, DINOV2_VITL14, DINOV2_VITL14_REG, DINOV2_VITG14, DINOV2_VITG14_REG


def convert_jimmy_params_to_flax_linen(jimmy_params: Dict[str, Any], flax_linen_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Jimmy/flax.nnx parameter format to flax.linen format.
    
    This function handles the structural differences between Jimmy's parameter format
    and the standard flax.linen format expected by our models.
    
    Key conversions:
    - Jimmy: flat structure with "raw_value" wrappers (e.g., "patch_embed/proj/kernel/raw_value")
    - flax.linen: nested structure (e.g., "params/PatchEmbed_0/proj/kernel")
    - Layer scaling: handles "ls1/gamma" and "ls2/gamma" parameters
    
    Args:
        jimmy_params: Parameters in Jimmy format (flat structure with raw_value)
        flax_linen_params: Parameters in flax.linen format (nested structure)
        
    Returns:
        Converted parameters in flax.linen format ready for model.apply()
    """
    converted_params = {"params": {}}
    
    # Helper function to extract raw_value from Jimmy format
    def extract_raw_value(param):
        if isinstance(param, dict) and "raw_value" in param:
            return param["raw_value"]
        return param
    
    # Convert patch embedding
    if "patch_embed" in jimmy_params:
        converted_params["params"]["PatchEmbed_0"] = {
            "proj": {
                "kernel": extract_raw_value(jimmy_params["patch_embed"]["proj"]["kernel"]),
                "bias": extract_raw_value(jimmy_params["patch_embed"]["proj"]["bias"])
            }
        }
    
    # Convert class token
    if "cls_token" in jimmy_params:
        converted_params["params"]["cls_token"] = extract_raw_value(jimmy_params["cls_token"])
    
    # Convert register tokens (if present)
    if "register_tokens" in flax_linen_params["params"]:
        # Use the structure from flax_linen_params but we might not have this in Jimmy params
        converted_params["params"]["register_tokens"] = flax_linen_params["params"]["register_tokens"]
    
    # Convert positional embedding
    if "pos_embed" in jimmy_params:
        converted_params["params"]["pos_embed"] = extract_raw_value(jimmy_params["pos_embed"])
    
    # Convert transformer blocks
    for i in range(12):  # DinoV2-S has 12 blocks
        block_key = f"blocks.{i}"
        if block_key in jimmy_params:
            block_params = {
                "norm1": {
                    "scale": extract_raw_value(jimmy_params[block_key]["norm1"]["scale"]),
                    "bias": extract_raw_value(jimmy_params[block_key]["norm1"]["bias"])
                },
                "attn": {
                    "qkv": {
                        "kernel": extract_raw_value(jimmy_params[block_key]["attn"]["qkv"]["kernel"]),
                        "bias": extract_raw_value(jimmy_params[block_key]["attn"]["qkv"]["bias"])
                    },
                    "proj": {
                        "kernel": extract_raw_value(jimmy_params[block_key]["attn"]["proj"]["kernel"]),
                        "bias": extract_raw_value(jimmy_params[block_key]["attn"]["proj"]["bias"])
                    }
                },
                "norm2": {
                    "scale": extract_raw_value(jimmy_params[block_key]["norm2"]["scale"]),
                    "bias": extract_raw_value(jimmy_params[block_key]["norm2"]["bias"])
                },
                "mlp": {
                    "fc1": {
                        "kernel": extract_raw_value(jimmy_params[block_key]["mlp"]["fc1"]["kernel"]),
                        "bias": extract_raw_value(jimmy_params[block_key]["mlp"]["fc1"]["bias"])
                    },
                    "fc2": {
                        "kernel": extract_raw_value(jimmy_params[block_key]["mlp"]["fc2"]["kernel"]),
                        "bias": extract_raw_value(jimmy_params[block_key]["mlp"]["fc2"]["bias"])
                    }
                }
            }
            
            # Add layer scaling parameters if they exist in Jimmy params
            if "ls1" in jimmy_params[block_key]:
                block_params["ls1"] = {
                    "gamma": extract_raw_value(jimmy_params[block_key]["ls1"]["gamma"])
                }
            if "ls2" in jimmy_params[block_key]:
                block_params["ls2"] = {
                    "gamma": extract_raw_value(jimmy_params[block_key]["ls2"]["gamma"])
                }
            
            converted_params["params"][f"ViTBlock_{i}"] = block_params
    
    # Convert final layer norm
    if "norm" in jimmy_params:
        converted_params["params"]["LayerNorm_0"] = {
            "scale": extract_raw_value(jimmy_params["norm"]["scale"]),
            "bias": extract_raw_value(jimmy_params["norm"]["bias"])
        }
    
    return converted_params


def load_model(specifications: dict,
               rngs: jax.random.PRNGKey,
               url: Optional[str] = None,
               pretrained: bool = True,
               **kwargs) -> tuple[nn.Module, Dict[str, Any]]:
    """
    Load a model based on the given specifications.
    
    This function creates a model instance and optionally loads pretrained weights.
    It handles the conversion from Jimmy format to flax.linen format automatically.
    
    Args:
        specifications (dict): Model specifications from configs.py containing:
            - "name": Model identifier (e.g., "dinov2_vits14")
            - "class": Model type (e.g., "dinov2")
            - "config": Architecture parameters (embed_dim, num_heads, etc.)
        rngs (jax.random.PRNGKey): Random number generator key for model initialization
        url (Optional[str], optional): URL to download pretrained weights from
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True
        **kwargs: Additional keyword arguments to override config parameters

    Returns:
        tuple[nn.Module, Dict[str, Any]]: The loaded model and its parameters
        
    Raises:
        NotImplementedError: If the specified model class is not implemented
        
    Example:
        specs = DINOV2_VITS14
        model, params = load_model(specs, rngs, url="https://...", pretrained=True)
        output = model.apply(params, input_data)
    """
    cls = specifications["class"]
    config = specifications["config"] | kwargs

    match cls:
        case "dinov2":
            model = vit.DinoV2(**config)
        case _:
            raise NotImplementedError(f"{cls} not implemented.")

    # Initialize model parameters
    dummy_input = jnp.ones((1, config.get("img_size", 224), config.get("img_size", 224), config.get("in_channels", 3)))
    params = model.init(rngs, dummy_input)

    if pretrained:
        # Load pretrained weights
        raw = load(
            name=specifications["name"],
            url=url,
            params=params,
            specifications=specifications,
        )
        # Convert Jimmy/flax.nnx format to flax.linen format
        params = convert_jimmy_params_to_flax_linen(raw["params"], params)

    return model, params


def download_and_save_weights(specifications: Dict[str, Any],
                             url: str,
                             rngs: jax.random.PRNGKey,
                             weights_dir: Optional[str] = None,
                             overwrite: bool = True,
                             **kwargs) -> Dict[str, Any]:
    """
    Download pretrained weights, convert them to flax.linen format, test the model,
    and save them to the weights directory.
    
    This is the main function that orchestrates the entire download-convert-test-save
    workflow. It ensures the model works correctly before saving.
    
    Workflow:
    1. Download pretrained weights from URL (Jimmy format)
    2. Convert parameter structure to flax.linen format
    3. Test model functionality with dummy input
    4. Save weights in standardized pickle format
    
    Args:
        specifications: Model specifications from configs.py containing name, class, config
        url: URL to download pretrained weights from (e.g., Hugging Face)
        rngs: Random number generator key for model initialization
        weights_dir: Directory to save weights. If None, uses default weights directory
        overwrite: Whether to overwrite existing files
        **kwargs: Additional keyword arguments to override config parameters
        
    Returns:
        Dictionary containing:
            - 'model_name': Name of the model
            - 'weights_dir': Directory where weights were saved
            - 'weights_path': Path to the saved pickle file
            - 'output_shape': Shape of model output (for verification)
            - 'model_config': Model configuration used
            
    Raises:
        Exception: If model initialization, testing, or saving fails
        
    Example:
        result = download_and_save_weights(
            DINOV2_VITS14, 
            "https://huggingface.co/.../dinov2_vits14.jim",
            rngs
        )
        print(f"Saved to: {result['weights_path']}")
    """
    print(f"🔄 Downloading and converting {specifications['name']}...")
    
    # Step 1: Download and convert weights
    model, params = load_model(specifications, rngs, url=url, pretrained=True, **kwargs)
    print(f"   ✓ Model loaded and weights converted successfully!")
    
    # Step 2: Test the model to ensure it works
    print("   🧪 Testing model initialization...")
    config = specifications['config']
    img_size = config.get('img_size', 224)
    dummy_input = jnp.ones((1, img_size, img_size, 3))
    output = model.apply(params, dummy_input)
    print(f"   ✓ Model test successful! Output shape: {output.shape}")
    
    # Step 3: Set up weights directory
    if weights_dir is None:
        weights_dir = Path(__file__).parent
    else:
        weights_dir = Path(weights_dir)
    
    weights_dir.mkdir(exist_ok=True)
    
    # Step 4: Save weights using the name from specifications
    model_name = specifications['name']
    base_name = f"{model_name}_local"
    
    print(f"   💾 Saving weights to {weights_dir}...")
    
    # Save using our custom save_pkl function
    save_path = save_pkl(
        params=params,
        specifications=specifications,
        name=base_name,
        model_dir=str(weights_dir),
        overwrite=overwrite
    )
    
    print(f"   ✓ Weights saved successfully!")
    print(f"      - Complete: {save_path}")
    
    return {
        'model_name': model_name,
        'weights_dir': str(weights_dir),
        'weights_path': str(save_path),
        'output_shape': output.shape,
        'model_config': specifications['config']
    }


# =============================================================================
# Convenience functions for downloading specific model variants
# =============================================================================

def download_dinov2_vits14(rngs: jax.random.PRNGKey = jax.random.PRNGKey(42),
                          url: str = "https://huggingface.co/poiretclement/dinov2_jax/resolve/main/dinov2_vits14.jim",
                          weights_dir: Optional[str] = None,
                          overwrite: bool = True) -> Dict[str, Any]:
    """
    Download and save DinoV2-S weights.
    
    Convenience function that downloads DinoV2-Small (384 dimensions, 12 layers)
    weights and saves them in the standardized local format.
    
    Args:
        rngs: Random number generator key for model initialization. Defaults to jax.random.PRNGKey(42)
        url: URL to download pretrained weights from. Defaults to the official Hugging Face URL
        weights_dir: Directory to save weights. If None, uses default weights directory
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with information about the saved files
        
    Example:
        # Use all defaults
        result = download_dinov2_vits14()
        
        # Or specify custom parameters
        rngs = jax.random.PRNGKey(123)
        result = download_dinov2_vits14(rngs, url="https://custom-url.com/model.jim")
    """
    return download_and_save_weights(DINOV2_VITS14, url, rngs, weights_dir, overwrite)


def download_dinov2_vits14_reg(rngs: jax.random.PRNGKey = jax.random.PRNGKey(42),
                              url: str = "https://huggingface.co/poiretclement/dinov2_jax/resolve/main/dinov2_vits14_reg.jim",
                              weights_dir: Optional[str] = None,
                              overwrite: bool = True) -> Dict[str, Any]:
    """
    Download and save DinoV2-S with register tokens weights.
    
    Same as DinoV2-S but includes 4 register tokens for improved performance.
    
    Args:
        rngs: Random number generator key for model initialization
        url: URL to download pretrained weights from. Defaults to the official Hugging Face URL
        weights_dir: Directory to save weights. If None, uses default weights directory
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with information about the saved files
    """
    return download_and_save_weights(DINOV2_VITS14_REG, url, rngs, weights_dir, overwrite)


def download_dinov2_vitb14(rngs: jax.random.PRNGKey = jax.random.PRNGKey(42),
                          url: str = "https://huggingface.co/poiretclement/dinov2_jax/resolve/main/dinov2_vitb14.jim",
                          weights_dir: Optional[str] = None,
                          overwrite: bool = True) -> Dict[str, Any]:
    """
    Download and save DinoV2-B weights.
    
    Downloads DinoV2-Base (768 dimensions, 12 layers) weights.
    
    Args:
        rngs: Random number generator key for model initialization
        url: URL to download pretrained weights from. Defaults to the official Hugging Face URL
        weights_dir: Directory to save weights. If None, uses default weights directory
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with information about the saved files
    """
    return download_and_save_weights(DINOV2_VITB14, url, rngs, weights_dir, overwrite)


def download_dinov2_vitb14_reg(rngs: jax.random.PRNGKey = jax.random.PRNGKey(42),
                              url: str = "https://huggingface.co/poiretclement/dinov2_jax/resolve/main/dinov2_vitb14_reg.jim",
                              weights_dir: Optional[str] = None,
                              overwrite: bool = True) -> Dict[str, Any]:
    """
    Download and save DinoV2-B with register tokens weights.
    
    Same as DinoV2-B but includes 4 register tokens for improved performance.
    
    Args:
        rngs: Random number generator key for model initialization
        url: URL to download pretrained weights from. Defaults to the official Hugging Face URL
        weights_dir: Directory to save weights. If None, uses default weights directory
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with information about the saved files
    """
    return download_and_save_weights(DINOV2_VITB14_REG, url, rngs, weights_dir, overwrite)


def download_dinov2_vitl14(rngs: jax.random.PRNGKey = jax.random.PRNGKey(42),
                          url: str = "https://huggingface.co/poiretclement/dinov2_jax/resolve/main/dinov2_vitl14.jim",
                          weights_dir: Optional[str] = None,
                          overwrite: bool = True) -> Dict[str, Any]:
    """
    Download and save DinoV2-L weights.
    
    Downloads DinoV2-Large (1024 dimensions, 24 layers) weights.
    
    Args:
        rngs: Random number generator key for model initialization
        url: URL to download pretrained weights from. Defaults to the official Hugging Face URL
        weights_dir: Directory to save weights. If None, uses default weights directory
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with information about the saved files
    """
    return download_and_save_weights(DINOV2_VITL14, url, rngs, weights_dir, overwrite)


def download_dinov2_vitl14_reg(rngs: jax.random.PRNGKey = jax.random.PRNGKey(42),
                              url: str = "https://huggingface.co/poiretclement/dinov2_jax/resolve/main/dinov2_vitl14_reg.jim",
                              weights_dir: Optional[str] = None,
                              overwrite: bool = True) -> Dict[str, Any]:
    """
    Download and save DinoV2-L with register tokens weights.
    
    Same as DinoV2-L but includes 4 register tokens for improved performance.
    
    Args:
        rngs: Random number generator key for model initialization
        url: URL to download pretrained weights from. Defaults to the official Hugging Face URL
        weights_dir: Directory to save weights. If None, uses default weights directory
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with information about the saved files
    """
    return download_and_save_weights(DINOV2_VITL14_REG, url, rngs, weights_dir, overwrite)


def download_dinov2_vitg14(rngs: jax.random.PRNGKey = jax.random.PRNGKey(42),
                          url: str = "https://huggingface.co/poiretclement/dinov2_jax/resolve/main/dinov2_vitg14.jim",
                          weights_dir: Optional[str] = None,
                          overwrite: bool = True) -> Dict[str, Any]:
    """
    Download and save DinoV2-G weights.
    
    Downloads DinoV2-Giant (1536 dimensions, 40 layers) weights.
    
    Args:
        rngs: Random number generator key for model initialization
        url: URL to download pretrained weights from. Defaults to the official Hugging Face URL
        weights_dir: Directory to save weights. If None, uses default weights directory
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with information about the saved files
    """
    return download_and_save_weights(DINOV2_VITG14, url, rngs, weights_dir, overwrite)


def download_dinov2_vitg14_reg(rngs: jax.random.PRNGKey = jax.random.PRNGKey(42),
                              url: str = "https://huggingface.co/poiretclement/dinov2_jax/resolve/main/dinov2_vitg14_reg.jim",
                              weights_dir: Optional[str] = None,
                              overwrite: bool = True) -> Dict[str, Any]:
    """
    Download and save DinoV2-G with register tokens weights.
    
    Same as DinoV2-G but includes 4 register tokens for improved performance.
    
    Args:
        rngs: Random number generator key for model initialization
        url: URL to download pretrained weights from. Defaults to the official Hugging Face URL
        weights_dir: Directory to save weights. If None, uses default weights directory
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with information about the saved files
    """
    return download_and_save_weights(DINOV2_VITG14_REG, url, rngs, weights_dir, overwrite)


# =============================================================================
# Summary
# =============================================================================
"""
This module provides a complete workflow for downloading, converting, and saving
pretrained model weights:

1. Use convenience functions (e.g., download_dinov2_vits14) for specific models
2. Or use download_and_save_weights for custom specifications
3. Load saved weights with load_pretrained_dinov2 from loader.py

All functions handle the conversion from Jimmy/flax.nnx format to flax.linen format
automatically, ensuring compatibility with the rest of the codebase.
"""
