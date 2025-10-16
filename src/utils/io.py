import logging
import shutil
import pickle
from pathlib import Path
from typing import Dict, Optional, Any
from urllib.request import urlretrieve

import orbax
import py7zr
import flax.linen as nn
from flax.training import orbax_utils

DEFAULT_MODEL_DIR = Path.home() / ".jimmy" / "hub" / "checkpoints"
FILTERS = [{"id": py7zr.FILTER_ZSTD, "level": 3}]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_orbax(
    params: Dict[str, Any],
    specifications: Dict,
    name: str,
    model_dir: Optional[str] = None,
    overwrite: bool = False,
    compress: bool = True,
) -> Path:
    """
    Save model parameters and specifications using Orbax checkpointing.

    Args:
        params (Dict[str, Any]): Model parameters to save (flax.linen format).
        specifications (Dict): Model specifications to save.
        name (str): Name of the model.
        model_dir (Optional[str], optional): Directory to save the model. Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        compress (bool, optional): Whether to compress the saved files. Defaults to True.

    Returns:
        Path: Path to the saved model.
    """
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    ckpt_dir = model_dir / name
    compressed_ckpt = model_dir / f"{name}.jim"

    if overwrite:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        if compressed_ckpt.exists():
            shutil.rmtree(compressed_ckpt)
    model_dir.mkdir(exist_ok=True, parents=True)

    ckpt = {
        "params": params,
        "specifications": specifications,
    }

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(ckpt_dir, ckpt, save_args=save_args)

    if compress:
        with py7zr.SevenZipFile(compressed_ckpt, "w",
                                filters=FILTERS) as archive:
            archive.writeall(ckpt_dir, arcname=name)
        shutil.rmtree(ckpt_dir)

        logger.info(f"Saved {name} to {compressed_ckpt}.")
        return compressed_ckpt

    logger.info(f"Saved {name} to {ckpt_dir}.")
    return ckpt_dir


def save_pkl(
    params: Dict[str, Any],
    specifications: Dict,
    name: str,
    model_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """
    Save model parameters and specifications using pickle serialization.
    
    Args:
        params (Dict[str, Any]): Model parameters to save (flax.linen format).
        specifications (Dict): Model specifications to save.
        name (str): Name of the model.
        model_dir (Optional[str], optional): Directory to save the model. Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.

    Returns:
        Path: Path to the saved pickle file.
    """
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Save directly as pickle file
    pkl_file = model_dir / f"{name}.pkl"

    if overwrite and pkl_file.exists():
        pkl_file.unlink()

    ckpt = {
        "name": specifications["name"],
        "class": specifications["class"],
        "config": specifications["config"],
        "params": params,
    }

    with open(pkl_file, 'wb') as f:
        pickle.dump(ckpt, f)

    logger.info(f"Saved {name} to {pkl_file}.")
    return pkl_file


def load(name: str,
         params: Dict[str, Any],
         specifications: Dict,
         url: Optional[str] = None,
         model_dir: Optional[str] = None):
    """
    Load model parameters and specifications.

    Args:
        name (str): Name of the model to load.
        params (Dict[str, Any]): Initial model parameters structure (flax.linen format).
        specifications (Dict): Model specifications.
        url (Optional[str], optional): URL to download the model from. Defaults to None.
        model_dir (Optional[str], optional): Directory to load the model from. Defaults to None.

    Returns:
        Dict: Loaded model parameters and specifications.

    Raises:
        ValueError: If the model is not found and no URL is provided.
    """
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir = model_dir / name
    compressed_ckpt = model_dir / f"{name}.jim"

    if not ckpt_dir.exists() and not compressed_ckpt.exists() and url is None:
        raise ValueError(f"{name} not found in {model_dir}.")

    if (not ckpt_dir.exists() and not compressed_ckpt.exists()) and url:
        logger.info(f"Downloading {name} to {compressed_ckpt}...")
        urlretrieve(url, compressed_ckpt)

    if not ckpt_dir.exists() and compressed_ckpt.exists():
        archive = py7zr.SevenZipFile(compressed_ckpt, mode="r")
        archive.extractall(model_dir)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # Load without target structure to avoid parameter structure mismatch
    raw = orbax_checkpointer.restore(ckpt_dir)

    return raw
