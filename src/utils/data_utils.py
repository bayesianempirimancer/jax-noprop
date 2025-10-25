"""Utilities for loading and working with training data."""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import jax.numpy as jnp

from jax import Array
from ..expfam.ef import MultivariateNormal_tril

def infer_dimensions(eta_data: Array, metadata: Optional[Dict] = None) -> int:
    """
    Infer eta dimensions from the data or metadata.
    
    Args:
        eta_data: Input eta data array
        metadata: Optional metadata dictionary containing dimension info
        
    Returns:
        int: The inferred eta dimension
    """
    if metadata is not None and 'eta_dim' in metadata:
        eta_dim = metadata['eta_dim']
        print(f"Using eta_dim from metadata: {eta_dim}")
        
        # Print additional metadata info if available
        if 'ef_distribution_name' in metadata:
            print(f"Exponential family: {metadata['ef_distribution_name']}")
        if 'x_shape' in metadata:
            print(f"Data shape x: {metadata['x_shape']}")
        if 'x_dim' in metadata:
            print(f"Data x dimension: {metadata['x_dim']}")
    else:
        eta_dim = eta_data.shape[-1]
        print(f"Inferred eta_dim from data shape: {eta_dim}")
    
    return eta_dim

def load_data_with_metadata(data_file: str) -> Dict:
    """
    Load training data using metadata to determine format - completely dimension-agnostic.
    
    Args:
        data_file: Path to the training data pickle file
    
    Returns:
        Dictionary with train/val/test splits and metadata in whatever format is specified
    """
    import pickle
    from pathlib import Path
    
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading data from {data_file}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract format information from metadata
    metadata = data.get('metadata', {})
    eta_dim = metadata.get('eta_dim', data['train']['eta'].shape[-1])
    ef_name = metadata.get('ef_distribution_name', 'unknown')
    
    print(f"📊 Data format from metadata:")
    print(f"  eta_dim: {eta_dim}")
    print(f"  ef_distribution_name: {ef_name}")
    print(f"  Using data as-is (dimension-agnostic)")
    
    # Purge cov_TT to save memory
    for split in ['train', 'val', 'test']:
        if split in data and "cov_TT" in data[split]:
            del data[split]["cov_TT"]
    import gc; gc.collect()
    print("✅ Purged cov_tt elements from memory for optimization")
    
    return data

def list_data_files(data_dir: str = "data") -> list[Path]:
    """List all available training data files."""
    data_path = Path(data_dir)
    
    # If relative path doesn't exist, try relative to current file's parent
    if not data_path.exists():
        # Try relative to the src directory (where this file is located)
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        data_path = project_root / "data"
    
    if not data_path.exists():
        return []
    return list(data_path.glob("training_data_*.pkl"))


def load_latest_data(data_dir: str = "data", purge_cov_tt: bool = True) -> Tuple[Dict[str, Array], Dict[str, Array], str]:
    """Load the most recently created training data file."""
    data_files = list_data_files(data_dir)
    if not data_files:
        raise FileNotFoundError(f"No training data files found in {data_dir}")
    
    # Sort by modification time, newest first
    latest_file = sorted(data_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return load_training_data(str(latest_file), purge_cov_tt=purge_cov_tt)


def inspect_data(train_data: Dict[str, Array], val_data: Dict[str, Array], config_hash: str = "unknown"):
    """Print information about the loaded data."""
    print(f"📊 Dataset Information:")
    print(f"Config hash: {config_hash}")
    print(f"Train data: eta {train_data['eta'].shape}, y {train_data['y'].shape}")
    print(f"Val data: eta {val_data['eta'].shape}, y {val_data['y'].shape}")
    
    # Check if cov_tt data is present
    has_cov = 'cov' in train_data or 'cov' in val_data
    if has_cov:
        print(f"⚠️  Covariance data (cov_tt) is present - using extra memory!")
    else:
        print(f"✅ Covariance data (cov_tt) has been purged - memory optimized!")
    
    print(f"\n📈 Data Statistics:")
    print(f"Train eta range: [{train_data['eta'].min():.3f}, {train_data['eta'].max():.3f}]")
    print(f"Train y range: [{train_data['y'].min():.3f}, {train_data['y'].max():.3f}]")
    print(f"Val eta range: [{val_data['eta'].min():.3f}, {val_data['eta'].max():.3f}]")
    print(f"Val y range: [{val_data['y'].min():.3f}, {val_data['y'].max():.3f}]")
    
    print(f"\n🔍 Sample Data:")
    print(f"First 3 train etas:\n{train_data['eta'][:3]}")
    print(f"First 3 train ys:\n{train_data['y'][:3]}")


# Convenience function for interactive use
def quick_load(data_file: Optional[str] = None, purge_cov_tt: bool = True) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    """
    Quick load function for interactive use.
    
    Args:
        data_file: Optional path to specific data file. If None, loads the latest.
        purge_cov_tt: If True, removes cov_tt elements from memory to save space
        
    Returns:
        Tuple of (train_data, val_data)
    """
    if data_file is None:
        train_data, val_data, config_hash = load_latest_data(purge_cov_tt=purge_cov_tt)
    else:
        train_data, val_data, config_hash = load_training_data(data_file, purge_cov_tt=purge_cov_tt)
    
    inspect_data(train_data, val_data, config_hash)
    return train_data, val_data


def load_3d_gaussian_data(data_dir: Path, format: str = "tril", purge_cov_tt: bool = True) -> Tuple[Dict[str, Array], MultivariateNormal_tril]:
    """
    Load 3D Gaussian dataset and optionally convert to tril format.
    
    Args:
        data_dir: Directory containing data files
        format: "full" or "tril" format
        purge_cov_tt: If True, removes cov_tt elements from memory to save space
        
    Returns:
        Tuple of (data_dict, exponential_family_object)
    """
    # Find largest 3D dataset
    suitable_files = []
    for data_file in data_dir.glob("*.pkl"):
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            if data["train_eta"].shape[1] == 12:  # 3D Gaussian
                suitable_files.append((data_file, data["train_eta"].shape[0]))
        except Exception:
            continue
    
    if not suitable_files:
        raise FileNotFoundError("No 3D Gaussian datasets found!")
    
    best_file, n_samples = max(suitable_files, key=lambda x: x[1])
    ef = MultivariateNormal_tril(x_shape=(3,))
    
    print(f"Loading 3D Gaussian data from {best_file.name}")
    print(f"Dataset size: {n_samples} training samples")
    
    with open(best_file, 'rb') as f:
        data = pickle.load(f)
    
    # Purge cov_tt elements if requested to save memory
    if purge_cov_tt:
        # Remove cov_tt elements from the loaded data
        if "train" in data and "cov" in data["train"]:
            del data["train"]["cov"]
        if "val" in data and "cov" in data["val"]:
            del data["val"]["cov"]
        if "test" in data and "cov" in data["test"]:
            del data["test"]["cov"]
        # Force garbage collection to free memory immediately
        import gc
        gc.collect()
    
    if format == "tril":
        # Convert from full matrix format to triangular format
        def convert_to_tril(y_full):
            """Convert from full 12D format to tril 9D format."""
            batch_size = y_full.shape[0]
            
            # Extract mean (first 3 components)
            mean = y_full[:, :3]
            
            # Extract full matrix (last 9 components, reshaped to 3x3)
            full_matrix = y_full[:, 3:].reshape(batch_size, 3, 3)
            
            # Convert to lower triangular using ef.flatten_LT
            tril_matrix = ef.flatten_LT(full_matrix)
            
            # Combine mean and triangular matrix
            return jnp.concatenate([mean, tril_matrix], axis=1)
        
        # Convert all datasets
        converted_data = {
            "train_eta": data["train_eta"],
            "train_y": convert_to_tril(data["train_y"]),
            "val_eta": data["val_eta"],
            "val_y": convert_to_tril(data["val_y"])
        }
        
        print(f"Converted from full format (12D) to tril format (9D)")
    else:
        converted_data = data
    
    print(f"Output dimension: {converted_data['train_y'].shape[1]}")
    
    return converted_data, ef


def compute_ground_truth_3d_tril(eta: jnp.ndarray, ef: MultivariateNormal_tril) -> jnp.ndarray:
    """
    Compute ground truth statistics for 3D Gaussian using analytical formulas.
    Returns in triangular format.
    
    For multivariate Gaussian: Σ^{-1} = -2η₂, μ = Ση₁
    
    Args:
        eta: Natural parameters [batch_size, 12] where first 3 are η₁, last 9 are η₂ (flattened)
        ef: MultivariateNormal_tril exponential family object
        
    Returns:
        Expected sufficient statistics [batch_size, 9] in tril format
    """
    batch_size = eta.shape[0]
    
    # Extract η₁ (first 3 components) and η₂ (last 9 components)
    eta1 = eta[:, :3]  # Shape: (batch_size, 3)
    eta2 = eta[:, 3:].reshape(batch_size, 3, 3)  # Shape: (batch_size, 3, 3)
    
    # Compute Σ^{-1} = -2η₂
    Sigma_inv = -2.0 * eta2  # Shape: (batch_size, 3, 3)
    
    # Compute Σ = (Σ^{-1})^{-1}
    Sigma = jnp.linalg.inv(Sigma_inv)  # Shape: (batch_size, 3, 3)
    
    # Compute μ = Ση₁
    mu = jnp.einsum('bij,bj->bi', Sigma, eta1)  # Shape: (batch_size, 3)
    
    # Expected sufficient statistics: [μ, Σ + μμ^T]
    mu_muT = jnp.einsum('bi,bj->bij', mu, mu)  # Shape: (batch_size, 3, 3)
    expected_xxT = Sigma + mu_muT  # Shape: (batch_size, 3, 3)
    
    # Convert to triangular format using ef.flatten_LT
    expected_xxT_tril = ef.flatten_LT(expected_xxT)
    
    # Combine mean and triangular part
    expected_stats = jnp.concatenate([mu, expected_xxT_tril], axis=1)  # Shape: (batch_size, 9)
    
    return expected_stats

def load_ef_data(data_file="data/easy_3d_gaussian.pkl"):
    """Load and prepare exponential family training data with memory optimization."""
    print(f"Loading test data from {data_file}...")
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    train = data["train"]
    val = data["val"]
    test = data["test"]

    if "cov_TT" in data["train"]: del data["train"]["cov_TT"]
    if "cov_TT" in data["val"]: del data["val"]["cov_TT"]
    if "cov_TT" in data["test"]: del data["test"]["cov_TT"]
    import gc; gc.collect()
    
    return train, val, test, data.get('metadata', {})

def load_standardized_ep_data(data_file="data/easy_3d_gaussian.pkl"):
    """Load and prepare standardized exponential family training data with memory optimization."""
    print(f"Loading test data from {data_file}...")
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    eta_data = data["train"]["eta"]
    ground_truth = data["train"]["mu_T"]
    print(f"Data shapes: eta={eta_data.shape}, ground_truth={ground_truth.shape}")
    
    # Purge cov_TT to save memory
    if "cov_TT" in data["train"]: del data["train"]["cov_TT"]
    if "cov_TT" in data["val"]: del data["val"]["cov_TT"]
    if "cov_TT" in data["test"]: del data["test"]["cov_TT"]
    import gc; gc.collect()
    print("✅ Purged cov_tt elements from memory for optimization")
    
    return eta_data, ground_truth, data.get('metadata', {})
