"""
Base model class with common functionality for all ET models.

This provides the standard interface and common methods that all models should implement.
"""

import os
import pickle
from typing import Dict, Optional, Type, TypeVar, Generic
import jax.numpy as jnp
from flax import linen as nn

# Type variable for configuration classes
ConfigType = TypeVar('ConfigType')

class BaseModel(nn.Module, Generic[ConfigType]):
    """
    Base class for all Exponential Family (ET) models.
    
    This class provides the standard interface and common functionality
    that all ET models should implement.
    """
    
    config: ConfigType
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through the model.
        
        Args:
            eta: Natural parameters of shape (batch_size, eta_dim)
            training: Whether in training mode
            rngs: Random number generator keys for stochastic operations
            **kwargs: Additional model-specific arguments
            
        Returns:
            Tuple of (predictions, internal_loss):
            - predictions: Model output of shape (batch_size, output_dim)
            - internal_loss: Internal regularization loss (scalar, usually 0.0)
        """
        raise NotImplementedError("Subclasses must implement __call__ method")
    
    def predict(self, params: Dict, eta: jnp.ndarray, rngs: dict = None, **kwargs) -> jnp.ndarray:
        """
        Make predictions for inference.
        
        Args:
            params: Model parameters
            eta: Natural parameters of shape (batch_size, eta_dim)
            rngs: Random number generator keys for stochastic operations (optional)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def loss(self, params: Dict, eta: jnp.ndarray, targets: jnp.ndarray, 
             training: bool = True, rngs: dict = None, **kwargs) -> jnp.ndarray:
        """
        Compute training loss.
        
        Args:
            params: Model parameters
            eta: Natural parameters of shape (batch_size, eta_dim)
            targets: Target values of shape (batch_size, output_dim)
            training: Whether in training mode
            rngs: Random number generator keys for stochastic operations (optional)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Loss value (scalar)
        """
        raise NotImplementedError("Subclasses must implement loss method")
    
    @classmethod
    def from_config(cls, config: ConfigType, **kwargs):
        """
        Create model from configuration.
        
        Args:
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Initialized model
        """
        return cls(config=config, **kwargs)
    
    def save(self, save_directory: str, params: Optional[Dict] = None):
        """
        Save model and configuration to directory.
        
        Creates two files:
        - model.pkl: Contains both config and params for exact preservation
        - config.yaml: Human-readable config for inspection and version control
        
        Args:
            save_directory: Directory to save to
            params: Model parameters to save (optional)
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Prepare model data
        model_data = {
            'config': self.config,
            'params': params,
            'model_type': self.config.model_type,
            'model_name': self.config.model_name
        }
        
        # Save complete model as pickle (config + params)
        model_path = os.path.join(save_directory, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        # Save human-readable config as YAML
        try:
            import yaml
            config_dict = self.config.to_dict()
            config_path = os.path.join(save_directory, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=True)
        except ImportError:
            # Fallback to JSON if YAML not available
            import json
            config_dict = self.config.to_dict()
            config_path = os.path.join(save_directory, "config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)
    
    @classmethod
    def load(cls, model_directory: str, load_params: bool = False, **kwargs):
        """
        Load model from directory.
        
        Args:
            model_directory: Directory containing the saved model
            load_params: Whether to also load the model parameters
            **kwargs: Additional arguments to override config
            
        Returns:
            Model instance (and optionally params if load_params=True)
        """
        from pathlib import Path
        
        model_path = Path(model_directory)
        
        # Try to load from the new pickle format first
        model_pkl_path = model_path / "model.pkl"
        if model_pkl_path.exists():
            with open(model_pkl_path, 'rb') as f:
                model_data = pickle.load(f)
            
            config = model_data['config']
            params = model_data.get('params')
            
            # Override config with any additional kwargs
            if kwargs:
                config_dict = config.to_dict()
                config_dict.update(kwargs)
                config_class = config.__class__
                config = config_class(**config_dict)
            
            model = cls.from_config(config)
            
            if load_params:
                return model, params
            else:
                return model
        
        # Fallback to old format detection
        config_files = {
            "config.json": "json",
            "config.pkl": "pickle", 
            "config.yaml": "yaml",
            "config.yml": "yaml",
            "config.toml": "toml",
            "config.py": "python"
        }
        
        config_format = None
        for filename, fmt in config_files.items():
            if (model_path / filename).exists():
                config_format = fmt
                break
        
        if config_format is None:
            raise FileNotFoundError(f"No config file found in {model_directory}")
        
        # Load config based on format
        if config_format == "json":
            import json
            config_path = model_path / "config.json"
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # Get the config class from the model class
            config_class = cls.__orig_bases__[0].__args__[0]  # Extract ConfigType from BaseModel[ConfigType]
            config = config_class(**config_dict)
            
        elif config_format == "pickle":
            config_path = model_path / "config.pkl"
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
                
        elif config_format == "yaml":
            try:
                import yaml
                config_path = model_path / "config.yaml"
                if not config_path.exists():
                    config_path = model_path / "config.yml"
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                config_class = cls.__orig_bases__[0].__args__[0]
                config = config_class(**config_dict)
            except ImportError:
                raise ImportError("PyYAML is required for YAML format. Install with: pip install PyYAML")
                
        elif config_format == "toml":
            try:
                import toml
                config_path = model_path / "config.toml"
                with open(config_path, 'r') as f:
                    config_dict = toml.load(f)
                config_class = cls.__orig_bases__[0].__args__[0]
                config = config_class(**config_dict)
            except ImportError:
                raise ImportError("toml is required for TOML format. Install with: pip install toml")
                
        elif config_format == "python":
            config_path = model_path / "config.py"
            # Execute the Python file and extract the config
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            config = config_module.config
        
        # Override with any additional kwargs
        if kwargs:
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config_class = config.__class__
            config = config_class(**config_dict)
        
        return cls.from_config(config)
    
    def get_config(self) -> ConfigType:
        """Get model configuration."""
        return self.config
