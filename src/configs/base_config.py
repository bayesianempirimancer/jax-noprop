"""
Lightweight base configuration utilities for frozen dataclasses.

This module provides the essential functionality for working with frozen dataclasses
in JAX/Flax applications, focusing on the ability to merge updates into FrozenDict
fields while maintaining immutability and hashability.

Usage Examples:
    # Basic usage with FrozenDict fields
    @dataclass(frozen=True)
    class MyConfig(BaseConfig):
        model_name: str = "my_model"
        main: FrozenDict = field(default_factory=lambda: FrozenDict({
            "dropout_rate": 0.1,
            "encoder": FrozenDict({
                "dropout_rate": 0.1,
                "hidden_dims": (64, 32)
            })
        }))
    
    # Create config instance
    config = MyConfig()
    
    # Merge updates into FrozenDict fields
    from dataclasses import replace
    updated_main = config.merge_frozen_dict('main', {
        "dropout_rate": 0.3,
        "encoder": {
            "dropout_rate": 0.5  # Recursively merges nested FrozenDicts
        }
    })
    new_config = replace(config, main=updated_main)
    
    # Multiple field updates
    updated_main = config.merge_frozen_dict('main', {"dropout_rate": 0.2})
    new_config = replace(config, main=updated_main)
"""

from dataclasses import dataclass, fields
from typing import Dict, Any, TypeVar, Union
import copy
import json
import yaml
from omegaconf import OmegaConf, DictConfig 
from pathlib import Path
from flax.core import FrozenDict


T = TypeVar('T', bound='BaseConfig')


@dataclass(frozen=True)
class BaseConfig:
    """
    Lightweight base configuration class for frozen dataclasses.
    
    Provides only the essential functionality needed for JAX/Flax compatibility:
    - Immutability (frozen=True)
    - Hashability (required for JAX compilation)
    - Simple update mechanism for creating modified configs
    """
    
    # === MODEL IDENTIFICATION ===
    model_name: str = "base_model_network"
    
    def merge_frozen_dict(self, field_name: str, updates: Union[FrozenDict, dict]) -> FrozenDict:
        """
        Merge updates into a FrozenDict field of this config, creating a new FrozenDict.
        Always returns a FrozenDict for consistency.
        
        Args:
            field_name: Name of the field (e.g., 'main', 'crn', 'encoder', 'decoder')
            updates: FrozenDict or dict with updates to apply to the field
            
        Returns:
            New FrozenDict with merged values
            
        Raises:
            AttributeError: If the field doesn't exist on this config
        """
        # Get the base from self
        if not hasattr(self, field_name):
            raise AttributeError(f"{self.__class__.__name__} has no field '{field_name}'")
        
        base = getattr(self, field_name)
        
        # Convert to dict for merging
        if isinstance(base, FrozenDict):
            result = dict(base)
        else:
            result = copy.deepcopy(base)
        
        # Convert updates to dict if needed
        if isinstance(updates, FrozenDict):
            updates_dict = dict(updates)
        else:
            updates_dict = updates
        
        # Merge updates into result
        for key, value in updates_dict.items():
            if key in result and isinstance(result[key], (FrozenDict, dict)) and isinstance(value, (FrozenDict, dict)):
                # Recursively merge nested FrozenDicts/dicts
                # Temporarily set the field to access it in recursion
                temp_field = result[key]
                if isinstance(temp_field, FrozenDict):
                    result[key] = self.merge_frozen_dict_impl(temp_field, value)
                else:
                    result[key] = self.merge_frozen_dict_impl(temp_field, value)
            else:
                # Replace or add the value
                if isinstance(value, FrozenDict):
                    result[key] = dict(value)  # Convert FrozenDict to dict for merging
                elif isinstance(value, dict):
                    result[key] = copy.deepcopy(value)
                else:
                    result[key] = value
        
        # Always return FrozenDict
        return FrozenDict(result)
    
    def merge_frozen_dict_impl(self, base: Union[FrozenDict, dict], updates: Union[FrozenDict, dict]) -> FrozenDict:
        """
        Internal helper method to merge updates into a base FrozenDict/dict.
        This is used recursively for nested merging.
        
        Args:
            base: Base FrozenDict or dict to merge into
            updates: FrozenDict or dict with updates to apply
            
        Returns:
            New FrozenDict with merged values
        """
        # Convert to dict for merging
        if isinstance(base, FrozenDict):
            result = dict(base)
        else:
            result = copy.deepcopy(base)
        
        # Convert updates to dict if needed
        if isinstance(updates, FrozenDict):
            updates_dict = dict(updates)
        else:
            updates_dict = updates
        
        # Merge updates into result
        for key, value in updates_dict.items():
            if key in result and isinstance(result[key], (FrozenDict, dict)) and isinstance(value, (FrozenDict, dict)):
                # Recursively merge nested FrozenDicts/dicts
                result[key] = self.merge_frozen_dict_impl(result[key], value)
            else:
                # Replace or add the value
                if isinstance(value, FrozenDict):
                    result[key] = dict(value)  # Convert FrozenDict to dict for merging
                elif isinstance(value, dict):
                    result[key] = copy.deepcopy(value)
                else:
                    result[key] = value
        
        # Always return FrozenDict
        return FrozenDict(result)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary using introspection."""
        result = {}
        for field_info in fields(self):
            result[field_info.name] = getattr(self, field_info.name)
        return result
    
    def save_yaml(self, filepath: Union[str, Path]):
        """
        Save configuration to a YAML file.
        
        Args:
            filepath: Path to save the YAML file to
            
        Example:
            config.save_yaml('config.yaml')
        """
        if yaml is None:
            raise ImportError("PyYAML is required for save_yaml(). Install it with: pip install pyyaml")
        
        config_dict = self.to_dict()
        # Convert FrozenDict to dict for YAML serialization
        config_dict = self._frozen_dict_to_dict(config_dict)
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=True)
    
    def _frozen_dict_to_dict(self, obj: Any) -> Any:
        """Recursively convert FrozenDict objects to regular dicts for YAML serialization."""
        if isinstance(obj, FrozenDict):
            return {k: self._frozen_dict_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {k: self._frozen_dict_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._frozen_dict_to_dict(item) for item in obj]
        else:
            return obj
    
    @classmethod
    def load_yaml(cls: type[T], filepath: Union[str, Path]) -> T:
        """
        Load configuration from a YAML file.
        
        Args:
            filepath: Path to the YAML file to load
            
        Returns:
            Config instance reconstructed from the YAML file
            
        Example:
            config = MyConfig.load_yaml('config.yaml')
        """
        if yaml is None:
            raise ImportError("PyYAML is required for load_yaml(). Install it with: pip install pyyaml")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            raise ValueError(f"YAML file is empty or invalid: {filepath}")
        
        # Convert dict to FrozenDict where appropriate
        config_dict = cls._dict_to_frozen_dict(config_dict)
        
        return cls(**config_dict)
    
    @staticmethod
    def _dict_to_frozen_dict(obj: Any) -> Any:
        """Recursively convert dict objects to FrozenDict where appropriate."""
        if isinstance(obj, dict):
            # Check if this looks like it should be a FrozenDict (has nested structure)
            # For now, convert all dicts to FrozenDict to match the config structure
            return FrozenDict({k: BaseConfig._dict_to_frozen_dict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [BaseConfig._dict_to_frozen_dict(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(BaseConfig._dict_to_frozen_dict(item) for item in obj)
        else:
            return obj
    
    def save_json(self, filepath: Union[str, Path]):
        """
        Save configuration to a JSON file.
        
        Args:
            filepath: Path to save the JSON file to
            
        Example:
            config.save_json('config.json')
        """
        config_dict = self.to_dict()
        # Convert FrozenDict to dict for JSON serialization
        config_dict = self._frozen_dict_to_dict(config_dict)
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
    
    @classmethod
    def load_json(cls: type[T], filepath: Union[str, Path]) -> T:
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to the JSON file to load
            
        Returns:
            Config instance reconstructed from the JSON file
            
        Example:
            config = MyConfig.load_json('config.json')
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        if config_dict is None:
            raise ValueError(f"JSON file is empty or invalid: {filepath}")
        
        # Convert dict to FrozenDict where appropriate
        config_dict = cls._dict_to_frozen_dict(config_dict)
        
        return cls(**config_dict)
    
    def save_omegaconf(self, filepath: Union[str, Path]):
        """
        Save configuration to a file using OmegaConf (Hydra's config format).
        
        Args:
            filepath: Path to save the config file to (typically .yaml)
            
        Example:
            config.save_omegaconf('config.yaml')
        """
        if OmegaConf is None:
            raise ImportError("OmegaConf is required for save_omegaconf(). Install it with: pip install omegaconf")
        
        config_dict = self.to_dict()
        # Convert FrozenDict to dict for OmegaConf serialization
        config_dict = self._frozen_dict_to_dict(config_dict)
        
        # Create OmegaConf DictConfig
        conf = OmegaConf.create(config_dict)
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        OmegaConf.save(conf, filepath)
    
    @classmethod
    def load_omegaconf(cls: type[T], filepath: Union[str, Path]) -> T:
        """
        Load configuration from a file using OmegaConf (Hydra's config format).
        
        Args:
            filepath: Path to the config file to load (typically .yaml)
            
        Returns:
            Config instance reconstructed from the OmegaConf file
            
        Example:
            config = MyConfig.load_omegaconf('config.yaml')
        """
        if OmegaConf is None:
            raise ImportError("OmegaConf is required for load_omegaconf(). Install it with: pip install omegaconf")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        # Load using OmegaConf
        conf = OmegaConf.load(filepath)
        
        # Convert OmegaConf DictConfig to regular dict
        config_dict = OmegaConf.to_container(conf, resolve=True)
        
        if config_dict is None:
            raise ValueError(f"Config file is empty or invalid: {filepath}")
        
        # Convert dict to FrozenDict where appropriate
        config_dict = cls._dict_to_frozen_dict(config_dict)
        
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"{self.__class__.__name__}({', '.join(f'{f.name}={getattr(self, f.name)}' for f in fields(self))})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()