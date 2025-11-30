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

from dataclasses import dataclass, fields, replace
from typing import Dict, Any, TypeVar, Union, Optional
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
        
        # Use merge_frozen_dict_impl for the actual merging
        return self.merge_frozen_dict_impl(base, updates)
    
    def merge_updates(self: T, updates: Dict[str, Any]) -> T:
        """
        Merge a dictionary of updates into the config.
        
        Args:
            updates: Dictionary where keys match config fields. 
                     Values can be nested dicts (merged) or values (replaced).
                     
        Returns:
            New config instance with updates applied.
        """
        changes = {}
        for field_name, update_val in updates.items():
            if not hasattr(self, field_name):
                continue
                
            current_val = getattr(self, field_name)
            
            # Apply filter_none if it's a dict, to avoid merging Nones?
            # Or assume caller handles it. 
            # config.py uses filter_none.
            # Let's just merge what is given.
            
            if isinstance(current_val, (FrozenDict, dict)) and isinstance(update_val, (dict, FrozenDict)):
                # Use existing merge logic for dict fields
                changes[field_name] = self.merge_frozen_dict(field_name, update_val)
            else:
                # Direct replacement for non-dict fields
                changes[field_name] = update_val
                
        return replace(self, **changes)

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
    
    @staticmethod
    def filter_none(d: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out None values from a dictionary.
        
        Args:
            d: Dictionary to filter
            
        Returns:
            Dictionary with None values removed
        """
        return {k: v for k, v in d.items() if v is not None}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary using introspection."""
        result = {}
        for field_info in fields(self):
            result[field_info.name] = getattr(self, field_info.name)
        return result
    
    def _prepare_for_serialization(self) -> Dict[str, Any]:
        """Prepare config for serialization by converting to dict and handling FrozenDicts.
        
        Returns:
            Dictionary ready for serialization (YAML, JSON, etc.)
        """
        config_dict = self.to_dict()
        return self._frozen_dict_to_dict(config_dict)
    
    @staticmethod
    def _prepare_save_path(filepath: Union[str, Path]) -> Path:
        """Prepare file path for saving: convert to Path and create parent directories.
        
        Args:
            filepath: Path to file (str or Path)
            
        Returns:
            Path object with parent directories created
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return filepath
    
    @staticmethod
    def _prepare_load_path(filepath: Union[str, Path]) -> Path:
        """Prepare file path for loading: convert to Path and check existence.
        
        Args:
            filepath: Path to file (str or Path)
            
        Returns:
            Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        return filepath
    
    @classmethod
    def _load_from_dict(cls: type[T], config_dict: Optional[Dict[str, Any]], filepath: Path) -> T:
        """Generic helper to load config from a dictionary.
        
        Args:
            config_dict: Dictionary loaded from file (may be None)
            filepath: Path to file (for error messages)
            
        Returns:
            Config instance
            
        Raises:
            ValueError: If config_dict is None or empty
        """
        if config_dict is None:
            raise ValueError(f"Config file is empty or invalid: {filepath}")
        
        # Convert dict to FrozenDict where appropriate
        config_dict = cls._dict_to_frozen_dict(config_dict)
        
        return cls(**config_dict)
    
    def save_yaml(self, filepath: Union[str, Path], desired_order: Optional[list] = None):
        """
        Save configuration to a YAML file.
        
        Args:
            filepath: Path to save the YAML file to
            desired_order: Optional list of keys to order the output. Default order provided if None.
            
        Example:
            config.save_yaml('config.yaml')
        """
        if yaml is None:
            raise ImportError("PyYAML is required for save_yaml(). Install it with: pip install pyyaml")
        
        config_dict = self._prepare_for_serialization()
        
        # Reorder keys to desired order
        # Any additional keys will be appended at the end
        if desired_order is None:
            ordered_dict = config_dict
        else:
            ordered_dict = {}
            # Add keys in desired order
            for key in desired_order:
                if key in config_dict:
                    ordered_dict[key] = config_dict[key]
            # Add any remaining keys that weren't in the desired order
            for key, value in config_dict.items():
                if key not in ordered_dict:
                    ordered_dict[key] = value
        
        filepath = self._prepare_save_path(filepath)
        
        with open(filepath, 'w') as f:
            yaml.dump(ordered_dict, f, default_flow_style=False, indent=2, sort_keys=False)
    
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
        
        filepath = cls._prepare_load_path(filepath)
        
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls._load_from_dict(config_dict, filepath)
    
    @staticmethod
    def _dict_to_frozen_dict(obj: Any) -> Any:
        """Recursively convert dict objects to FrozenDict where appropriate.
        
        Also converts lists to tuples to ensure hashability.
        """
        if isinstance(obj, dict):
            # Check if this looks like it should be a FrozenDict (has nested structure)
            # For now, convert all dicts to FrozenDict to match the config structure
            return FrozenDict({k: BaseConfig._dict_to_frozen_dict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            # Convert lists to tuples for hashability (needed for Flax modules)
            return tuple(BaseConfig._dict_to_frozen_dict(item) for item in obj)
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
        config_dict = self._prepare_for_serialization()
        filepath = self._prepare_save_path(filepath)
        
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
        filepath = cls._prepare_load_path(filepath)
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls._load_from_dict(config_dict, filepath)
    
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
        
        config_dict = self._prepare_for_serialization()
        
        # Create OmegaConf DictConfig
        conf = OmegaConf.create(config_dict)
        
        filepath = self._prepare_save_path(filepath)
        
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
        
        filepath = cls._prepare_load_path(filepath)
        
        # Load using OmegaConf
        conf = OmegaConf.load(filepath)
        
        # Convert OmegaConf DictConfig to regular dict
        config_dict = OmegaConf.to_container(conf, resolve=True)
        
        return cls._load_from_dict(config_dict, filepath)
    
    @classmethod
    def merge_with_defaults(cls: type[T], loaded_config: T, default_config: Optional[T] = None) -> T:
        """
        Merge a loaded config (e.g., from YAML) with default values from a default config.
        This ensures all default values are preserved even if not specified in the loaded config.
        
        Args:
            loaded_config: Config loaded from YAML or custom class
            default_config: Default config instance to merge with. If None, creates a new default instance.
            
        Returns:
            Merged config with all default values filled in
            
        Example:
            loaded = MyConfig.load_yaml('config.yaml')
            merged = MyConfig.merge_with_defaults(loaded)
        """
        if default_config is None:
            default_config = cls()
        
        # Get both configs as dicts
        loaded_dict = loaded_config.to_dict()
        default_dict = default_config.to_dict()
        
        # Merge recursively, preserving loaded values but filling in defaults
        merged_dict = {}
        for key in default_dict:
            if key in loaded_dict:
                # Key exists in loaded config - merge recursively if it's a FrozenDict
                if isinstance(default_dict[key], FrozenDict) and isinstance(loaded_dict[key], FrozenDict):
                    # Use merge_frozen_dict_impl for recursive merging
                    # Start with defaults (base) and apply loaded (updates) to override defaults
                    # This preserves loaded values while filling in missing keys from defaults
                    merged_dict[key] = default_config.merge_frozen_dict_impl(default_dict[key], loaded_dict[key])
                else:
                    # Use loaded value (overrides default)
                    merged_dict[key] = loaded_dict[key]
            else:
                # Key only in defaults - use default value
                merged_dict[key] = default_dict[key]
        
        # Add any keys that are only in loaded (shouldn't happen with proper configs, but be safe)
        for key in loaded_dict:
            if key not in merged_dict:
                merged_dict[key] = loaded_dict[key]
        
        # Create new config instance with merged values
        return loaded_config.__class__(**merged_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"{self.__class__.__name__}({', '.join(f'{f.name}={getattr(self, f.name)}' for f in fields(self))})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()