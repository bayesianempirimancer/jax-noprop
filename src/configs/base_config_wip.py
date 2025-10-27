"""
Lightweight base configuration utilities for frozen dataclasses.

This module provides the essential functionality for working with frozen dataclasses
in JAX/Flax applications, focusing on the ability to create updated configs with
modified values while maintaining immutability and hashability.

Usage Examples:
    # Basic usage with a config class
    @dataclass(frozen=True)
    class MyConfig(BaseConfigWIP):
        model_name: str = "my_model"
        config: dict = field(default_factory=lambda: {
            "dropout_rate": 0.1,
            "encoder_config": {
                "dropout_rate": 0.1,
                "hidden_dims": (64, 32)
            }
        })
    
    # Create config instance
    config = MyConfig()
    
    # Clean interface for config dict updates
    new_config = config.update_config({"dropout_rate": 0.3})
    new_config = config.update_config({
        "encoder_config": {
            "dropout_rate": 0.5  # Only updates encoder dropout
        }
    })
    
    # Update multiple entries in the same nested path
    new_config = config.update_config({
        "encoder_config": {
            "dropout_rate": 0.4,
            "hidden_dims": (128, 64),
            "activation": "swish"  # Updates multiple fields in encoder_config
        }
    })
    
    # Add new fields to config
    new_config = config.append({"new_param": 42})
    new_config = config.append({
        "new_nested": {
            "sub_param": "value"
        }
    })
    
    # Multiple config updates at once (mixed levels)
    new_config = config.update_config({
        "dropout_rate": 0.2,
        "encoder_config": {
            "dropout_rate": 0.4,
            "hidden_dims": (128, 64)  # Updates both dropout and hidden_dims
        }
    })
    
    # Error handling - invalid keys raise KeyError with helpful messages
    try:
        config.update_config({"invalid_key": "value"})
    except KeyError as e:
        print(f"Error: {e}")  # Shows available config keys
"""

from dataclasses import dataclass, fields, replace
from typing import Dict, Any, TypeVar, Type
import copy

T = TypeVar('T', bound='BaseConfigWIP')

@dataclass(frozen=True)
class BaseConfigWIP:
    """
    Lightweight base configuration class for frozen dataclasses.
    
    Provides only the essential functionality needed for JAX/Flax compatibility:
    - Immutability (frozen=True)
    - Hashability (required for JAX compilation)
    - Simple update mechanism for creating modified configs
    """
    
    @staticmethod
    def _deep_merge_dicts(base_dict: dict, update_dict: dict, path: str = "") -> dict:
        """
        Recursively merge update_dict into base_dict, creating a new dict.
        Validates that all keys in update_dict exist in base_dict.

        Args:
            base_dict: The base dictionary to merge into
            update_dict: The dictionary with updates to apply
            path: Current path for error reporting

        Returns:
            New dictionary with merged values

        Raises:
            KeyError: If a key in update_dict doesn't exist in base_dict
        """
        result = copy.deepcopy(base_dict)
        
        for key, value in update_dict.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in result:
                raise KeyError(f"Key '{current_path}' not found in config. Available keys: {list(result.keys())}. Use append() to add new fields.")
            
            if isinstance(result[key], dict) and isinstance(value, dict):
                # Both are dicts, merge recursively
                result[key] = BaseConfigWIP._deep_merge_dicts(result[key], value, current_path)
            else:
                # Replace the value
                result[key] = copy.deepcopy(value)
        
        return result
    
    def update_config(self: T, updates: Dict[str, Any]) -> T:
        """
        Create a new config instance with updated values in the config dictionary.

        Args:
            updates: Dictionary of config keys and new values to update.
                    Supports nested updates by providing nested dictionaries.

        Returns:
            New config instance with updated config values

        Raises:
            KeyError: If any key in updates doesn't exist in the config dict
            TypeError: If updates is not a dictionary
            AttributeError: If the config class doesn't have a 'config' field

        Example:
            config = MyConfig(model_name="test", config={"dropout_rate": 0.1, "encoder": {"dim": 64}})
            new_config = config.update_config({
                "dropout_rate": 0.2,
                "encoder": {"dim": 128}  # This will merge with existing encoder dict
            })
            # Result: MyConfig(model_name="test", config={"dropout_rate": 0.2, "encoder": {"dim": 128}})
        """
        if not isinstance(updates, dict):
            raise TypeError(f"Expected updates to be a dictionary, got {type(updates).__name__}")

        if not updates:
            return self

        # Check if config field exists
        if not hasattr(self, 'config'):
            raise AttributeError("Config class must have a 'config' field to use update_config()")

        # Get current field values
        current_values = {}
        for field_info in fields(self):
            current_values[field_info.name] = getattr(self, field_info.name)

        # Apply updates to the config dict recursively with validation
        updated_config = self._deep_merge_dicts(current_values['config'], updates)
        current_values['config'] = updated_config

        # Create new instance with updated values
        return replace(self, **current_values)

    def append(self: T, new_fields: Dict[str, Any]) -> T:
        """
        Create a new config instance with new fields added to the config dictionary.
        
        Unlike update_config(), this method allows adding new keys that don't exist
        in the current config. Supports nested additions by providing nested dictionaries.

        Args:
            new_fields: Dictionary of new field names and values to add.
                       Supports nested additions by providing nested dictionaries.

        Returns:
            New config instance with added fields

        Raises:
            TypeError: If new_fields is not a dictionary
            AttributeError: If the config class doesn't have a 'config' field

        Example:
            config = MyConfig(model_name="test", config={"dropout_rate": 0.1})
            new_config = config.append({
                "new_param": 42,
                "new_nested": {
                    "sub_param": "value"
                }
            })
            # Result: MyConfig(model_name="test", config={
            #     "dropout_rate": 0.1,
            #     "new_param": 42,
            #     "new_nested": {"sub_param": "value"}
            # })
        """
        if not isinstance(new_fields, dict):
            raise TypeError(f"Expected new_fields to be a dictionary, got {type(new_fields).__name__}")

        if not new_fields:
            return self

        # Check if config field exists
        if not hasattr(self, 'config'):
            raise AttributeError("Config class must have a 'config' field to use append()")

        # Get current field values
        current_values = {}
        for field_info in fields(self):
            current_values[field_info.name] = getattr(self, field_info.name)

        # Deep merge new fields into config (allows adding new keys)
        updated_config = self._deep_merge_dicts_append(current_values['config'], new_fields)
        current_values['config'] = updated_config

        # Create new instance with updated values
        return replace(self, **current_values)

    @staticmethod
    def _deep_merge_dicts_append(base_dict: dict, new_dict: dict, path: str = "") -> dict:
        """
        Recursively merge new_dict into base_dict, creating a new dict.
        Unlike _deep_merge_dicts, this allows adding new keys that don't exist in base_dict.

        Args:
            base_dict: The base dictionary to merge into
            new_dict: The dictionary with new fields to add
            path: Current path for error reporting (unused in append mode)

        Returns:
            New dictionary with merged values
        """
        result = copy.deepcopy(base_dict)
        
        for key, value in new_dict.items():
            if isinstance(result.get(key), dict) and isinstance(value, dict):
                # Both are dicts, merge recursively
                result[key] = BaseConfigWIP._deep_merge_dicts_append(result[key], value, path)
            else:
                # Add or replace the value
                result[key] = copy.deepcopy(value)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary using introspection."""
        result = {}
        for field_info in fields(self):
            result[field_info.name] = getattr(self, field_info.name)
        return result
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"{self.__class__.__name__}({', '.join(f'{f.name}={getattr(self, f.name)}' for f in fields(self))})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()