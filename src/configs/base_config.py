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
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"{self.__class__.__name__}({', '.join(f'{f.name}={getattr(self, f.name)}' for f in fields(self))})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()