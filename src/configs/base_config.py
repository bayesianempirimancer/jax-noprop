"""
Simplified base model configuration for model architecture parameters.

This module defines the core model parameters that are specific to the model
architecture, independent of training. Streamlined to focus on essential parameters.
"""

from dataclasses import dataclass, fields
from typing import Dict, Any, Tuple


@dataclass(frozen=True)
class BaseConfig:
    """
    Base model configuration containing only the essential model identification.
    
    Specific model configs should define their own parameters in config_dict
    for hierarchical organization and automatic namespace conversion.
    """
    
    # === MODEL IDENTIFICATION ===
    model_name: str = "base_model_network"
    
    # === OUTPUT DIRECTORY ===
    output_dir_parent: str = "artifacts"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary using introspection."""
        result = {}
        for field_info in fields(self):
            result[field_info.name] = getattr(self, field_info.name)
        return result
    
    def get_architecture_summary(self) -> str:
        """Get a summary of the entire configuration."""
        if hasattr(self, 'config_dict'):
            # Use the hierarchical config structure
            return self._format_config_dict(self.config_dict)
        else:
            # Fallback to the old method for backward compatibility
            return self._get_legacy_architecture_summary()
    
    def _format_config_dict(self, config_dict, indent=0) -> str:
        """Format the config_dict as a readable string."""
        lines = []
        indent_str = "  " * indent
        
        for key, value in config_dict.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.append(self._format_config_dict(value, indent + 1))
            else:
                # Format the value nicely
                if isinstance(value, tuple):
                    value_str = f"({', '.join(map(str, value))})"
                elif isinstance(value, str):
                    value_str = f'"{value}"'
                else:
                    value_str = str(value)
                lines.append(f"{indent_str}{key}: {value_str}")
        
        return "\n".join(lines)
    
    def _get_legacy_architecture_summary(self) -> str:
        """Legacy architecture summary for backward compatibility."""
        try:
            layers = [str(self.input_dim)] + [str(size) for size in self.hidden_sizes] + [str(self.output_dim)]
            arch_str = " -> ".join(layers)

            features = []
            if hasattr(self, 'use_resnet') and self.use_resnet:
                features.append(f"ResNet({self.num_resnet_blocks} blocks)")
            if hasattr(self, 'use_batch_norm') and self.use_batch_norm:
                features.append("BatchNorm")
            if hasattr(self, 'use_layer_norm') and self.use_layer_norm:
                features.append("LayerNorm")
            if hasattr(self, 'dropout_rate') and self.dropout_rate > 0:
                features.append(f"Dropout({self.dropout_rate})")

            feature_str = f" + {', '.join(features)}" if features else ""
            return f"{self.model_name.upper()}: {arch_str} ({self.activation}){feature_str}"
        except AttributeError:
            # If attributes don't exist, just return basic info
            return f"{self.model_name.upper()}: Configuration summary not available"
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Load configuration from a pretrained model directory.
        
        Args:
            model_name_or_path: Path to the pretrained model directory
            **kwargs: Additional parameters to override the loaded config
            
        Returns:
            Config instance loaded from the pretrained model
        """
        import json
        from pathlib import Path
        
        model_path = Path(model_name_or_path)
        config_path = model_path / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Override with any additional kwargs
        config_dict.update(kwargs)
        
        return cls(**config_dict)
    
    @classmethod
    def create(cls, **kwargs):
        """
        Create a configuration with specified parameters.
        
        Args:
            **kwargs: Configuration parameters to override defaults
            
        Returns:
            Config instance
        """
        return cls(**kwargs)
    
    @classmethod
    def create_from_args(cls, args):
        """
        Create a configuration instance from command line arguments.
        
        Args:
            args: argparse.Namespace object containing all configuration parameters
            
        Returns:
            Config instance
        """
        # Convert args to dictionary
        args_dict = vars(args)
        
        # Add model-specific args to model_kwargs
        model_attributes = cls.get_model_attributes()
        filtered_kwargs = {}
        
        for attr_name, attr_value in args_dict.items():
            if attr_name in model_attributes and attr_value is not None:
                filtered_kwargs[attr_name] = attr_value
        
        # Convert hidden_sizes from list to tuple if provided as list
        if 'hidden_sizes' in filtered_kwargs and isinstance(filtered_kwargs['hidden_sizes'], list):
            filtered_kwargs['hidden_sizes'] = tuple(filtered_kwargs['hidden_sizes'])
        
        return cls(**filtered_kwargs)
    
    @classmethod
    def get_model_attributes(cls):
        """
        Get the list of model-specific attributes that can be set via command line arguments.
        
        Subclasses should override this to specify which parameters they support.
        """
        return ['model_name', 'output_dir_parent']  # Base class attributes
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"{self.__class__.__name__}(model_name={self.model_name}, output_dir_parent={self.output_dir_parent})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', output_dir_parent='{self.output_dir_parent}')"
    
    def to_namespace(self):
        """
        Convert configuration to namespace with attribute access.
        
        This allows accessing config values as attributes instead of dictionary keys:
        config.to_namespace().z_shape instead of config["z_shape"]
        """
        from argparse import Namespace
        
        def dict_to_namespace(d):
            """Convert nested dictionary to Namespace with attribute access."""
            if isinstance(d, dict):
                return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d
        
        return dict_to_namespace(self.to_dict())
    
    def __post_init__(self):
        """Flatten config_dict items directly into the config object if config_dict exists."""
        if hasattr(self, 'config_dict'):
            from argparse import Namespace
            
            def dict_to_namespace(d):
                if isinstance(d, dict):
                    return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
                return d
            
            # Flatten config_dict items directly into the config object
            for key, value in self.config_dict.items():
                if isinstance(value, dict):
                    # For nested dicts, create a namespace
                    object.__setattr__(self, key, dict_to_namespace(value))
                else:
                    # For simple values, set directly
                    object.__setattr__(self, key, value)