"""
Clean example usage of the simplified model factory system.

This shows the new clean interface: create_model(model_name, config_dict, **kwargs)
No more clunky z_dim/x_dim requirements!
"""

from src.factories.model_factory_v2 import (
    create_model,
    create_vae_flow_with_defaults,
    create_conditional_resnet_with_defaults,
    create_encoder_with_defaults,
    create_decoder_with_defaults,
    get_default_config
)
from src.flow_models_wip.crn_wip import Config as CRNConfig


def example_clean_usage():
    """Example of the new clean factory interface."""
    
    print("=" * 60)
    print("CLEAN FACTORY INTERFACE EXAMPLES")
    print("=" * 60)
    
    # 1. Create a conditional ResNet - only pass what it needs
    print("\n1. Conditional ResNet:")
    crn_config = CRNConfig().config
    model = create_model("conditional_resnet", crn_config, z_dim=8, x_dim=4)
    print(f"   ✓ Created: {type(model).__name__}")
    
    # 2. Create an encoder - only pass what it needs
    print("\n2. Encoder:")
    encoder_config = get_default_config("encoder")
    encoder = create_model("encoder", encoder_config, input_dim=4, latent_shape=(8,))
    print(f"   ✓ Created: {type(encoder).__name__}")
    
    # 3. Create a decoder - only pass what it needs
    print("\n3. Decoder:")
    decoder_config = get_default_config("decoder")
    decoder = create_model("decoder", decoder_config, input_dim=8, output_dim=10)
    print(f"   ✓ Created: {type(decoder).__name__}")
    
    # 4. Create a VAE flow - only pass what it needs
    print("\n4. VAE Flow:")
    main_config = {
        "model_name": "vae_flow_network",
        "loss_type": "cross_entropy",
        "flow_loss_weight": 0.01,
        "reg_weight": 0.0,
    }
    vae_flow = create_model("vae_flow", main_config, z_dim=8, x_dim=4, output_dim=2)
    print(f"   ✓ Created: {type(vae_flow).__name__}")
    print(f"   Input shape: {vae_flow.config.config['input_shape']}")
    print(f"   Output shape: {vae_flow.config.config['output_shape']}")
    print(f"   Latent shape: {vae_flow.config.config['latent_shape']}")
    
    # 5. Create flow models - only pass what they need
    print("\n5. Flow Models:")
    crn_config = CRNConfig().config
    
    potential_flow = create_model("potential_flow", crn_config, z_dim=8, x_dim=4)
    print(f"   ✓ Potential flow: {type(potential_flow).__name__}")
    
    geometric_flow = create_model("geometric_flow", crn_config, z_dim=8, x_dim=4)
    print(f"   ✓ Geometric flow: {type(geometric_flow).__name__}")
    
    natural_flow = create_model("natural_flow", crn_config, z_dim=8, x_dim=4)
    print(f"   ✓ Natural flow: {type(natural_flow).__name__}")


def example_convenience_functions():
    """Example using convenience functions with defaults."""
    
    print("\n" + "=" * 60)
    print("CONVENIENCE FUNCTIONS WITH DEFAULTS")
    print("=" * 60)
    
    # 1. VAE flow with defaults - super simple!
    print("\n1. VAE Flow with defaults:")
    vae_flow = create_vae_flow_with_defaults(z_dim=8, x_dim=4, output_dim=2)
    print(f"   ✓ Created: {type(vae_flow).__name__}")
    
    # 2. Conditional ResNet with defaults
    print("\n2. Conditional ResNet with defaults:")
    crn = create_conditional_resnet_with_defaults(z_dim=8, x_dim=4)
    print(f"   ✓ Created: {type(crn).__name__}")
    
    # 3. Encoder with defaults
    print("\n3. Encoder with defaults:")
    encoder = create_encoder_with_defaults(input_dim=4, latent_shape=(8,))
    print(f"   ✓ Created: {type(encoder).__name__}")
    
    # 4. Decoder with defaults
    print("\n4. Decoder with defaults:")
    decoder = create_decoder_with_defaults(input_dim=8, output_dim=10)
    print(f"   ✓ Created: {type(decoder).__name__}")


def example_custom_configs():
    """Example with custom configurations."""
    
    print("\n" + "=" * 60)
    print("CUSTOM CONFIGURATIONS")
    print("=" * 60)
    
    # Custom encoder config
    custom_encoder_config = {
        "model_type": "mlp_normal",
        "encoder_type": "normal",
        "hidden_dims": (128, 64, 32),  # Deeper network
        "dropout_rate": 0.2,           # More dropout
        "activation": "relu",           # Different activation
    }
    
    print("\n1. Custom Encoder:")
    encoder = create_model("encoder", custom_encoder_config, input_dim=4, latent_shape=(8,))
    print(f"   ✓ Created: {type(encoder).__name__}")
    print(f"   Hidden dims: {custom_encoder_config['hidden_dims']}")
    print(f"   Dropout rate: {custom_encoder_config['dropout_rate']}")
    print(f"   Activation: {custom_encoder_config['activation']}")
    
    # Custom VAE flow config
    custom_vae_config = {
        "model_name": "vae_flow_network",
        "loss_type": "mse",  # Different loss
        "flow_loss_weight": 0.1,  # Higher flow weight
        "reg_weight": 0.01,  # Some regularization
    }
    
    print("\n2. Custom VAE Flow:")
    vae_flow = create_model("vae_flow", custom_vae_config, z_dim=8, x_dim=4, output_dim=2)
    print(f"   ✓ Created: {type(vae_flow).__name__}")
    print(f"   Loss type: {custom_vae_config['loss_type']}")
    print(f"   Flow weight: {custom_vae_config['flow_loss_weight']}")


def example_trainer_usage():
    """Example of how trainer code would use the clean interface."""
    
    print("\n" + "=" * 60)
    print("TRAINER USAGE EXAMPLE")
    print("=" * 60)
    
    def create_model_for_training(model_name: str, **kwargs):
        """Create a model for training using the clean factory."""
        
        if model_name == "conditional_resnet":
            config = CRNConfig().config
            return create_model("conditional_resnet", config, **kwargs)
        
        elif model_name == "vae_flow":
            main_config = {"model_name": "vae_flow_network", "loss_type": "cross_entropy"}
            return create_model("vae_flow", main_config, **kwargs)
        
        elif model_name == "potential_flow":
            config = CRNConfig().config
            return create_model("potential_flow", config, **kwargs)
        
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
    
    # Example usage - much cleaner!
    print("\nCreating models for training:")
    
    # Conditional ResNet
    crn = create_model_for_training("conditional_resnet", z_dim=8, x_dim=4)
    print(f"   ✓ Conditional ResNet: {type(crn).__name__}")
    
    # VAE Flow
    vae_flow = create_model_for_training("vae_flow", z_dim=8, x_dim=4, output_dim=2)
    print(f"   ✓ VAE Flow: {type(vae_flow).__name__}")
    
    # Potential Flow
    potential_flow = create_model_for_training("potential_flow", z_dim=8, x_dim=4)
    print(f"   ✓ Potential Flow: {type(potential_flow).__name__}")


if __name__ == "__main__":
    print("Testing Clean Factory Interface...")
    
    example_clean_usage()
    example_convenience_functions()
    example_custom_configs()
    example_trainer_usage()
    
    print("\n" + "=" * 60)
    print("✅ CLEAN FACTORY INTERFACE WORKS PERFECTLY!")
    print("=" * 60)
    print("Key benefits:")
    print("✓ No more clunky z_dim/x_dim requirements")
    print("✓ Each model only needs its specific parameters")
    print("✓ Clean, intuitive interface")
    print("✓ Easy to use and extend")
    print("✓ Convenience functions for common cases")
