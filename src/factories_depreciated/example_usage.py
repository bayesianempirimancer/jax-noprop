"""
Example usage of the unified model factory system.

This shows how to use the new factory functions to create models
with a simple, consistent interface.

DIMENSION PARAMETERS EXPLAINED:
- z_dim: Latent space dimension (used by flow models and VAE)
- x_dim: Conditional input dimension (used by flow models and VAE)
- input_dim: Data input dimension (used by encoder/decoder)
- output_dim: Data output dimension (used by decoder)
- latent_shape: Latent space shape tuple (used by encoder)

Note: z_dim/x_dim are required by the factory interface but may not be
used by all model types (e.g., encoder only uses input_dim/latent_shape).
"""

from src.factories.model_factory import (
    create_model, 
    create_flow_model, 
    create_vae_flow_model,
    get_default_config
)
from src.flow_models_wip.crn_wip import Config as CRNConfig


def example_basic_usage():
    """Example of basic model creation."""
    
    # 1. Create a conditional ResNet
    # Uses: z_dim (latent space), x_dim (conditional input)
    crn_config = CRNConfig()
    model = create_model(
        model_name="conditional_resnet",
        config_dict=crn_config.config,
        z_dim=8,  # Latent space dimension
        x_dim=4   # Conditional input dimension
    )
    print(f"Created conditional ResNet: {type(model).__name__}")
    
    # 2. Create an encoder
    # Uses: input_dim (data input), latent_shape (latent output)
    encoder_config = get_default_config("encoder")
    encoder = create_model(
        model_name="encoder",
        config_dict=encoder_config,
        z_dim=8,  # Not used by encoder, but required by factory
        x_dim=4,  # Not used by encoder, but required by factory
        input_dim=4,     # Data input dimension
        latent_shape=(8,)  # Latent output shape
    )
    print(f"Created encoder: {type(encoder).__name__}")
    
    # 3. Create a decoder
    # Uses: input_dim (latent input), output_dim (data output)
    decoder_config = get_default_config("decoder")
    decoder = create_model(
        model_name="decoder",
        config_dict=decoder_config,
        z_dim=8,  # Not used by decoder, but required by factory
        x_dim=4,  # Not used by decoder, but required by factory
        input_dim=8,   # Latent input dimension
        output_dim=10  # Data output dimension (classes)
    )
    print(f"Created decoder: {type(decoder).__name__}")


def example_flow_models():
    """Example of creating flow models."""
    
    # Create a potential flow model
    crn_config = CRNConfig()
    potential_flow = create_flow_model(
        flow_type="potential",
        backbone_config=crn_config.config,
        z_dim=8,
        x_dim=4
    )
    print(f"Created potential flow: {type(potential_flow).__name__}")
    
    # Create a geometric flow model
    geometric_flow = create_flow_model(
        flow_type="geometric",
        backbone_config=crn_config.config,
        z_dim=8,
        x_dim=4
    )
    print(f"Created geometric flow: {type(geometric_flow).__name__}")


def example_vae_flow():
    """Example of creating a VAE with flow model."""
    
    # Main VAE config
    main_config = {
        "model_name": "vae_flow_network",
        "loss_type": "cross_entropy",
        "flow_loss_weight": 0.01,
        "reg_weight": 0.0,
    }
    
    # CRN config for the flow backbone
    crn_config = CRNConfig().config
    
    # Encoder config
    encoder_config = get_default_config("encoder")
    
    # Decoder config
    decoder_config = get_default_config("decoder")
    
    # Create VAE flow model
    # Uses: z_dim (latent space), x_dim (conditional input), output_dim (classes)
    vae_flow = create_vae_flow_model(
        main_config=main_config,
        crn_config=crn_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        z_dim=8,      # Latent space dimension
        x_dim=4,      # Conditional input dimension  
        output_dim=2  # Output classes
    )
    print(f"Created VAE flow: {type(vae_flow).__name__}")
    print(f"  Input shape: {vae_flow.config.config['input_shape']}")
    print(f"  Output shape: {vae_flow.config.config['output_shape']}")
    print(f"  Latent shape: {vae_flow.config.config['latent_shape']}")


def example_trainer_usage():
    """Example of how trainer code would use the factory."""
    
    # This is how the trainer would create models
    def create_model_for_training(model_name: str, z_dim: int, x_dim: int):
        """Create a model for training using the factory."""
        
        if model_name == "conditional_resnet":
            config = CRNConfig()
            return create_model(
                model_name="conditional_resnet",
                config_dict=config.config,
                z_dim=z_dim,
                x_dim=x_dim
            )
        
        elif model_name == "potential_flow":
            config = CRNConfig()
            return create_flow_model(
                flow_type="potential",
                backbone_config=config.config,
                z_dim=z_dim,
                x_dim=x_dim
            )
        
        elif model_name == "vae_flow":
            # Use default configs
            main_config = {"model_name": "vae_flow_network", "loss_type": "cross_entropy"}
            crn_config = CRNConfig().config
            encoder_config = get_default_config("encoder")
            decoder_config = get_default_config("decoder")
            
            return create_vae_flow_model(
                main_config=main_config,
                crn_config=crn_config,
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                z_dim=z_dim,
                x_dim=x_dim
            )
        
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
    
    # Example usage
    models = {}
    for model_name in ["conditional_resnet", "potential_flow", "vae_flow"]:
        models[model_name] = create_model_for_training(model_name, z_dim=8, x_dim=4)
        print(f"Created {model_name}: {type(models[model_name]).__name__}")


if __name__ == "__main__":
    print("=== Basic Usage ===")
    example_basic_usage()
    
    print("\n=== Flow Models ===")
    example_flow_models()
    
    print("\n=== VAE Flow ===")
    example_vae_flow()
    
    print("\n=== Trainer Usage ===")
    example_trainer_usage()
