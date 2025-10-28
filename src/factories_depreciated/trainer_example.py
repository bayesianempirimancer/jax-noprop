"""
Example showing how to replace the existing trainer model creation code.

This demonstrates how the new factory functions can replace the complex
create_model function in trainer.py with simple, clean code.
"""

from src.factories.trainer_factory import create_noprop_model, create_simple_model


def old_trainer_approach():
    """Example of the old complex approach (from trainer.py)."""
    
    # This is how it was done before - complex and inconsistent
    def create_model_old(training_protocol: str, model: str, config, z_shape, x_ndims: int = 1):
        """Old complex create_model function."""
        
        # For wrapper models, ensure the config has the right model type for the CRN backbone
        if model in ["potential_flow", "geometric_flow", "natural_flow"]:
            config.config_dict['model'] = "conditional_resnet_mlp"  # Use CRN MLP as backbone
            print(f"Using CRN MLP backbone for {model} wrapper")
        else:
            print(f"Using direct {model} architecture")
        
        # Create the appropriate model based on training protocol
        if training_protocol == "fm":
            return NoPropFM(config=config, z_shape=z_shape, x_ndims=x_ndims, model=model)
        elif training_protocol == "ct":
            return NoPropCT(config=config, z_shape=z_shape, x_ndims=x_ndims, model=model)
        elif training_protocol == "df":
            return NoPropDF(config=config, z_shape=z_shape, x_ndims=x_ndims, model=model)
        else:
            raise ValueError(f"Unsupported training protocol: {training_protocol}")
    
    print("Old approach: Complex, inconsistent, hard to maintain")


def new_trainer_approach():
    """Example of the new simple approach using factory functions."""
    
    # This is how it's done now - simple and consistent
    def create_model_new(training_protocol: str, model: str, z_shape, x_ndims: int = 1, **kwargs):
        """New simple create_model function using factory."""
        return create_noprop_model(
            training_protocol=training_protocol,
            model_name=model,
            z_shape=z_shape,
            x_ndims=x_ndims,
            **kwargs
        )
    
    print("New approach: Simple, consistent, easy to maintain")
    
    # Example usage
    z_shape = (8,)
    x_ndims = 1
    
    # Create different models easily
    models = {}
    
    # FM models
    models["fm_conditional_resnet"] = create_model_new("fm", "conditional_resnet", z_shape, x_ndims)
    models["fm_potential_flow"] = create_model_new("fm", "potential_flow", z_shape, x_ndims)
    
    # CT models  
    models["ct_conditional_resnet"] = create_model_new("ct", "conditional_resnet", z_shape, x_ndims)
    models["ct_geometric_flow"] = create_model_new("ct", "geometric_flow", z_shape, x_ndims)
    
    # DF models
    models["df_conditional_resnet"] = create_model_new("df", "conditional_resnet", z_shape, x_ndims)
    models["df_vae_flow"] = create_model_new("df", "vae_flow", z_shape, x_ndims)
    
    print(f"Created {len(models)} models successfully!")
    for name, model in models.items():
        print(f"  {name}: {type(model).__name__}")


def simple_model_creation():
    """Example of creating individual model components."""
    
    print("\n=== Simple Model Creation ===")
    
    z_dim, x_dim = 8, 4
    
    # Create individual models easily
    models = {}
    
    # Basic models
    models["conditional_resnet"] = create_simple_model("conditional_resnet", z_dim, x_dim)
    models["potential_flow"] = create_simple_model("potential_flow", z_dim, x_dim)
    models["geometric_flow"] = create_simple_model("geometric_flow", z_dim, x_dim)
    models["natural_flow"] = create_simple_model("natural_flow", z_dim, x_dim)
    models["vae_flow"] = create_simple_model("vae_flow", z_dim, x_dim)
    
    print(f"Created {len(models)} simple models:")
    for name, model in models.items():
        print(f"  {name}: {type(model).__name__}")


def trainer_integration_example():
    """Example of how to integrate into existing trainer code."""
    
    print("\n=== Trainer Integration Example ===")
    
    # This is how you would modify the existing trainer.py
    def load_training_data(data_path: str):
        """Load training data (unchanged)."""
        # ... existing code ...
        return data, x_dim, y_dim
    
    def main_new():
        """New main function using factory functions."""
        # Load data
        data, x_dim, y_dim = load_training_data("data.pkl")
        
        # Set up dimensions
        z_shape = (y_dim,)
        x_ndims = 1
        
        # Create model using factory - much simpler!
        model = create_noprop_model(
            training_protocol="fm",  # or "ct", "df"
            model_name="conditional_resnet",  # or "potential_flow", "vae_flow", etc.
            z_shape=z_shape,
            x_ndims=x_ndims
        )
        
        print(f"Created model: {type(model).__name__}")
        
        # Rest of training code remains the same...
        # trainer = NoPropTrainer(model)
        # results = trainer.train(...)
    
    print("Trainer integration: Replace complex create_model with simple factory call")


if __name__ == "__main__":
    print("=== Model Factory Examples ===")
    
    old_trainer_approach()
    new_trainer_approach()
    simple_model_creation()
    trainer_integration_example()
    
    print("\n=== Benefits of New Approach ===")
    print("✓ Simple, consistent interface")
    print("✓ Easy to add new model types")
    print("✓ Automatic shape handling")
    print("✓ Clean separation of concerns")
    print("✓ Easy to test and maintain")
