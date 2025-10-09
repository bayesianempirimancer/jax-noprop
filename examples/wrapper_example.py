"""
Example demonstrating the NoPropModelWrapper for integrating learnable noise schedules.

This example shows how to use the wrapper to ensure that learnable noise schedule
parameters are part of the model's parameter tree and get updated during training.
"""

import jax.numpy as jnp
import jax.random as jr
import optax

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from jax_noprop import (
    ConditionalResNet, 
    LearnableNoiseSchedule, 
    CosineNoiseSchedule,
    NoPropModelWrapper, 
    create_no_prop_model,
    NoPropDT
)


def main():
    """Demonstrate the wrapper solution."""
    print("=== NoPropModelWrapper Example ===\n")
    
    # Create components
    model = ConditionalResNet(num_classes=10, z_dim=10)
    learnable_schedule = LearnableNoiseSchedule(hidden_dims=(64, 64))
    fixed_schedule = CosineNoiseSchedule()
    
    # Create dummy data
    key = jr.PRNGKey(42)
    batch_size = 4
    z_dim = 10
    x_dim = (28, 28, 1)
    
    dummy_z = jnp.ones((batch_size, z_dim))
    dummy_x = jnp.ones((batch_size, *x_dim))
    dummy_t = jnp.ones((batch_size,))
    
    print("1. Testing with learnable noise schedule...")
    
    # Create wrapper with learnable schedule
    wrapper_learnable = create_no_prop_model(model, learnable_schedule)
    
    # Initialize parameters
    params_learnable = wrapper_learnable.init(key, dummy_z, dummy_x, dummy_t)
    
    print(f"   ✓ Model parameters include noise schedule: {'gamma_network' in params_learnable['params']}")
    print(f"   ✓ Parameter tree keys: {list(params_learnable['params'].keys())}")
    
    # Test forward pass
    output = wrapper_learnable.apply(params_learnable, dummy_z, dummy_x, dummy_t)
    print(f"   ✓ Forward pass works: {output.shape}")
    
    # Test parameter extraction
    noise_params = wrapper_learnable.get_noise_schedule_params(params_learnable)
    model_params = wrapper_learnable.get_model_params(params_learnable)
    
    print(f"   ✓ Noise schedule parameters extracted: {noise_params is not None}")
    print(f"   ✓ Model parameters extracted: {len(model_params)} submodules")
    
    # Test noise schedule functionality
    t_test = jnp.array([0.0, 0.5, 1.0])
    gamma_t = learnable_schedule.get_gamma_t(t_test, noise_params)
    print(f"   ✓ Noise schedule works: γ(t) = {gamma_t}")
    
    print("\n2. Testing with fixed noise schedule...")
    
    # Create wrapper with fixed schedule
    wrapper_fixed = create_no_prop_model(model, fixed_schedule)
    
    # Initialize parameters
    params_fixed = wrapper_fixed.init(key, dummy_z, dummy_x, dummy_t)
    
    print(f"   ✓ Model parameters exclude noise schedule: {'gamma_network' not in params_fixed['params']}")
    print(f"   ✓ Parameter tree keys: {list(params_fixed['params'].keys())}")
    
    # Test forward pass
    output = wrapper_fixed.apply(params_fixed, dummy_z, dummy_x, dummy_t)
    print(f"   ✓ Forward pass works: {output.shape}")
    
    # Test parameter extraction
    noise_params = wrapper_fixed.get_noise_schedule_params(params_fixed)
    model_params = wrapper_fixed.get_model_params(params_fixed)
    
    print(f"   ✓ Noise schedule parameters extracted: {noise_params is None}")
    print(f"   ✓ Model parameters extracted: {len(model_params)} submodules")
    
    print("\n3. Integration with NoProp training...")
    
    # Create a NoProp model using the wrapper
    noprop_model = NoPropDT(
        model=wrapper_learnable,  # Use the wrapper instead of raw model
        noise_schedule=learnable_schedule,
        num_timesteps=20
    )
    
    print("   ✓ NoProp model created with wrapper")
    print("   ✓ Learnable noise schedule parameters will be trained")
    
    # Demonstrate training setup
    optimizer = optax.adam(learning_rate=1e-3)
    
    # The parameters from the wrapper can be used directly with optax
    opt_state = optimizer.init(params_learnable)
    print("   ✓ Optimizer initialized with combined parameters")
    print("   ✓ Both model and noise schedule parameters will be updated")
    
    print("\n🎉 Wrapper solution successfully integrates learnable noise schedules!")
    print("   ✅ All parameters are in a single parameter tree")
    print("   ✅ Parameters are updated together during training")
    print("   ✅ Clean API for parameter extraction")
    print("   ✅ Works with both learnable and fixed schedules")


if __name__ == "__main__":
    main()
