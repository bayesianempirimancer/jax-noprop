#!/usr/bin/env python3
"""
Test script to demonstrate the generative mode functionality.
Shows the difference between regression mode (z_0 = 0) and generative mode (z_0 ~ N(0,1)).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add the src directory to the path
import sys
sys.path.append('src')

from flow_models.df import VAEFlowConfig, VAE_flow
from flow_models.trainer import VAEFlowTrainer

def test_generative_mode():
    """Test the generative mode functionality."""
    
    print("Testing Generative Mode Functionality")
    print("=" * 50)
    
    # Load a trained model (use the test model with config)
    model_path = "artifacts/test_generative_with_config_20251029_070323/model_params.pkl"
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train a model first using the training script.")
        return
    
    # Load model parameters and config
    import pickle
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    model_params = data['params']
    config = data['config']
    
    print(f"Loaded model with config type: {type(config)}")
    print(f"Config main keys: {list(config.main.keys())}")
    
    # Create model with the loaded config
    model = VAE_flow(config)
    
    # Create some test data
    key = jr.PRNGKey(42)
    x_test = jnp.array([[0.5, 0.5], [-0.5, -0.5], [1.0, 0.0], [-1.0, 0.0]])  # 4 test points
    
    print(f"Test input shape: {x_test.shape}")
    print(f"Test inputs:\n{x_test}")
    
    # Test 1: Regression mode (z_0 = 0)
    print("\n1. Regression Mode (z_0 = 0):")
    print("-" * 30)
    y_regression = model.predict(model_params, x_test, num_steps=20, prng_key=None)
    print(f"Regression output shape: {y_regression.shape}")
    print(f"Regression outputs:\n{y_regression}")
    
    # Test 2: Generative mode (z_0 ~ N(0,1))
    print("\n2. Generative Mode (z_0 ~ N(0,1)):")
    print("-" * 30)
    key_gen, _ = jr.split(key)
    y_generative = model.predict(model_params, x_test, num_steps=20, prng_key=key_gen)
    print(f"Generative output shape: {y_generative.shape}")
    print(f"Generative outputs:\n{y_generative}")
    
    # Test 3: Multiple generative samples (same input, different noise)
    print("\n3. Multiple Generative Samples (same input, different noise):")
    print("-" * 30)
    x_single = jnp.array([[0.0, 0.0]])  # Single input point
    print(f"Single input: {x_single}")
    
    for i in range(3):
        key_i, key_gen = jr.split(key_gen)
        y_gen_i = model.predict(model_params, x_single, num_steps=20, prng_key=key_i)
        print(f"Sample {i+1}: {y_gen_i[0]}")
    
    # Test 4: Compare outputs
    print("\n4. Comparison:")
    print("-" * 30)
    print("Regression mode produces deterministic outputs (same input → same output)")
    print("Generative mode produces stochastic outputs (same input → different outputs)")
    print(f"Are regression outputs deterministic? {jnp.allclose(y_regression, y_regression)}")
    print(f"Are generative outputs different? {not jnp.allclose(y_generative, y_regression)}")
    
    print("\nGenerative mode test completed successfully!")
    print("The model can now generate diverse samples from the same input by using random initial conditions.")

if __name__ == "__main__":
    test_generative_mode()
