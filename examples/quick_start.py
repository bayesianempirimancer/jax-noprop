"""
Quick start example for NoProp implementations.

This script demonstrates the basic usage of all three NoProp variants
with a simple example.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Import our NoProp implementations
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from jax_noprop import NoPropDT, NoPropCT, NoPropFM, SimpleCNN
from jax_noprop.utils import one_hot_encode
from jax_noprop.noise_schedules import LinearNoiseSchedule


def main():
    print("NoProp Quick Start Example")
    print("=" * 40)
    
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create dummy data
    batch_size = 32
    num_classes = 10
    image_size = 28
    
    # Generate random images and labels
    key, data_key = jax.random.split(key)
    images = jax.random.normal(data_key, (batch_size, image_size, image_size, 1))
    
    key, label_key = jax.random.split(key)
    labels = jax.random.randint(label_key, (batch_size,), 0, num_classes)
    labels_onehot = one_hot_encode(labels, num_classes)
    
    print(f"Data shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Labels (one-hot): {labels_onehot.shape}")
    
    # Create model
    model = SimpleCNN(num_classes=num_classes)
    
    # Initialize model parameters
    key, init_key = jax.random.split(key)
    dummy_z = jnp.ones((1, num_classes))
    dummy_t = jnp.ones((1,))
    params = model.init(init_key, dummy_z, images[:1], dummy_t)
    
    print(f"\nModel initialized with {sum(x.size for x in jax.tree_leaves(params))} parameters")
    
    # Test all three NoProp variants
    variants = {
        "DT": NoPropDT(model, num_timesteps=10),
        "CT": NoPropCT(model, num_timesteps=1000),
        "FM": NoPropFM(model, num_timesteps=1000)
    }
    
    for variant_name, noprop in variants.items():
        print(f"\n--- NoProp-{variant_name} ---")
        
        # Sample random timesteps
        key, t_key = jax.random.split(key)
        if variant_name == "DT":
            t = noprop.sample_timestep(t_key, batch_size)
        else:
            t = noprop.sample_timestep(t_key, batch_size)
        
        print(f"Sampled timesteps: {t[:5]}...")
        
        # Add noise to targets
        key, noise_key = jax.random.split(key)
        if variant_name == "FM":
            # For FM, we need to sample from base distribution
            z_t, noise = noprop.sample_base_distribution(noise_key, labels_onehot.shape), None
        else:
            z_t, noise = noprop.add_noise_to_target(labels_onehot, noise_key, t)
        
        print(f"Noisy target shape: {z_t.shape}")
        print(f"Noisy target norm: {jnp.linalg.norm(z_t, axis=-1)[:5]}")
        
        # Compute loss
        key, loss_key = jax.random.split(key)
        if variant_name == "FM":
            loss, metrics = noprop.compute_flow_matching_loss(
                params, z_t, labels_onehot, images, t, loss_key
            )
        else:
            loss, metrics = noprop.compute_loss(
                params, z_t, images, labels_onehot, t, loss_key
            )
        
        print(f"Loss: {loss:.4f}")
        print(f"Metrics: {metrics}")
        
        # Generate predictions (using fewer steps for demo)
        key, gen_key = jax.random.split(key)
        if variant_name == "DT":
            pred = noprop.generate(params, images, gen_key, num_steps=5)
        else:
            pred = noprop.generate(params, images, gen_key, num_steps=10)
        
        print(f"Generated predictions shape: {pred.shape}")
        print(f"Prediction norm: {jnp.linalg.norm(pred, axis=-1)[:5]}")
        
        # Compute accuracy
        pred_classes = jnp.argmax(pred, axis=-1)
        true_classes = jnp.argmax(labels_onehot, axis=-1)
        accuracy = jnp.mean(pred_classes == true_classes)
        print(f"Accuracy: {accuracy:.4f}")
    
    print(f"\n" + "=" * 40)
    print("Quick start example completed!")
    print("\nTo train models on real datasets, run:")
    print("  python examples/train_mnist.py")
    print("  python examples/train_cifar.py")


if __name__ == "__main__":
    main()
