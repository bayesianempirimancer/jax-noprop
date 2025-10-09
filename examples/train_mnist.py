"""
Example training script for NoProp on MNIST dataset.

This script demonstrates how to train all three NoProp variants (DT, CT, FM)
on the MNIST dataset using JAX/Flax.
"""

import argparse
import time
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
import matplotlib.pyplot as plt

# Import our NoProp implementations
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from jax_noprop import NoPropDT, NoPropCT, NoPropFM, ConditionalResNet, SimpleCNN
from jax_noprop.utils import (
    create_train_state, train_step, eval_step, 
    one_hot_encode, create_data_iterators
)
from jax_noprop.noise_schedules import LinearNoiseSchedule


def load_mnist():
    """Load MNIST dataset."""
    try:
        from torchvision import datasets, transforms
        import torch
        
        # Load MNIST
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, 
            transform=transforms.ToTensor()
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True,
            transform=transforms.ToTensor()
        )
        
        # Convert to numpy
        train_images = np.array([img.numpy() for img, _ in train_dataset])
        train_labels = np.array([label for _, label in train_dataset])
        test_images = np.array([img.numpy() for img, _ in test_dataset])
        test_labels = np.array([label for _, label in test_dataset])
        
        # Reshape to [N, H, W, C] format
        train_images = train_images.transpose(0, 2, 3, 1)
        test_images = test_images.transpose(0, 2, 3, 1)
        
        return (train_images, train_labels), (test_images, test_labels)
        
    except ImportError:
        print("torchvision not available, using dummy data")
        # Create dummy data for testing
        train_images = np.random.rand(1000, 28, 28, 1).astype(np.float32) * 255
        train_labels = np.random.randint(0, 10, 1000)
        test_images = np.random.rand(200, 28, 28, 1).astype(np.float32) * 255
        test_labels = np.random.randint(0, 10, 200)
        
        return (train_images, train_labels), (test_images, test_labels)


def train_model(
    model: Any,
    train_data: tuple,
    test_data: tuple,
    args: argparse.Namespace,
    variant: str
) -> Dict[str, Any]:
    """Train a NoProp model.
    
    Args:
        model: NoProp model (DT, CT, or FM)
        train_data: Training data tuple
        test_data: Test data tuple
        args: Command line arguments
        variant: Model variant name
        
    Returns:
        Training results dictionary
    """
    print(f"\nTraining NoProp-{variant}")
    print("=" * 50)
    
    # Load data
    (train_images, train_labels), (test_images, test_labels) = train_data, test_data
    
    # Convert labels to one-hot
    train_labels = one_hot_encode(train_labels, 10)
    test_labels = one_hot_encode(test_labels, 10)
    
    # Create data iterators
    key = jax.random.PRNGKey(42)
    train_iter, test_iter = create_data_iterators(
        (train_images, train_labels),
        (test_images, test_labels),
        args.batch_size,
        key
    )
    
    # Initialize model
    key, init_key = jax.random.split(key)
    dummy_x = jnp.ones((1, 28, 28, 1))
    dummy_z = jnp.ones((1, 10))
    dummy_t = jnp.ones((1,))
    
    if variant == "DT":
        params = model.model.init(init_key, dummy_z, dummy_x)
    else:  # CT or FM
        params = model.model.init(init_key, dummy_z, dummy_x, dummy_t)
    
    # Create training state
    state = create_train_state(
        model, params, 
        learning_rate=args.learning_rate,
        optimizer=args.optimizer
    )
    
    # Training loop
    train_losses = []
    test_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_losses = []
        
        # Training
        for batch_idx, (x, y) in enumerate(train_iter()):
            key, train_key = jax.random.split(key)
            state, loss, metrics = train_step(state, x, y, train_key)
            epoch_losses.append(loss)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        # Evaluation
        test_accs = []
        for x, y in test_iter():
            key, eval_key = jax.random.split(key)
            metrics = eval_step(state, x, y, eval_key)
            test_accs.append(metrics["accuracy"])
        
        avg_accuracy = np.mean(test_accs)
        test_accuracies.append(avg_accuracy)
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
    
    training_time = time.time() - start_time
    
    return {
        "variant": variant,
        "train_losses": train_losses,
        "test_accuracies": test_accuracies,
        "final_accuracy": test_accuracies[-1],
        "training_time": training_time,
        "state": state
    }


def main():
    parser = argparse.ArgumentParser(description="Train NoProp on MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--variants", nargs="+", default=["DT", "CT", "FM"], 
                       help="NoProp variants to train")
    parser.add_argument("--model_type", type=str, default="simple", 
                       choices=["simple", "resnet"], help="Model architecture")
    
    args = parser.parse_args()
    
    print("Loading MNIST dataset...")
    train_data, test_data = load_mnist()
    print(f"Train data shape: {train_data[0].shape}")
    print(f"Test data shape: {test_data[0].shape}")
    
    # Create models
    if args.model_type == "simple":
        base_model = SimpleCNN(num_classes=10)
    else:
        base_model = ConditionalResNet(num_classes=10, depth=18)
    
    models = {}
    if "DT" in args.variants:
        models["DT"] = NoPropDT(base_model, num_timesteps=10)
    if "CT" in args.variants:
        models["CT"] = NoPropCT(base_model, num_timesteps=1000)
    if "FM" in args.variants:
        models["FM"] = NoPropFM(base_model, num_timesteps=1000)
    
    # Train models
    results = {}
    for variant, model in models.items():
        results[variant] = train_model(model, train_data, test_data, args, variant)
    
    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for variant, result in results.items():
        print(f"NoProp-{variant}:")
        print(f"  Final Accuracy: {result['final_accuracy']:.4f}")
        print(f"  Training Time: {result['training_time']:.2f}s")
        print()
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for variant, result in results.items():
        plt.plot(result['train_losses'], label=f'NoProp-{variant}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    for variant, result in results.items():
        plt.plot(result['test_accuracies'], label=f'NoProp-{variant}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.title('Test Accuracy')
    
    plt.tight_layout()
    plt.savefig('mnist_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Results saved to 'mnist_results.png'")


if __name__ == "__main__":
    main()
