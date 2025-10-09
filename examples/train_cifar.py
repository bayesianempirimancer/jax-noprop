"""
Example training script for NoProp on CIFAR-10/100 datasets.

This script demonstrates how to train NoProp variants on CIFAR datasets
with ResNet backbones.
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

from jax_noprop import NoPropDT, NoPropCT, NoPropFM, ConditionalResNet
from jax_noprop.utils import (
    create_train_state, train_step, eval_step, 
    one_hot_encode, create_data_iterators
)
from jax_noprop.noise_schedules import LinearNoiseSchedule


def load_cifar(dataset: str = "cifar10"):
    """Load CIFAR-10 or CIFAR-100 dataset."""
    try:
        from torchvision import datasets, transforms
        import torch
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
        # Load dataset
        if dataset == "cifar10":
            train_dataset = datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform
            )
            num_classes = 10
        elif dataset == "cifar100":
            train_dataset = datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform
            )
            num_classes = 100
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Convert to numpy
        train_images = np.array([img.numpy() for img, _ in train_dataset])
        train_labels = np.array([label for _, label in train_dataset])
        test_images = np.array([img.numpy() for img, _ in test_dataset])
        test_labels = np.array([label for _, label in test_dataset])
        
        # Reshape to [N, H, W, C] format
        train_images = train_images.transpose(0, 2, 3, 1)
        test_images = test_images.transpose(0, 2, 3, 1)
        
        return (train_images, train_labels), (test_images, test_labels), num_classes
        
    except ImportError:
        print("torchvision not available, using dummy data")
        # Create dummy data for testing
        num_classes = 10 if dataset == "cifar10" else 100
        train_images = np.random.rand(1000, 32, 32, 3).astype(np.float32)
        train_labels = np.random.randint(0, num_classes, 1000)
        test_images = np.random.rand(200, 32, 32, 3).astype(np.float32)
        test_labels = np.random.randint(0, num_classes, 200)
        
        return (train_images, train_labels), (test_images, test_labels), num_classes


def train_model(
    model: Any,
    train_data: tuple,
    test_data: tuple,
    num_classes: int,
    args: argparse.Namespace,
    variant: str
) -> Dict[str, Any]:
    """Train a NoProp model on CIFAR.
    
    Args:
        model: NoProp model (DT, CT, or FM)
        train_data: Training data tuple
        test_data: Test data tuple
        num_classes: Number of classes
        args: Command line arguments
        variant: Model variant name
        
    Returns:
        Training results dictionary
    """
    print(f"\nTraining NoProp-{variant} on {args.dataset.upper()}")
    print("=" * 50)
    
    # Load data
    (train_images, train_labels), (test_images, test_labels) = train_data, test_data
    
    # Convert labels to one-hot
    train_labels = one_hot_encode(train_labels, num_classes)
    test_labels = one_hot_encode(test_labels, num_classes)
    
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
    dummy_x = jnp.ones((1, 32, 32, 3))
    dummy_z = jnp.ones((1, num_classes))
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
        "dataset": args.dataset,
        "train_losses": train_losses,
        "test_accuracies": test_accuracies,
        "final_accuracy": test_accuracies[-1],
        "training_time": training_time,
        "state": state
    }


def main():
    parser = argparse.ArgumentParser(description="Train NoProp on CIFAR")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                       choices=["cifar10", "cifar100"], help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer")
    parser.add_argument("--variants", nargs="+", default=["DT", "CT", "FM"], 
                       help="NoProp variants to train")
    parser.add_argument("--resnet_depth", type=int, default=18, 
                       choices=[18, 50, 152], help="ResNet depth")
    
    args = parser.parse_args()
    
    print(f"Loading {args.dataset.upper()} dataset...")
    train_data, test_data, num_classes = load_cifar(args.dataset)
    print(f"Train data shape: {train_data[0].shape}")
    print(f"Test data shape: {test_data[0].shape}")
    print(f"Number of classes: {num_classes}")
    
    # Create models with ResNet backbone
    base_model = ConditionalResNet(
        num_classes=num_classes, 
        depth=args.resnet_depth
    )
    
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
        results[variant] = train_model(
            model, train_data, test_data, num_classes, args, variant
        )
    
    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for variant, result in results.items():
        print(f"NoProp-{variant} on {result['dataset'].upper()}:")
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
    plt.title(f'Training Loss - {args.dataset.upper()}')
    
    plt.subplot(1, 2, 2)
    for variant, result in results.items():
        plt.plot(result['test_accuracies'], label=f'NoProp-{variant}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.title(f'Test Accuracy - {args.dataset.upper()}')
    
    plt.tight_layout()
    plt.savefig(f'{args.dataset}_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to '{args.dataset}_results.png'")


if __name__ == "__main__":
    main()
