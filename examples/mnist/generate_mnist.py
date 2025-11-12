#!/usr/bin/env python3
"""
Generate MNIST dataset with one-hot encoded labels for flow models.

This script loads MNIST data, flattens images to 784 dimensions, converts labels
to one-hot encoding, and saves the dataset in the expected format for regression tasks.

NOTE: This script should be called from the project root directory:
    python examples/mnist/generate_mnist.py [args]

All paths (output_dir) are relative to the project root directory.
"""

import numpy as np
import pickle
import argparse
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torchvision
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import tensorflow_datasets as tfds
    TFDS_AVAILABLE = True
except ImportError:
    TFDS_AVAILABLE = False


def load_mnist_torchvision():
    """Load MNIST dataset using torchvision."""
    print("Loading MNIST using torchvision...")
    
    # Define transforms: convert to numpy and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
    ])
    
    # Load training and test sets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Extract images and labels
    train_images = []
    train_labels = []
    for img, label in train_dataset:
        train_images.append(img.numpy().flatten())  # Flatten to (784,)
        train_labels.append(label)
    
    test_images = []
    test_labels = []
    for img, label in test_dataset:
        test_images.append(img.numpy().flatten())  # Flatten to (784,)
        test_labels.append(label)
    
    # Convert to numpy arrays
    train_images = np.array(train_images, dtype=np.float32)  # (60000, 784)
    train_labels = np.array(train_labels, dtype=np.int32)  # (60000,)
    test_images = np.array(test_images, dtype=np.float32)  # (10000, 784)
    test_labels = np.array(test_labels, dtype=np.int32)  # (10000,)
    
    print(f"Loaded MNIST:")
    print(f"  Train: {train_images.shape[0]} samples, images shape: {train_images.shape}")
    print(f"  Test: {test_images.shape[0]} samples, images shape: {test_images.shape}")
    
    return train_images, train_labels, test_images, test_labels


def load_mnist_tfds():
    """Load MNIST dataset using tensorflow_datasets."""
    print("Loading MNIST using tensorflow_datasets...")
    
    # Load dataset
    ds_train, ds_test = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True
    )
    
    # Convert to numpy arrays
    train_images = []
    train_labels = []
    for img, label in tfds.as_numpy(ds_train):
        # img is (28, 28, 1), normalize to [0, 1] and flatten
        img_flat = img.astype(np.float32).reshape(-1) / 255.0  # (784,)
        train_images.append(img_flat)
        train_labels.append(label)
    
    test_images = []
    test_labels = []
    for img, label in tfds.as_numpy(ds_test):
        # img is (28, 28, 1), normalize to [0, 1] and flatten
        img_flat = img.astype(np.float32).reshape(-1) / 255.0  # (784,)
        test_images.append(img_flat)
        test_labels.append(label)
    
    # Convert to numpy arrays
    train_images = np.array(train_images, dtype=np.float32)  # (60000, 784)
    train_labels = np.array(train_labels, dtype=np.int32)  # (60000,)
    test_images = np.array(test_images, dtype=np.float32)  # (10000, 784)
    test_labels = np.array(test_labels, dtype=np.int32)  # (10000,)
    
    print(f"Loaded MNIST:")
    print(f"  Train: {train_images.shape[0]} samples, images shape: {train_images.shape}")
    print(f"  Test: {test_images.shape[0]} samples, images shape: {test_images.shape}")
    
    return train_images, train_labels, test_images, test_labels


def convert_to_one_hot(labels, num_classes=10):
    """Convert integer labels to one-hot encoding."""
    return np.eye(num_classes, dtype=np.float32)[labels]


def save_dataset(x_data: np.ndarray, y_data: np.ndarray, filepath: str,
                 train_ratio: float = 0.8, seed: int = 42):
    """
    Save the dataset in the formatted format with train/val splits.
    
    Args:
        x_data: Flattened images [n_samples, 784]
        y_data: One-hot encoded labels [n_samples, 10] or integer labels [n_samples]
        filepath: Path to save the pickle file
        train_ratio: Fraction of data for training (default: 0.8)
        seed: Random seed for splitting
    """
    # Check if y_data is already one-hot encoded
    if len(y_data.shape) == 2 and y_data.shape[1] > 1:
        # Already one-hot encoded
        y_onehot = y_data
        y_int = np.argmax(y_data, axis=1)  # For stratification
        num_classes = y_onehot.shape[1]
        print(f"Using one-hot encoded labels (already converted):")
        print(f"  y shape: {y_onehot.shape}")
    else:
        # Convert integer labels to one-hot encoding
        num_classes = len(np.unique(y_data))
        y_onehot = convert_to_one_hot(y_data, num_classes=num_classes)
        y_int = y_data  # For stratification
        print(f"Converting labels to one-hot encoding:")
        print(f"  Original y shape: {y_data.shape}")
        print(f"  One-hot y shape: {y_onehot.shape}")
        print(f"  Number of classes: {num_classes}")
    
    # Split into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_onehot,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=y_int  # Use integer labels for stratification
    )
    
    dataset = {
        'train': {
            'x': x_train,
            'y': y_train
        },
        'val': {
            'x': x_val,
            'y': y_val
        }
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nDataset saved to {filepath}")
    print(f"Dataset info:")
    print(f"  Train samples: {x_train.shape[0]}")
    print(f"  Val samples: {x_val.shape[0]}")
    print(f"  Input dim (flattened images): {x_data.shape[1]}")
    print(f"  Output dim (one-hot): {y_train.shape[1]}")
    print(f"  Classes: {num_classes}")
    print(f"  Image pixel range: [{x_data.min():.4f}, {x_data.max():.4f}]")


def main():
    """Main function to generate and save the MNIST dataset."""
    parser = argparse.ArgumentParser(description='Generate MNIST dataset with one-hot labels')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Directory to save the dataset (default: ./data)')
    parser.add_argument('--filename', type=str, default='mnist.pkl',
                       help='Output filename (default: mnist.pkl)')
    parser.add_argument('--use_tfds', action='store_true',
                       help='Use tensorflow_datasets instead of torchvision')
    
    args = parser.parse_args()
    
    # Check which library is available
    if args.use_tfds:
        if not TFDS_AVAILABLE:
            print("ERROR: tensorflow_datasets not available. Install with: pip install tensorflow_datasets")
            sys.exit(1)
        train_images, train_labels, test_images, test_labels = load_mnist_tfds()
    else:
        if not TORCHVISION_AVAILABLE:
            if TFDS_AVAILABLE:
                print("WARNING: torchvision not available. Falling back to tensorflow_datasets.")
                train_images, train_labels, test_images, test_labels = load_mnist_tfds()
            else:
                print("ERROR: Neither torchvision nor tensorflow_datasets is available.")
                print("Install one of them with:")
                print("  pip install torchvision")
                print("  or")
                print("  pip install tensorflow_datasets")
                sys.exit(1)
        else:
            train_images, train_labels, test_images, test_labels = load_mnist_torchvision()
    
    # Combine train and test sets for full dataset
    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    
    print(f"\nCombined dataset:")
    print(f"  Total samples: {all_images.shape[0]}")
    print(f"  Image shape: {all_images.shape[1]} (flattened 28x28)")
    
    # Save dataset
    output_path = os.path.join(args.output_dir, args.filename)
    save_dataset(
        x_data=all_images,
        y_data=all_labels,
        filepath=output_path,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print(f"\n✅ Successfully generated MNIST dataset!")
    print(f"   Saved to: {output_path}")


if __name__ == '__main__':
    main()

