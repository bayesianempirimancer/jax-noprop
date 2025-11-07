#!/usr/bin/env python3
"""
Generate two moons dataset for VAE_flow testing.

This script creates a two moons classification dataset with 10,000 points,
ensuring both moons are evenly and completely sampled.

NOTE: This script should be called from the project root directory:
    python examples/two_moons/generate_two_moons.py [args]

All paths (output_dir) are relative to the project root directory.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import argparse
import os


def generate_two_moons_dataset(
    n_samples: int = 10000,
    noise: float = 0.1,
    seed: int = 42
) -> tuple:
    """
    Generate two moons dataset with balanced sampling.
    
    Args:
        n_samples: Total number of samples
        noise: Standard deviation of Gaussian noise added to the data
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (x_data, y_data) where:
        - x_data: 2D coordinates [n_samples, 2]
        - y_data: Class labels [n_samples] (0 or 1)
    """
    # Generate the two moons dataset
    x_data, y_data = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=seed
    )
    
    # Ensure both classes are represented
    unique_classes, counts = np.unique(y_data, return_counts=True)
    print(f"Class distribution:")
    for cls, count in zip(unique_classes, counts):
        print(f"  Class {cls}: {count} samples ({count/n_samples*100:.1f}%)")
    
    return x_data, y_data


def visualize_dataset(x_data: np.ndarray, y_data: np.ndarray, save_path: str = None):
    """
    Visualize the two moons dataset.
    
    Args:
        x_data: 2D coordinates [n_samples, 2]
        y_data: Class labels [n_samples] (integer) or [n_samples, num_classes] (one-hot)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # Convert one-hot to integer labels if needed
    if len(y_data.shape) == 2 and y_data.shape[1] > 1:
        # One-hot encoded: convert to integer labels
        y_int = np.argmax(y_data, axis=1)
    else:
        # Already integer labels
        y_int = y_data
    
    # Plot the two moons
    colors = ['red', 'blue']
    labels = ['Moon 0', 'Moon 1']
    
    for i in range(2):
        mask = y_int == i
        plt.scatter(x_data[mask, 0], x_data[mask, 1], 
                   c=colors[i], label=labels[i], alpha=0.6, s=20)
    
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Two Moons Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset visualization saved to {save_path}")
    
    plt.show()


def save_dataset(x_data: np.ndarray, y_data: np.ndarray, filepath: str, 
                 train_ratio: float = 0.80, seed: int = 42):
    """
    Save the dataset in the formatted format with train/val splits.
    
    NOTE: Data is ALWAYS shuffled before splitting. This cannot be disabled.
    y_data is converted to one-hot encoding before saving.
    
    Args:
        x_data: 2D coordinates [n_samples, 2]
        y_data: Class labels [n_samples] (integer labels 0 or 1)
        filepath: Path to save the pickle file
        train_ratio: Fraction of data for training (default: 0.80)
        seed: Random seed for splitting (also used for shuffling)
    """
    # Convert y_data to one-hot encoding
    num_classes = len(np.unique(y_data))
    y_onehot = np.eye(num_classes)[y_data.astype(int)]  # [n_samples, num_classes]
    
    print(f"Converting labels to one-hot encoding:")
    print(f"  Original y shape: {y_data.shape}")
    print(f"  One-hot y shape: {y_onehot.shape}")
    print(f"  Number of classes: {num_classes}")
    
    # Split into train and validation sets
    # IMPORTANT: shuffle=True is MANDATORY - data must always be shuffled before splitting
    # This ensures proper randomization and prevents data leakage
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_onehot, 
        train_size=train_ratio, 
        random_state=seed,
        shuffle=True,  # MANDATORY: Data is always shuffled before splitting
        stratify=y_data  # Use original integer labels for stratification
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
    
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved to {filepath}")
    print(f"Dataset info:")
    print(f"  Train samples: {x_train.shape[0]}")
    print(f"  Val samples: {x_val.shape[0]}")
    print(f"  Input dim: {x_data.shape[1]}")
    print(f"  Output dim (one-hot): {y_train.shape[1]}")
    print(f"  Classes: {num_classes}")


def load_dataset(filepath: str) -> dict:
    """
    Load the dataset from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Dictionary containing the dataset
    """
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Dataset loaded from {filepath}")
    return dataset


def main():
    """Main function to generate and save the two moons dataset."""
    parser = argparse.ArgumentParser(description='Generate two moons dataset')
    parser.add_argument('--n_samples', type=int, default=10000, 
                       help='Number of samples to generate')
    parser.add_argument('--noise', type=float, default=0.1, 
                       help='Noise level for the dataset')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./data', 
                       help='Directory to save the dataset')
    parser.add_argument('--filename', type=str, default='two_moons_xy_format.pkl', 
                       help='Filename for the dataset')
    parser.add_argument('--train_ratio', type=float, default=0.80,
                       help='Fraction of data for training (default: 0.80)')
    parser.add_argument('--visualize', action='store_true', 
                       help='Show visualization of the dataset')
    parser.add_argument('--save_plot', action='store_true', 
                       help='Save visualization plot')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Two Moons Dataset Generator")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Number of samples: {args.n_samples}")
    print(f"  Noise level: {args.noise}")
    print(f"  Seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Filename: {args.filename}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate the dataset
    print("Generating two moons dataset...")
    x_data, y_data = generate_two_moons_dataset(
        n_samples=args.n_samples,
        noise=args.noise,
        seed=args.seed
    )
    
    # Save the dataset
    filepath = os.path.join(args.output_dir, args.filename)
    save_dataset(x_data, y_data, filepath, 
                 train_ratio=args.train_ratio, 
                 seed=args.seed)
    
    # Visualize if requested
    if args.visualize or args.save_plot:
        plot_path = None
        if args.save_plot:
            plot_path = os.path.join(args.output_dir, 'two_moons_visualization.png')
        
        print("Creating visualization...")
        visualize_dataset(x_data, y_data, save_path=plot_path)
    
    # Test loading the dataset
    print("\nTesting dataset loading...")
    loaded_dataset = load_dataset(filepath)
    
    # Verify the data structure
    assert 'train' in loaded_dataset, "Missing 'train' key!"
    assert 'val' in loaded_dataset, "Missing 'val' key!"
    assert 'x' in loaded_dataset['train'], "Missing 'x' in train!"
    assert 'y' in loaded_dataset['train'], "Missing 'y' in train!"
    assert 'x' in loaded_dataset['val'], "Missing 'x' in val!"
    assert 'y' in loaded_dataset['val'], "Missing 'y' in val!"
    print("Dataset verification passed! ✅")
    
    print(f"\nDataset ready for VAE_flow training!")
    print(f"Use: x_train = dataset['train']['x'], y_train = dataset['train']['y']")
    print(f"     x_val = dataset['val']['x'], y_val = dataset['val']['y']")
    print(f"Note: y is one-hot encoded with shape (n_samples, 2)")


if __name__ == "__main__":
    main()
