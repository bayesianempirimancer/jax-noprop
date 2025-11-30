#!/usr/bin/env python3
"""
Generate spherical two moons dataset for VAE_flow testing.

This script creates a spherical two moons dataset by:
1. Loading or generating the two moons dataset (2D)
2. Adding a third dimension with value 1: [x, y, 1]
3. Normalizing each point to put it on the unit sphere

NOTE: This script should be called from the project root directory:
    python examples/two_moons/generate_spherical_two_moons.py [args]

All paths (output_dir) are relative to the project root directory.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import argparse
import os
import sys
from pathlib import Path

# Add project root to path to import generate_two_moons functions
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.two_moons.generate_two_moons import generate_two_moons_dataset, load_dataset


def create_spherical_two_moons(x_data_2d: np.ndarray) -> np.ndarray:
    """
    Convert 2D two moons data to 3D spherical data.
    
    Process:
    1. Standardize 2D data: center and scale so x and y have mean 0 and std 1
    2. Add third dimension with value 1: [x_std, y_std, 1]
    3. Normalize each point to unit sphere: point / ||point||
    
    Args:
        x_data_2d: 2D coordinates [n_samples, 2]
        
    Returns:
        3D coordinates on unit sphere [n_samples, 3]
    """
    n_samples = x_data_2d.shape[0]
    
    # Step 1: Standardize 2D data (center and scale to mean=0, std=1)
    x_mean = np.mean(x_data_2d, axis=0)
    x_std = np.std(x_data_2d, axis=0)
    x_data_standardized = (x_data_2d - x_mean) / x_std
    
    # Step 2: Add third dimension with value 1
    x_data_3d = np.zeros((n_samples, 3))
    x_data_3d[:, :2] = x_data_standardized
    x_data_3d[:, 2] = 1.0
    
    # Step 3: Normalize to unit sphere
    norms = np.linalg.norm(x_data_3d, axis=1, keepdims=True)
    x_data_spherical = x_data_3d / norms
    
    # Verify all points are on unit sphere
    norms_after = np.linalg.norm(x_data_spherical, axis=1)
    assert np.allclose(norms_after, 1.0), "Points are not on unit sphere!"
    
    return x_data_spherical


def visualize_spherical_dataset(x_data: np.ndarray, y_data: np.ndarray, save_path: str = None):
    """
    Visualize the spherical two moons dataset in 3D.
    
    Args:
        x_data: 3D coordinates on unit sphere [n_samples, 3]
        y_data: Class labels [n_samples] (integer) or [n_samples, num_classes] (one-hot)
        save_path: Path to save the plot (optional)
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Convert one-hot to integer labels if needed
    if len(y_data.shape) == 2 and y_data.shape[1] > 1:
        y_int = np.argmax(y_data, axis=1)
    else:
        y_int = y_data
    
    # 3D scatter plot
    ax1 = fig.add_subplot(121, projection='3d')
    colors = ['red', 'blue']
    labels = ['Moon 0', 'Moon 1']
    
    for i in range(2):
        mask = y_int == i
        ax1.scatter(x_data[mask, 0], x_data[mask, 1], x_data[mask, 2],
                   c=colors[i], label=labels[i], alpha=0.6, s=20)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Spherical Two Moons (3D)')
    ax1.legend()
    
    # 2D projection (first two dimensions)
    ax2 = fig.add_subplot(122)
    for i in range(2):
        mask = y_int == i
        ax2.scatter(x_data[mask, 0], x_data[mask, 1],
                   c=colors[i], label=labels[i], alpha=0.6, s=20)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Spherical Two Moons (XY Projection)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset visualization saved to {save_path}")
    
    plt.show()


def save_dataset(x_data: np.ndarray, y_data: np.ndarray, filepath: str, 
                 train_ratio: float = 0.80, seed: int = 42):
    """
    Save the dataset in the formatted format with train/val splits.
    
    NOTE: Data is ALWAYS shuffled before splitting. This cannot be disabled.
    y_data should already be one-hot encoded.
    
    Args:
        x_data: 3D coordinates on unit sphere [n_samples, 3]
        y_data: Class labels [n_samples, num_classes] (one-hot encoded) or [n_samples] (integer)
        filepath: Path to save the pickle file
        train_ratio: Fraction of data for training (default: 0.80)
        seed: Random seed for splitting (also used for shuffling)
    """
    # Check if y_data is already one-hot encoded
    if len(y_data.shape) == 2 and y_data.shape[1] > 1:
        y_onehot = y_data
        y_int = np.argmax(y_data, axis=1)  # For stratification
        num_classes = y_onehot.shape[1]
        print(f"Using one-hot encoded labels (already converted):")
        print(f"  y shape: {y_onehot.shape}")
    else:
        num_classes = len(np.unique(y_data))
        y_onehot = np.eye(num_classes)[y_data.astype(int)]  # [n_samples, num_classes]
        y_int = y_data  # For stratification
        print(f"Converting labels to one-hot encoding:")
        print(f"  Original y shape: {y_data.shape}")
        print(f"  One-hot y shape: {y_onehot.shape}")
        print(f"  Number of classes: {num_classes}")
    
    # Split into train and validation sets
    # IMPORTANT: shuffle=True is MANDATORY - data must always be shuffled before splitting
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_onehot, 
        train_size=train_ratio, 
        random_state=seed,
        shuffle=True,  # MANDATORY: Data is always shuffled before splitting
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
    
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved to {filepath}")
    print(f"Dataset info:")
    print(f"  Train samples: {x_train.shape[0]}")
    print(f"  Val samples: {x_val.shape[0]}")
    print(f"  Input dim: {x_data.shape[1]}")
    print(f"  Output dim (one-hot): {y_train.shape[1]}")
    print(f"  Classes: {num_classes}")


def main():
    """Main function to generate and save the spherical two moons dataset."""
    parser = argparse.ArgumentParser(description='Generate spherical two moons dataset')
    parser.add_argument('--input_data', type=str, default='./data/two_moons.pkl',
                       help='Path to input two moons dataset (will generate if not found)')
    parser.add_argument('--n_samples', type=int, default=10000, 
                       help='Number of samples to generate (if generating from scratch)')
    parser.add_argument('--noise', type=float, default=0.1, 
                       help='Noise level for the dataset (if generating from scratch)')
    parser.add_argument('--scale_factor', type=float, default=8.0,
                       help='Scale factor for the dataset (if generating from scratch)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./data', 
                       help='Directory to save the dataset')
    parser.add_argument('--filename', type=str, default='spherical_two_moons.pkl', 
                       help='Filename for the dataset')
    parser.add_argument('--train_ratio', type=float, default=0.80,
                       help='Fraction of data for training (default: 0.80)')
    parser.add_argument('--visualize', action='store_true', 
                       help='Show visualization of the dataset')
    parser.add_argument('--save_plot', action='store_true', 
                       help='Save visualization plot')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Spherical Two Moons Dataset Generator")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Input data: {args.input_data}")
    print(f"  Seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Filename: {args.filename}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or generate the 2D two moons dataset
    if os.path.exists(args.input_data):
        print(f"Loading existing two moons dataset from {args.input_data}...")
        dataset_2d = load_dataset(args.input_data)
        x_data_2d = np.concatenate([dataset_2d['train']['x'], dataset_2d['val']['x']], axis=0)
        y_data = np.concatenate([dataset_2d['train']['y'], dataset_2d['val']['y']], axis=0)
        print(f"Loaded {x_data_2d.shape[0]} samples from existing dataset")
    else:
        print(f"Input dataset not found. Generating two moons dataset...")
        x_data_2d, y_data = generate_two_moons_dataset(
            n_samples=args.n_samples,
            noise=args.noise,
            scale_factor=args.scale_factor,
            center=True,
            seed=args.seed
        )
        print(f"Generated {x_data_2d.shape[0]} samples")
    
    # Convert to spherical coordinates
    print("\nConverting to spherical coordinates...")
    print(f"  Input shape: {x_data_2d.shape}")
    x_data_spherical = create_spherical_two_moons(x_data_2d)
    print(f"  Output shape: {x_data_spherical.shape}")
    
    # Verify points are on unit sphere
    norms = np.linalg.norm(x_data_spherical, axis=1)
    print(f"  Norm statistics: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
    assert np.allclose(norms, 1.0, atol=1e-6), "Points are not on unit sphere!"
    print("  ✓ All points verified to be on unit sphere")
    
    # Save the dataset
    filepath = os.path.join(args.output_dir, args.filename)
    save_dataset(x_data_spherical, y_data, filepath, 
                 train_ratio=args.train_ratio, 
                 seed=args.seed)
    
    # Visualize if requested
    if args.visualize or args.save_plot:
        plot_path = None
        if args.save_plot:
            plot_path = os.path.join(args.output_dir, 'spherical_two_moons_visualization.png')
        
        print("Creating visualization...")
        visualize_spherical_dataset(x_data_spherical, y_data, save_path=plot_path)
    
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
    print(f"Note: x is 3D coordinates on unit sphere with shape (n_samples, 3)")
    print(f"Note: y is one-hot encoded with shape (n_samples, 2)")


if __name__ == "__main__":
    main()

