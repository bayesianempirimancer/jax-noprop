#!/usr/bin/env python3
"""Quick script to explore the two moons dataset."""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
with open('data/two_moons.pkl', 'rb') as f:
    data = pickle.load(f)

x_train, y_train = data['train']['x'], data['train']['y']
x_val, y_val = data['val']['x'], data['val']['y']

# Convert one-hot to integer labels
y_train_int = np.argmax(y_train, axis=1)
y_val_int = np.argmax(y_val, axis=1)

print("=" * 60)
print("Two Moons Dataset Info")
print("=" * 60)
print(f"Training set: {x_train.shape[0]} samples")
print(f"Validation set: {x_val.shape[0]} samples")
print(f"Input dimension: {x_train.shape[1]}")
print(f"Output dimension: {y_train.shape[1]}")
print(f"\nTraining class distribution:")
unique_train, counts_train = np.unique(y_train_int, return_counts=True)
for cls, count in zip(unique_train, counts_train):
    print(f"  Class {cls}: {count} samples ({count/len(y_train_int)*100:.1f}%)")
print(f"\nValidation class distribution:")
unique_val, counts_val = np.unique(y_val_int, return_counts=True)
for cls, count in zip(unique_val, counts_val):
    print(f"  Class {cls}: {count} samples ({count/len(y_val_int)*100:.1f}%)")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['red', 'blue']
labels = ['Class 0', 'Class 1']

for ax, x_data, y_data, title in zip(axes, [x_train, x_val], [y_train_int, y_val_int], 
                                     ['Training Set', 'Validation Set']):
    for i in range(2):
        mask = y_data == i
        ax.scatter(x_data[mask, 0], x_data[mask, 1], 
                  c=colors[i], label=labels[i], alpha=0.6, s=20)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

plt.tight_layout()
plt.savefig('data/two_moons_exploration.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to data/two_moons_exploration.png")
plt.show()


