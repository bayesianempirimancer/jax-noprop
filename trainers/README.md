# NoProp Trainers

Simple and concise trainers for NoProp-CT and NoProp-FM models that work with torchvision and Hugging Face datasets.

## Features

- **Easy to use**: Simple API for training NoProp models
- **Dataset compatibility**: Works with torchvision, Hugging Face, and numpy datasets
- **Automatic data handling**: Converts torch tensors and various data formats to JAX arrays
- **Built-in evaluation**: Includes training and validation metrics
- **Flexible configuration**: Customizable hyperparameters and model settings
- **Uses built-in train_step**: Leverages the optimized train_step methods from the models
- **Timing metrics**: Tracks training time per epoch and inference time per sample
- **Clean output**: Suppressed JAX warnings for cleaner training logs

## Quick Start

### Basic Usage

```python
from trainer import NoPropTrainer

# Train NoProp-CT
ct_trainer = NoPropTrainer(
    model_type="ct",
    input_dim=2,
    target_dim=2,
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=100
)

# Train the model
history = ct_trainer.fit(x_train, y_train, x_val, y_val)

# Make predictions
predictions = ct_trainer.predict(x_test)

# Evaluate
loss, accuracy = ct_trainer.evaluate(x_test, y_test)
```

### NoProp-FM Usage

```python
# Train NoProp-FM
fm_trainer = NoPropTrainer(
    model_type="fm",
    input_dim=2,
    target_dim=2,
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=100
)

# Train the model
history = fm_trainer.fit(x_train, y_train, x_val, y_val)
```

## Dataset Compatibility

### Torchvision Datasets

```python
from torchvision import datasets, transforms

# Load MNIST
dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor()
)

# Extract data and labels
x_data = dataset.data.numpy()  # Shape: (N, 28, 28)
y_data = dataset.targets.numpy()  # Shape: (N,)

# Train (data will be automatically flattened)
trainer = NoPropTrainer(
    model_type="ct",
    input_dim=28*28,  # 784 for MNIST
    target_dim=10,    # 10 classes
    batch_size=32,
    num_epochs=100
)
trainer.fit(x_data, y_data)
```

### Hugging Face Datasets

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Extract data (assuming you have embeddings)
x_data = np.random.randn(1000, 768)  # BERT embeddings
y_data = np.random.randint(0, 2, (1000,))  # Binary labels

# Train
trainer = NoPropTrainer(
    model_type="fm",
    input_dim=768,
    target_dim=2,
    batch_size=16,
    num_epochs=50
)
trainer.fit(x_data, y_data)
```

### Numpy Arrays

```python
import numpy as np

# Create data
x_data = np.random.randn(1000, 784)  # Flattened images
y_data = np.random.randint(0, 10, (1000,))  # Labels

# Train
trainer = NoPropTrainer(
    model_type="ct",
    input_dim=784,
    target_dim=10,
    batch_size=32,
    num_epochs=100
)
trainer.fit(x_data, y_data)
```

## Configuration Options

### NoPropTrainer Parameters

- `model_type`: Model type ("ct" for NoProp-CT, "fm" for NoProp-FM)
- `input_dim`: Input dimension
- `target_dim`: Target dimension (number of classes)
- `hidden_dims`: Hidden layer dimensions tuple (default: (64, 64, 64))
- `learning_rate`: Learning rate for optimizer (default: 1e-3)
- `batch_size`: Batch size for training (default: 32)
- `num_epochs`: Number of training epochs (default: 100)
- `noise_schedule`: Noise schedule for CT ("linear" or "cosine", default: "linear")
- `sigma_t`: Flow matching noise level for FM (default: 0.05)
- `integration_method`: ODE integration method (default: "euler")
- `num_steps`: Number of integration steps (default: 10)
- `random_seed`: Random seed for reproducibility (default: 42)

## Training History

The `fit()` method returns a dictionary with training history and timing metrics:

```python
history = trainer.fit(x_train, y_train, x_val, y_val)

# Access training metrics
train_losses = history['train_losses']
val_losses = history['val_losses']
train_accs = history['train_accs']
val_accs = history['val_accs']
epoch_times = history['epoch_times']
inference_times = history['inference_times']
params = history['params']

# Timing analysis
avg_epoch_time = np.mean(epoch_times)
total_training_time = np.sum(epoch_times)
inference_time_per_sample = inference_times[-1]  # Last measured time
```

## Examples

### Two Moons Dataset

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from trainer import NoPropTrainer

# Generate data
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train NoProp-CT
ct_trainer = NoPropTrainer(
    model_type="ct",
    input_dim=2,
    target_dim=2,
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=50
)

ct_history = ct_trainer.fit(x_train, y_train, x_val, y_val)

# Evaluate
ct_loss, ct_acc = ct_trainer.evaluate(x_val, y_val)
print(f"NoProp-CT: Loss = {ct_loss:.4f}, Accuracy = {ct_acc:.4f}")
print(f"Average epoch time: {np.mean(ct_history['epoch_times']):.3f}s")
print(f"Inference time per sample: {ct_history['inference_times'][-1]*1000:.2f}ms")
```

### Image Classification

```python
# For image data (e.g., CIFAR-10)
trainer = NoPropTrainer(
    model_type="ct",
    input_dim=32*32*3,  # CIFAR-10: 32x32x3
    target_dim=10,      # 10 classes
    hidden_dims=(128, 64, 32),
    learning_rate=1e-3,
    batch_size=64,
    num_epochs=100
)

# Data will be automatically flattened
trainer.fit(x_train, y_train, x_val, y_val)
```

## Files

- `trainer.py`: Main trainer implementation using the built-in train_step methods
- `test_trainer.py`: Comprehensive test script demonstrating usage
- `README.md`: This documentation

## Requirements

- JAX
- Flax
- Optax
- NumPy
- tqdm
- scikit-learn (for examples)

## Notes

- **Use `trainer.py`**: This is the working implementation that uses the built-in `train_step` methods from the models
- **Data handling**: Data is automatically converted to JAX arrays and flattened if needed
- **Performance**: The trainer uses the optimized JIT-compiled `train_step` methods from the models
- **Compatibility**: Works with torchvision, Hugging Face, and numpy datasets
- **Training loop**: Follows the same pattern as the working two_moons example

## Troubleshooting

- **JIT compilation errors**: Use `trainer.py` which avoids these issues
- **Import errors**: Make sure you're running from the correct directory with proper path setup
- **NaN values**: Check your data preprocessing and model configuration
- **Memory issues**: Reduce batch size or model dimensions