"""
Utility functions for NoProp training and evaluation.

This module provides helper functions for creating training states,
managing training loops, and handling data loading.
"""

from typing import Any, Dict, Optional, Tuple, Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training import train_state


@struct.dataclass
class TrainState:
    """Training state for NoProp models."""
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState


def create_train_state(
    model: Any,
    params: Dict[str, Any],
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
    weight_decay: float = 1e-4
) -> TrainState:
    """Create a training state for NoProp models.
    
    Args:
        model: The NoProp model (NoPropDT, NoPropCT, or NoPropFM)
        params: Model parameters
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer type ("adam", "adamw", "sgd")
        weight_decay: Weight decay for regularization
        
    Returns:
        TrainState object
    """
    # Create optimizer
    if optimizer == "adam":
        tx = optax.adam(learning_rate, weight_decay=weight_decay)
    elif optimizer == "adamw":
        tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    elif optimizer == "sgd":
        tx = optax.sgd(learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Create training state
    state = TrainState(
        step=0,
        apply_fn=model,
        params=params,
        tx=tx,
        opt_state=tx.init(params)
    )
    
    return state


def train_step(
    state: TrainState,
    x: jnp.ndarray,
    target: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[TrainState, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Single training step for NoProp models.
    
    Args:
        state: Current training state
        x: Input data [batch_size, height, width, channels]
        target: Clean target [batch_size, num_classes]
        key: Random key
        
    Returns:
        Tuple of (updated_state, loss, metrics)
    """
    # Compute loss and gradients
    (loss, metrics), grads = jax.value_and_grad(
        state.apply_fn.train_step, has_aux=True
    )(state.params, x, target, key)
    
    # Update parameters
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    
    # Update state
    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state
    )
    
    return new_state, loss, metrics


def eval_step(
    state: TrainState,
    x: jnp.ndarray,
    target: jnp.ndarray,
    key: jax.random.PRNGKey,
    num_steps: Optional[int] = None
) -> Dict[str, jnp.ndarray]:
    """Single evaluation step for NoProp models.
    
    Args:
        state: Current training state
        x: Input data [batch_size, height, width, channels]
        target: Clean target [batch_size, num_classes]
        key: Random key
        num_steps: Number of steps for generation (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    return state.apply_fn.evaluate(
        state.params, x, target, key, num_steps
    )


def one_hot_encode(labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """Convert integer labels to one-hot encoding.
    
    Args:
        labels: Integer labels [batch_size]
        num_classes: Number of classes
        
    Returns:
        One-hot encoded labels [batch_size, num_classes]
    """
    return jax.nn.one_hot(labels, num_classes)


def normalize_images(images: jnp.ndarray) -> jnp.ndarray:
    """Normalize images to [-1, 1] range.
    
    Args:
        images: Images in [0, 255] range [batch_size, height, width, channels]
        
    Returns:
        Normalized images in [-1, 1] range
    """
    return (images / 255.0) * 2.0 - 1.0


def denormalize_images(images: jnp.ndarray) -> jnp.ndarray:
    """Denormalize images from [-1, 1] to [0, 255] range.
    
    Args:
        images: Images in [-1, 1] range [batch_size, height, width, channels]
        
    Returns:
        Denormalized images in [0, 255] range
    """
    return ((images + 1.0) / 2.0) * 255.0


def compute_accuracy(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Compute classification accuracy.
    
    Args:
        predictions: Model predictions [batch_size, num_classes]
        targets: Target labels (one-hot or integer) [batch_size, num_classes] or [batch_size]
        
    Returns:
        Accuracy as a scalar
    """
    if targets.ndim > 1:
        # One-hot encoded targets
        pred_classes = jnp.argmax(predictions, axis=-1)
        target_classes = jnp.argmax(targets, axis=-1)
    else:
        # Integer targets
        pred_classes = jnp.argmax(predictions, axis=-1)
        target_classes = targets
    
    return jnp.mean(pred_classes == target_classes)


def create_data_iterators(
    train_data: Tuple[jnp.ndarray, jnp.ndarray],
    test_data: Tuple[jnp.ndarray, jnp.ndarray],
    batch_size: int,
    key: jax.random.PRNGKey
) -> Tuple[Any, Any]:
    """Create data iterators for training and testing.
    
    Args:
        train_data: Tuple of (train_images, train_labels)
        test_data: Tuple of (test_images, test_labels)
        batch_size: Batch size for training
        key: Random key for shuffling
        
    Returns:
        Tuple of (train_iterator, test_iterator)
    """
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    
    # Normalize images
    train_images = normalize_images(train_images)
    test_images = normalize_images(test_images)
    
    # Create training iterator
    num_train = train_images.shape[0]
    num_batches = num_train // batch_size
    
    def train_iterator():
        # Shuffle data
        key_shuffle = jax.random.fold_in(key, 0)
        perm = jax.random.permutation(key_shuffle, num_train)
        train_images_shuffled = train_images[perm]
        train_labels_shuffled = train_labels[perm]
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            yield (
                train_images_shuffled[start_idx:end_idx],
                train_labels_shuffled[start_idx:end_idx]
            )
    
    # Create test iterator
    num_test = test_images.shape[0]
    num_test_batches = num_test // batch_size
    
    def test_iterator():
        for i in range(num_test_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            yield (
                test_images[start_idx:end_idx],
                test_labels[start_idx:end_idx]
            )
    
    return train_iterator, test_iterator


def save_checkpoint(
    state: TrainState,
    path: str,
    step: Optional[int] = None
) -> None:
    """Save training checkpoint.
    
    Args:
        state: Training state to save
        path: Path to save checkpoint
        step: Step number (optional)
    """
    # This is a placeholder - in practice you'd use flax.training.checkpoints
    # or another checkpointing library
    pass


def load_checkpoint(
    path: str,
    state: TrainState
) -> TrainState:
    """Load training checkpoint.
    
    Args:
        path: Path to checkpoint
        state: Initial state to load into
        
    Returns:
        Loaded training state
    """
    # This is a placeholder - in practice you'd use flax.training.checkpoints
    # or another checkpointing library
    return state
