"""
Base training configuration for model-agnostic training.

This module defines the core training parameters that any trainer needs,
independent of the specific model architecture. This allows BaseETTrainer
to be truly model-agnostic.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class BaseTrainingConfig:
    """
    Base training configuration containing all training-specific parameters.
    
    This configuration is model-agnostic and contains only the parameters
    needed for training, optimization, and loss computation. Model-specific
    architecture parameters should be defined in separate config classes.
    """
    
    # === OPTIMIZER PARAMETERS ===
    learning_rate: float = 0.001
    optimizer: str = "adamw"  # 'adam', 'adamw', 'sgd', 'rmsprop'
    weight_decay: float = 1e-4
    
    # Adam/AdamW specific parameters
    beta1: float = 0.9  # First moment decay rate
    beta2: float = 0.999  # Second moment decay rate
    eps: float = 1e-8  # Numerical stability parameter
    
    # SGD/RMSprop specific parameters
    momentum: float = 0.9  # Momentum for SGD/RMSprop
    nesterov: bool = False  # Nesterov momentum for SGD
    rmsprop_decay: float = 0.9  # Decay rate for RMSprop
    
    # === LOSS FUNCTION PARAMETERS ===
    loss_function: str = "mse"  # 'mse', 'mae', 'huber', 'model_specific'
    loss_alpha: float = 1.0  # Weight for primary loss
    loss_beta: float = 0.0  # Weight for secondary loss (if applicable)
    huber_delta: float = 1.0  # Delta parameter for Huber loss
    
    # === REGULARIZATION PARAMETERS ===
    l1_reg_weight: float = 0.0  # L1 regularization weight
    # Note: dropout_rate is specified in the model config, not here
    # The trainer controls when dropout is active via the training flag
    
    # === TRAINING CONTROL PARAMETERS ===
    use_mini_batching: bool = True  # Whether to use mini-batching (False = process entire dataset at once)
    batch_size: int = 32
    random_batch_sampling: bool = True  # Whether to use random sampling (True) or sequential batching (False)
    dropout_epochs: Optional[int] = None  # Number of epochs to use dropout (None = use dropout for entire training)
    
    # === TRAINING MONITORING PARAMETERS ===
    eval_steps: int = 10  # Steps between evaluations
    save_steps: Optional[int] = None  # Steps between model saves (None = no intermediate saves)
    early_stopping_patience: int = 50  # Epochs to wait before early stopping
    early_stopping_min_delta: float = 1e-4  # Minimum change to qualify as improvement
    
    # === SYSTEM PARAMETERS ===
    random_seed: int = 42  # Random seed for reproducibility
    device: str = "auto"  # Device to use ('auto', 'cpu', 'gpu')
    compile_model: bool = True  # Whether to compile model with JAX
    memory_efficient: bool = False  # Whether to use memory-efficient training
    
    # === LOGGING PARAMETERS ===
    log_frequency: int = 10  # Steps between logging
    track_metrics: list = field(default_factory=lambda: ['loss', 'accuracy'])  # Metrics to track
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'momentum': self.momentum,
            'nesterov': self.nesterov,
            'rmsprop_decay': self.rmsprop_decay,
            'loss_function': self.loss_function,
            'loss_alpha': self.loss_alpha,
            'loss_beta': self.loss_beta,
            'huber_delta': self.huber_delta,
            'l1_reg_weight': self.l1_reg_weight,
            'batch_size': self.batch_size,
            'use_mini_batching': self.use_mini_batching,
            'random_batch_sampling': self.random_batch_sampling,
            'eval_steps': self.eval_steps,
            'save_steps': self.save_steps,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'random_seed': self.random_seed,
            'device': self.device,
            'compile_model': self.compile_model,
            'memory_efficient': self.memory_efficient,
            'log_frequency': self.log_frequency,
            'track_metrics': self.track_metrics,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseTrainingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get only the training-specific parameters."""
        return self.to_dict()
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.optimizer_type not in ['adam', 'adamw', 'sgd', 'rmsprop']:
            raise ValueError(f"optimizer_type must be one of ['adam', 'adamw', 'sgd', 'rmsprop'], got {self.optimizer_type}")
        if self.loss_function not in ['mse', 'mae', 'huber', 'model_specific']:
            raise ValueError(f"loss_function must be one of ['mse', 'mae', 'huber', 'model_specific'], got {self.loss_function}")
        if self.beta1 <= 0 or self.beta1 >= 1:
            raise ValueError("beta1 must be in (0, 1)")
        if self.beta2 <= 0 or self.beta2 >= 1:
            raise ValueError("beta2 must be in (0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.momentum < 0 or self.momentum >= 1:
            raise ValueError("momentum must be in [0, 1)")
        if self.rmsprop_decay < 0 or self.rmsprop_decay >= 1:
            raise ValueError("rmsprop_decay must be in [0, 1)")
        if self.loss_alpha < 0:
            raise ValueError("loss_alpha must be non-negative")
        if self.loss_beta < 0:
            raise ValueError("loss_beta must be non-negative")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if self.l1_reg_weight < 0:
            raise ValueError("l1_reg_weight must be non-negative")
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.log_frequency <= 0:
            raise ValueError("log_frequency must be positive")
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"BaseTrainingConfig(optimizer={self.optimizer}, lr={self.learning_rate}, loss={self.loss_function})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"BaseTrainingConfig("
                f"learning_rate={self.learning_rate}, "
                f"optimizer='{self.optimizer}', "
                f"loss_function='{self.loss_function}', "
                f"batch_size={self.batch_size}, "
                f"use_mini_batching={self.use_mini_batching})")
    
    # Class attribute that can be overridden by subclasses
    training_switches = [
        'learning_rate', 'batch_size', 'weight_decay', 'beta1', 'beta2', 'eps',
        'loss_function', 'l1_reg_weight', 'use_mini_batching', 'random_batch_sampling',
        'eval_steps', 'save_steps', 'early_stopping_patience', 'early_stopping_min_delta',
        'log_frequency', 'random_seed', 'optimizer'
    ]
    
    @classmethod
    def create_from_args(cls, args) -> 'BaseTrainingConfig':
        """
        Create training configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            TrainingConfig instance
        """
        # Build training config kwargs - only include explicitly provided arguments
        training_kwargs = {}
        
        # Get the training switches from the class (or instance if needed)
        if hasattr(cls, 'training_switches'):
            training_attributes = cls.training_switches
        else:
            # Fallback to default list
            training_attributes = [
                'learning_rate', 'batch_size', 'weight_decay', 'beta1', 'beta2', 'eps',
                'loss_function', 'l1_reg_weight', 'use_mini_batching', 'random_batch_sampling',
                'eval_steps', 'save_steps', 'early_stopping_patience', 'early_stopping_min_delta',
                'log_frequency', 'random_seed', 'optimizer'
            ]
        
        for attribute in training_attributes:
            if hasattr(args, attribute) and getattr(args, attribute) is not None:
                training_kwargs[attribute] = getattr(args, attribute)
        
        return cls(**training_kwargs)


