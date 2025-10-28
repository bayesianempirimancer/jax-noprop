"""
Training script for NoProp models (FM, CT, DF) with CRN architectures.

This script provides training for all training protocols (FM, CT, DF) using all model architectures
(potential_flow, geometric_flow, natural_flow, conditional_resnet_mlp).
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple

import jax
import jax.numpy as jnp

# Add the project root to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# from src.configs.base_trainer import BaseETTrainer
# from src.configs.base_training_config import BaseTrainingConfig
try:
    from .trainer import NoPropTrainer
    from .fm import NoPropFM, Config as FMConfig
    from .ct import NoPropCT, Config as CTConfig
    from .df import NoPropDF, Config as DFConfig
except ImportError:
    from trainer import NoPropTrainer
    from fm import NoPropFM, Config as FMConfig
    from ct import NoPropCT, Config as CTConfig
    from df import NoPropDF, Config as DFConfig


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common arguments for all model types."""
    
    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data file (.pkl)")
    
    # Training protocol selection
    parser.add_argument("--training-protocol", type=str, default="fm",
                       choices=["fm", "ct", "df"],
                       help="Training protocol: fm (flow matching), ct (continuous-time), df (diffusion) (default: fm)")
    
    # Model architecture selection
    parser.add_argument("--model", type=str, default="potential_flow",
                       choices=["potential_flow", "geometric_flow", "natural_flow", "convex_potential_flow", "conditional_resnet_mlp", "convex_conditional_resnet", "bilinear_conditional_resnet"],
                       help="Model architecture to use (default: potential_flow)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=400,
                       help="Number of training epochs (default: 400)")
    parser.add_argument("--dropout-epochs", type=int, default=300,
                       help="Number of epochs with dropout (default: 300)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size (default: 256)")
    parser.add_argument("--eval-steps", type=int, default=10,
                       help="Steps between detailed evaluation (default: 10)")
    
    # Model-specific arguments
    parser.add_argument("--loss-type", type=str, default="mse",
                       choices=["mse", "snr_weighted_mse"],
                       help="Loss function type (default: mse)")
    parser.add_argument("--noise-schedule", type=str, default="linear",
                       choices=["linear", "sigmoid", "cosine", "simple_learnable", "learnable"],
                       help="Noise schedule type (default: linear)")
    parser.add_argument("--sigma-t", type=float, default=0.1,
                       help="Noise level for FM/DF models (default: 0.1)")
    parser.add_argument("--reg-weight", type=float, default=0.0,
                       help="Regularization weight (default: 0.0)")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: auto-generated)")
    
    return parser


def create_model(training_protocol: str, model: str, config, z_shape: Tuple[int, ...], x_ndims: int = 1):
    """Create model with specified training protocol and architecture."""
    
    # For wrapper models, ensure the config has the right model type for the CRN backbone
    if model in ["potential_flow", "geometric_flow", "natural_flow"]:
        config.config_dict['model'] = "conditional_resnet_mlp"  # Use CRN MLP as backbone
        print(f"Using CRN MLP backbone for {model} wrapper")
    else:
        print(f"Using direct {model} architecture")
    
    # Create the appropriate model based on training protocol
    if training_protocol == "fm":
        return NoPropFM(config=config, z_shape=z_shape, x_ndims=x_ndims, model=model)
    elif training_protocol == "ct":
        return NoPropCT(config=config, z_shape=z_shape, x_ndims=x_ndims, model=model)
    elif training_protocol == "df":
        return NoPropDF(config=config, z_shape=z_shape, x_ndims=x_ndims, model=model)
    else:
        raise ValueError(f"Unsupported training protocol: {training_protocol}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="NoProp Model Training")
    parser = add_common_arguments(parser)
    args = parser.parse_args()
    
    print("NoProp Model Training Script (flow_models - All Training Protocols)")
    print("="*60)
    print(f"Training Protocol: {args.training_protocol.upper()}")
    print(f"Model Architecture: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Dropout epochs: {args.dropout_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    
    # Load training data
    print("\n1. Loading training data...")
    from .trainer import load_training_data
    data, x_dim, y_dim = load_training_data(args.data)
    
    # Standardize targets (y) to help with training
    print("Standardizing targets...")
    
    # Standardize targets (y)
    train_y_mean = jnp.mean(data['train']['y'], axis=0)
    train_y_std = jnp.std(data['train']['y'], axis=0)
    
    # Apply target standardization to all splits
    data['train']['y'] = (data['train']['y'] - train_y_mean) / train_y_std
    data['val']['y'] = (data['val']['y'] - train_y_mean) / train_y_std
    if 'test' in data:
        data['test']['y'] = (data['test']['y'] - train_y_mean) / train_y_std
    
    print(f"Target standardization: mean={train_y_mean}, std={train_y_std}")
    
    # Store standardization parameters for later use
    standardization_params = {
        'y_mean': train_y_mean,
        'y_std': train_y_std
    }
    
    # Create model configuration
    print("\n2. Creating model configuration...")
    config_kwargs = {
        'loss_type': args.loss_type,
        'noise_schedule': args.noise_schedule,
        'sigma_t': args.sigma_t,
        'reg_weight': args.reg_weight,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
    }
    
    # Create config based on training protocol
    if args.training_protocol == "fm":
        config = FMConfig()
    elif args.training_protocol == "ct":
        config = CTConfig()
    elif args.training_protocol == "df":
        config = DFConfig()
    else:
        raise ValueError(f"Unsupported training protocol: {args.training_protocol}")
    
    print(f"Configuration: {config}")
    
    # Set up output directory
    print("\n3. Setting up output directory...")
    if args.output_dir is None:
        # Extract dataset name from data file path
        data_file = Path(args.data)
        dataset_name = data_file.stem  # Get filename without extension
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"artifacts/{args.training_protocol}_{args.model}_{dataset_name}_{timestamp}"
    else:
        output_dir = args.output_dir
    print(f"Output directory: {output_dir}")
    
    # Create model
    print("\n4. Creating model...")
    z_shape = (x_dim,)
    x_ndims = 1  # x_ndims = 1 for the cases we're considering
    
    model = create_model(args.training_protocol, args.model, config, z_shape, x_ndims)
    print(f"Model created: {type(model).__name__}")
    
    # Create trainer
    print("\n5. Creating trainer...")
    trainer = NoPropTrainer(model)
    
    # Train model
    print("\n6. Training model...")
    results = trainer.train(
        train_x=data['train']['x'],
        train_y=data['train']['y'],
        val_x=data['val']['x'],
        val_y=data['val']['y'],
        num_epochs=args.epochs,
        test_x=data.get('test', {}).get('x'),
        test_y=data.get('test', {}).get('y'),
        dropout_epochs=args.dropout_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        output_dir=output_dir
    )
    
    # Add standardization parameters to results
    results['standardization_params'] = standardization_params
    
    # Save results and generate plots
    print("\n7. Saving results and generating plots...")
    trainer.save_results(results, output_dir)
    
    print(f"\nSUCCESS! {args.training_protocol.upper()} training with {args.model} completed")
    print(f"Results saved to: {output_dir}")
    print(f"Final validation MSE: {results['final_val_mse']:.6f}")
    if results['final_test_mse'] is not None:
        print(f"Final test MSE: {results['final_test_mse']:.6f}")


if __name__ == "__main__":
    main()
