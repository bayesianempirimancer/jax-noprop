"""
Utility functions for training scripts to ensure consistent directory and file naming.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional
import pickle


def get_save_directory(save_dir: Optional[str], task: str, model_type: str, unconditional: bool = False) -> str:
    """
    Generate a consistent save directory path.
    
    Args:
        save_dir: User-specified save directory. If None, auto-generates one.
        task: Task identifier ('reg', 'gen', or 'seq')
        model_type: Model type ('flow_matching', 'diffusion', or 'ct')
        unconditional: Whether this is unconditional generation (only applies to 'gen' task)
        
    Returns:
        Path to save directory (as string)
    """
    if save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        # For unconditional generation, use '_uncond_gen' instead of '_gen'
        if task == 'gen' and unconditional:
            task_suffix = 'uncond_gen'
        else:
            task_suffix = task
        save_dir = f"artifacts/{model_type}_{task_suffix}/{timestamp}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir


def save_training_artifacts(
    save_dir: str,
    history: dict,
    trainer,
    config,
    verbose: bool = True
) -> None:
    """
    Save training artifacts (results, params, config) with consistent naming.
    
    Args:
        save_dir: Directory to save artifacts
        history: Training history dictionary
        trainer: Trainer instance with save_params method
        config: Config instance with save_yaml method
        verbose: Whether to print save messages
    """
    save_path = Path(save_dir)
    
    if verbose:
        print(f"Saving results to {save_dir}...")
    
    # Save training results
    with open(save_path / 'training_results.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # Save model parameters
    trainer.save_params(str(save_path / 'model_params.pkl'))
    
    # Save config
    config.save_yaml(save_path / 'config.yaml')
    
    if verbose:
        print(f"Config saved to {save_path / 'config.yaml'}")

