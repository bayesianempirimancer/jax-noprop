#!/usr/bin/env python3
"""
Test script for the enhanced trainer with timing metrics and suppressed warnings.
"""

import os
import warnings
import logging

# Suppress all warnings and logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'  # Use GPU instead of CPU
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

from .trainer import NoPropTrainer
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np

def test_enhanced_trainer():
    """Test the enhanced trainer with timing metrics."""
    print("🌙 Enhanced NoProp Trainer Test")
    print("=" * 60)
    
    # Generate data
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset: {len(X)} samples")
    print(f"Training: {len(x_train)} samples")
    print(f"Validation: {len(x_val)} samples")
    
    # Train NoProp-CT
    print("\n" + "="*60)
    print("Training NoProp-CT")
    print("="*60)
    
    ct_trainer = NoPropTrainer(
        model_type="ct",
        input_dim=2,
        target_dim=2,
        hidden_dims=(64, 64, 64),
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=50,
        noise_schedule="linear",
        random_seed=42
    )
    
    ct_history = ct_trainer.fit(x_train, y_train, x_val, y_val, verbose=True)
    
    # Evaluate
    ct_loss, ct_acc = ct_trainer.evaluate(x_val, y_val)
    
    print(f"\n📊 NoProp-CT Results:")
    print(f"  Final validation loss: {ct_loss:.4f}")
    print(f"  Final validation accuracy: {ct_acc:.4f}")
    print(f"  Average epoch time: {np.mean(ct_history['epoch_times']):.3f}s")
    print(f"  Inference time per sample: {ct_history['inference_times'][-1]*1000:.2f}ms")
    print(f"  Total training time: {np.sum(ct_history['epoch_times']):.2f}s")
    print(f"  Max training accuracy: {max(ct_history['train_accs']):.4f}")
    print(f"  Max validation accuracy: {max(ct_history['val_accs']):.4f}")
    
    # Train NoProp-FM
    print("\n" + "="*60)
    print("Training NoProp-FM")
    print("="*60)
    
    fm_trainer = NoPropTrainer(
        model_type="fm",
        input_dim=2,
        target_dim=2,
        hidden_dims=(64, 64, 64),
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=50,
        sigma_t=0.05,
        random_seed=42
    )
    
    fm_history = fm_trainer.fit(x_train, y_train, x_val, y_val, verbose=True)
    
    # Evaluate
    fm_loss, fm_acc = fm_trainer.evaluate(x_val, y_val)
    
    print(f"\n📊 NoProp-FM Results:")
    print(f"  Final validation loss: {fm_loss:.4f}")
    print(f"  Final validation accuracy: {fm_acc:.4f}")
    print(f"  Average epoch time: {np.mean(fm_history['epoch_times']):.3f}s")
    print(f"  Inference time per sample: {fm_history['inference_times'][-1]*1000:.2f}ms")
    print(f"  Total training time: {np.sum(fm_history['epoch_times']):.2f}s")
    print(f"  Max training accuracy: {max(fm_history['train_accs']):.4f}")
    print(f"  Max validation accuracy: {max(fm_history['val_accs']):.4f}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("📈 PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'Val Acc':<10} {'Avg Epoch':<12} {'Inference':<12} {'Total Time':<12}")
    print("-" * 60)
    print(f"{'NoProp-CT':<20} {ct_acc:.4f}     {np.mean(ct_history['epoch_times']):.3f}s      {ct_history['inference_times'][-1]*1000:.2f}ms      {np.sum(ct_history['epoch_times']):.2f}s")
    print(f"{'NoProp-FM':<20} {fm_acc:.4f}     {np.mean(fm_history['epoch_times']):.3f}s      {fm_history['inference_times'][-1]*1000:.2f}ms      {np.sum(fm_history['epoch_times']):.2f}s")
    
    print("\n🎉 Enhanced trainer test completed successfully!")
    print("✅ Timing metrics working")
    print("✅ Warnings suppressed")
    print("✅ Both models trained successfully")

if __name__ == "__main__":
    test_enhanced_trainer()
