#!/usr/bin/env python3
"""
Test script to verify preprocessing and inversion routines are correct.
"""

import numpy as np
from preprocess_stock_data import preprocess_stock_data, invert_preprocessing, convert_log_normalized_to_original


def test_preprocessing_inversion():
    """Test that preprocessing and inversion are correct."""
    print("=" * 60)
    print("Testing Preprocessing and Inversion")
    print("=" * 60)
    
    # Create synthetic data
    n_samples = 10
    y_seq_len = 48
    
    # Create synthetic log-normalized data
    # Simulate log(price/prev_close), log(1+vol) - log(1+prev_avg_vol), etc.
    np.random.seed(42)
    
    # Generate synthetic previous closes and avg volumes
    previous_closes = 100.0 + 20.0 * np.random.randn(n_samples)
    previous_closes = np.maximum(previous_closes, 50.0)  # Ensure positive
    
    previous_avg_volumes = 1000000.0 + 500000.0 * np.random.randn(n_samples)
    previous_avg_volumes = np.maximum(previous_avg_volumes, 100000.0)  # Ensure positive
    
    # Generate synthetic log-normalized sequences
    # For x sequences (variable length)
    x_sequences = []
    for i in range(n_samples):
        seq_len = np.random.randint(10, 20)
        x_seq = np.zeros((seq_len, 4))
        
        # Generate log-normalized values
        x_seq[:, 0] = 0.01 * np.random.randn(seq_len)  # log(price/prev_close)
        x_seq[:, 1] = 0.1 * np.random.randn(seq_len)   # log(1+vol) - log(1+prev_avg_vol)
        x_seq[:, 2] = 0.01 * np.random.randn(seq_len)  # log(bid/prev_close)
        x_seq[:, 3] = 0.01 * np.random.randn(seq_len)  # log(ask/prev_close)
        
        x_sequences.append(x_seq)
    
    # For y sequences
    y_sequences = np.zeros((n_samples, y_seq_len, 4))
    y_sequences[:, :, 0] = 0.01 * np.random.randn(n_samples, y_seq_len)  # log(price/prev_close)
    y_sequences[:, :, 1] = 0.1 * np.random.randn(n_samples, y_seq_len)   # log(1+vol) - log(1+prev_avg_vol)
    y_sequences[:, :, 2] = 0.01 * np.random.randn(n_samples, y_seq_len)  # log(bid/prev_close)
    y_sequences[:, :, 3] = 0.01 * np.random.randn(n_samples, y_seq_len)  # log(ask/prev_close)
    
    # Store original values
    x_original_log_norm = [x.copy() for x in x_sequences]
    y_original_log_norm = y_sequences.copy()
    
    print(f"\nOriginal log-normalized data:")
    print(f"  x sequences: {len(x_sequences)} sequences (variable length)")
    print(f"  y sequences: {y_sequences.shape}")
    print(f"  Sample y[0, 0, :] = {y_sequences[0, 0, :]}")
    
    # Step 1: Apply preprocessing
    print(f"\n{'='*60}")
    print("Step 1: Applying preprocessing...")
    print(f"{'='*60}")
    preprocessed_x, preprocessed_y, params = preprocess_stock_data(
        x_sequences, y_sequences, previous_closes, previous_avg_volumes
    )
    
    print(f"  Preprocessed x sequences: {len(preprocessed_x)} sequences")
    print(f"  Preprocessed y sequences: {preprocessed_y.shape}")
    print(f"  Sample preprocessed_y[0, 0, :] = {preprocessed_y[0, 0, :]}")
    print(f"\n  Preprocessing parameters:")
    print(f"    std_log_price: {params['std_log_price']:.6f}")
    print(f"    std_log_volume_diff: {params['std_log_volume_diff']:.6f}")
    print(f"    volume_scale_factor: {params['volume_scale_factor']:.6f}")
    
    # Step 2: Invert preprocessing
    print(f"\n{'='*60}")
    print("Step 2: Inverting preprocessing...")
    print(f"{'='*60}")
    x_inverted, y_inverted = invert_preprocessing(preprocessed_x, preprocessed_y, params)
    
    print(f"  Inverted x sequences: {len(x_inverted)} sequences")
    print(f"  Inverted y sequences: {y_inverted.shape}")
    print(f"  Sample inverted_y[0, 0, :] = {y_inverted[0, 0, :]}")
    
    # Step 3: Verify inversion
    print(f"\n{'='*60}")
    print("Step 3: Verifying inversion...")
    print(f"{'='*60}")
    
    # Check x sequences
    x_max_diff = 0.0
    for i, (x_orig, x_inv) in enumerate(zip(x_original_log_norm, x_inverted)):
        diff = np.abs(x_orig - x_inv).max()
        x_max_diff = max(x_max_diff, diff)
        if diff > 1e-5:
            print(f"  ⚠️  x[{i}] max difference: {diff:.2e}")
            print(f"     Original[0, :] = {x_orig[0, :]}")
            print(f"     Inverted[0, :]  = {x_inv[0, :]}")
    
    # Check y sequences
    y_diff = np.abs(y_original_log_norm - y_inverted)
    y_max_diff = y_diff.max()
    y_mean_diff = y_diff.mean()
    
    print(f"\n  y sequences:")
    print(f"    Max difference: {y_max_diff:.2e}")
    print(f"    Mean difference: {y_mean_diff:.2e}")
    
    if y_max_diff > 1e-5:
        print(f"  ⚠️  WARNING: Large differences found!")
        print(f"     Sample differences[0, 0, :] = {y_diff[0, 0, :]}")
        print(f"     Original[0, 0, :] = {y_original_log_norm[0, 0, :]}")
        print(f"     Inverted[0, 0, :] = {y_inverted[0, 0, :]}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  x sequences max difference: {x_max_diff:.2e}")
    print(f"  y sequences max difference: {y_max_diff:.2e}")
    print(f"  y sequences mean difference: {y_mean_diff:.2e}")
    
    tolerance = 1e-5
    if x_max_diff < tolerance and y_max_diff < tolerance:
        print(f"\n  ✅ SUCCESS: All differences are below tolerance ({tolerance:.2e})")
        return True
    else:
        print(f"\n  ❌ FAILURE: Some differences exceed tolerance ({tolerance:.2e})")
        return False


def test_full_pipeline():
    """Test the full pipeline from log-normalized to original domain."""
    print(f"\n{'='*60}")
    print("Testing Full Pipeline (log-normalized -> original)")
    print(f"{'='*60}")
    
    # Create synthetic log-normalized data
    n_samples = 5
    y_seq_len = 48
    
    np.random.seed(42)
    
    previous_closes = np.array([100.0, 150.0, 200.0, 120.0, 180.0])
    previous_avg_volumes = np.array([1000000.0, 2000000.0, 1500000.0, 800000.0, 2500000.0])
    
    # Create log-normalized sequences
    x_sequences = []
    for i in range(n_samples):
        seq_len = 15
        x_seq = np.zeros((seq_len, 4))
        x_seq[:, 0] = 0.01 * np.random.randn(seq_len)  # log(price/prev_close)
        x_seq[:, 1] = 0.1 * np.random.randn(seq_len)   # log(1+vol) - log(1+prev_avg_vol)
        x_seq[:, 2] = 0.01 * np.random.randn(seq_len)  # log(bid/prev_close)
        x_seq[:, 3] = 0.01 * np.random.randn(seq_len)  # log(ask/prev_close)
        x_sequences.append(x_seq)
    
    y_sequences = np.zeros((n_samples, y_seq_len, 4))
    y_sequences[:, :, 0] = 0.01 * np.random.randn(n_samples, y_seq_len)
    y_sequences[:, :, 1] = 0.1 * np.random.randn(n_samples, y_seq_len)
    y_sequences[:, :, 2] = 0.01 * np.random.randn(n_samples, y_seq_len)
    y_sequences[:, :, 3] = 0.01 * np.random.randn(n_samples, y_seq_len)
    
    print(f"\n  Log-normalized data:")
    print(f"    Sample y[0, 0, 0] (log price) = {y_sequences[0, 0, 0]:.6f}")
    print(f"    Sample y[0, 0, 1] (log vol diff) = {y_sequences[0, 0, 1]:.6f}")
    
    # Convert to original domain
    params = {
        'previous_closes': previous_closes,
        'previous_avg_volumes': previous_avg_volumes,
    }
    
    x_original, y_original = convert_log_normalized_to_original(x_sequences, y_sequences, params)
    
    print(f"\n  Original domain data:")
    print(f"    Sample y[0, 0, 0] (price) = {y_original[0, 0, 0]:.2f}")
    print(f"    Sample y[0, 0, 1] (volume) = {y_original[0, 0, 1]:.2f}")
    
    # Verify conversion
    # Price: price = prev_close * exp(log_normalized)
    expected_price = previous_closes[0] * np.exp(y_sequences[0, 0, 0])
    actual_price = y_original[0, 0, 0]
    price_diff = abs(expected_price - actual_price)
    
    # Volume: vol = (1+prev_avg_vol) * exp(log_normalized) - 1
    expected_volume = (1.0 + previous_avg_volumes[0]) * np.exp(y_sequences[0, 0, 1]) - 1.0
    actual_volume = y_original[0, 0, 1]
    volume_diff = abs(expected_volume - actual_volume)
    
    print(f"\n  Verification:")
    print(f"    Expected price: {expected_price:.2f}, Actual: {actual_price:.2f}, Diff: {price_diff:.2e}")
    print(f"    Expected volume: {expected_volume:.2f}, Actual: {actual_volume:.2f}, Diff: {volume_diff:.2e}")
    
    tolerance = 1e-5
    if price_diff < tolerance and volume_diff < tolerance:
        print(f"\n  ✅ SUCCESS: Conversions are correct")
        return True
    else:
        print(f"\n  ❌ FAILURE: Conversion errors exceed tolerance")
        return False


if __name__ == '__main__':
    success1 = test_preprocessing_inversion()
    success2 = test_full_pipeline()
    
    if success1 and success2:
        print(f"\n{'='*60}")
        print("✅ ALL TESTS PASSED")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("❌ SOME TESTS FAILED")
        print(f"{'='*60}")

