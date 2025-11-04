#!/usr/bin/env python3
"""
Test script to verify preprocessing and inversion routines with raw data.
"""

import numpy as np
from preprocess_stock_data import preprocess_stock_data, invert_preprocessing, convert_log_normalized_to_original


def test_preprocessing_with_raw_data():
    """Test preprocessing with raw price/volume data."""
    print("=" * 60)
    print("Testing Preprocessing with Raw Data")
    print("=" * 60)
    
    # Create synthetic RAW data (prices, volumes, bid, ask)
    n_samples = 10
    y_seq_len = 48
    
    np.random.seed(42)
    
    # Generate synthetic previous closes and avg volumes
    previous_closes = 100.0 + 20.0 * np.random.randn(n_samples)
    previous_closes = np.maximum(previous_closes, 50.0)  # Ensure positive
    
    previous_avg_volumes = 1000000.0 + 500000.0 * np.random.randn(n_samples)
    previous_avg_volumes = np.maximum(previous_avg_volumes, 100000.0)  # Ensure positive
    
    # Generate synthetic RAW sequences (actual prices and volumes)
    x_sequences = []
    for i in range(n_samples):
        seq_len = np.random.randint(10, 20)
        prev_close = previous_closes[i]
        prev_avg_vol = previous_avg_volumes[i]
        
        x_seq = np.zeros((seq_len, 4))
        # Generate prices around previous close (±2%)
        x_seq[:, 0] = prev_close * (1.0 + 0.02 * np.random.randn(seq_len))
        x_seq[:, 0] = np.maximum(x_seq[:, 0], prev_close * 0.95)  # Ensure positive
        
        # Generate volumes around previous avg volume (±20%)
        x_seq[:, 1] = prev_avg_vol * (1.0 + 0.2 * np.random.randn(seq_len))
        x_seq[:, 1] = np.maximum(x_seq[:, 1], 1000.0)  # Ensure positive
        
        # Bid and ask close to price
        x_seq[:, 2] = x_seq[:, 0] * 0.999  # Bid slightly below price
        x_seq[:, 3] = x_seq[:, 0] * 1.001  # Ask slightly above price
        
        x_sequences.append(x_seq)
    
    # For y sequences
    y_sequences = np.zeros((n_samples, y_seq_len, 4))
    for i in range(n_samples):
        prev_close = previous_closes[i]
        prev_avg_vol = previous_avg_volumes[i]
        
        # Generate prices around previous close
        y_sequences[i, :, 0] = prev_close * (1.0 + 0.02 * np.random.randn(y_seq_len))
        y_sequences[i, :, 0] = np.maximum(y_sequences[i, :, 0], prev_close * 0.95)
        
        # Generate volumes
        y_sequences[i, :, 1] = prev_avg_vol * (1.0 + 0.2 * np.random.randn(y_seq_len))
        y_sequences[i, :, 1] = np.maximum(y_sequences[i, :, 1], 1000.0)
        
        # Bid and ask
        y_sequences[i, :, 2] = y_sequences[i, :, 0] * 0.999
        y_sequences[i, :, 3] = y_sequences[i, :, 0] * 1.001
    
    print(f"\nOriginal raw data:")
    print(f"  x sequences: {len(x_sequences)} sequences (variable length)")
    print(f"  y sequences: {y_sequences.shape}")
    print(f"  Sample y[0, 0, :] (raw) = {y_sequences[0, 0, :]}")
    print(f"    Price: {y_sequences[0, 0, 0]:.2f}, Volume: {y_sequences[0, 0, 1]:.2f}")
    
    # Step 1: Apply preprocessing (raw -> standardized)
    print(f"\n{'='*60}")
    print("Step 1: Applying preprocessing (raw -> standardized)...")
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
    
    # Step 2: Invert preprocessing (standardized -> log-normalized)
    print(f"\n{'='*60}")
    print("Step 2: Inverting preprocessing (standardized -> log-normalized)...")
    print(f"{'='*60}")
    x_log_normalized, y_log_normalized = invert_preprocessing(preprocessed_x, preprocessed_y, params)
    
    print(f"  Log-normalized x sequences: {len(x_log_normalized)} sequences")
    print(f"  Log-normalized y sequences: {y_log_normalized.shape}")
    print(f"  Sample y_log_normalized[0, 0, :] = {y_log_normalized[0, 0, :]}")
    
    # Step 3: Convert log-normalized to original domain
    print(f"\n{'='*60}")
    print("Step 3: Converting to original domain (log-normalized -> raw)...")
    print(f"{'='*60}")
    x_original, y_original = convert_log_normalized_to_original(x_log_normalized, y_log_normalized, params)
    
    print(f"  Original x sequences: {len(x_original)} sequences")
    print(f"  Original y sequences: {y_original.shape}")
    print(f"  Sample y_original[0, 0, :] = {y_original[0, 0, :]}")
    print(f"    Price: {y_original[0, 0, 0]:.2f}, Volume: {y_original[0, 0, 1]:.2f}")
    
    # Step 4: Verify full round-trip
    print(f"\n{'='*60}")
    print("Step 4: Verifying full round-trip (raw -> standardized -> log-normalized -> raw)...")
    print(f"{'='*60}")
    
    # Check x sequences
    x_max_diff = 0.0
    for i, (x_orig_raw, x_orig_recovered) in enumerate(zip(x_sequences, x_original)):
        diff = np.abs(x_orig_raw - x_orig_recovered).max()
        x_max_diff = max(x_max_diff, diff)
        if diff > 1e-4:
            print(f"  ⚠️  x[{i}] max difference: {diff:.2e}")
            print(f"     Original[0, :] = {x_orig_raw[0, :]}")
            print(f"     Recovered[0, :] = {x_orig_recovered[0, :]}")
    
    # Check y sequences
    y_diff = np.abs(y_sequences - y_original)
    y_max_diff = y_diff.max()
    y_mean_diff = y_diff.mean()
    
    print(f"\n  y sequences:")
    print(f"    Max difference: {y_max_diff:.2e}")
    print(f"    Mean difference: {y_mean_diff:.2e}")
    
    # Check relative differences for prices (more meaningful)
    price_rel_diff = np.abs((y_sequences[:, :, 0] - y_original[:, :, 0]) / (y_sequences[:, :, 0] + 1e-8)).max()
    volume_rel_diff = np.abs((y_sequences[:, :, 1] - y_original[:, :, 1]) / (y_sequences[:, :, 1] + 1e-8)).max()
    
    print(f"    Price max relative difference: {price_rel_diff:.2e}")
    print(f"    Volume max relative difference: {volume_rel_diff:.2e}")
    
    if y_max_diff > 1e-2:
        print(f"  ⚠️  WARNING: Large absolute differences found!")
        print(f"     Sample differences[0, 0, :] = {y_diff[0, 0, :]}")
        print(f"     Original[0, 0, :] = {y_sequences[0, 0, :]}")
        print(f"     Recovered[0, 0, :] = {y_original[0, 0, :]}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  x sequences max difference: {x_max_diff:.2e}")
    print(f"  y sequences max difference: {y_max_diff:.2e}")
    print(f"  y sequences mean difference: {y_mean_diff:.2e}")
    print(f"  Price max relative difference: {price_rel_diff:.2e}")
    print(f"  Volume max relative difference: {volume_rel_diff:.2e}")
    
    # Tolerances
    abs_tolerance = 1e-2  # Allow for some floating point errors
    rel_tolerance = 1e-3  # Relative tolerance for prices/volumes
    
    if x_max_diff < abs_tolerance and y_max_diff < abs_tolerance and price_rel_diff < rel_tolerance:
        print(f"\n  ✅ SUCCESS: All differences are within tolerance")
        print(f"     (absolute: {abs_tolerance:.2e}, relative: {rel_tolerance:.2e})")
        return True
    else:
        print(f"\n  ❌ FAILURE: Some differences exceed tolerance")
        return False


if __name__ == '__main__':
    success = test_preprocessing_with_raw_data()
    
    if success:
        print(f"\n{'='*60}")
        print("✅ ALL TESTS PASSED")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("❌ TESTS FAILED")
        print(f"{'='*60}")

