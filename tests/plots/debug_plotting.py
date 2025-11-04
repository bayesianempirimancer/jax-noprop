#!/usr/bin/env python3
"""Debug script to test the plotting conversion logic."""

import pickle
import numpy as np
from src.embeddings.positional_encoding import rotary_positional_encoding

# Load data
data_path = 'data/stock_sequences_projected_updated.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Get first sample from validation set
y_sample = data['val']['y'][0:1]  # Shape: (1, 48, 20)
batch_size, seq_len, embed_dim = y_sample.shape

print(f"Sample y shape: {y_sample.shape}")
print(f"y_sample stats: min={y_sample.min():.6f}, max={y_sample.max():.6f}, std={y_sample.std():.6f}")

# Get projection matrix
projection_matrix = data['projection']['matrix']  # [20, 4]
input_dim = data['projection']['input_dim']  # 4
rope_base = data.get('rope', {}).get('base', 10000.0)

# Get day-of-week embeddings
day_embeddings_dict = data.get('day_of_week', {}).get('embeddings', {})
day_embeddings = {int(k): np.array(v) for k, v in day_embeddings_dict.items()}
all_days_of_week = data.get('day_of_week', {}).get('days_of_week_val', None)
if all_days_of_week is None:
    all_days_of_week = np.zeros(batch_size, dtype=np.int32)
sample_days_of_week = all_days_of_week[:batch_size]

print(f"\nDay-of-week for first sample: {sample_days_of_week[0]}")

# Step 1: Remove RoPE
print(f"\nStep 1: Removing RoPE...")
position_offset = -(seq_len - 1)
max_pos_needed = abs(position_offset) + seq_len
rope_encoding_full = np.array(rotary_positional_encoding(max_pos_needed, embed_dim, base=rope_base))
start_idx = abs(position_offset)
end_idx = start_idx + seq_len
extracted = rope_encoding_full[start_idx:end_idx]
rope_encoding = np.flip(extracted, axis=0)
rope_encoding[:, 0::2] = -rope_encoding[:, 0::2]  # Flip sin components

y_no_rope = y_sample - rope_encoding[None, :, :]
print(f"  After RoPE removal: min={y_no_rope.min():.6f}, max={y_no_rope.max():.6f}, std={y_no_rope.std():.6f}")

# Step 2: Remove day-of-week embeddings
print(f"\nStep 2: Removing day-of-week embeddings...")
sample_embeddings = np.array([day_embeddings.get(day, day_embeddings.get(0, np.zeros(embed_dim))) 
                             for day in sample_days_of_week])
y_no_pos = y_no_rope - sample_embeddings[:, None, :]
print(f"  After day-of-week removal: min={y_no_pos.min():.6f}, max={y_no_pos.max():.6f}, std={y_no_pos.std():.6f}")

# Step 3: Inverse project to 4D
print(f"\nStep 3: Inverse projecting to 4D...")
proj_pinv = np.linalg.pinv(projection_matrix)
y_4d = y_no_pos.reshape(-1, embed_dim) @ proj_pinv.T
y_4d = y_4d.reshape(batch_size, seq_len, input_dim)

print(f"  y_4d shape: {y_4d.shape}")
print(f"  Dim0 (price) stats: min={y_4d[0, :, 0].min():.6f}, max={y_4d[0, :, 0].max():.6f}, mean={y_4d[0, :, 0].mean():.6f}, std={y_4d[0, :, 0].std():.6f}")
print(f"  Dim0 first 10 values: {y_4d[0, :10, 0]}")

# Step 4: Convert to original domain
print(f"\nStep 4: Converting to original domain...")
previous_closes = data.get('previous_closes', {}).get('val', None)
if previous_closes is not None:
    prev_close = previous_closes[0]
    print(f"  Previous close: {prev_close:.2f}")
    
    log_norm = y_4d[0, :, 0]
    print(f"  log_norm stats: min={log_norm.min():.6f}, max={log_norm.max():.6f}, mean={log_norm.mean():.6f}, std={log_norm.std():.6f}")
    print(f"  log_norm first 10 values: {log_norm[:10]}")
    
    exp_vals = np.exp(log_norm)
    print(f"  exp(log_norm) stats: min={exp_vals.min():.6f}, max={exp_vals.max():.6f}, mean={exp_vals.mean():.6f}, std={exp_vals.std():.6f}")
    
    prices = prev_close * exp_vals
    print(f"  Prices stats: min={prices.min():.2f}, max={prices.max():.2f}, mean={prices.mean():.2f}, std={prices.std():.2f}")
    print(f"  Prices first 10 values: {prices[:10]}")
    
    if prices.std() < 0.01:
        print(f"\n⚠️  PROBLEM: Prices are constant (std={prices.std():.6f})")
        print(f"   This means log_norm values are all very close to 0")
        print(f"   Check if positional encoding removal is working correctly")
    else:
        print(f"\n✓ Prices have variation (std={prices.std():.2f})")

