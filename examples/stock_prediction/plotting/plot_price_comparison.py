"""
Plot actual vs predicted prices during trading hours.
"""
import os
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_price_comparison(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    data_path: str,
    output_dir: str,
    num_samples: int = 8,
    start_time: str = "10:30",
    end_time: str = "14:30"
):
    """
    Plot actual vs predicted prices during trading hours (10:30 AM - 2:30 PM).
    
    Args:
        y_real: Real sequences [batch, seq_len, embed_dim] (20D)
        y_pred: Predicted sequences [batch, seq_len, embed_dim] (20D)
        data_path: Path to processed data file containing projection matrix
        output_dir: Directory to save the plot
        num_samples: Number of samples to plot
        start_time: Start time in format "HH:MM"
        end_time: End time in format "HH:MM"
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data file to get preprocessing parameters
    # Note: Since CRN handles embeddings internally, outputs are already 2D
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check if data has projection info (old format) or is 2D (new format)
    has_projection = 'projection' in data and data['projection'] is not None
    if has_projection:
        projection_matrix = data['projection']['matrix']  # [20, 4]
        input_dim = data['projection']['input_dim']  # 2 (price, volume)
    else:
        # New format: data is already 2D, no projection needed
        # y_real and y_pred are already in 2D format (price, volume)
        input_dim = 2
        projection_matrix = None
    
    # Get RoPE parameters
    rope_base = data.get('rope', {}).get('base', 10000.0)
    
    # Get sequence dimensions first
    batch_size, seq_len, feature_dim = y_real.shape
    
    # Check if outputs are already 2D (new format) or need projection (old format)
    if has_projection and feature_dim != input_dim:
        # Old format: outputs are in embed_dim, need to remove embeddings and project
        embed_dim = feature_dim
        
        # Get day-of-week embeddings (if available)
        day_embeddings_dict = data.get('day_of_week', {}).get('embeddings', {})
        day_embeddings = {int(k): np.array(v) for k, v in day_embeddings_dict.items()}
        
        # Get days_of_week labels for validation set
        all_days_of_week = data.get('day_of_week', {}).get('days_of_week_val', None)
        if all_days_of_week is None:
            all_days_of_week = np.zeros(batch_size, dtype=np.int32)
        if len(all_days_of_week) < batch_size:
            all_days_of_week = np.concatenate([
                all_days_of_week,
                np.zeros(batch_size - len(all_days_of_week), dtype=np.int32)
            ])
        sample_days_of_week = all_days_of_week[:batch_size]
        
        # Remove RoPE and day-of-week embeddings, then project back to 2D
        from src.embeddings.positional_encoding import rotary_positional_encoding
        
        position_offset = -(seq_len - 1)
        max_pos_needed = abs(position_offset) + seq_len
        rope_encoding_full = np.array(rotary_positional_encoding(max_pos_needed, embed_dim, base=rope_base))
        
        start_idx = abs(position_offset)
        end_idx = start_idx + seq_len
        extracted = rope_encoding_full[start_idx:end_idx]
        rope_encoding = np.flip(extracted, axis=0)
        rope_encoding[:, 0::2] = -rope_encoding[:, 0::2]
        
        norms = np.linalg.norm(rope_encoding, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        rope_encoding = rope_encoding / norms
        
        y_real_no_rope = y_real - rope_encoding[None, :, :]
        y_pred_no_rope = y_pred - rope_encoding[None, :, :]
        
        sample_embeddings = np.array([day_embeddings.get(day, day_embeddings.get(0, np.zeros(embed_dim))) 
                                     for day in sample_days_of_week])
        y_real_no_pos = y_real_no_rope - sample_embeddings[:, None, :]
        y_pred_no_pos = y_pred_no_rope - sample_embeddings[:, None, :]
        
        proj_pinv = np.linalg.pinv(projection_matrix)
        y_real_4d = y_real_no_pos.reshape(-1, embed_dim) @ proj_pinv.T
        y_real_4d = y_real_4d.reshape(batch_size, seq_len, input_dim)
        
        y_pred_4d = y_pred_no_pos.reshape(-1, embed_dim) @ proj_pinv.T
        y_pred_4d = y_pred_4d.reshape(batch_size, seq_len, input_dim)
    else:
        # New format: outputs are already 2D (price, volume)
        # No need to remove embeddings or project - CRN handles this internally
        y_real_4d = y_real  # Already 2D
        y_pred_4d = y_pred  # Already 2D
    
    # Debug: Check 2D values
    print(f"  DEBUG: Data shape (should be 2D): {y_real_4d.shape}")
    print(f"    y_real_4d[0, :, 0] (price) range: [{y_real_4d[0, :, 0].min():.6f}, {y_real_4d[0, :, 0].max():.6f}], std: {y_real_4d[0, :, 0].std():.6f}")
    if y_real_4d.shape[2] > 1:
        print(f"    y_real_4d[0, :, 1] (volume) range: [{y_real_4d[0, :, 1].min():.6f}, {y_real_4d[0, :, 1].max():.6f}], std: {y_real_4d[0, :, 1].std():.6f}")
    
    # Convert log-normalized prices back to original domain
    # Get previous closes for validation set
    metadata = data.get('metadata', {})
    previous_closes = metadata.get('previous_closes_val', None)
    if previous_closes is None:
        # Try old format
        previous_closes = data.get('previous_closes', {}).get('val', None)
    
    # Get standardization parameters from metadata (needed to reverse standardization)
    std_log_price = metadata.get('std_log_price', 1.0)
    std_log_volume_diff = metadata.get('std_log_volume_diff', 1.0)
    volume_scale_factor = metadata.get('volume_scale_factor', 0.05)
    
    if previous_closes is not None and len(previous_closes) >= batch_size:
        # Note: y_real should be the first batch_size samples from validation set
        # so we use previous_closes[:batch_size] to match
        sample_previous_closes = previous_closes[:batch_size]
        
        # Debug: Check if previous_closes are all the same
        if len(sample_previous_closes) > 1:
            if np.allclose(sample_previous_closes, sample_previous_closes[0]):
                print(f"  WARNING: All previous_closes in batch are the same: {sample_previous_closes[0]:.2f}")
            else:
                print(f"  DEBUG: previous_closes vary: min={sample_previous_closes.min():.2f}, max={sample_previous_closes.max():.2f}, std={sample_previous_closes.std():.2f}")
        
        # y_real_4d and y_pred_4d are currently preprocessed:
        # - Price: log10(price/prev_close) (no standardization)
        # Inverse: price = prev_close * 10^(log10_normalized)
        y_real_price = np.zeros((batch_size, seq_len))
        y_pred_price = np.zeros((batch_size, seq_len))
        
        for i in range(batch_size):
            prev_close = sample_previous_closes[i]
            # Price is already in log10 space (no standardization reversal needed)
            log_norm_real = y_real_4d[i, :, 0]  # log10(price / prev_close)
            log_norm_pred = y_pred_4d[i, :, 0]  # log10(price / prev_close)
            
            # Convert: price = prev_close * 10^(log10_norm)
            pow10_real = np.clip(np.power(10.0, log_norm_real), 1e-10, 1e10)
            pow10_pred = np.clip(np.power(10.0, log_norm_pred), 1e-10, 1e10)
            y_real_price[i, :] = prev_close * pow10_real
            y_pred_price[i, :] = prev_close * pow10_pred
            
            # Debug: Check if prices have variation (only for first sample)
            if i == 0 and batch_size > 0:
                price_std = y_real_price[i].std()
                price_range = y_real_price[i].max() - y_real_price[i].min()
                print(f"  DEBUG (sample {i}): Converted prices - range: [{y_real_price[i].min():.2f}, {y_real_price[i].max():.2f}], std: {price_std:.2f}, range: {price_range:.2f}")
                if price_std < 0.01:
                    print(f"    ⚠️  WARNING: Prices appear constant (std={price_std:.6f})")
                    print(f"       This is likely because log_norm values are very small (std={log_norm_real.std():.6f})")
                    print(f"       Small log_norm -> 10^(log_norm) ≈ 1 -> prices ≈ prev_close (constant)")
            
            # Ensure prices are positive (safety check)
            y_real_price[i, :] = np.maximum(y_real_price[i, :], 1e-6)
            y_pred_price[i, :] = np.maximum(y_pred_price[i, :], 1e-6)
    else:
        # If no previous closes available, plot log-normalized values
        y_real_price = y_real_4d[:, :, 0]  # [batch, seq_len]
        y_pred_price = y_pred_4d[:, :, 0]  # [batch, seq_len]
    
    # Create time labels (5-minute intervals from start_time to end_time)
    start_dt = datetime.strptime(start_time, "%H:%M")
    time_delta = timedelta(minutes=5)
    time_labels = [start_dt + i * time_delta for i in range(seq_len)]
    time_strs = [t.strftime("%H:%M") for t in time_labels]
    
    # Select samples to plot
    num_samples = min(num_samples, batch_size)
    n_cols = 2
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Actual vs Predicted Price ({start_time} - {end_time} Window)', 
                 fontsize=16, fontweight='bold')
    
    for idx in range(num_samples):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Plot actual and predicted
        ax.plot(range(seq_len), y_real_price[idx], 
               label='Actual', color='blue', linewidth=2, marker='o', markersize=4, alpha=0.7)
        ax.plot(range(seq_len), y_pred_price[idx], 
               label='Predicted', color='red', linewidth=2, marker='s', markersize=4, 
               linestyle='--', alpha=0.7)
        
        # Formatting
        ax.set_title(f'Sample {idx+1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (5-min intervals)', fontsize=10)
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set x-axis ticks (every 6 timesteps = 30 minutes)
        tick_step = max(1, seq_len // 8)
        ax.set_xticks(range(0, seq_len, tick_step))
        ax.set_xticklabels([time_strs[i] for i in range(0, seq_len, tick_step)], 
                          rotation=45, ha='right', fontsize=8)
    
    # Hide unused subplots
    for idx in range(num_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, f'price_comparison_{start_time.replace(":", "_")}_to_{end_time.replace(":", "_")}.png')
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved price comparison plot to {plot_path}")

