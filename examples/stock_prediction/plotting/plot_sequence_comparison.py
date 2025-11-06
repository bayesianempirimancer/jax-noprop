"""
Plot generated sequences vs real sequences.
"""
import os
import pickle
import numpy as np
from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt


def plot_sequence_comparison(
    y_real: np.ndarray,
    y_gen: np.ndarray,
    output_dir: str,
    data_path: Optional[str] = None
):
    """
    Plot generated sequences vs real sequences.
    
    If data_path is provided, removes positional embeddings and projects back to 2D
    for meaningful visualization (shows actual price/volume features).
    Otherwise, plots the raw embedding dimensions (may show positional embedding patterns).
    
    Args:
        y_real: Real sequences [batch, seq_len, feature_dim]
        y_gen: Generated sequences [batch, seq_len, feature_dim]
        output_dir: Directory to save the plot
        data_path: Optional path to data file for preprocessing reversal
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    seq_len = y_real.shape[1]
    feature_dim = y_real.shape[2]
    batch_size = y_real.shape[0]
    
    # If data_path is provided, handle 2D data or remove embeddings (old format)
    if data_path is not None:
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check if data has projection info (old format) or is 2D (new format)
            has_projection = 'projection' in data and data['projection'] is not None
            if has_projection:
                projection_matrix = data['projection']['matrix']  # [20, 4]
                input_dim = data['projection']['input_dim']  # 2 (price, volume)
                rope_base = data.get('rope', {}).get('base', 10000.0)
            else:
                # New format: data is already 2D, no projection needed
                input_dim = 2
                projection_matrix = None
                rope_base = 10000.0
            
            # Check if outputs are already 2D (new format) or need projection (old format)
            feature_dim = y_real.shape[2]
            if has_projection and feature_dim != input_dim:
                # Old format: outputs are in embed_dim, need to remove embeddings and project
                embed_dim = feature_dim
                
                # Get day-of-week embeddings and labels
                day_embeddings_dict = data.get('day_of_week', {}).get('embeddings', {})
                day_embeddings = {int(k): np.array(v) for k, v in day_embeddings_dict.items()}
                all_days_of_week = data.get('day_of_week', {}).get('days_of_week_val', None)
                if all_days_of_week is None:
                    all_days_of_week = np.zeros(batch_size, dtype=np.int32)
                if len(all_days_of_week) < batch_size:
                    all_days_of_week = np.concatenate([
                        all_days_of_week,
                        np.zeros(batch_size - len(all_days_of_week), dtype=np.int32)
                    ])
                sample_days_of_week = all_days_of_week[:batch_size]
                
                # Remove RoPE positional encodings (y sequences use shifted positions)
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
                y_gen_no_rope = y_gen - rope_encoding[None, :, :]
                
                sample_embeddings = np.array([day_embeddings.get(day, day_embeddings.get(0, np.zeros(embed_dim))) 
                                            for day in sample_days_of_week])
                y_real_no_pos = y_real_no_rope - sample_embeddings[:, None, :]
                y_gen_no_pos = y_gen_no_rope - sample_embeddings[:, None, :]
                
                proj_pinv = np.linalg.pinv(projection_matrix)
                y_real_4d = y_real_no_pos.reshape(-1, embed_dim) @ proj_pinv.T
                y_real_4d = y_real_4d.reshape(batch_size, seq_len, input_dim)
                y_gen_4d = y_gen_no_pos.reshape(-1, embed_dim) @ proj_pinv.T
                y_gen_4d = y_gen_4d.reshape(batch_size, seq_len, input_dim)
            else:
                # New format: outputs are already 2D (price, volume)
                # No need to remove embeddings or project - CRN handles this internally
                y_real_4d = y_real  # Already 2D
                y_gen_4d = y_gen  # Already 2D
            
            # Convert log-normalized values back to original domain
            # Get previous closes and avg volumes for validation set
            metadata = data.get('metadata', {})
            previous_closes = metadata.get('previous_closes_val', None)
            if previous_closes is None:
                previous_closes = data.get('previous_closes', {}).get('val', None)
            
            previous_avg_volumes = metadata.get('previous_avg_volumes_val', None)
            if previous_avg_volumes is None:
                previous_avg_volumes = data.get('previous_avg_volumes', {}).get('val', None)
            
            # Get standardization parameters from metadata (needed to reverse standardization)
            std_log_price = metadata.get('std_log_price', 1.0)
            std_log_volume_diff = metadata.get('std_log_volume_diff', 1.0)
            volume_scale_factor = metadata.get('volume_scale_factor', 0.05)
            
            if previous_closes is not None and len(previous_closes) >= batch_size:
                sample_previous_closes = previous_closes[:batch_size]
                sample_previous_avg_volumes = previous_avg_volumes[:batch_size] if previous_avg_volumes is not None else None
                
                # y_real_4d and y_gen_4d are currently preprocessed:
                # - Price/bid/ask: log10(price/prev_close) (no standardization)
                # - Volume: (log10(1+vol) - log10(1+prev_avg_vol)) / std_log_volume_diff * std_log_price
                
                y_real_4d_original = y_real_4d.copy()
                y_gen_4d_original = y_gen_4d.copy()
                
                for i in range(batch_size):
                    prev_close = sample_previous_closes[i]
                    
                    # Convert dim0 (price) from log10 to original
                    # Price is already in log10 space (no standardization reversal needed)
                    log_norm_real = y_real_4d[i, :, 0]  # log10(price / prev_close)
                    log_norm_gen = y_gen_4d[i, :, 0]    # log10(price / prev_close)
                    
                    # Convert: price = prev_close * 10^(log10_norm)
                    pow10_real = np.clip(np.power(10.0, log_norm_real), 1e-10, 1e10)
                    pow10_gen = np.clip(np.power(10.0, log_norm_gen), 1e-10, 1e10)
                    y_real_4d_original[i, :, 0] = prev_close * pow10_real
                    y_gen_4d_original[i, :, 0] = prev_close * pow10_gen
                    
                    # Ensure prices are positive (safety check)
                    y_real_4d_original[i, :, 0] = np.maximum(y_real_4d_original[i, :, 0], 1e-6)
                    y_gen_4d_original[i, :, 0] = np.maximum(y_gen_4d_original[i, :, 0], 1e-6)
                    
                    # Convert dim1 (volume) from standardized to original
                    if sample_previous_avg_volumes is not None and input_dim > 1:
                        prev_avg_vol = sample_previous_avg_volumes[i]
                        # Step 1: Reverse volume scaling: divide by volume_scale_factor
                        vol_scaled = y_real_4d[i, :, 1] / volume_scale_factor
                        vol_scaled_gen = y_gen_4d[i, :, 1] / volume_scale_factor
                        
                        # Step 2: Reverse standardization: multiply by std_log_volume_diff
                        vol_diff_real = vol_scaled * std_log_volume_diff  # log10(1+vol) - log10(1+prev_avg_vol)
                        vol_diff_gen = vol_scaled_gen * std_log_volume_diff
                        
                        # Step 3: Convert: 10^(vol_diff) = (1+vol) / (1+prev_avg_vol)
                        # So: vol = (1+prev_avg_vol) * 10^(vol_diff) - 1
                        y_real_4d_original[i, :, 1] = (1.0 + prev_avg_vol) * np.power(10.0, vol_diff_real) - 1.0
                        y_gen_4d_original[i, :, 1] = (1.0 + prev_avg_vol) * np.power(10.0, vol_diff_gen) - 1.0
                
                plot_data_real = y_real_4d_original
                plot_data_gen = y_gen_4d_original
            else:
                # If no previous closes available, plot log-normalized values
                plot_data_real = y_real_4d
                plot_data_gen = y_gen_4d
            
            # Use 2D features for plotting - only plot dim0 (Price) and dim1 (Volume)
            num_dims_to_plot = min(2, input_dim)  # Only Price and Volume
            dim_names = ['Price ($)', 'Volume'][:num_dims_to_plot]
        except Exception as e:
            print(f"  Warning: Could not load data_path for embedding removal: {e}")
            print(f"  Plotting raw feature dimensions instead")
            plot_data_real = y_real
            plot_data_gen = y_gen
            num_dims_to_plot = min(2, feature_dim)  # Price and volume
            dim_names = ['Price', 'Volume'][:num_dims_to_plot]
    else:
        # Plot raw feature dimensions (should be 2D: price, volume)
        plot_data_real = y_real
        plot_data_gen = y_gen
        num_dims_to_plot = min(2, feature_dim)  # Price and volume
        dim_names = ['Price', 'Volume'][:num_dims_to_plot]
    
    # Plot each dimension separately with many subplots
    # Each subplot shows one true vs one predicted sequence
    num_samples_to_plot = min(20, plot_data_real.shape[0])
    
    # Create separate figures for each dimension (dim0 and dim1)
    for dim_idx in range(num_dims_to_plot):
        # Calculate subplot layout: 4 rows x 5 cols for 20 samples
        n_cols = 5
        n_rows = (num_samples_to_plot + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if axes.ndim > 1 else axes.reshape(-1)
        
        for sample_idx in range(num_samples_to_plot):
            row = sample_idx // n_cols
            col = sample_idx % n_cols
            
            if n_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            # Plot single true and predicted sequence
            ax.plot(range(seq_len), plot_data_real[sample_idx, :, dim_idx], 
                   label='True', color='blue', linewidth=2, alpha=0.7)
            ax.plot(range(seq_len), plot_data_gen[sample_idx, :, dim_idx], 
                   label='Predicted', color='red', linewidth=2, linestyle='--', alpha=0.7)
            
            ax.set_title(f'Sample {sample_idx+1}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=9)
            ax.set_ylabel(dim_names[dim_idx], fontsize=9)
            ax.grid(True, alpha=0.3)
            if sample_idx == 0:
                ax.legend(fontsize=8)
        
        # Hide unused subplots
        for sample_idx in range(num_samples_to_plot, n_rows * n_cols):
            row = sample_idx // n_cols
            col = sample_idx % n_cols
            if n_rows == 1:
                axes[col].axis('off')
            else:
                axes[row, col].axis('off')
        
        fig.suptitle(f'True vs Predicted {dim_names[dim_idx]} Sequences ({num_samples_to_plot} samples)', 
                    fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        # Save separate file for each dimension
        plot_name = f'sequence_comparison_{dim_names[dim_idx].lower().replace(" ", "_").replace("$", "")}.png'
        plot_path = os.path.join(output_dir, plot_name)
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Saved sequence comparison plot to {plot_path}")

