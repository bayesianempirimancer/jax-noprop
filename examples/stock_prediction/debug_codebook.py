#!/usr/bin/env python3
"""
Debug codebook initialization and usage.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import jax.random as jr
import pickle
from flax.core import FrozenDict

from src.models.vae.vqvae import VQVAE, VQVAEConfig

def main():
    # Load small sample
    with open('data/stock_sequences_full_day_2d.pkl', 'rb') as f:
        data = pickle.load(f)
    
    train_sequences = data['train']['sequences'][:10]
    y_seq_len = 12
    
    train_hour_sequences = []
    for seq in train_sequences:
        for i in range(len(seq) - y_seq_len + 1):
            train_hour_sequences.append(seq[i:i+y_seq_len])
            break
    
    train_data = jnp.array(train_hour_sequences)
    
    config = VQVAEConfig(
        main=FrozenDict({
            "input_shape": (12, 2),
            "codebook_size": 512,
            "embedding_dim": 32,
            "output_shape": (12, 2),
            "recon_loss_type": "mse",
            "recon_weight": 1.0,
            "vq_weight": 1.0,
            "commitment_weight": 0.25,
        }),
        encoder=FrozenDict({
            "model_type": "mlp",
            "encoder_type": "none",
            "input_shape": (12, 2),
            "latent_shape": (32,),
            "hidden_dims": (64, 32),
            "activation": "swish",
            "dropout_rate": 0.1,
        }),
        decoder=FrozenDict({
            "model_type": "mlp",
            "decoder_type": "none",
            "latent_shape": (32,),
            "output_shape": (12, 2),
            "hidden_dims": (32, 64),
            "activation": "swish",
            "dropout_rate": 0.1,
        }),
    )
    
    model = VQVAE(config=config)
    key = jr.PRNGKey(42)
    
    # Initialize
    params = model.init(key, train_data, key)
    
    # Check codebook initialization
    embedding = params['vq']['embedding']
    print(f"Codebook shape: {embedding.shape}")
    print(f"Codebook mean: {jnp.mean(embedding):.4f}")
    print(f"Codebook std: {jnp.std(embedding):.4f}")
    print(f"Codebook min: {jnp.min(embedding):.4f}, max: {jnp.max(embedding):.4f}")
    
    # Encode a batch
    z_e, z_q_st, indices = model.apply(params, train_data, method='encode', training=False)
    print(f"\nEncoder output (z_e) shape: {z_e.shape}")
    print(f"z_e mean: {jnp.mean(z_e):.4f}, std: {jnp.std(z_e):.4f}")
    print(f"z_e min: {jnp.min(z_e):.4f}, max: {jnp.max(z_e):.4f}")
    
    print(f"\nIndices shape: {indices.shape}")
    unique_indices = jnp.unique(indices)
    print(f"Unique indices: {unique_indices}")
    print(f"Number of unique tokens used: {len(unique_indices)}")
    
    # Check distances
    z_e_flat = z_e.reshape(-1, z_e.shape[-1])
    dists = jnp.sum((z_e_flat[:, None, :] - embedding[None, :, :]) ** 2, axis=2)
    min_dists = jnp.min(dists, axis=1)
    print(f"\nMin distances to codebook: mean={jnp.mean(min_dists):.4f}, std={jnp.std(min_dists):.4f}")
    print(f"Min distance range: [{jnp.min(min_dists):.4f}, {jnp.max(min_dists):.4f}]")
    
    # Check which token is closest
    closest_indices = jnp.argmin(dists, axis=1)
    print(f"Closest token indices: {jnp.unique(closest_indices)}")
    print(f"All tokens map to same? {len(jnp.unique(closest_indices)) == 1}")

if __name__ == "__main__":
    main()

