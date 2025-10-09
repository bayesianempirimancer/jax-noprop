"""Example usage of NoPropFM for flow matching training."""

import jax
import jax.numpy as jnp
from jax_noprop import NoPropFM
from jax_noprop.models import ConditionalResNetFM


def main():
    """Demonstrate NoPropFM training on a flow matching task."""
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)
    
    # Create model
    model = ConditionalResNetFM(
        hidden_dims=(64, 64),
        output_dim=32,
    )
    
    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    z_init = jnp.ones((1, 32))
    x_init = jnp.ones((1, 32))
    t_init = jnp.array([[0.5]])
    params = model.init(init_rng, z_init, x_init, t_init)
    
    # Create NoProp wrapper
    wrapper = NoPropFM(
        model=lambda p, z, x, t: model.apply(p, z, x, t),
        noise_scale=0.01,
        learning_rate=0.001,
        num_noise_samples=2,
        time_steps=10,
    )
    
    # Training loop
    num_epochs = 10
    batch_size = 32
    
    print("Training NoPropFM for flow matching...")
    for epoch in range(num_epochs):
        # Generate random batch
        rng, batch_rng, data_rng = jax.random.split(rng, 3)
        
        # Base distribution (e.g., Gaussian)
        z0 = jax.random.normal(batch_rng, (batch_size, 32))
        
        # Target data (e.g., mixture of Gaussians)
        x = jax.random.normal(data_rng, (batch_size, 32)) + 2.0
        
        # Random time
        t = jax.random.uniform(batch_rng, shape=())
        batch = (z0, x, t)
        
        # Training step (uses default flow matching loss)
        rng, train_rng = jax.random.split(rng)
        params, metrics = wrapper.train_step(params, batch, None, train_rng)
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: loss = {metrics['loss']:.6f}, "
                  f"velocity_norm = {metrics['velocity_norm']:.6f}")
    
    print("Training complete!")
    
    # Sample from the learned flow
    print("\nSampling from learned flow...")
    rng, sample_rng = jax.random.split(rng)
    z0_sample = jax.random.normal(sample_rng, (100, 32))
    
    # Target distribution
    rng, target_rng = jax.random.split(rng)
    x_target = jax.random.normal(target_rng, (100, 32)) + 2.0
    
    # Generate samples
    samples = wrapper.sample_conditional_flow(params, z0_sample, x_target)
    
    print(f"\nGenerated samples:")
    print(f"  Shape: {samples.shape}")
    print(f"  Mean: {jnp.mean(samples):.6f} (target ~2.0)")
    print(f"  Std: {jnp.std(samples):.6f}")
    
    # Compare with target
    print(f"\nTarget distribution:")
    print(f"  Mean: {jnp.mean(x_target):.6f}")
    print(f"  Std: {jnp.std(x_target):.6f}")


if __name__ == "__main__":
    main()
