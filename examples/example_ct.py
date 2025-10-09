"""Example usage of NoPropCT for continuous time training."""

import jax
import jax.numpy as jnp
from jax_noprop import NoPropCT
from jax_noprop.models import ConditionalResNetCT


def main():
    """Demonstrate NoPropCT training on a simple ODE task."""
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)
    
    # Create model
    model = ConditionalResNetCT(
        hidden_dims=(64, 64),
        output_dim=32,
    )
    
    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    z_init = jnp.ones((1, 32))
    x_init = jnp.ones((1, 16))
    t_init = jnp.array([[0.5]])
    params = model.init(init_rng, z_init, x_init, t_init)
    
    # Create NoProp wrapper
    wrapper = NoPropCT(
        model=lambda p, z, x, t: model.apply(p, z, x, t),
        noise_scale=0.01,
        learning_rate=0.001,
        num_noise_samples=2,
        time_steps=10,
    )
    
    # Define loss function
    def loss_fn(z_final):
        # Target final state is zeros
        target = jnp.zeros_like(z_final)
        return jnp.mean((z_final - target) ** 2)
    
    # Training loop
    num_epochs = 10
    batch_size = 32
    
    print("Training NoPropCT...")
    for epoch in range(num_epochs):
        # Generate random batch
        rng, batch_rng = jax.random.split(rng)
        z0 = jax.random.normal(batch_rng, (batch_size, 32))
        x = jax.random.normal(batch_rng, (batch_size, 16))
        t = 1.0  # Integrate to t=1
        batch = (z0, x, t)
        
        # Training step
        rng, train_rng = jax.random.split(rng)
        params, metrics = wrapper.train_step(params, batch, loss_fn, train_rng)
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: loss = {metrics['loss']:.6f}")
    
    print("Training complete!")
    
    # Test trajectory integration
    rng, test_rng = jax.random.split(rng)
    z0_test = jax.random.normal(test_rng, (5, 32))
    x_test = jax.random.normal(test_rng, (5, 16))
    z_final = wrapper.integrate_trajectory(params, z0_test, x_test, t0=0.0, t1=1.0)
    
    print(f"\nTest trajectory:")
    print(f"  Initial state mean: {jnp.mean(z0_test):.6f}")
    print(f"  Final state mean: {jnp.mean(z_final):.6f}")
    print(f"  Final state std: {jnp.std(z_final):.6f}")


if __name__ == "__main__":
    main()
