"""Example usage of NoPropDT for discrete time training."""

import jax
import jax.numpy as jnp
from jax_noprop import NoPropDT
from jax_noprop.models import ConditionalResNetDT


def main():
    """Demonstrate NoPropDT training on a simple task."""
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)
    
    # Create model
    model = ConditionalResNetDT(
        hidden_dims=(64, 64),
        output_dim=32,
    )
    
    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    z_init = jnp.ones((1, 32))
    x_init = jnp.ones((1, 16))
    params = model.init(init_rng, z_init, x_init)
    
    # Create NoProp wrapper
    wrapper = NoPropDT(
        model=lambda p, z, x: model.apply(p, z, x),
        noise_scale=0.01,
        learning_rate=0.001,
        num_noise_samples=2,
    )
    
    # Define loss function (e.g., regression to target)
    def loss_fn(output):
        # Target is zeros for this example
        target = jnp.zeros_like(output)
        return jnp.mean((output - target) ** 2)
    
    # Training loop
    num_epochs = 10
    batch_size = 32
    
    print("Training NoPropDT...")
    for epoch in range(num_epochs):
        # Generate random batch
        rng, batch_rng = jax.random.split(rng)
        z = jax.random.normal(batch_rng, (batch_size, 32))
        x = jax.random.normal(batch_rng, (batch_size, 16))
        batch = (z, x)
        
        # Training step
        rng, train_rng = jax.random.split(rng)
        params, metrics = wrapper.train_step(params, batch, loss_fn, train_rng)
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: loss = {metrics['loss']:.6f}")
    
    print("Training complete!")
    
    # Test the trained model
    rng, test_rng = jax.random.split(rng)
    z_test = jax.random.normal(test_rng, (5, 32))
    x_test = jax.random.normal(test_rng, (5, 16))
    output = model.apply(params, z_test, x_test)
    
    print(f"\nTest output shape: {output.shape}")
    print(f"Test output mean: {jnp.mean(output):.6f}")
    print(f"Test output std: {jnp.std(output):.6f}")


if __name__ == "__main__":
    main()
