"""Tests for NoProp implementations."""

import jax
import jax.numpy as jnp
import pytest
from jax_noprop import NoPropDT, NoPropCT, NoPropFM
from jax_noprop.models import ConditionalResNetDT, ConditionalResNetCT, ConditionalResNetFM


class TestNoPropDT:
    """Tests for discrete time NoProp."""
    
    def test_initialization(self):
        """Test that NoPropDT initializes correctly."""
        model = ConditionalResNetDT()
        wrapper = NoPropDT(
            model=lambda params, z, x: model.apply(params, z, x),
            noise_scale=0.01,
            learning_rate=0.001,
        )
        assert wrapper.noise_scale == 0.01
        assert wrapper.learning_rate == 0.001
    
    def test_forward_with_noise(self):
        """Test forward pass with noise injection."""
        # Initialize model
        model = ConditionalResNetDT(output_dim=16)
        rng = jax.random.PRNGKey(0)
        
        # Initialize parameters
        z_init = jnp.ones((1, 16))
        x_init = jnp.ones((1, 8))
        params = model.init(rng, z_init, x_init)
        
        # Create wrapper
        wrapper = NoPropDT(
            model=lambda p, z, x: model.apply(p, z, x),
            noise_scale=0.1,
        )
        
        # Test forward pass
        z = jax.random.normal(rng, (4, 16))
        x = jax.random.normal(rng, (4, 8))
        rng_forward = jax.random.PRNGKey(1)
        
        output, info = wrapper.forward_with_noise(params, (z, x), rng_forward)
        
        assert output.shape == (4, 16)
        assert 'noise' in info
        assert info['noise'].shape == z.shape
    
    def test_train_step(self):
        """Test a complete training step."""
        # Initialize model
        model = ConditionalResNetDT(output_dim=16)
        rng = jax.random.PRNGKey(0)
        
        z_init = jnp.ones((1, 16))
        x_init = jnp.ones((1, 8))
        params = model.init(rng, z_init, x_init)
        
        # Create wrapper
        wrapper = NoPropDT(
            model=lambda p, z, x: model.apply(p, z, x),
            noise_scale=0.01,
            learning_rate=0.001,
            num_noise_samples=2,
        )
        
        # Create batch
        z = jax.random.normal(rng, (4, 16))
        x = jax.random.normal(rng, (4, 8))
        batch = (z, x)
        
        # Define loss function
        def loss_fn(output):
            target = jnp.zeros_like(output)
            return jnp.mean((output - target) ** 2)
        
        # Perform training step
        rng_train = jax.random.PRNGKey(2)
        new_params, metrics = wrapper.train_step(params, batch, loss_fn, rng_train)
        
        assert 'loss' in metrics
        assert isinstance(metrics['loss'], jnp.ndarray)
        # Check that parameters were updated
        assert not jnp.allclose(
            params['params']['Dense_0']['kernel'],
            new_params['params']['Dense_0']['kernel']
        )


class TestNoPropCT:
    """Tests for continuous time NoProp."""
    
    def test_initialization(self):
        """Test that NoPropCT initializes correctly."""
        model = ConditionalResNetCT()
        wrapper = NoPropCT(
            model=lambda params, z, x, t: model.apply(params, z, x, t),
            noise_scale=0.01,
            learning_rate=0.001,
            time_steps=10,
        )
        assert wrapper.noise_scale == 0.01
        assert wrapper.learning_rate == 0.001
        assert wrapper.time_steps == 10
    
    def test_forward_with_noise(self):
        """Test forward pass with noise injection."""
        model = ConditionalResNetCT(output_dim=16)
        rng = jax.random.PRNGKey(0)
        
        z_init = jnp.ones((1, 16))
        x_init = jnp.ones((1, 8))
        t_init = jnp.array([[0.5]])
        params = model.init(rng, z_init, x_init, t_init)
        
        wrapper = NoPropCT(
            model=lambda p, z, x, t: model.apply(p, z, x, t),
            noise_scale=0.1,
        )
        
        z = jax.random.normal(rng, (4, 16))
        x = jax.random.normal(rng, (4, 8))
        t = jnp.ones((4, 1)) * 0.5
        rng_forward = jax.random.PRNGKey(1)
        
        output, info = wrapper.forward_with_noise(params, (z, x, t), rng_forward)
        
        assert output.shape == (4, 16)
        assert 'noise' in info
        assert 'time' in info
    
    def test_integrate_trajectory(self):
        """Test trajectory integration."""
        model = ConditionalResNetCT(output_dim=16)
        rng = jax.random.PRNGKey(0)
        
        z_init = jnp.ones((1, 16))
        x_init = jnp.ones((1, 8))
        t_init = jnp.array([[0.5]])
        params = model.init(rng, z_init, x_init, t_init)
        
        wrapper = NoPropCT(
            model=lambda p, z, x, t: model.apply(p, z, x, t),
            time_steps=5,
        )
        
        z0 = jax.random.normal(rng, (4, 16))
        x = jax.random.normal(rng, (4, 8))
        
        z_final = wrapper.integrate_trajectory(params, z0, x, t0=0.0, t1=1.0)
        
        assert z_final.shape == z0.shape
    
    def test_train_step(self):
        """Test a complete training step."""
        model = ConditionalResNetCT(output_dim=16)
        rng = jax.random.PRNGKey(0)
        
        z_init = jnp.ones((1, 16))
        x_init = jnp.ones((1, 8))
        t_init = jnp.array([[0.5]])
        params = model.init(rng, z_init, x_init, t_init)
        
        wrapper = NoPropCT(
            model=lambda p, z, x, t: model.apply(p, z, x, t),
            noise_scale=0.01,
            learning_rate=0.001,
            num_noise_samples=2,
            time_steps=5,
        )
        
        z0 = jax.random.normal(rng, (4, 16))
        x = jax.random.normal(rng, (4, 8))
        t = jnp.array(0.5)
        batch = (z0, x, t)
        
        def loss_fn(z_final):
            target = jnp.zeros_like(z_final)
            return jnp.mean((z_final - target) ** 2)
        
        rng_train = jax.random.PRNGKey(2)
        new_params, metrics = wrapper.train_step(params, batch, loss_fn, rng_train)
        
        assert 'loss' in metrics


class TestNoPropFM:
    """Tests for flow matching NoProp."""
    
    def test_initialization(self):
        """Test that NoPropFM initializes correctly."""
        model = ConditionalResNetFM()
        wrapper = NoPropFM(
            model=lambda params, z, x, t: model.apply(params, z, x, t),
            noise_scale=0.01,
            learning_rate=0.001,
        )
        assert wrapper.noise_scale == 0.01
        assert wrapper.learning_rate == 0.001
    
    def test_forward_with_noise(self):
        """Test forward pass with noise injection."""
        model = ConditionalResNetFM(output_dim=16)
        rng = jax.random.PRNGKey(0)
        
        z_init = jnp.ones((1, 16))
        x_init = jnp.ones((1, 16))
        t_init = jnp.array([[0.5]])
        params = model.init(rng, z_init, x_init, t_init)
        
        wrapper = NoPropFM(
            model=lambda p, z, x, t: model.apply(p, z, x, t),
            noise_scale=0.1,
        )
        
        z = jax.random.normal(rng, (4, 16))
        x = jax.random.normal(rng, (4, 16))
        t = jnp.ones((4, 1)) * 0.5
        rng_forward = jax.random.PRNGKey(1)
        
        velocity, info = wrapper.forward_with_noise(params, (z, x, t), rng_forward)
        
        assert velocity.shape == (4, 16)
        assert 'noise' in info
    
    def test_compute_target_velocity(self):
        """Test target velocity computation."""
        model = ConditionalResNetFM()
        wrapper = NoPropFM(
            model=lambda params, z, x, t: model.apply(params, z, x, t),
        )
        
        z_t = jnp.ones((4, 16))
        x = jnp.ones((4, 16)) * 2.0
        t = 0.5
        
        v_target = wrapper.compute_target_velocity(z_t, x, t)
        
        # At t=0.5, v = (x - z_t) / (1 - 0.5) = (2 - 1) / 0.5 = 2
        expected = jnp.ones((4, 16)) * 2.0
        assert jnp.allclose(v_target, expected, atol=1e-5)
    
    def test_sample_conditional_flow(self):
        """Test sampling from conditional flow."""
        model = ConditionalResNetFM(output_dim=16)
        rng = jax.random.PRNGKey(0)
        
        z_init = jnp.ones((1, 16))
        x_init = jnp.ones((1, 16))
        t_init = jnp.array([[0.5]])
        params = model.init(rng, z_init, x_init, t_init)
        
        wrapper = NoPropFM(
            model=lambda p, z, x, t: model.apply(p, z, x, t),
            time_steps=10,
        )
        
        z0 = jax.random.normal(rng, (4, 16))
        x = jax.random.normal(rng, (4, 16))
        
        z_final = wrapper.sample_conditional_flow(params, z0, x)
        
        assert z_final.shape == z0.shape
    
    def test_train_step(self):
        """Test a complete training step."""
        model = ConditionalResNetFM(output_dim=16)
        rng = jax.random.PRNGKey(0)
        
        z_init = jnp.ones((1, 16))
        x_init = jnp.ones((1, 16))
        t_init = jnp.array([[0.5]])
        params = model.init(rng, z_init, x_init, t_init)
        
        wrapper = NoPropFM(
            model=lambda p, z, x, t: model.apply(p, z, x, t),
            noise_scale=0.01,
            learning_rate=0.001,
            num_noise_samples=2,
        )
        
        z0 = jax.random.normal(rng, (4, 16))
        x = jax.random.normal(rng, (4, 16))
        t = jnp.array(0.5)
        batch = (z0, x, t)
        
        rng_train = jax.random.PRNGKey(2)
        new_params, metrics = wrapper.train_step(params, batch, None, rng_train)
        
        assert 'loss' in metrics
        assert 'velocity_norm' in metrics
