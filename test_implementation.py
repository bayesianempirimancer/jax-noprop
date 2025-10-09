#!/usr/bin/env python3
"""
Simple test script to verify the NoProp implementation works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import jax
import jax.numpy as jnp
import numpy as np

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from jax_noprop import NoPropDT, NoPropCT, NoPropFM, ConditionalResNet, SimpleCNN
        from jax_noprop.noise_schedules import LinearNoiseSchedule, CosineNoiseSchedule
        from jax_noprop.utils import one_hot_encode, create_train_state
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_noise_schedules():
    """Test noise schedule implementations."""
    print("Testing noise schedules...")
    
    try:
        from jax_noprop.noise_schedules import LinearNoiseSchedule, CosineNoiseSchedule
        
        # Test linear schedule
        linear = LinearNoiseSchedule()
        t = jnp.array([0.0, 0.5, 1.0])
        alpha_t = linear.get_alpha_t(t)
        sigma_t = linear.get_sigma_t(t)
        
        assert jnp.allclose(alpha_t, jnp.array([1.0, 0.5, 0.0]))
        assert jnp.allclose(sigma_t, jnp.array([0.0, jnp.sqrt(0.5), 1.0]))
        
        # Test cosine schedule
        cosine = CosineNoiseSchedule()
        alpha_t_cos = cosine.get_alpha_t(t)
        sigma_t_cos = cosine.get_sigma_t(t)
        
        assert alpha_t_cos.shape == t.shape
        assert sigma_t_cos.shape == t.shape
        
        print("✓ Noise schedules working correctly")
        return True
    except Exception as e:
        print(f"✗ Noise schedule test failed: {e}")
        return False


def test_models():
    """Test model initialization and forward pass."""
    print("Testing models...")
    
    try:
        from jax_noprop.models import SimpleCNN, ConditionalResNet
        
        # Test SimpleCNN
        model = SimpleCNN(num_classes=10)
        key = jax.random.PRNGKey(42)
        
        dummy_z = jnp.ones((2, 10))
        dummy_x = jnp.ones((2, 28, 28, 1))
        dummy_t = jnp.ones((2,))
        
        params = model.init(key, dummy_z, dummy_x, dummy_t)
        output = model.apply(params, dummy_z, dummy_x, dummy_t)
        
        assert output.shape == (2, 10)
        
        # Test ConditionalResNet
        resnet = ConditionalResNet(num_classes=10, depth=18)
        params_resnet = resnet.init(key, dummy_z, dummy_x, dummy_t)
        output_resnet = resnet.apply(params_resnet, dummy_z, dummy_x, dummy_t)
        
        assert output_resnet.shape == (2, 10)
        
        print("✓ Models working correctly")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_noprop_variants():
    """Test NoProp variant initialization and basic functionality."""
    print("Testing NoProp variants...")
    
    try:
        from jax_noprop import NoPropDT, NoPropCT, NoPropFM
        from jax_noprop.models import SimpleCNN
        
        model = SimpleCNN(num_classes=10)
        
        # Test NoProp-DT
        noprop_dt = NoPropDT(model, num_timesteps=10)
        assert noprop_dt.num_timesteps == 10
        
        # Test NoProp-CT
        noprop_ct = NoPropCT(model, num_timesteps=1000)
        assert noprop_ct.num_timesteps == 1000
        
        # Test NoProp-FM
        noprop_fm = NoPropFM(model, num_timesteps=1000)
        assert noprop_fm.num_timesteps == 1000
        
        print("✓ NoProp variants initialized correctly")
        return True
    except Exception as e:
        print(f"✗ NoProp variant test failed: {e}")
        return False


def test_training_step():
    """Test a single training step."""
    print("Testing training step...")
    
    try:
        from jax_noprop import NoPropDT
        from jax_noprop.models import SimpleCNN
        from jax_noprop.utils import one_hot_encode
        
        # Create model and data
        model = SimpleCNN(num_classes=10)
        noprop_dt = NoPropDT(model, num_timesteps=10)
        
        key = jax.random.PRNGKey(42)
        dummy_x = jax.random.normal(key, (4, 28, 28, 1))
        dummy_y = one_hot_encode(jnp.array([0, 1, 2, 3]), 10)
        
        # Initialize parameters
        key, init_key = jax.random.split(key)
        dummy_z = jnp.ones((1, 10))
        params = model.init(init_key, dummy_z, dummy_x[:1])
        
        # Test training step
        key, train_key = jax.random.split(key)
        updated_params, loss, metrics = noprop_dt.train_step(
            params, dummy_x, dummy_y, train_key
        )
        
        assert isinstance(loss, jnp.ndarray)
        assert "loss" in metrics
        
        print("✓ Training step working correctly")
        return True
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running NoProp implementation tests...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_noise_schedules,
        test_models,
        test_noprop_variants,
        test_training_step,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
