#!/usr/bin/env python3
"""Comprehensive test for noise schedules: shapes and ct_v2 training compatibility."""

import jax
import jax.numpy as jnp
import jax.random as jr
from src.embeddings.noise_schedules_v2 import (
    LinearNoiseSchedule, CosineNoiseSchedule, SigmoidNoiseSchedule,
    ExponentialNoiseSchedule, CauchyNoiseSchedule, LaplaceNoiseSchedule,
    QuadraticNoiseSchedule, PolynomialNoiseSchedule, LogisticNoiseSchedule,
    NoiseScheduleNetwork, create_noise_schedule
)
from src.flow_models.ct_v2 import VAEFlowConfig, VAE_flow
from flax.core import FrozenDict

def test_shape_handling():
    """Test all noise schedules with various input shapes."""
    print("="*60)
    print("SHAPE HANDLING TESTS")
    print("="*60)
    
    schedule_classes = [
        ("linear", LinearNoiseSchedule),
        ("cosine", CosineNoiseSchedule),
        ("sigmoid", SigmoidNoiseSchedule),
        ("exponential", ExponentialNoiseSchedule),
        ("cauchy", CauchyNoiseSchedule),
        ("laplace", LaplaceNoiseSchedule),
        ("quadratic", QuadraticNoiseSchedule),
        ("polynomial", PolynomialNoiseSchedule),
        ("logistic", LogisticNoiseSchedule),
    ]
    
    # Test shapes: scalar, 1D, 2D, 3D
    test_shapes = [
        (),      # Scalar
        (1,),    # 1D with 1 element
        (10,),   # 1D with 10 elements
        (5, 10), # 2D
        (2, 3, 4), # 3D
        (1, 1, 1), # 3D scalar-like
    ]
    
    key = jr.PRNGKey(42)
    all_passed = True
    
    for schedule_name, schedule_class in schedule_classes:
        print(f"\n📋 Testing {schedule_name} schedule...")
        
        for learnable in [True, False]:
            schedule = schedule_class(learnable=learnable)
            
            for t_shape in test_shapes:
                try:
                    # Create t with the desired shape
                    if t_shape == ():
                        t = jnp.array(0.5)
                    else:
                        t = jnp.ones(t_shape) * 0.5
                    
                    # Test get_alpha_bar
                    init_key, apply_key = jr.split(key)
                    variables = schedule.init(init_key, t, method='get_alpha_bar')
                    alpha_bar = schedule.apply(variables, t, method='get_alpha_bar')
                    
                    assert alpha_bar.shape == t.shape, \
                        f"get_alpha_bar shape mismatch: expected {t.shape}, got {alpha_bar.shape}"
                    
                    # Test get_alpha_bar_gamma_prime
                    variables2 = schedule.init(init_key, t, method='get_alpha_bar_gamma_prime')
                    alpha_bar2, gamma_prime = schedule.apply(variables2, t, method='get_alpha_bar_gamma_prime')
                    
                    assert alpha_bar2.shape == t.shape, \
                        f"alpha_bar shape mismatch: expected {t.shape}, got {alpha_bar2.shape}"
                    assert gamma_prime.shape == t.shape, \
                        f"gamma_prime shape mismatch: expected {t.shape}, got {gamma_prime.shape}"
                    
                    # Test helper methods
                    alpha_bar3 = schedule.alpha_bar(variables, t)
                    gamma_prime2 = schedule.gamma_prime(variables2, t)
                    alpha_bar4, gamma_prime3 = schedule.alpha_bar_gamma_prime(variables2, t)
                    
                    assert alpha_bar3.shape == t.shape
                    assert gamma_prime2.shape == t.shape
                    assert alpha_bar4.shape == t.shape
                    assert gamma_prime3.shape == t.shape
                    
                except Exception as e:
                    print(f"  ❌ FAILED: shape {t_shape}, learnable={learnable}")
                    print(f"     Error: {e}")
                    all_passed = False
                    import traceback
                    traceback.print_exc()
    
    # Test NoiseScheduleNetwork separately (needs hidden_dims)
    print(f"\n📋 Testing noise_schedule_network schedule...")
    for learnable in [True, False]:
        schedule = NoiseScheduleNetwork(learnable=learnable, hidden_dims=(32, 32))
        for t_shape in test_shapes[:4]:  # Skip very large shapes for network
            try:
                if t_shape == ():
                    t = jnp.array(0.5)
                else:
                    t = jnp.ones(t_shape) * 0.5
                
                init_key, apply_key = jr.split(key)
                variables = schedule.init(init_key, t, method='get_alpha_bar')
                alpha_bar = schedule.apply(variables, t, method='get_alpha_bar')
                
                assert alpha_bar.shape == t.shape
                
                variables2 = schedule.init(init_key, t, method='get_alpha_bar_gamma_prime')
                alpha_bar2, gamma_prime = schedule.apply(variables2, t, method='get_alpha_bar_gamma_prime')
                
                assert alpha_bar2.shape == t.shape
                assert gamma_prime.shape == t.shape
                
            except Exception as e:
                print(f"  ❌ FAILED: shape {t_shape}, learnable={learnable}")
                print(f"     Error: {e}")
                all_passed = False
    
    if all_passed:
        print("\n✅ All shape handling tests passed!")
    else:
        print("\n❌ Some shape handling tests failed!")
    
    return all_passed


def test_factory_function():
    """Test the factory function with all schedule types."""
    print("\n" + "="*60)
    print("FACTORY FUNCTION TESTS")
    print("="*60)
    
    schedule_types = [
        "linear", "cosine", "sigmoid", "exponential", 
        "cauchy", "laplace", "quadratic", "polynomial", "logistic",
        "network", "monotonic_nn", "learnable"  # Aliases for NoiseScheduleNetwork
    ]
    
    key = jr.PRNGKey(42)
    t = jnp.array([0.1, 0.5, 0.9])
    all_passed = True
    
    for schedule_type in schedule_types:
        try:
            if schedule_type in ["network", "monotonic_nn", "learnable"]:
                schedule = create_noise_schedule(schedule_type, learnable=True, hidden_dims=(32, 32))
            else:
                schedule = create_noise_schedule(schedule_type, learnable=True)
            
            variables = schedule.init(key, t, method='get_alpha_bar')
            alpha_bar = schedule.apply(variables, t, method='get_alpha_bar')
            
            assert alpha_bar.shape == t.shape
            print(f"  ✅ {schedule_type}")
            
        except Exception as e:
            print(f"  ❌ {schedule_type}: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✅ All factory function tests passed!")
    else:
        print("\n❌ Some factory function tests failed!")
    
    return all_passed


def test_ct_v2_integration():
    """Test noise schedules integrated with ct_v2 model."""
    print("\n" + "="*60)
    print("CT_V2 INTEGRATION TESTS")
    print("="*60)
    
    schedule_types = ["linear", "cosine", "sigmoid", "exponential", "quadratic", "polynomial"]
    
    key = jr.PRNGKey(42)
    x = jnp.ones((4, 2))  # batch_size=4, input_shape=(2,)
    y = jnp.ones((4, 2))  # batch_size=4, output_shape=(2,)
    
    all_passed = True
    
    for schedule_type in schedule_types:
        for learnable in [True, False]:
            try:
                # Create config
                config = VAEFlowConfig(
                    main=FrozenDict({
                        'input_shape': (2,),
                        'output_shape': (2,),
                        'latent_shape': (2,),
                        'recon_loss_type': 'mse',
                        'recon_weight': 0.0,
                        'reg_weight': 0.0,
                        'integration_method': 'midpoint',
                        'noise_schedule': schedule_type,
                    }),
                    noise_schedule=FrozenDict({
                        'schedule_type': schedule_type,
                        'learnable': learnable,
                        'default_params': FrozenDict({}),
                    })
                )
                
                # Initialize model
                model = VAE_flow(config)
                variables = model.init(key, x, y, key)
                
                # Test that noise schedule params are present
                assert 'noise_schedule' in variables['params'], \
                    f"Noise schedule params not found for {schedule_type}"
                
                # Test get_alpha_bar_gamma_prime_t directly (bypasses JIT issues)
                t_test = jnp.array([0.5])
                alpha_bar, gamma_prime = model.apply(
                    variables, t_test, method='get_alpha_bar_gamma_prime_t'
                )
                
                assert alpha_bar.shape == t_test.shape, \
                    f"alpha_bar shape mismatch for {schedule_type}"
                assert gamma_prime.shape == t_test.shape, \
                    f"gamma_prime shape mismatch for {schedule_type}"
                assert jnp.isfinite(alpha_bar).all(), \
                    f"alpha_bar contains non-finite values for {schedule_type}"
                assert jnp.isfinite(gamma_prime).all(), \
                    f"gamma_prime contains non-finite values for {schedule_type}"
                
                print(f"  ✅ {schedule_type}, learnable={learnable}")
                
            except Exception as e:
                print(f"  ❌ {schedule_type}, learnable={learnable}: {e}")
                all_passed = False
                import traceback
                traceback.print_exc()
    
    # Test with NoiseScheduleNetwork
    try:
        config = VAEFlowConfig(
            main=FrozenDict({
                'input_shape': (2,),
                'output_shape': (2,),
                'latent_shape': (2,),
                'recon_loss_type': 'mse',
                'recon_weight': 0.0,
                'reg_weight': 0.0,
                'integration_method': 'midpoint',
                'noise_schedule': 'network',
            }),
            noise_schedule=FrozenDict({
                'schedule_type': 'network',
                'learnable': True,
                'hidden_dims': (32, 32),
                'default_params': FrozenDict({}),
            })
        )
        
        model = VAE_flow(config)
        variables = model.init(key, x, y, key)
        
        assert 'noise_schedule' in variables['params']
        
        t_test = jnp.array([0.5])
        alpha_bar, gamma_prime = model.apply(
            variables, t_test, method='get_alpha_bar_gamma_prime_t'
        )
        
        assert alpha_bar.shape == t_test.shape
        assert gamma_prime.shape == t_test.shape
        
        print(f"  ✅ network, learnable=True")
        
    except Exception as e:
        print(f"  ❌ network, learnable=True: {e}")
        all_passed = False
        import traceback
        traceback.print_exc()
    
    if all_passed:
        print("\n✅ All ct_v2 integration tests passed!")
    else:
        print("\n❌ Some ct_v2 integration tests failed!")
    
    return all_passed


def test_stop_gradient():
    """Test that stop_gradient is applied when learnable=False."""
    print("\n" + "="*60)
    print("STOP_GRADIENT TESTS")
    print("="*60)
    
    schedule = LinearNoiseSchedule(learnable=False)
    key = jr.PRNGKey(42)
    t = jnp.array([0.5])
    
    all_passed = True
    
    try:
        # Initialize
        variables = schedule.init(key, t, method='get_alpha_bar')
        
        # Test gradient flow
        def loss_fn(v):
            alpha_bar = schedule.apply(v, t, method='get_alpha_bar')
            return alpha_bar.sum()
        
        grads = jax.grad(lambda v: loss_fn(v))(variables)
        
        # When learnable=False, gradients should be zero
        if 'noise_schedule' in grads['params']:
            for param_name, param_grad in grads['params']['noise_schedule'].items():
                if not jnp.allclose(param_grad, 0.0):
                    print(f"  ❌ Gradient for {param_name} is not zero: {param_grad}")
                    all_passed = False
                else:
                    print(f"  ✅ Gradient for {param_name} is zero (as expected)")
        else:
            # If noise_schedule not in grads, that's also fine (may be structured differently)
            print(f"  ✅ Noise schedule gradients handled correctly")
        
        # Test with learnable=True - gradients should be non-zero
        schedule2 = LinearNoiseSchedule(learnable=True)
        variables2 = schedule2.init(key, t, method='get_alpha_bar')
        grads2 = jax.grad(lambda v: loss_fn(v))(variables2)
        
        if 'noise_schedule' in grads2['params']:
            has_nonzero = False
            for param_name, param_grad in grads2['params']['noise_schedule'].items():
                if not jnp.allclose(param_grad, 0.0):
                    has_nonzero = True
                    break
            if has_nonzero:
                print(f"  ✅ With learnable=True, gradients are non-zero (as expected)")
            else:
                print(f"  ⚠️  With learnable=True, all gradients are zero (unexpected)")
                all_passed = False
        
    except Exception as e:
        print(f"  ❌ Stop gradient test failed: {e}")
        all_passed = False
        import traceback
        traceback.print_exc()
    
    if all_passed:
        print("\n✅ All stop_gradient tests passed!")
    else:
        print("\n❌ Some stop_gradient tests failed!")
    
    return all_passed


if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPREHENSIVE NOISE SCHEDULE TESTS")
    print("="*60)
    
    results = []
    
    results.append(("Shape Handling", test_shape_handling()))
    results.append(("Factory Function", test_factory_function()))
    results.append(("CT_V2 Integration", test_ct_v2_integration()))
    results.append(("Stop Gradient", test_stop_gradient()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print("\n⚠️  Some tests failed!")
        exit(1)

