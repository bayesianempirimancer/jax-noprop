"""Test noise schedules with various input shapes for t.

This test ensures that all noise schedules can handle various input shapes,
including scalar (t.shape = ()), and that output shapes always match input shapes.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from src.embeddings.noise_schedules_v2 import (
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    SigmoidNoiseSchedule,
    ExponentialNoiseSchedule,
    CauchyNoiseSchedule,
    LaplaceNoiseSchedule,
    QuadraticNoiseSchedule,
    PolynomialNoiseSchedule,
    LogisticNoiseSchedule,
    NoiseScheduleNetwork,
    create_noise_schedule,
)

# Test configurations
TEST_SCHEDULES = {
    "linear": LinearNoiseSchedule(),
    "cosine": CosineNoiseSchedule(),
    "sigmoid": SigmoidNoiseSchedule(),
    "exponential": ExponentialNoiseSchedule(),
    "cauchy": CauchyNoiseSchedule(),
    "laplace": LaplaceNoiseSchedule(),
    "quadratic": QuadraticNoiseSchedule(),
    "polynomial": PolynomialNoiseSchedule(),
    "logistic": LogisticNoiseSchedule(),
    "network": NoiseScheduleNetwork(hidden_dims=(32, 32)),
}

# Various test shapes for t
TEST_SHAPES = [
    (),  # Scalar
    (1,),  # Single element 1D
    (10,),  # 1D array
    (5, 10),  # 2D array
    (2, 3, 4),  # 3D array
    (1, 1, 1),  # 3D with singleton dimensions
]

def test_schedule_shape_handling(schedule_name, schedule, key):
    """Test a schedule with various input shapes."""
    print(f"\n{'='*60}")
    print(f"Testing {schedule_name} schedule")
    print(f"{'='*60}")
    
    all_passed = True
    
    for shape in TEST_SHAPES:
        # Generate test data
        if shape == ():
            t = jnp.array(0.5)  # Scalar
        else:
            t = jr.uniform(key, shape, minval=0.0, maxval=1.0)
        
        t_original_shape = t.shape
        print(f"\n  Testing with t.shape = {t_original_shape}")
        
        # Initialize schedule parameters
        init_key, apply_key = jr.split(key, 2)
        
        try:
            # Initialize by calling get_alpha_bar (which uses @nn.compact)
            params = schedule.init(init_key, t, method='get_alpha_bar')
            alpha_bar = schedule.apply(params, t, method='get_alpha_bar')
            
            # Test get_alpha_bar_gamma_prime (can use same params)
            alpha_bar_prime, gamma_prime = schedule.apply(
                params, t, method='get_alpha_bar_gamma_prime'
            )
            
            # Check shapes
            if alpha_bar.shape != t_original_shape:
                print(f"    ❌ FAIL: alpha_bar.shape = {alpha_bar.shape}, expected {t_original_shape}")
                all_passed = False
            else:
                print(f"    ✓ alpha_bar.shape = {alpha_bar.shape} ✓")
            
            if alpha_bar_prime.shape != t_original_shape:
                print(f"    ❌ FAIL: alpha_bar_prime.shape = {alpha_bar_prime.shape}, expected {t_original_shape}")
                all_passed = False
            else:
                print(f"    ✓ alpha_bar_prime.shape = {alpha_bar_prime.shape} ✓")
            
            if gamma_prime.shape != t_original_shape:
                print(f"    ❌ FAIL: gamma_prime.shape = {gamma_prime.shape}, expected {t_original_shape}")
                all_passed = False
            else:
                print(f"    ✓ gamma_prime.shape = {gamma_prime.shape} ✓")
            
            # Check values are reasonable
            if jnp.any(jnp.isnan(alpha_bar)) or jnp.any(jnp.isinf(alpha_bar)):
                print(f"    ❌ FAIL: alpha_bar contains NaN or Inf")
                all_passed = False
            elif jnp.any((alpha_bar < 0) | (alpha_bar > 1)):
                print(f"    ⚠ WARNING: alpha_bar out of [0, 1] range: min={alpha_bar.min():.6f}, max={alpha_bar.max():.6f}")
            
            if jnp.any(jnp.isnan(gamma_prime)) or jnp.any(jnp.isinf(gamma_prime)):
                print(f"    ❌ FAIL: gamma_prime contains NaN or Inf")
                all_passed = False
            
            # Check that alpha_bar_prime matches alpha_bar (should be identical)
            if not jnp.allclose(alpha_bar, alpha_bar_prime):
                print(f"    ⚠ WARNING: alpha_bar != alpha_bar_prime (should match)")
                max_diff = jnp.abs(alpha_bar - alpha_bar_prime).max()
                print(f"      Max difference: {max_diff:.2e}")
            
        except Exception as e:
            print(f"    ❌ FAIL: Exception raised: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_helper_methods():
    """Test helper methods on base NoiseSchedule."""
    print(f"\n{'='*60}")
    print("Testing helper methods (alpha_bar, gamma_prime, etc.)")
    print(f"{'='*60}")
    
    # Use LinearNoiseSchedule as test case
    schedule = LinearNoiseSchedule()
    key = jr.PRNGKey(42)
    
    test_shapes = [(), (5,), (3, 4)]
    all_passed = True
    
    for shape in test_shapes:
        if shape == ():
            t = jnp.array(0.5)
        else:
            t = jr.uniform(key, shape, minval=0.0, maxval=1.0)
        
        print(f"\n  Testing helper methods with t.shape = {shape}")
        
        try:
            # Initialize by calling get_alpha_bar
            # init() returns just the params dict, not wrapped
            variables = schedule.init(key, t, method='get_alpha_bar')
            
            # Test each helper method - these use apply internally and expect variables dict
            alpha_bar = schedule.alpha_bar(variables, t)
            gamma = schedule.gamma(variables, t)
            gamma_prime = schedule.gamma_prime(variables, t)
            alpha_bar_gamma_prime = schedule.alpha_bar_gamma_prime(variables, t)
            alpha_bar_prime = schedule.alpha_bar_prime(variables, t)
            
            # Check shapes
            methods = {
                'alpha_bar': alpha_bar,
                'gamma': gamma,
                'gamma_prime': gamma_prime,
                'alpha_bar_gamma_prime': alpha_bar_gamma_prime,
                'alpha_bar_prime': alpha_bar_prime,
            }
            
            for method_name, result in methods.items():
                if isinstance(result, tuple):
                    # alpha_bar_gamma_prime returns tuple
                    for i, val in enumerate(result):
                        if val.shape != t.shape:
                            print(f"    ❌ FAIL: {method_name}[{i}].shape = {val.shape}, expected {t.shape}")
                            all_passed = False
                        else:
                            print(f"    ✓ {method_name}[{i}].shape = {val.shape} ✓")
                else:
                    if result.shape != t.shape:
                        print(f"    ❌ FAIL: {method_name}.shape = {result.shape}, expected {t.shape}")
                        all_passed = False
                    else:
                        print(f"    ✓ {method_name}.shape = {result.shape} ✓")
                        
        except Exception as e:
            print(f"    ❌ FAIL: Exception raised: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_factory_function():
    """Test create_noise_schedule factory function."""
    print(f"\n{'='*60}")
    print("Testing create_noise_schedule factory function")
    print(f"{'='*60}")
    
    key = jr.PRNGKey(42)
    test_shapes = [(), (5,), (2, 3)]
    all_passed = True
    
    schedule_types = [
        "linear", "cosine", "sigmoid", "exponential", 
        "cauchy", "laplace", "quadratic", "polynomial", "logistic", "network"
    ]
    
    for schedule_type in schedule_types:
        print(f"\n  Testing schedule_type = {schedule_type}")
        
        try:
            if schedule_type == "network":
                schedule = create_noise_schedule(schedule_type, hidden_dims=(16, 16))
            else:
                schedule = create_noise_schedule(schedule_type)
            
            for shape in test_shapes:
                if shape == ():
                    t = jnp.array(0.5)
                else:
                    t = jr.uniform(key, shape, minval=0.0, maxval=1.0)
                
                # Initialize by calling get_alpha_bar
                params = schedule.init(key, t, method='get_alpha_bar')
                alpha_bar, gamma_prime = schedule.apply(
                    params, t, method='get_alpha_bar_gamma_prime'
                )
                
                if alpha_bar.shape != t.shape or gamma_prime.shape != t.shape:
                    print(f"    ❌ FAIL: shapes don't match for shape {shape}")
                    all_passed = False
                else:
                    print(f"      ✓ t.shape={shape} → alpha_bar.shape={alpha_bar.shape}, gamma_prime.shape={gamma_prime.shape}")
                    
        except Exception as e:
            print(f"    ❌ FAIL: Exception raised: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("="*60)
    print("NOISE SCHEDULE SHAPE HANDLING TESTS")
    print("="*60)
    
    key = jr.PRNGKey(12345)
    all_results = []
    
    # Test each schedule individually
    for schedule_name, schedule in TEST_SCHEDULES.items():
        test_key, key = jr.split(key)
        passed = test_schedule_shape_handling(schedule_name, schedule, test_key)
        all_results.append((schedule_name, passed))
    
    # Test helper methods
    test_key, key = jr.split(key)
    helper_passed = test_helper_methods()
    all_results.append(("helper_methods", helper_passed))
    
    # Test factory function
    test_key, key = jr.split(key)
    factory_passed = test_factory_function()
    all_results.append(("factory_function", factory_passed))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for name, passed in all_results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(passed for _, passed in all_results)
    
    if all_passed:
        print(f"\n{'='*60}")
        print("✅ ALL TESTS PASSED!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("❌ SOME TESTS FAILED!")
        print(f"{'='*60}")
        exit(1)

