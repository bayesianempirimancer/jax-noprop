# SNR Weighting Corrections for NoProp-CT

## Overview

Thank you for pointing out this critical issue! After carefully reviewing the [NoProp paper](https://arxiv.org/html/2503.24322v1) again, I've now implemented the correct SNR-weighted loss function for the NoProp-CT variant.

## Key Correction: SNR-Weighted Loss Function

### The Issue
The original implementation was missing the crucial SNR (Signal-to-Noise Ratio) weighting in the NoProp-CT loss function. According to the paper, the continuous-time loss should be weighted by the derivative of the SNR with respect to time.

### The Correct Implementation

#### NoProp-CT Loss Function
**Before (Incorrect):**
```python
# Simple unweighted loss
main_loss = jnp.mean((v_pred - v_true) ** 2)
```

**After (Correct):**
```python
# SNR-weighted loss as described in the paper
snr_weight = self.noise_schedule.get_snr_derivative(t)
main_loss = jnp.mean(snr_weight * (v_pred - v_true) ** 2)
```

### Mathematical Foundation

According to the paper, the SNR is parameterized as:
```
SNR(t) = exp(-γ(t))
```

Where γ(t) is a learnable, monotonically decreasing function. The loss function should be weighted by:
```
w(t) = |dSNR/dt| = |γ'(t) * exp(-γ(t))|
```

### Implementation Details

#### 1. **LearnableNoiseSchedule Class**
I've added a new `LearnableNoiseSchedule` class that implements the trainable noise schedule from Appendix B:

```python
class LearnableNoiseSchedule(NoiseSchedule):
    """Learnable noise schedule as described in Appendix B of the NoProp paper.
    
    This implements the trainable noise schedule where:
    SNR(t) = exp(-γ(t))
    γ(t) = γ₀ + (γ₁ - γ₀)(1 - γ̄(t))
    
    where γ̄(t) is a normalized neural network output.
    """
    
    def get_snr_derivative(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute the derivative of SNR with respect to time."""
        gamma_t = self._compute_gamma_t(t)
        gamma_derivative = -(self.gamma_1 - self.gamma_0)
        snr_derivative = -gamma_derivative * jnp.exp(-gamma_t)
        return jnp.abs(snr_derivative)
```

#### 2. **SNR Weighting in Loss Function**
The NoProp-CT loss now properly weights the vector field loss by the SNR derivative:

```python
def compute_loss(self, params, z_t, x, target, t, key):
    # Compute the predicted vector field
    v_pred = self.vector_field(params, z_t, x, t)
    v_true = target - z_t
    
    # Compute SNR weighting factor
    if hasattr(self.noise_schedule, 'get_snr_derivative'):
        snr_weight = self.noise_schedule.get_snr_derivative(t)
    else:
        snr_weight = self._compute_snr_weight(t)  # Fallback for non-learnable schedules
    
    # SNR-weighted main loss
    main_loss = jnp.mean(snr_weight * (v_pred - v_true) ** 2)
    
    # Regularization term
    reg_loss = jnp.mean(v_pred ** 2)
    
    # Total loss
    total_loss = main_loss + self.eta * reg_loss
    
    return total_loss, metrics
```

#### 3. **Fallback for Linear Schedule**
For the linear noise schedule, I've implemented a fallback SNR weighting:

```python
def _compute_snr_weight(self, t: jnp.ndarray) -> jnp.ndarray:
    """Compute SNR weight for linear schedule.
    
    For linear schedule: α_t = 1 - t, σ_t = sqrt(t)
    SNR(t) = α_t² / σ_t² = (1-t)² / t
    dSNR/dt = -(1-t)(1+t)/t²
    """
    t_safe = jnp.maximum(t, 1e-8)
    snr_derivative = -(1 - t_safe) * (1 + t_safe) / (t_safe ** 2)
    snr_weight = jnp.abs(snr_derivative)
    snr_weight = snr_weight / jnp.mean(snr_weight)  # Normalize for stability
    return snr_weight
```

## Why This Matters

### 1. **Theoretical Correctness**
The SNR weighting is crucial for the continuous-time formulation to work correctly. It ensures that the loss function properly accounts for the varying noise levels at different timesteps.

### 2. **Training Stability**
The SNR weighting helps balance the contribution of different timesteps to the overall loss, preventing the model from being dominated by high-noise or low-noise regions.

### 3. **Performance Impact**
Without proper SNR weighting, the NoProp-CT model may not learn the correct vector field, leading to poor denoising performance during generation.

## Usage Examples

### Using Learnable Noise Schedule
```python
from jax_noprop import NoPropCT, LearnableNoiseSchedule

# Create learnable noise schedule
schedule = LearnableNoiseSchedule(gamma_0=0.0, gamma_1=1.0)

# Create NoProp-CT with learnable schedule
noprop_ct = NoPropCT(
    model=model,
    num_timesteps=1000,
    noise_schedule=schedule,
    eta=1.0
)
```

### Using Linear Schedule with SNR Weighting
```python
from jax_noprop import NoPropCT, LinearNoiseSchedule

# Linear schedule with automatic SNR weighting
schedule = LinearNoiseSchedule()
noprop_ct = NoPropCT(
    model=model,
    num_timesteps=1000,
    noise_schedule=schedule,
    eta=1.0
)
```

## Testing

All tests continue to pass after implementing the SNR weighting:
```
Tests passed: 5/5
🎉 All tests passed! The implementation is working correctly.
```

## Next Steps

1. **Neural Network γ(t)**: The current implementation uses a simplified γ(t) function. For full fidelity to the paper, this should be implemented as a neural network.

2. **Experimental Validation**: Test the SNR-weighted loss on real datasets to verify improved performance.

3. **Hyperparameter Tuning**: Experiment with different γ₀ and γ₁ values for the learnable schedule.

## Conclusion

The SNR weighting correction brings the NoProp-CT implementation much closer to the paper's methodology. This is a critical component that was missing from the initial implementation and should significantly improve the model's performance and theoretical correctness.

Thank you for catching this important detail! The implementation is now much more faithful to the original paper.
