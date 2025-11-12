# Noise Schedule Instantiation Parameters

This document lists all parameters that can be set when instantiating each noise schedule class.

## Base Class: `NoiseSchedule`

All noise schedules inherit these parameters:

- **`learnable`** (bool, default: `True`): Whether schedule parameters should be learnable. If False, stop_gradient is applied to outputs to freeze parameters.
- **`gamma_prime_max`** (float, default: `100.0`): Maximum value for clipping gamma_prime_t.

---

## 1. `LinearNoiseSchedule`

**Parameters:**
- **`alpha_bar_min`** (float, default: `0.01`): Initial and bound values for alpha_bar_min
- **`alpha_bar_max`** (float, default: `0.99`): Initial and bound values for alpha_bar_max

**Inherited from base:**
- `learnable` (default: `True`)
- `gamma_prime_max` (default: `100.0`)

---

## 2. `CosineNoiseSchedule`

**Parameters:**
- **`alpha_bar_min`** (float, default: `0.01`): Initial and bound values for alpha_bar_min
- **`alpha_bar_max`** (float, default: `0.99`): Initial and bound values for alpha_bar_max

**Inherited from base:**
- `learnable` (default: `True`)
- `gamma_prime_max` (default: `100.0`)

---

## 3. `SigmoidNoiseSchedule`

**Parameters:**
- **`alpha_bar_min`** (float, default: `0.01`): Initial and bound values for alpha_bar_min
- **`alpha_bar_max`** (float, default: `0.99`): Initial and bound values for alpha_bar_max

**Inherited from base:**
- `learnable` (default: `True`)
- `gamma_prime_max` (default: `100.0`)

---

## 4. `ExponentialNoiseSchedule`

**Parameters:**
- **`beta`** (float, default: `0.5`): Initial value for beta (exponential decay rate)
- **`alpha_bar_min`** (float, default: `0.01`): Initial value for alpha_bar_min
- **`alpha_bar_max`** (float, default: `0.99`): Initial value for alpha_bar_max

**Inherited from base:**
- `learnable` (default: `True`)
- `gamma_prime_max` (default: `100.0`)

---

## 5. `CauchyNoiseSchedule`

**Parameters:**
- **`log_scale`** (float, default: `-1.2`): Initial log_scale value (exp(-1.2) ≈ 0.3)
- **`alpha_bar_min`** (float, default: `0.01`): Initial value for alpha_bar_min
- **`alpha_bar_max`** (float, default: `0.99`): Initial value for alpha_bar_max

**Inherited from base:**
- `learnable` (default: `True`)
- `gamma_prime_max` (default: `100.0`)

---

## 6. `LaplaceNoiseSchedule`

**Parameters:**
- **`loc`** (float, default: `0.5`): Initial loc value
- **`log_scale`** (float, default: `-1.0`): Initial log_scale value
- **`alpha_bar_min`** (float, default: `0.01`): Initial value for alpha_bar_min
- **`alpha_bar_max`** (float, default: `0.99`): Initial value for alpha_bar_max

**Inherited from base:**
- `learnable` (default: `True`)
- `gamma_prime_max` (default: `100.0`)

---

## 7. `QuadraticNoiseSchedule`

**Parameters:**
- **`alpha_bar_min`** (float, default: `0.01`): Initial value for alpha_bar_min
- **`alpha_bar_max`** (float, default: `0.99`): Initial value for alpha_bar_max
- **`beta`** (float, default: `0.5`): Initial scale value

**Inherited from base:**
- `learnable` (default: `True`)
- `gamma_prime_max` (default: `100.0`)

---

## 8. `PolynomialNoiseSchedule`

**Parameters:**
- **`log_power`** (float, default: `0.0`): Initial log_power value (exp(0.0) = 1.0)
- **`alpha_bar_min`** (float, default: `0.05`): Initial value for alpha_bar_min
- **`alpha_bar_max`** (float, default: `0.95`): Initial value for alpha_bar_max

**Inherited from base:**
- `learnable` (default: `True`)
- `gamma_prime_max` (default: `100.0`)

---

## 9. `NoiseScheduleNetwork`

**Parameters:**
- **`hidden_dims`** (Tuple[int, ...], default: `(64, 64)`): Hidden dimensions for the neural network
- **`monotonic_network`** (nn.Module, default: `SimpleMonotonicNetwork`): The monotonic network module to use
- **`gamma_range`** (Tuple[float, float], default: `(-4.0, 4.0)`): Range for gamma values

**Inherited from base:**
- `learnable` (default: `True`)
- `gamma_prime_max` (default: `100.0`)

---

## Example Usage

```python
# Linear schedule with custom parameters
linear = LinearNoiseSchedule(
    alpha_bar_min=0.02,
    alpha_bar_max=0.98,
    gamma_prime_max=50.0,
    learnable=True
)

# Exponential schedule with custom beta
exponential = ExponentialNoiseSchedule(
    beta=1.0,
    alpha_bar_min=0.05,
    alpha_bar_max=0.95
)

# Neural network schedule with custom architecture
nn_schedule = NoiseScheduleNetwork(
    hidden_dims=(128, 128, 64),
    gamma_range=(-5.0, 5.0),
    gamma_prime_max=200.0
)
```

