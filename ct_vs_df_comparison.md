# CT vs DF Model Comparison - Key Differences

## Critical Differences Found

### 1. **Time Sampling Shape** ⚠️ POTENTIAL ISSUE
- **CT (line 235)**: `t = jr.uniform(t_key, batch_shape + self.z_ndims*(1,), minval=0.0, maxval=1.0)`
  - Shape: `(batch_size, 1, 1, ...)` if `z_ndims > 0`
  - Example: If `batch_shape=(256,)` and `z_ndims=1`, then `t.shape = (256, 1)`
  
- **DF (line 243)**: `t = jr.uniform(t_key, batch_shape, minval=0.0, maxval=1.0)`
  - Shape: `(batch_size,)`
  - Example: If `batch_shape=(256,)`, then `t.shape = (256,)`

**Impact**: CT samples time with extra dimensions, which may cause broadcasting issues or incorrect noise schedule evaluation.

### 2. **Time Expansion for Noise Schedule**
- **CT (line 236)**: `alpha_t, gamma_prime_t = self.apply(params, t, method='get_noise_params')`
  - Uses `t` directly (which already has extra dims from line 235)
  - Then squeezes `t` on line 239: `t = t.squeeze(tuple(range(-self.z_ndims, 0)))`
  
- **DF (line 247-248)**: 
  ```python
  t_expanded = jnp.expand_dims(t, axis=tuple(range(-self.z_ndims, 0)))
  alpha_t, gamma_prime_t = self.apply(params, t_expanded, method='get_noise_params')
  ```
  - Expands `t` from `(batch_size,)` to `(batch_size, 1, 1, ...)` before calling noise schedule
  - Then squeezes `alpha_t` and `gamma_prime_t` on lines 263-264

**Impact**: The order of operations is different - CT expands first then squeezes, DF expands before calling then squeezes after. This could lead to different shapes being passed to the noise schedule.

### 3. **SNR Weight Formula** ⚠️ MAJOR DIFFERENCE
- **CT (line 246)**: `snr_weight = (gamma_prime_t * alpha_t / (1.0 - alpha_t))`
  - Can explode when `alpha_t` is close to 1
  - With sigmoid schedule `k=10.0` (so `gamma_prime_t=10`), if `alpha_t=0.99`:
    - `snr_weight = 10 * 0.99 / 0.01 = 990` (very large!)
  
- **DF (line 269)**: `snr_weight = gamma_prime_t`
  - Simple, bounded by `gamma_prime_t` (typically ~10 for sigmoid schedule)
  - Much more stable

**Impact**: This is the PRIMARY cause of high initial losses in CT. The SNR weight can be 100x larger than DF.

### 4. **Flow Loss Target**
- **CT (line 250)**: `squared_error = jnp.mean((z_target_est - z_target) ** 2, ...)`
  - Compares model's direct estimate of `z_target` to actual `z_target`
  
- **DF (line 273)**: `squared_error = jnp.mean((noise - predicted_noise) ** 2, ...)`
  - Compares predicted noise to actual noise
  - Then computes `z_target_est` from predicted noise on line 258

**Impact**: Different loss formulations, but both should work. CT's direct approach might be more sensitive to initialization.

### 5. **Model Output Method**
- **CT (line 240)**: `z_target_est = self.apply(params, z_t, x, t, method='crn_output', ...)`
  - CRN directly outputs estimate of `z_target`
  
- **DF (line 257-258)**:
  ```python
  predicted_noise = self.apply(params, z_t, x, t, method='pred_noise', ...)
  z_target_est = (z_t - predicted_noise * sqrt_1_minus_alpha_t)/(sqrt_alpha_t)
  ```
  - CRN predicts noise, then `z_target_est` is computed from predicted noise

**Impact**: Different model architectures - CT predicts target directly, DF predicts noise.

### 6. **Lazy Flow Formula**
- **CT (line 172)**: `lazy_flow = gamma_prime_t * (jnp.sqrt(alpha_t)*z_target - 0.5*(1+alpha_t)*z_t)`
- **DF (line 178)**: `lazy_flow = gamma_prime_t * (0.5*(1-alpha_t)*z_t - jnp.sqrt(1-alpha_t)*predicted_noise)`

**Impact**: Different vector field formulations, but both are mathematically correct for their respective models.

## Summary of Issues

1. **PRIMARY ISSUE**: SNR weight formula in CT can explode (`gamma_prime_t * alpha_t / (1-alpha_t)`) when `alpha_t` is close to 1, leading to losses 100x larger than DF.

2. **POTENTIAL ISSUE**: Time sampling shape difference - CT samples `t` with extra dimensions `batch_shape + z_ndims*(1,)`, which may cause shape mismatches or incorrect noise schedule evaluation.

3. **MINOR**: Different loss formulations (direct target prediction vs noise prediction) may have different sensitivities to initialization.

## Recommendations

1. **Fix SNR weight**: Consider normalizing or clipping the SNR weight in CT, or use a different formula that doesn't explode.

2. **Fix time sampling**: Make CT's time sampling consistent with DF - sample `t` as `(batch_size,)` then expand before calling noise schedule, matching DF's pattern.

3. **Verify noise schedule input**: Ensure the noise schedule receives the correct shape in both models.

