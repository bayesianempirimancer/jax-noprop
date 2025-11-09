# CT vs DF Loss Calculation Comparison

## Key Differences in Loss Computation

### 1. **Flow Loss Target** ⚠️ CRITICAL DIFFERENCE

**CT (line 260-261):**
```python
squared_error = jnp.mean((z_target_est - z_target) ** 2, axis=tuple(range(-self.z_ndims, 0)))
snr_loss = jnp.mean(snr_weight * squared_error)
```
- Compares `z_target_est` (model's direct estimate) to `z_target` (ground truth)
- `z_target_est` comes from: `crn_output(z_t, x, t)` - direct CRN output

**DF (line 273-275):**
```python
squared_error = jnp.mean((noise - predicted_noise) ** 2, axis=tuple(range(-self.z_ndims, 0)))
flow_loss = jnp.mean(snr_weight * squared_error)
```
- Compares `predicted_noise` (model's noise prediction) to `noise` (ground truth noise)
- `predicted_noise` comes from: `pred_noise(z_t, x, t)` - CRN predicts noise
- Then derives `z_target_est = (z_t - predicted_noise * sqrt_1_minus_alpha_t)/(sqrt_alpha_t)`

**Impact**: CT directly predicts the target, DF predicts noise. This is a fundamental architectural difference.

### 2. **SNR Weight Formula** ⚠️ MAJOR DIFFERENCE

**CT (line 256):**
```python
snr_weight = (gamma_prime_t_squeezed * alpha_t_squeezed / (1.0 - alpha_t_squeezed))
```
- Can explode when `alpha_t` is close to 1
- Example: With `gamma_prime_t=10` and `alpha_t=0.99`: `snr_weight = 10 * 0.99 / 0.01 = 990`

**DF (line 269):**
```python
snr_weight = gamma_prime_t
```
- Simple, bounded by `gamma_prime_t` (typically ~10 for sigmoid schedule)
- Much more stable

**Impact**: This is the PRIMARY cause of high initial losses in CT. The SNR weight can be 100x larger than DF.

### 3. **Model Output Computation**

**CT (line 243):**
```python
z_target_est = self.apply(params, z_t, x, t, method='crn_output', training=training, rngs={'dropout': key})
```
- CRN directly outputs estimate of `z_target`
- Single forward pass

**DF (line 257-258):**
```python
predicted_noise = self.apply(params, z_t, x, t, method='pred_noise', training=training, rngs={'dropout': dropout_key1})
z_target_est = (z_t - predicted_noise * sqrt_1_minus_alpha_t)/(sqrt_alpha_t)
```
- CRN predicts noise, then `z_target_est` is computed from predicted noise
- Two-step process: predict noise, then derive target

**Impact**: Different model architectures - CT is more direct, DF uses noise prediction paradigm.

### 4. **Lazy Flow Computation Order**

**CT (line 247):**
```python
dz_dt = self.lazy_flow(z_t, z_target_est, alpha_t, gamma_prime_t)
# Then squeeze alpha_t and gamma_prime_t
```
- Computes `lazy_flow` BEFORE squeezing `alpha_t` and `gamma_prime_t`
- Needs expanded shapes for broadcasting

**DF (line 260, 263-264):**
```python
dz_dt = self.lazy_flow(z_t, predicted_noise, alpha_t, gamma_prime_t)
# Then squeeze alpha_t and gamma_prime_t
```
- Also computes `lazy_flow` BEFORE squeezing
- Same pattern (both fixed now)

**Impact**: Both now follow the same pattern after the time sampling fix.

### 5. **Lazy Flow Formula**

**CT (line 172):**
```python
lazy_flow = gamma_prime_t * (jnp.sqrt(alpha_t)*z_target - 0.5*(1+alpha_t)*z_t)
```
- Uses `z_target` directly in the formula

**DF (line 178):**
```python
lazy_flow = gamma_prime_t * (0.5*(1-alpha_t)*z_t - jnp.sqrt(1-alpha_t)*predicted_noise)
```
- Uses `predicted_noise` in the formula

**Impact**: Different vector field formulations, but both are mathematically correct for their respective models.

### 6. **Loss Component Names**

**CT:**
- `snr_loss` (the flow loss)
- `reg_loss` (regularization loss)
- `recon_loss` (reconstruction loss)
- Returns: `{'flow_loss': snr_loss, ...}`

**DF:**
- `flow_loss` (the flow loss, same as CT's snr_loss)
- `reg_loss` (regularization loss)
- `recon_loss` (reconstruction loss)
- Returns: `{'flow_loss': flow_loss, ...}`

**Impact**: Just naming difference, both represent the same concept.

### 7. **Total Loss Formula**

**CT (line 282):**
```python
total_loss = snr_loss + recon_weight * recon_loss + reg_weight * reg_loss
```

**DF (line 294):**
```python
total_loss = flow_loss + recon_weight * recon_loss + reg_weight * reg_loss
```

**Impact**: Identical formula, just different variable names (`snr_loss` vs `flow_loss`).

## Summary of Critical Differences

1. **Flow Loss Target**: CT compares `z_target_est` to `z_target`, DF compares `predicted_noise` to `noise`
2. **SNR Weight**: CT uses `(gamma_prime_t * alpha_t / (1-alpha_t))` which can explode, DF uses `gamma_prime_t`
3. **Model Architecture**: CT directly predicts target, DF predicts noise then derives target

## Recommendations

The **SNR weight formula** is the primary issue causing high initial losses in CT. The formula `(gamma_prime_t * alpha_t / (1-alpha_t))` can be 100x larger than DF's `gamma_prime_t` when `alpha_t` is close to 1.

The **flow loss target** difference (direct target prediction vs noise prediction) is an architectural choice and both are valid, but may have different sensitivities to initialization.

