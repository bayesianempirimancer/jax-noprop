# Square Root Relationship Correction

## Overview

I've corrected the noise schedule implementation to use the **correct square root relationship** as described in the NoProp paper. The key insight is that the backward process uses square roots of the signal strength coefficients.

## The Correct Mathematical Relationship

### **Backward Process Formula**
According to the NoProp paper, the backward process follows:
```
z_t = sqrt(ᾱ(t)) * z_1 + sqrt(1-ᾱ(t)) * ε
```

Where:
- **$z_1$** is the target/starting point of the backward process (clean data)
- **$z_t$** is the noisy data at time $t$
- **$\bar{\alpha}(t)$** is the signal strength coefficient (INCREASING with time)
- **$1 - \bar{\alpha}(t)$** represents the cumulative noise added by the backward process
- **$\epsilon$** is Gaussian noise

### **Boundary Conditions**
- **$t = 0$**: $z_0 = \sqrt{0} \cdot z_1 + \sqrt{1} \cdot \epsilon = \epsilon$ (pure noise)
- **$t = 1$**: $z_1 = \sqrt{1} \cdot z_1 + \sqrt{0} \cdot \epsilon = z_1$ (pure signal)

## Corrected Implementations

### **1. Noise Schedule Methods**

#### **LinearNoiseSchedule**
```python
def get_alpha_t(self, t: jnp.ndarray) -> jnp.ndarray:
    """Get ᾱ(t) - the signal strength coefficient.
    
    For linear schedule: ᾱ(t) = t (INCREASING function)
    """
    return t

def get_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
    """Get σ(t) - the noise coefficient.
    
    Following the paper: σ(t) = sqrt(1 - ᾱ(t))
    """
    alpha_bar_t = self.get_alpha_t(t)
    return jnp.sqrt(1.0 - alpha_bar_t)
```

**Results:**
- $t = [0.0, 0.5, 1.0]$
- $\bar{\alpha}(t) = [0.0, 0.5, 1.0]$ ✅ **INCREASING**
- $\sigma(t) = [1.0, 0.707, 0.0]$ ✅ **DECREASING**
- $\sqrt{\bar{\alpha}(t)} = [0.0, 0.707, 1.0]$ ✅ **INCREASING**

#### **CosineNoiseSchedule**
```python
def get_alpha_t(self, t: jnp.ndarray) -> jnp.ndarray:
    """Get ᾱ(t) - the signal strength coefficient.
    
    Cosine schedule: ᾱ(t) = sin(π/2 * t) (INCREASING function)
    """
    return jnp.sin(jnp.pi / 2 * t)

def get_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
    """Get σ(t) - the noise coefficient.
    
    Following the paper: σ(t) = sqrt(1 - ᾱ(t))
    """
    alpha_bar_t = self.get_alpha_t(t)
    return jnp.sqrt(1.0 - alpha_bar_t)
```

**Results:**
- $t = [0.0, 0.5, 1.0]$
- $\bar{\alpha}(t) = [0.0, 0.707, 1.0]$ ✅ **INCREASING**
- $\sigma(t) = [1.0, 0.541, 0.0]$ ✅ **DECREASING**
- $\sqrt{\bar{\alpha}(t)} = [0.0, 0.841, 1.0]$ ✅ **INCREASING**

### **2. Noise Addition Function**

```python
def add_noise(
    clean: jnp.ndarray,
    noise: jnp.ndarray, 
    alpha_t: jnp.ndarray,
    sigma_t: jnp.ndarray,
) -> jnp.ndarray:
    """Add noise to clean data using the backward process.
    
    Following the NoProp paper: z_t = sqrt(ᾱ(t)) * z_1 + sqrt(1-ᾱ(t)) * ε
    
    Args:
        clean: Clean data (z_1) [batch_size, ...]
        noise: Gaussian noise (ε) [batch_size, ...]
        alpha_t: ᾱ(t) values [batch_size, 1] or scalar
        sigma_t: sqrt(1-ᾱ(t)) values [batch_size, 1] or scalar
        
    Returns:
        Noisy data: sqrt(alpha_t) * clean + sigma_t * noise
    """
    return jnp.sqrt(alpha_t) * clean + sigma_t * noise
```

### **3. SNR Computation**

```python
def _compute_snr_inverse(self, t: jnp.ndarray) -> jnp.ndarray:
    """Compute the inverse SNR weighting factor for the loss.
    
    Following the NoProp paper notation:
    - ᾱ(t) is the signal strength coefficient
    - Backward process: z_t = sqrt(ᾱ(t)) * z_1 + sqrt(1-ᾱ(t)) * ε
    - SNR(t) = ᾱ(t) / (1 - ᾱ(t))
    - 1/SNR(t) = (1 - ᾱ(t)) / ᾱ(t)
    """
    alpha_bar_t = self.noise_schedule.get_alpha_t(t_safe)
    snr_inverse = (1.0 - alpha_bar_t) / (alpha_bar_t + 1e-8)
    return snr_inverse
```

## Verification Results

### **Mathematical Properties**
✅ **$\bar{\alpha}(t)$ is INCREASING** with time  
✅ **$\sigma(t) = \sqrt{1-\bar{\alpha}(t)}$ is DECREASING** with time  
✅ **$\sqrt{\bar{\alpha}(t)}$ is INCREASING** with time  
✅ **$1 - \bar{\alpha}(t)$ is DECREASING** (cumulative noise)  
✅ **Boundary conditions**: $\bar{\alpha}(0) = 0$, $\bar{\alpha}(1) = 1$  

### **Backward Process Verification**
✅ **At $t=0$**: $z_0 = \sqrt{0} \cdot z_1 + \sqrt{1} \cdot \epsilon = \epsilon$ (pure noise)  
✅ **At $t=1$**: $z_1 = \sqrt{1} \cdot z_1 + \sqrt{0} \cdot \epsilon = z_1$ (pure signal)  
✅ **Noise addition follows correct formula**  

### **SNR Properties**
✅ **SNR is INCREASING** with time (more signal, less noise)  
✅ **SNR computation matches paper formula**: $\text{SNR}(t) = \frac{\bar{\alpha}(t)}{1-\bar{\alpha}(t)}$  
✅ **1/SNR weighting is correct** for loss computation  

### **Test Results**
✅ **All tests pass** (5/5)  
✅ **Noise schedules work correctly**  
✅ **Integration with NoProp works**  

## Key Changes Made

### **1. Noise Coefficient Formula**
- **Before**: $\sigma(t) = \sqrt{1 - \bar{\alpha}(t)^2}$ (incorrect)
- **After**: $\sigma(t) = \sqrt{1 - \bar{\alpha}(t)}$ (correct)

### **2. Noise Addition**
- **Before**: $z_t = \bar{\alpha}(t) \cdot z_1 + \sigma(t) \cdot \epsilon$ (incorrect)
- **After**: $z_t = \sqrt{\bar{\alpha}(t)} \cdot z_1 + \sigma(t) \cdot \epsilon$ (correct)

### **3. SNR Computation**
- **Before**: $\text{SNR}(t) = \frac{\bar{\alpha}(t)^2}{1-\bar{\alpha}(t)^2}$ (incorrect)
- **After**: $\text{SNR}(t) = \frac{\bar{\alpha}(t)}{1-\bar{\alpha}(t)}$ (correct)

## Impact on NoProp Algorithms

### **NoProp-CT**
- **Loss weighting** now uses correct SNR formula
- **Vector field computation** uses correct noise parameters
- **ODE integration** follows proper signal/noise dynamics

### **NoProp-DT**
- **Noise addition** follows correct backward process
- **Denoising process** uses correct signal strength

### **NoProp-FM**
- **Flow matching** uses correct noise schedule
- **Generation process** follows proper dynamics

## Example Usage

```python
from jax_noprop.noise_schedules import LinearNoiseSchedule, add_noise

# Create noise schedule
schedule = LinearNoiseSchedule()

# Get parameters
t = jnp.array([0.5])
alpha_bar_t = schedule.get_alpha_t(t)  # [0.5]
sigma_t = schedule.get_sigma_t(t)      # [0.707]

# Add noise using backward process
clean = jnp.array([[1.0, 2.0]])  # z_1
noise = jnp.array([[0.1, 0.2]])  # ε
noisy = add_noise(clean, noise, alpha_bar_t, sigma_t)
# Result: sqrt(0.5) * [1.0, 2.0] + 0.707 * [0.1, 0.2]
#       = [0.707, 1.414] + [0.071, 0.141] = [0.778, 1.555]
```

## Conclusion

The noise schedule implementation now **correctly follows the NoProp paper's square root relationship**:

1. **Backward process**: $z_t = \sqrt{\bar{\alpha}(t)} \cdot z_1 + \sqrt{1-\bar{\alpha}(t)} \cdot \epsilon$
2. **Signal strength**: $\sqrt{\bar{\alpha}(t)}$ is INCREASING with time
3. **Noise strength**: $\sqrt{1-\bar{\alpha}(t)}$ is DECREASING with time
4. **SNR computation**: $\text{SNR}(t) = \frac{\bar{\alpha}(t)}{1-\bar{\alpha}(t)}$
5. **All mathematical properties are satisfied**

This brings the implementation into full alignment with the theoretical framework described in the NoProp paper and ensures that the backward denoising process behaves correctly with the proper square root relationship between signal and noise components.
