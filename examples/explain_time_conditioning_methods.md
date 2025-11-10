# Time Conditioning Methods for Standard Attention

This document explains the different approaches for incorporating time information into standard (non-Twisted) attention mechanisms.

## Overview

When using standard attention (without TwistedAttention), we can still incorporate time information through various conditioning methods. Each method applies time conditioning at different stages of the attention computation.

## Methods

### 1. **FiLM (Feature-wise Linear Modulation)** - `"film"`

**What it does:**
- Applies feature-wise affine transformation to the normalized input before attention
- Formula: `output = (1 + scale) * input + shift`
- Both `scale` and `shift` are learned functions of time embedding

**Where it's applied:**
- Before the attention computation (on the normalized input)

**Mathematical form:**
```
t_film = MLP(time_embedding)  # [batch, 2 * embed_dim]
scale, shift = split(t_film)  # Each: [batch, embed_dim]
xz_norm = (1 + scale) * xz_norm + shift
xz_attn = Attention(xz_norm)
```

**Pros:**
- Simple and effective
- Commonly used in conditional generation
- Allows time to modulate all features independently

**Cons:**
- Applied before attention, so doesn't directly affect attention weights
- Less expressive than methods that affect attention computation directly

---

### 2. **Time-Conditioned Bias** - `"bias"`

**What it does:**
- Adds a time-dependent bias to the attention output
- The bias is learned from the time embedding via an MLP

**Where it's applied:**
- After the attention computation (on the attention output)

**Mathematical form:**
```
xz_attn = Attention(xz_norm)
t_bias = MLP(time_embedding)  # [batch, embed_dim]
xz_attn = xz_attn + t_bias
```

**Pros:**
- Very simple implementation
- Adds time-dependent offset to attention outputs
- Minimal computational overhead

**Cons:**
- Doesn't affect attention weights themselves
- Less expressive than methods that modulate the attention computation
- Bias is additive, so it shifts but doesn't scale features

---

### 3. **Adaptive LayerNorm Zero (adaLN-Zero)** - `"adaln"`

**What it does:**
- Replaces standard LayerNorm with time-adaptive normalization
- Time embedding modulates both the scale and shift parameters of normalization
- **This is exactly what DiT (Diffusion Transformer) uses!**
- "Zero" refers to zero initialization: the MLP producing scale/shift starts at zero, so the model begins as identity mapping

**Where it's applied:**
- Replaces the standard LayerNorm before attention
- (In full DiT, also applied to norm2 and sometimes MLP layers)

**Mathematical form:**
```
t_params = MLP(time_embedding)  # [batch, 2 * embed_dim]
# MLP initialized with zeros (adaLN-Zero) for training stability
scale, shift = split(t_params)  # Each: [batch, embed_dim]

# Custom normalization with time-dependent parameters
xz_mean = mean(xz, axis=-1)
xz_var = var(xz, axis=-1)
xz_norm = (xz - xz_mean) / sqrt(xz_var + eps)
xz_norm = (1 + scale) * xz_norm + shift  # DiT formula

xz_attn = Attention(xz_norm)
```

**Key DiT Details:**
- **Zero initialization**: MLP weights/bias start at zero, so scale=0, shift=0 initially
- This ensures the model starts as identity mapping and gradually learns conditioning
- Improves training stability - prevents large conditioning effects early in training
- Each layer can learn different degrees of time dependence

**Pros:**
- **Proven in DiT** - state-of-the-art for diffusion transformers
- Time affects normalization, which influences all downstream computations
- Zero initialization provides training stability
- More expressive than simple bias or scaling

**Cons:**
- Slightly more complex than FiLM
- Requires custom normalization instead of standard LayerNorm

---

### 4. **Time-Conditioned Scaling** - `"scale"`

**What it does:**
- Multiplies the attention output by a time-dependent scaling factor
- The scaling factor is learned from time embedding and constrained to [0, 2] via sigmoid

**Where it's applied:**
- After the attention computation (on the attention output)

**Mathematical form:**
```
xz_attn = Attention(xz_norm)
t_scale = sigmoid(MLP(time_embedding)) * 2.0  # [batch, 1], range [0, 2]
xz_attn = xz_attn * t_scale
```

**Pros:**
- Simple multiplicative modulation
- Constrained scaling prevents extreme values
- Can amplify or dampen attention outputs based on time

**Cons:**
- Only affects magnitude, not direction
- Less expressive than methods that affect multiple aspects
- Single scalar per batch, not per-feature

---

### 5. **None** - `"none"`

**What it does:**
- No time conditioning applied
- Standard attention with no time information

**Use case:**
- When you want standard attention without any time dependency
- Useful for ablation studies or when time information isn't available

---

## Comparison Table

| Method | Where Applied | Expressiveness | Complexity | Common Use Cases |
|--------|---------------|----------------|------------|------------------|
| **FiLM** | Before attention | Medium | Low | Conditional generation, style transfer |
| **Bias** | After attention | Low | Very Low | Simple time-dependent offsets |
| **adaLN-Zero** | Replaces LayerNorm | High | Medium | **DiT (Diffusion Transformer)** - standard approach |
| **Scale** | After attention | Low | Very Low | Time-dependent magnitude modulation |
| **None** | N/A | None | None | Baseline, ablation studies |

## Recommendations

1. **For diffusion/flow models**: Use **adaLN-Zero** (`"adaln"`) - **This is what DiT uses!**
   - Proven state-of-the-art for diffusion transformers
   - Zero initialization provides training stability
   - Matches the exact approach in the DiT paper

2. **For simplicity**: Use **FiLM** - easy to understand and implement, generally effective

3. **For minimal overhead**: Use **Bias** or **Scale** - very lightweight

4. **For maximum expressiveness**: Use **TwistedAttention** - directly modulates QKV matrices
   - More expressive than adaLN but also more complex

## Implementation Notes

- All methods require time embedding to be computed first
- Time embedding dimension should match `time_embed_dim` in config
- Methods can be combined in theory, but typically only one is used at a time
- The choice of method can significantly affect model performance and training dynamics

