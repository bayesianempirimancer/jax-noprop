# Factory Interface Comparison: Old vs New

## The Problem with the Old Interface

The original factory interface was clunky because it required `z_dim` and `x_dim` parameters for ALL models, even when they weren't used:

```python
# OLD INTERFACE - Clunky and confusing
encoder = create_model("encoder", config, z_dim=8, x_dim=4, input_dim=4, latent_shape=(8,))
#                                 ^^^^^^ ^^^^^^ These are required but not used!
#                                 ^^^^^^ ^^^^^^ These are the actual parameters used
```

## The New Clean Interface

The new interface eliminates the clunky requirements and only asks for what each model actually needs:

```python
# NEW INTERFACE - Clean and intuitive
encoder = create_model("encoder", config, input_dim=4, latent_shape=(8,))
#                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Only what's needed!
```

## Side-by-Side Comparison

### 1. Conditional ResNet

**Old (Clunky):**
```python
model = create_model("conditional_resnet", config, z_dim=8, x_dim=4)
#                                                      ^^^^^^ ^^^^^^ Actually used
```

**New (Clean):**
```python
model = create_model("conditional_resnet", config, z_dim=8, x_dim=4)
#                                                      ^^^^^^ ^^^^^^ Actually used
```

### 2. Encoder

**Old (Clunky):**
```python
encoder = create_model("encoder", config, z_dim=8, x_dim=4, input_dim=4, latent_shape=(8,))
#                                 ^^^^^^ ^^^^^^ Required but not used!
#                                 ^^^^^^ ^^^^^^ Actually used
```

**New (Clean):**
```python
encoder = create_model("encoder", config, input_dim=4, latent_shape=(8,))
#                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Only what's needed!
```

### 3. Decoder

**Old (Clunky):**
```python
decoder = create_model("decoder", config, z_dim=8, x_dim=4, input_dim=8, output_dim=10)
#                                 ^^^^^^ ^^^^^^ Required but not used!
#                                 ^^^^^^ ^^^^^^ Actually used
```

**New (Clean):**
```python
decoder = create_model("decoder", config, input_dim=8, output_dim=10)
#                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Only what's needed!
```

### 4. VAE Flow

**Old (Clunky):**
```python
vae_flow = create_vae_flow_model(
    main_config=main_config,
    crn_config=crn_config,
    encoder_config=encoder_config,
    decoder_config=decoder_config,
    z_dim=8,      # Actually used
    x_dim=4,      # Actually used
    output_dim=2  # Actually used
)
```

**New (Clean):**
```python
# Option 1: Direct factory call
vae_flow = create_model("vae_flow", main_config, z_dim=8, x_dim=4, output_dim=2)

# Option 2: Convenience function (even cleaner!)
vae_flow = create_vae_flow_with_defaults(z_dim=8, x_dim=4, output_dim=2)
```

## Key Benefits of the New Interface

### 1. **No More Redundancy**
- Each model only requires the parameters it actually uses
- No more passing `z_dim`/`x_dim` to models that don't need them

### 2. **Cleaner Code**
- More intuitive and readable
- Less confusing for new users
- Easier to understand what each model needs

### 3. **Better Error Messages**
- Clear error messages when required parameters are missing
- No more wondering why `z_dim`/`x_dim` are required

### 4. **Convenience Functions**
- `create_vae_flow_with_defaults()` for common cases
- `create_conditional_resnet_with_defaults()` for quick setup
- `create_encoder_with_defaults()` for simple encoders

### 5. **Consistent Interface**
- All models use the same pattern: `create_model(name, config, **kwargs)`
- Easy to extend with new model types
- Predictable parameter handling

## Migration Guide

### For Conditional ResNet Models
```python
# Old
model = create_model("conditional_resnet", config, z_dim=8, x_dim=4)

# New (same!)
model = create_model("conditional_resnet", config, z_dim=8, x_dim=4)
```

### For Encoder Models
```python
# Old
encoder = create_model("encoder", config, z_dim=8, x_dim=4, input_dim=4, latent_shape=(8,))

# New
encoder = create_model("encoder", config, input_dim=4, latent_shape=(8,))
```

### For Decoder Models
```python
# Old
decoder = create_model("decoder", config, z_dim=8, x_dim=4, input_dim=8, output_dim=10)

# New
decoder = create_model("decoder", config, input_dim=8, output_dim=10)
```

### For VAE Flow Models
```python
# Old
vae_flow = create_vae_flow_model(main_config, crn_config, encoder_config, decoder_config, 
                                z_dim=8, x_dim=4, output_dim=2)

# New
vae_flow = create_model("vae_flow", main_config, z_dim=8, x_dim=4, output_dim=2)

# Or even simpler
vae_flow = create_vae_flow_with_defaults(z_dim=8, x_dim=4, output_dim=2)
```

## Conclusion

The new interface is much cleaner, more intuitive, and eliminates the clunky redundancy of the old system. Each model only requires the parameters it actually uses, making the code more readable and maintainable.
