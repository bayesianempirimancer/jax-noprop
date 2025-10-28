# Factory Function Dimension Parameters Guide

This guide explains which dimension parameters are actually used by different model types in our factory functions.

## Dimension Parameters

- **`z_dim`**: Latent space dimension (used by flow models and VAE)
- **`x_dim`**: Conditional input dimension (used by flow models and VAE)  
- **`input_dim`**: Data input dimension (used by encoder/decoder)
- **`output_dim`**: Data output dimension (used by decoder)
- **`latent_shape`**: Latent space shape tuple (used by encoder)

## Model-Specific Usage

### 1. Conditional ResNet Models
**Uses**: `z_dim`, `x_dim`
```python
model = create_model("conditional_resnet", config, z_dim=8, x_dim=4)
# z_dim: latent space dimension
# x_dim: conditional input dimension
```

### 2. Encoder Models  
**Uses**: `input_dim`, `latent_shape`
```python
encoder = create_model("encoder", config, z_dim=8, x_dim=4, 
                      input_dim=4, latent_shape=(8,))
# input_dim: data input dimension (e.g., image pixels)
# latent_shape: latent output shape (e.g., (8,) for 8D latent)
# Note: z_dim/x_dim are required by factory but not used by encoder
```

### 3. Decoder Models
**Uses**: `input_dim`, `output_dim`
```python
decoder = create_model("decoder", config, z_dim=8, x_dim=4,
                      input_dim=8, output_dim=10)
# input_dim: latent input dimension (e.g., 8D latent space)
# output_dim: data output dimension (e.g., 10 classes)
# Note: z_dim/x_dim are required by factory but not used by decoder
```

### 4. VAE Flow Models
**Uses**: `z_dim`, `x_dim`, `output_dim`
```python
vae_flow = create_vae_flow_model(
    main_config=main_config,
    crn_config=crn_config, 
    encoder_config=encoder_config,
    decoder_config=decoder_config,
    z_dim=8,      # Latent space dimension
    x_dim=4,      # Conditional input dimension
    output_dim=2  # Output classes
)
# z_dim: latent space dimension
# x_dim: conditional input dimension  
# output_dim: output classes
```

## Key Points

1. **`z_dim` and `x_dim`** are required by the factory interface for consistency, but not all model types use them.

2. **`input_dim` and `output_dim`** are the actual dimensions used by encoder/decoder models.

3. **`latent_shape`** is used by encoders to specify the shape of the latent representation.

4. The factory functions automatically set the correct shapes in the model configs based on these parameters.

## Redundancy Explanation

You might notice that some examples pass both `z_dim` and `input_dim` with the same value. This is because:

- `z_dim` is required by the factory interface for consistency
- `input_dim` is the actual parameter used by the specific model (encoder/decoder)
- For VAE flow models, `z_dim` is actually used and `input_dim` is not needed

The factory functions handle this internally by using the appropriate parameters for each model type.
