# Two Moons Dataset Example

This directory contains a complete example for training flow models (Flow Matching, Diffusion, and CT) on the two moons dataset. The example demonstrates three different tasks: regression/classification, conditional generation, and unconditional generation.

## Overview

The two moons dataset is a classic synthetic dataset for binary classification. In this example:
- **x**: 2D coordinates `(n_samples, 2)`
- **y**: One-hot encoded class labels `(n_samples, 2)` where each row is either `[1, 0]` or `[0, 1]`

The dataset is automatically split into training (80%) and validation (20%) sets with mandatory shuffling.

## Pipeline

### Step 1: Generate the Dataset

First, generate the two moons dataset using the provided script:

```bash
python examples/two_moons/generate_two_moons.py
```

**Default parameters:**
- `--n_samples`: 10000 (total samples)
- `--noise`: 0.1 (Gaussian noise level)
- `--scale_factor`: 10.0 (scale factor to multiply x coordinates by)
- `--center`: `True` (center the data after scaling)
- `--no_center`: Optional flag to disable centering
- `--seed`: 42 (random seed)
- `--output_dir`: `./data` (output directory)
- `--filename`: `two_moons.pkl` (output filename)
- `--train_ratio`: 0.80 (80% for training, 20% for validation)
- `--visualize`: Optional flag to display the dataset
- `--save_plot`: Optional flag to save a visualization plot

**Example with custom parameters:**
```bash
python examples/two_moons/generate_two_moons.py \
    --n_samples 10000 \
    --noise 0.1 \
    --scale_factor 10.0 \
    --seed 42 \
    --output_dir ./data \
    --filename two_moons.pkl \
    --train_ratio 0.80 \
    --save_plot
```

**Note:** The `--scale_factor` parameter controls the scale of the dataset. After scaling, the data is centered by default (unless `--no_center` is specified). This is useful for testing model performance at different scales.

The script will:
1. Generate the two moons dataset
2. Convert labels to one-hot encoding
3. Shuffle and split into train/validation sets
4. Save to `data/two_moons.pkl`
5. Optionally create a visualization plot

### Step 2: Train Models

After generating the dataset, you can train models using the scripts in `src/flow_models/`:

#### Task 1: Regression/Classification

Train models to predict class labels (y) from coordinates (x) using `train.py`:

```bash
# Flow Matching
python -m src.flow_models.train --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type flow_matching

# Diffusion
python -m src.flow_models.train --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type diffusion

# CT (Continuous-Time)
python -m src.flow_models.train --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type ct
```

**Note:** You can also use a custom config class by specifying `--config_class examples.two_moons.config.Config`. This can be used:
- With `--config_file` to load YAML using the custom class
- Without `--config_file` to use the class's default values directly

**Default parameters for regression/classification:**
- `--config_file`: `examples/two_moons/config.yaml` (YAML config file with all default parameters)
- `--model_type`: `flow_matching` (choices: `flow_matching`, `diffusion`, `ct`)
- `--data_path`: `data/two_moons.pkl` (required)
- `--num_epochs`: 50
- `--batch_size`: 256
- `--learning_rate`: 0.001
- `--optimizer`: `adam`
- `--seed`: 42
- `--warmup_steps`: 0 (number of training steps for learning rate warmup)
- `--warmup_epochs`: `None` (number of epochs for warmup, overrides `--warmup_steps` if provided)

**Shape arguments:** You can specify shapes using either `--input_dim`, `--output_dim`, `--latent_dim` (converted to `(dim,)` tuples) or `--input_shape`, `--output_shape`, `--latent_shape` (explicit tuples). If neither config file nor shape arguments are provided, you must specify at least `--input_shape`, `--output_shape`, and `--latent_shape` via command-line arguments.

All other parameters (shapes, architecture, loss weights, etc.) are loaded from the config file but can be overridden via command-line arguments.

#### Task 2: Conditional Generation

Train models to generate coordinates (x) conditioned on class labels (y) using `train_gen.py`:

```bash
# Flow Matching
python -m src.flow_models.train_gen --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type flow_matching

# Diffusion
python -m src.flow_models.train_gen --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type diffusion

# CT
python -m src.flow_models.train_gen --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type ct
```

**Default parameters for conditional generation:**
- `--config_file`: `examples/two_moons/config.yaml` (YAML config file with all default parameters)
- `--model_type`: `flow_matching` (choices: `flow_matching`, `diffusion`, `ct`)
- `--data_path`: `data/two_moons.pkl` (required)
- `--num_epochs`: 50
- `--batch_size`: 256
- `--learning_rate`: 0.001
- `--seed`: 42
- `--unconditional`: `False` (set to `True` for unconditional generation)
- `--warmup_steps`: 0 (number of training steps for learning rate warmup)
- `--warmup_epochs`: `None` (number of epochs for warmup, overrides `--warmup_steps` if provided)

**Shape arguments:** For conditional generation, the config file's `input_shape` and `output_shape` are automatically reversed (y→x generation). For unconditional generation, `input_shape` is set to `()` (empty tuple) and `output_shape` is taken from the config. You can override shapes using `--input_shape`, `--output_shape`, `--latent_shape` or their `_dim` equivalents.

All other parameters (shapes, architecture, loss weights, noise schedules, etc.) are loaded from the config file but can be overridden via command-line arguments.

#### Task 3: Unconditional Generation

Train models to generate coordinates (x) without conditioning using `train_gen.py` with the `--unconditional` flag:

```bash
# Flow Matching
python -m src.flow_models.train_gen --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type flow_matching --unconditional

# Diffusion
python -m src.flow_models.train_gen --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type diffusion --unconditional

# CT
python -m src.flow_models.train_gen --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type ct --unconditional
```

## Complete Training Example

To train all three models on all three tasks sequentially:

```bash
# Regression/Classification
python -m src.flow_models.train --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type flow_matching
python -m src.flow_models.train --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type diffusion
python -m src.flow_models.train --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type ct

# Conditional Generation
python -m src.flow_models.train_gen --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type flow_matching
python -m src.flow_models.train_gen --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type diffusion
python -m src.flow_models.train_gen --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type ct

# Unconditional Generation
python -m src.flow_models.train_gen --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type flow_matching --unconditional
python -m src.flow_models.train_gen --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type diffusion --unconditional
python -m src.flow_models.train_gen --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type ct --unconditional
```

## Output Structure

### Data Generation Output

The data generation script saves:
- `data/two_moons.pkl`: Dataset file with structure:
  ```python
  {
      'train': {
          'x': (8000, 2),  # Coordinates
          'y': (8000, 2)   # One-hot encoded labels
      },
      'val': {
          'x': (2000, 2),  # Coordinates
          'y': (2000, 2)   # One-hot encoded labels
      }
  }
  ```
- `data/two_moons_visualization.png`: Optional visualization plot

### Training Output

Both `train.py` and `train_gen.py` save results to timestamped directories in `artifacts/` with the structure `artifacts/{model_type}_{task}/{YYYYMMDD_HHMM}/`:

**For `train.py` (regression/classification):**
- `artifacts/{model_type}_reg/{YYYYMMDD_HHMM}/`
  - `history.pkl`: Training history (losses, metrics)
  - `params.pkl`: Trained model parameters
  - `config.yaml`: Configuration used for training (human-readable, with hierarchical key ordering)
  - `loss_trends.png`: Loss trends plot (2x3 layout)
  - `data_visualization.png`: Data visualization
  - `predictions.png`: Model predictions visualization
  - `trajectories.png`: Sample trajectories
  - `trajectory_diagnostics.png`: Trajectory diagnostics

**For `train_gen.py` (conditional generation):**
- `artifacts/{model_type}_gen/{YYYYMMDD_HHMM}/`
  - `history.pkl`: Training history
  - `params.pkl`: Trained model parameters
  - `config.yaml`: Configuration used for training (human-readable, with hierarchical key ordering)
  - `loss_trends.png`: Loss trends plot (2x3 layout: Total Loss, Flow Loss, Reconstruction Loss, Regularization Loss, Chamfer Distance, Empty)
  - `conditional_generation.png`: Generation visualization
  - `latent_trajectories.png`: Latent space trajectories

**For `train_gen.py` (unconditional generation):**
- `artifacts/{model_type}_uncond_gen/{YYYYMMDD_HHMM}/`
  - `history.pkl`: Training history
  - `params.pkl`: Trained model parameters
  - `config.yaml`: Configuration used for training (human-readable, with hierarchical key ordering)
  - `loss_trends.png`: Loss trends plot (2x3 layout)
  - `unconditional_generation.png`: Generation visualization
  - `latent_trajectories.png`: Latent space trajectories

**Examples:**
- `artifacts/flow_matching_reg/20251108_1042/` - Flow matching regression
- `artifacts/diffusion_gen/20251108_1043/` - Diffusion conditional generation
- `artifacts/ct_uncond_gen/20251108_1102/` - CT unconditional generation

## Customizing Training

### Common Parameters

Both scripts support many customization options. Here are some commonly modified parameters:

```bash
# Increase training epochs
--num_epochs 200

# Adjust learning rate
--learning_rate 0.0005

# Change reconstruction weight
--recon_weight 4.0

# Use different latent dimension (requires linear encoder/decoder)
--latent_dim 8
# Or use explicit shape
--latent_shape 8

# Change noise schedule (for diffusion/CT)
--noise_schedule linear

# Make noise schedule learnable
--noise_schedule_learnable

# Use different encoder/decoder types
--encoder_model_type mlp
--decoder_model_type mlp

# Add learning rate warmup (in steps or epochs)
--warmup_steps 100
# Or specify warmup in epochs (overrides warmup_steps)
--warmup_epochs 2.0
```

### Example: Custom Training Run

```bash
python -m src.flow_models.train_gen \
    --config_file examples/two_moons/config.yaml \
    --model_type flow_matching \
    --unconditional \
    --data_path data/two_moons.pkl \
    --num_epochs 200 \
    --learning_rate 0.0005 \
    --recon_weight 4.0 \
    --latent_dim 2 \
    --encoder_model_type identity \
    --batch_size 256 \
    --seed 42
```

## Configuration File

This directory contains both `config.py` (Python config class) and `config.yaml` (YAML config file) that provide default configuration values for all model parameters. You have three options for customizing these parameters:

### Option 1: YAML Config File (Recommended)

The easiest way to use configuration is via the YAML file:

```bash
python -m src.flow_models.train --config_file examples/two_moons/config.yaml --data_path data/two_moons.pkl --model_type flow_matching
```

The YAML file contains all default parameters in a human-readable format. The config loading system automatically merges the YAML values with default values from the Config class, ensuring all parameters are present even if not explicitly specified in the YAML. You can edit `config.yaml` directly to change defaults, and command-line arguments will override any values in the config file.

**Note:** Saved config files maintain a hierarchical key order (`model_name`, `main`, `noise_schedule`, `encoder`, `decoder`, `crn`) for better readability.

### Option 2: Custom Config Class

If you want to use a custom config class (like the one in `config.py`), you can specify it:

**Option 2a: With YAML file (loads YAML using custom class)**
```bash
python -m src.flow_models.train \
    --config_file examples/two_moons/config.yaml \
    --config_class examples.two_moons.config.Config \
    --data_path data/two_moons.pkl \
    --model_type flow_matching
```

**Option 2b: Python class only (uses default values from the class)**
```bash
python -m src.flow_models.train \
    --config_class examples.two_moons.config.Config \
    --data_path data/two_moons.pkl \
    --model_type flow_matching
```

When using `--config_class` without `--config_file`, the config class is instantiated with its default values. You can still override any parameters via command-line arguments.

### Option 3: Command Line Arguments (Quick Overrides)

Many common parameters can be overridden using command line flags. This is convenient for quick experiments and parameter sweeps. Command line arguments **take precedence** over config file values.

**Available command line overrides:**
- `--recon_weight`: Override reconstruction loss weight
- `--reg_weight`: Override regularization weight
- `--use_snr_weight`: Override SNR weighting (True/False)
- `--integration_method`: Override ODE integration method (`euler`, `midpoint`, `heun`, `rk4`, etc.)
- `--noise_schedule`: Override noise schedule type (`linear`, `exponential`, `sigmoid`, `cosine`, etc.)
- `--noise_schedule_learnable`: Toggle learnable noise schedule
- `--crn_type`: Override CRN type (`vanilla`, `geometric`, `potential`, etc.)
- `--network_type`: Override network backbone (`mlp`, `bilinear`, `convex`)
- `--hidden_dims`: Override hidden layer dimensions
- `--encoder_model_type`: Override encoder type (`identity`, `linear`, `mlp`, `mlp_normal`, `resnet`, `resnet_normal`)
- `--decoder_model_type`: Override decoder model type (`identity`, `mlp`, `resnet`)
- `--decoder_type`: Override decoder type (`linear`, `softmax`, `none`)
- `--warmup_steps`: Number of training steps for learning rate warmup
- `--warmup_epochs`: Number of epochs for warmup (overrides `--warmup_steps`)
- And many more (see `--help` for each training script)

**Example:**
```bash
# Override just a few parameters via command line
python -m src.flow_models.train \
    --config_file examples/two_moons/config.yaml \
    --model_type flow_matching \
    --data_path data/two_moons.pkl \
    --recon_weight 4.0 \
    --noise_schedule linear
```

### Option 4: Direct Config File Editing (Finer-Grained Control)

For more comprehensive control or to modify parameters not exposed via command line, you can directly edit `examples/two_moons/config.yaml` or `examples/two_moons/config.py`. This gives you access to **all** configuration options, including:

- **Main configuration** (`main` dict):
  - Data shapes (input_shape, output_shape, latent_shape)
  - Loss types and weights (recon_loss_type, recon_weight, reg_weight)
  - Flow model settings (use_snr_weight, integration_method, sigma)
  
- **Noise schedule** (`noise_schedule` dict):
  - Schedule type and learnability
  - Hidden dimensions for learnable schedules
  - Detailed parameters for different schedule types (alpha_bar_min/max, s, k, beta, etc.)

- **CRN network** (`crn` dict):
  - Model and network types
  - Hidden dimensions
  - Time embedding configuration
  - Activation functions, batch norm, dropout

- **Encoder/Decoder** (`encoder` and `decoder` dicts):
  - Model types and architectures
  - Hidden dimensions for MLP variants
  - Activation functions and dropout rates

**Example: Editing config.py**

```python
# In examples/two_moons/config.py, modify the main configuration:
main: FrozenDict = field(default_factory=lambda: FrozenDict({
    "input_shape": (2,),
    "output_shape": (2,),
    "latent_shape": (2,),
    "recon_loss_type": "mse",
    "recon_weight": 4.0,  # Changed from 1.0
    "reg_weight": 0.1,     # Changed from 0.0
    # ... other parameters
}))

# Or modify CRN hidden dimensions:
crn: FrozenDict = field(default_factory=lambda: FrozenDict({
    "model_type": "vanilla",
    "network_type": "mlp",
    "hidden_dims": (64, 64, 64, 64),  # Changed from (32, 32, 32, 32, 32, 32)
    # ... other parameters
}))
```

**Important Notes:**
- YAML config files are recommended for most use cases as they're human-readable and easy to edit
- The config loading system automatically merges YAML values with default values from the Config class, ensuring all parameters are present even if not explicitly specified in the YAML
- The Python config class (`config.py`) uses `FrozenDict` from Flax, which means values are immutable at runtime
- After editing a config file, restart your training script to use the new values
- Command line arguments will always override config file values if both are specified
- Saved config files maintain a hierarchical key order for better readability
- See the comments in `config.py` and the structure of `config.yaml` for detailed explanations of each parameter

### Which Method Should I Use?

- **Use YAML config file** (recommended) when:
  - You want a simple, human-readable configuration
  - You want to set comprehensive defaults for all your experiments
  - You're working with the standard unified Config class

- **Use custom config class** when:
  - You need custom logic or validation in your config
  - You want to extend the base Config class with additional functionality
  - You prefer Python code over YAML for configuration
  - You want to use the class defaults directly (without a YAML file) by specifying only `--config_class`

- **Use command line arguments** when:
  - You want to quickly test different values for common parameters
  - You're running parameter sweeps or hyperparameter optimization
  - You want to override just a few parameters without modifying config files

- **Edit config files directly** when:
  - You need to modify parameters not exposed via command line
  - You need fine-grained control over noise schedule parameters, activation functions, or other advanced options

## Notes

1. **Data Format**: The dataset uses one-hot encoded labels. Each label is a 2D vector: `[1, 0]` for class 0 and `[0, 1]` for class 1.

2. **Data Shuffling**: The data generation script always shuffles the data before splitting to ensure proper train/validation distribution.

3. **Default Encoder/Decoder**: With `latent_dim=2` (default), the models automatically use identity encoders and decoders. For `latent_dim>2`, linear encoders/decoders are used.

4. **Config Files**: Each training run saves a `config.yaml` file with the exact configuration used, making it easy to reproduce results. The config file is saved in the same directory as the training results. The saved config includes all parameters (merged with defaults) and maintains a hierarchical key order for readability.

5. **Learning Rate Warmup**: Both `train.py` and `train_gen.py` support learning rate warmup via `--warmup_steps` or `--warmup_epochs`. This helps stabilize early training iterations. Warmup uses a linear schedule from 0 to the specified learning rate over the warmup period.

6. **Directory Structure**: Results are saved to `artifacts/{model_type}_{task}/{YYYYMMDD_HHMM}/` where:
   - `{model_type}` is one of: `flow_matching`, `diffusion`, `ct`
   - `{task}` is one of: `reg` (regression), `gen` (conditional generation), `uncond_gen` (unconditional generation)
   - `{YYYYMMDD_HHMM}` is a timestamp (year, month, day, hour, minute)

7. **Shape vs Dimension Arguments**: You can specify shapes using either `--input_dim`, `--output_dim`, `--latent_dim` (which are converted to `(dim,)` tuples) or `--input_shape`, `--output_shape`, `--latent_shape` (explicit tuples). For unconditional generation, `input_shape` is automatically set to `()` (empty tuple).

8. **Data Scaling**: The `generate_two_moons.py` script supports `--scale_factor` to scale the dataset and `--no_center` to disable automatic centering. This is useful for testing model performance at different scales.

9. **GPU Warnings**: You may see GPU autotuning warnings during training. These are harmless and don't affect the results.

## Troubleshooting

- **File not found**: Make sure to run commands from the project root directory.
- **Shape mismatches**: Ensure you're using `data/two_moons.pkl` for this example.
- **Import errors**: Make sure you're using the `numpyro` conda environment: `conda activate numpyro` or use `conda run -n numpyro` before your command.
- **Config file not found**: Make sure the path to `config.yaml` is correct relative to the project root, or use an absolute path.
- **No config file**: If you don't provide a config file, you must specify `--input_shape`, `--output_shape`, and `--latent_shape` (or their `_dim` equivalents) via command-line arguments. The encoder and decoder will automatically use the shapes specified in the main config.
- **Shape errors**: Make sure you're using the correct shape format. Use `--input_dim 2` (converted to `(2,)`) or `--input_shape 2` (explicit tuple). For unconditional generation, `input_shape` is automatically set to `()`.
- **Warmup issues**: If warmup seems to have a negative effect, try reducing the number of warmup steps/epochs or disabling it entirely with `--warmup_steps 0`.

