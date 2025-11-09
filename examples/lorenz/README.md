# Lorenz System Example

This directory contains a complete example for training flow models (Flow Matching, Diffusion, and CT) on the Lorenz system dataset. The example demonstrates sequence modeling tasks: conditional generation (predicting future states from past states) and unconditional generation (generating full trajectories).

## Overview

The Lorenz system is a classic chaotic dynamical system defined by:
- dx/dt = σ(y - x)
- dy/dt = x(ρ - z) - y
- dz/dt = xy - βz

Where typically σ=10, ρ=28, β=8/3.

In this example:
- **x**: Input sequences of past states `(n_samples, input_seq_len, 3)` where each sequence contains `input_seq_len` time steps of 3D coordinates
- **y**: Output sequences of future states `(n_samples, output_seq_len, 3)` where each sequence contains `output_seq_len` time steps of 3D coordinates

The dataset is automatically split into training (80%) and validation (20%) sets with mandatory shuffling.

## Pipeline

### Step 1: Generate the Dataset

First, generate the Lorenz system dataset using the provided script:

```bash
python examples/lorenz/generate_lorenz.py
```

**Default parameters:**
- `--n_trajectories`: 1000 (number of trajectories to generate)
- `--trajectory_length`: 200 (length of each trajectory)
- `--input_seq_len`: 20 (length of input sequences)
- `--output_seq_len`: 20 (length of output sequences)
- `--stride`: 1 (stride for sliding window)
- `--sigma`: 10.0 (Lorenz parameter σ)
- `--rho`: 28.0 (Lorenz parameter ρ)
- `--beta`: 8.0/3.0 (Lorenz parameter β)
- `--noise`: 0.0 (Gaussian noise level)
- `--t_span`: [0.0, 20.0] (time span for integration)
- `--seed`: 42 (random seed)
- `--output_dir`: `./data` (output directory)
- `--filename`: `lorenz.pkl` (output filename)
- `--train_ratio`: 0.80 (80% for training, 20% for validation)
- `--visualize`: Optional flag to display the dataset
- `--save_plot`: Optional flag to save a visualization plot

**Example with custom parameters:**
```bash
python examples/lorenz/generate_lorenz.py \
    --n_trajectories 1000 \
    --trajectory_length 200 \
    --input_seq_len 20 \
    --output_seq_len 20 \
    --sigma 10.0 \
    --rho 28.0 \
    --beta 2.67 \
    --seed 42 \
    --output_dir ./data \
    --filename lorenz.pkl \
    --train_ratio 0.80 \
    --save_plot
```

The script will:
1. Generate multiple Lorenz trajectories with random initial conditions
2. Split each trajectory into input-output sequence pairs using a sliding window
3. Shuffle and split into train/validation sets
4. Save to `data/lorenz.pkl`
5. Optionally create a visualization plot

### Step 2: Train Models

After generating the dataset, you can train models using `train_seq.py`:

#### Task 1: Conditional Generation

Train models to generate future states (y) conditioned on past states (x):

```bash
# Flow Matching
python -m src.flow_models.train_seq --config_file examples/lorenz/config.yaml --data_path data/lorenz.pkl --model_type flow_matching

# Diffusion
python -m src.flow_models.train_seq --config_file examples/lorenz/config.yaml --data_path data/lorenz.pkl --model_type diffusion

# CT (Continuous-Time)
python -m src.flow_models.train_seq --config_file examples/lorenz/config.yaml --data_path data/lorenz.pkl --model_type ct
```

**Note:** You can also use a custom config class by specifying `--config_class examples.lorenz.config.Config`. This can be used:
- With `--config_file` to load YAML using the custom class
- Without `--config_file` to use the class's default values directly

**Default parameters for conditional generation:**
- `--config_file`: `examples/lorenz/config.yaml` (YAML config file with all default parameters)
- `--model_type`: `flow_matching` (choices: `flow_matching`, `diffusion`, `ct`)
- `--data_path`: `data/lorenz.pkl` (required)
- `--num_epochs`: 50
- `--batch_size`: 32
- `--learning_rate`: 0.001
- `--seed`: 42
- `--warmup_steps`: 0 (number of training steps for learning rate warmup)
- `--warmup_epochs`: `None` (number of epochs for warmup, overrides `--warmup_steps` if provided)

**Shape arguments:** For sequences, shapes are specified as `(seq_len, embed_dim)`. The config file sets `input_shape=(20, 3)`, `output_shape=(20, 3)`, and `latent_shape=(20, 3)`. You can override these using `--input_shape`, `--output_shape`, `--latent_shape` or their component arguments (`--x_seq_len`, `--z_seq_len`, `--embed_dim`).

All other parameters (architecture, loss weights, noise schedules, etc.) are loaded from the config file but can be overridden via command-line arguments.

#### Task 2: Unconditional Generation

Train models to generate full trajectories without conditioning using `train_seq.py` with the `--unconditional` flag:

```bash
# Flow Matching
python -m src.flow_models.train_seq --config_file examples/lorenz/config.yaml --data_path data/lorenz.pkl --model_type flow_matching --unconditional

# Diffusion
python -m src.flow_models.train_seq --config_file examples/lorenz/config.yaml --data_path data/lorenz.pkl --model_type diffusion --unconditional

# CT
python -m src.flow_models.train_seq --config_file examples/lorenz/config.yaml --data_path data/lorenz.pkl --model_type ct --unconditional
```

## Complete Training Example

To train all three models on conditional generation sequentially:

```bash
# Conditional Generation
python -m src.flow_models.train_seq --config_file examples/lorenz/config.yaml --data_path data/lorenz.pkl --model_type flow_matching
python -m src.flow_models.train_seq --config_file examples/lorenz/config.yaml --data_path data/lorenz.pkl --model_type diffusion
python -m src.flow_models.train_seq --config_file examples/lorenz/config.yaml --data_path data/lorenz.pkl --model_type ct
```

## Output Structure

### Data Generation Output

The data generation script saves:
- `data/lorenz.pkl`: Dataset file with structure:
  ```python
  {
      'train': {
          'x': (n_train, 20, 3),  # Input sequences (past states)
          'y': (n_train, 20, 3)   # Output sequences (future states)
      },
      'val': {
          'x': (n_val, 20, 3),    # Input sequences (past states)
          'y': (n_val, 20, 3)     # Output sequences (future states)
      }
  }
  ```
- `data/lorenz_visualization.png`: Optional visualization plot showing sample trajectories

### Training Output

`train_seq.py` saves results to timestamped directories in `artifacts/` with the structure `artifacts/{model_type}_seq/{YYYYMMDD_HHMM}/`:

**For conditional generation:**
- `artifacts/{model_type}_seq/{YYYYMMDD_HHMM}/`
  - `history.pkl`: Training history
  - `params.pkl`: Trained model parameters
  - `config.yaml`: Configuration used for training (human-readable, with hierarchical key ordering)
  - `loss_trends.png`: Loss trends plot
  - `sequence_comparison.png`: Sequence generation visualization
  - `latent_trajectories.png`: Latent space trajectories

**For unconditional generation:**
- `artifacts/{model_type}_seq/{YYYYMMDD_HHMM}/`
  - Same structure as conditional generation

**Examples:**
- `artifacts/flow_matching_seq/20251109_1200/` - Flow matching sequence modeling
- `artifacts/diffusion_seq/20251109_1201/` - Diffusion sequence modeling
- `artifacts/ct_seq/20251109_1202/` - CT sequence modeling

## Customizing Training

### Common Parameters

```bash
# Increase training epochs
--num_epochs 200

# Adjust learning rate
--learning_rate 0.0005

# Change reconstruction weight
--recon_weight 4.0

# Use different sequence lengths
--input_seq_len 30
--output_seq_len 30
# Or use explicit shapes
--input_shape 30 3
--output_shape 30 3

# Change noise schedule (for diffusion/CT)
--noise_schedule linear

# Make noise schedule learnable
--noise_schedule_learnable

# Use different encoder/decoder types
--encoder_model_type mlp
--decoder_model_type mlp

# Adjust transformer parameters (for transformer_seq2seq CRN)
--num_layers 6
--num_heads 8
--mlp_ratio 4.0

# Add learning rate warmup (in steps or epochs)
--warmup_steps 100
# Or specify warmup in epochs (overrides warmup_steps)
--warmup_epochs 2.0
```

### Example: Custom Training Run

```bash
python -m src.flow_models.train_seq \
    --config_file examples/lorenz/config.yaml \
    --model_type flow_matching \
    --data_path data/lorenz.pkl \
    --num_epochs 200 \
    --learning_rate 0.0005 \
    --recon_weight 4.0 \
    --input_seq_len 20 \
    --output_seq_len 20 \
    --batch_size 32 \
    --seed 42
```

## Configuration File

This directory contains both `config.py` (Python config class) and `config.yaml` (YAML config file) that provide default configuration values for all model parameters. The configuration system works the same way as the two_moons example:

### Option 1: YAML Config File (Recommended)

```bash
python -m src.flow_models.train_seq --config_file examples/lorenz/config.yaml --data_path data/lorenz.pkl --model_type flow_matching
```

### Option 2: Custom Config Class

**With YAML file:**
```bash
python -m src.flow_models.train_seq \
    --config_file examples/lorenz/config.yaml \
    --config_class examples.lorenz.config.Config \
    --data_path data/lorenz.pkl \
    --model_type flow_matching
```

**Python class only:**
```bash
python -m src.flow_models.train_seq \
    --config_class examples.lorenz.config.Config \
    --data_path data/lorenz.pkl \
    --model_type flow_matching
```

### Option 3: Command Line Arguments

Many common parameters can be overridden using command line flags. See the two_moons README for a complete list of available overrides.

## Notes

1. **Data Format**: The dataset uses 3D sequences (x, y, z coordinates) from the Lorenz system. Each sequence represents a time series of states.

2. **Sequence Splitting**: The data generation script uses a sliding window approach to create input-output pairs from full trajectories. The `--stride` parameter controls the overlap between windows.

3. **Transformer Architecture**: The default config uses `transformer_seq2seq` for the CRN, which is well-suited for sequence modeling tasks. You can override this with `--crn_type` and `--network_type`.

4. **Config Files**: Each training run saves a `config.yaml` file with the exact configuration used, making it easy to reproduce results. The config file is saved in the same directory as the training results.

5. **Learning Rate Warmup**: `train_seq.py` supports learning rate warmup via `--warmup_steps` or `--warmup_epochs`. This helps stabilize early training iterations.

6. **Directory Structure**: Results are saved to `artifacts/{model_type}_seq/{YYYYMMDD_HHMM}/` where:
   - `{model_type}` is one of: `flow_matching`, `diffusion`, `ct`
   - `{YYYYMMDD_HHMM}` is a timestamp (year, month, day, hour, minute)

7. **GPU Warnings**: You may see GPU autotuning warnings during training. These are harmless and don't affect the results.

## Troubleshooting

- **File not found**: Make sure to run commands from the project root directory.
- **Shape mismatches**: Ensure you're using `data/lorenz.pkl` for this example and that sequence lengths match your config.
- **Import errors**: Make sure you're using the `numpyro` conda environment: `conda activate numpyro` or use `conda run -n numpyro` before your command.
- **Config file not found**: Make sure the path to `config.yaml` is correct relative to the project root, or use an absolute path.
- **Sequence length errors**: Make sure `input_seq_len` and `output_seq_len` match the shapes in your config file or are specified via command-line arguments.

