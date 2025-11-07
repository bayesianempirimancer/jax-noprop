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
- `--seed`: 42 (random seed)
- `--output_dir`: `./data` (output directory)
- `--filename`: `two_moons_xy_format.pkl` (output filename)
- `--train_ratio`: 0.80 (80% for training, 20% for validation)
- `--visualize`: Optional flag to display the dataset
- `--save_plot`: Optional flag to save a visualization plot

**Example with custom parameters:**
```bash
python examples/two_moons/generate_two_moons.py \
    --n_samples 10000 \
    --noise 0.1 \
    --seed 42 \
    --output_dir ./data \
    --filename two_moons_xy_format.pkl \
    --train_ratio 0.80 \
    --save_plot
```

The script will:
1. Generate the two moons dataset
2. Convert labels to one-hot encoding
3. Shuffle and split into train/validation sets
4. Save to `data/two_moons_xy_format.pkl`
5. Optionally create a visualization plot

### Step 2: Train Models

After generating the dataset, you can train models using the scripts in `src/flow_models/`:

#### Task 1: Regression/Classification

Train models to predict class labels (y) from coordinates (x) using `train.py`:

```bash
# Flow Matching
python -m src.flow_models.train --model_type flow_matching --data_path data/two_moons_xy_format.pkl

# Diffusion
python -m src.flow_models.train --model_type diffusion --data_path data/two_moons_xy_format.pkl

# CT (Continuous-Time)
python -m src.flow_models.train --model_type ct --data_path data/two_moons_xy_format.pkl
```

**Default parameters for regression/classification:**
- `--model_type`: `flow_matching` (choices: `flow_matching`, `diffusion`, `ct`)
- `--data_path`: `data/two_moons_formatted.pkl` (use `data/two_moons_xy_format.pkl` for this example)
- `--input_dim`: 2
- `--output_dim`: 2
- `--latent_dim`: 2
- `--crn_type`: `vanilla`
- `--network_type`: `mlp`
- `--hidden_dims`: `[32, 32, 32, 32, 32, 32]`
- `--num_epochs`: 50
- `--batch_size`: 256
- `--learning_rate`: 0.001
- `--optimizer`: `adam`
- `--recon_weight`: 1.0
- `--encoder_type`: `identity` (auto-selected for latent_dim=2)
- `--decoder_model`: `identity` (auto-selected)
- `--decoder_type`: `none` (auto-selected for latent_dim=2)
- `--seed`: 42

#### Task 2: Conditional Generation

Train models to generate coordinates (x) conditioned on class labels (y) using `train_gen.py`:

```bash
# Flow Matching
python -m src.flow_models.train_gen --model_type flow_matching --data_path data/two_moons_xy_format.pkl

# Diffusion
python -m src.flow_models.train_gen --model_type diffusion --data_path data/two_moons_xy_format.pkl

# CT
python -m src.flow_models.train_gen --model_type ct --data_path data/two_moons_xy_format.pkl
```

**Default parameters for conditional generation:**
- `--model_type`: `flow_matching`
- `--data_path`: `data/two_moons_formatted.pkl` (use `data/two_moons_xy_format.pkl` for this example)
- `--input_dim`: 2
- `--output_dim`: 2
- `--latent_dim`: 2
- `--num_epochs`: 50
- `--batch_size`: 256
- `--learning_rate`: 0.001
- `--recon_weight`: 1.0
- `--noise_schedule`: `exponential`
- `--noise_schedule_learnable`: `False`
- `--encoder_model_type`: `None` (auto: `identity` for latent_dim=2, `linear` for latent_dim>2)
- `--decoder_model_type`: `None` (auto: `identity`)
- `--decoder_type`: `None` (auto: `none` for latent_dim=2, `linear` for latent_dim>2)
- `--seed`: 42
- `--unconditional`: `False` (set to `True` for unconditional generation)

#### Task 3: Unconditional Generation

Train models to generate coordinates (x) without conditioning using `train_gen.py` with the `--unconditional` flag:

```bash
# Flow Matching
python -m src.flow_models.train_gen --model_type flow_matching --unconditional --data_path data/two_moons_xy_format.pkl

# Diffusion
python -m src.flow_models.train_gen --model_type diffusion --unconditional --data_path data/two_moons_xy_format.pkl

# CT
python -m src.flow_models.train_gen --model_type ct --unconditional --data_path data/two_moons_xy_format.pkl
```

## Complete Training Example

To train all three models on all three tasks sequentially:

```bash
# Regression/Classification
python -m src.flow_models.train --model_type flow_matching --data_path data/two_moons_xy_format.pkl
python -m src.flow_models.train --model_type diffusion --data_path data/two_moons_xy_format.pkl
python -m src.flow_models.train --model_type ct --data_path data/two_moons_xy_format.pkl

# Conditional Generation
python -m src.flow_models.train_gen --model_type flow_matching --data_path data/two_moons_xy_format.pkl
python -m src.flow_models.train_gen --model_type diffusion --data_path data/two_moons_xy_format.pkl
python -m src.flow_models.train_gen --model_type ct --data_path data/two_moons_xy_format.pkl

# Unconditional Generation
python -m src.flow_models.train_gen --model_type flow_matching --unconditional --data_path data/two_moons_xy_format.pkl
python -m src.flow_models.train_gen --model_type diffusion --unconditional --data_path data/two_moons_xy_format.pkl
python -m src.flow_models.train_gen --model_type ct --unconditional --data_path data/two_moons_xy_format.pkl
```

## Output Structure

### Data Generation Output

The data generation script saves:
- `data/two_moons_xy_format.pkl`: Dataset file with structure:
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

Both `train.py` and `train_gen.py` save results to timestamped directories in `artifacts/`:

**For `train.py` (regression/classification):**
- `artifacts/two_moons_YYYYMMDD_HHMMSS/{model_type}/`
  - `training_results.pkl`: Training history (losses, metrics)
  - `model_params.pkl`: Trained model parameters
  - `config.yaml`: Configuration used for training (human-readable)
  - `training_progress.png`: Loss trends plot
  - `data_visualization.png`: Data visualization
  - `trajectories.png`: Sample trajectories
  - `trajectory_diagnostics.png`: Trajectory diagnostics

**For `train_gen.py` (generation):**
- `artifacts/two_moons_YYYYMMDD_HHMMSS_gen/{model_type}/`
  - `training_results.pkl`: Training history
  - `model_params.pkl`: Trained model parameters
  - `config.yaml`: Configuration used for training (human-readable)
  - `loss_trends.png`: Loss trends plot
  - `conditional_generation.png` or `unconditional_generation.png`: Generation visualization
  - `latent_trajectories.png`: Latent space trajectories

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

# Change noise schedule (for diffusion/CT)
--noise_schedule linear

# Make noise schedule learnable
--noise_schedule_learnable

# Use different encoder/decoder types
--encoder_model_type mlp
--decoder_model_type mlp
```

### Example: Custom Training Run

```bash
python -m src.flow_models.train_gen \
    --model_type flow_matching \
    --unconditional \
    --data_path data/two_moons_xy_format.pkl \
    --num_epochs 200 \
    --learning_rate 0.0005 \
    --recon_weight 4.0 \
    --latent_dim 2 \
    --encoder_model_type identity \
    --batch_size 256 \
    --seed 42
```

## Notes

1. **Data Format**: The dataset uses one-hot encoded labels. Each label is a 2D vector: `[1, 0]` for class 0 and `[0, 1]` for class 1.

2. **Data Shuffling**: The data generation script always shuffles the data before splitting to ensure proper train/validation distribution.

3. **Default Encoder/Decoder**: With `latent_dim=2` (default), the models automatically use identity encoders and decoders. For `latent_dim>2`, linear encoders/decoders are used.

4. **Config Files**: Each training run saves a `config.yaml` file with the exact configuration used, making it easy to reproduce results.

5. **GPU Warnings**: You may see GPU autotuning warnings during training. These are harmless and don't affect the results.

## Troubleshooting

- **File not found**: Make sure to run commands from the project root directory.
- **Shape mismatches**: Ensure you're using `data/two_moons_xy_format.pkl` (not `data/two_moons_formatted.pkl`) for this example.
- **Import errors**: Make sure you're using the `numpyro` conda environment: `conda activate numpyro`

