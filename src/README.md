# Source Code Documentation

Complete implementation of MACE-PINN architecture with training scripts, utilities, and analysis tools. This directory contains all production code used in the thesis research.

---

## Installation

### System Requirements

**Hardware:**
- GPU: NVIDIA GPU with CUDA 12.0+ (A100/V100 recommended)
- RAM: 16GB minimum, 32GB recommended
- Storage: 5GB for code + data

**Software:**
- Python 3.8 or higher
- CUDA 12.0 or higher
- cuDNN 8.9 or higher

---

### Quick Install
```bash
# Clone repository
git clone https://github.com/rushirb2001/thesis-mace-pinn.git
cd thesis-mace-pinn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import jax; print(f'JAX devices: {jax.devices()}')"
```

---

### Dependencies
```txt
# Core ML Stack
jax[cuda12]>=0.4.20
jaxlib>=0.4.20
flax>=0.7.0
optax>=0.1.7

# Scientific Computing
numpy>=1.24.0
scipy>=1.10.0
h5py>=3.8.0

# Visualization
matplotlib>=3.7.0
imageio>=2.31.0
pillow>=10.0.0

# Experiment Tracking
wandb>=0.15.0
tqdm>=4.65.0

# Data Generation (optional)
# MATLAB R2021a+ with Chebfun
```

Save as `requirements.txt` in root directory.

---

## Project Structure
```
src/
├── models/                 # Training scripts
│   ├── train.py           # Base training loop
│   ├── train_final.py     # Gray-Scott production
│   ├── train_gls.py       # Ginzburg-Landau
│   ├── train_mlp.py       # MLP variant
│   ├── train_mlp_gls.py   # MLP for GLS
│   └── train_gated.py     # Gated architecture
├── config/                 # Experiment configurations
│   ├── config_greyscott.py
│   └── config_gls.py
├── visualization/          # Plotting utilities
│   ├── generate_gifs.py
│   ├── plot_timeseries.py
│   └── gif_generator.py
├── processing/             # Data processing
│   ├── create_datasets.py
│   └── create_datasets_v2.py
├── data_generation/        # MATLAB scripts
│   ├── gen_grey_scott.m
│   ├── data_gen_grey_scott_square.m
│   └── data_gen_ginz_lan_square.m
├── legacy/                 # Development history
└── tests/                  # Unit tests
```

---

## Usage

### Training Gray-Scott

#### Quick Start
```bash
# Train with default configuration
python src/models/train_final.py

# Train specific variation
python src/models/train_final.py --variation 5 --epochs 50000

# Enable WandB logging
python src/models/train_final.py --wandb-project "my-pinn" --wandb-run "gs-var5"
```

#### Advanced Configuration
```bash
# Custom hyperparameters
python src/models/train_final.py \
    --variation 5 \
    --epochs 50000 \
    --learning-rate 1e-3 \
    --colloc-points 10000 \
    --data-fraction 0.05 \
    --batch-size 256 \
    --save-dir ./checkpoints/gs_var5
```

---

### Training Ginzburg-Landau
```bash
# Train on GLS system
python src/models/train_gls.py --variation 2 --epochs 50000

# With custom architecture
python src/models/train_mlp_gls.py \
    --u-layers 64 64 64 \
    --v-layers 128 128 128 128 \
    --fourier-scale 2.0
```

---

### Configuration Files

Edit `src/config/config_greyscott.py` for Gray-Scott:
```python
config = {
    # Data
    'data_path': 'data/input/mat/grey_scott_variation5.mat',
    'variation': 5,
    
    # Architecture
    'u_layers': [64, 64, 64, 1],
    'v_layers': [128, 128, 128, 128, 1],
    'fourier_features': 32,
    'fourier_scale': 2.0,
    
    # Training
    'epochs': 50000,
    'learning_rate': 1e-3,
    'decay_steps': 10000,
    'decay_rate': 0.95,
    
    # Loss weights (initial)
    'lambda_ic_u': 1.0,
    'lambda_res_u': 100.0,
    'lambda_data_u': 1.0,
    'lambda_ic_v': 1.0,
    'lambda_res_v': 100.0,
    'lambda_data_v': 1.0,
    
    # Adaptive weighting
    'alpha': 0.2,          # EMA smoothing
    'update_freq': 1000,   # Weight update frequency
    
    # Sampling
    'colloc_points': 10000,
    'data_fraction': 0.05,  # 5% supervised
    
    # Logging
    'log_interval': 100,
    'save_interval': 5000,
    'wandb_project': 'mace-pinn-thesis',
}
```

---

### Generating Visualizations

#### Create GIFs
```bash
# Generate all visualizations
python src/visualization/generate_gifs.py

# Specific variation
python src/visualization/generate_gifs.py --variation 5 --output data/output/gif/

# Custom settings
python src/visualization/generate_gifs.py \
    --variation 5 \
    --fps 15 \
    --dpi 150 \
    --show-colorbar
```

#### Plot Time Series
```bash
# Create comparison plots
python src/visualization/plot_timeseries.py \
    --ground-truth data/input/mat/grey_scott_variation5.mat \
    --prediction checkpoints/gs_var5/final.pkl \
    --output analysis/comparison.png
```

---

### Data Processing

#### Create H5 Datasets
```bash
# Process all .mat files to H5
python src/processing/create_datasets.py \
    --input data/input/mat/ \
    --output data/processed/h5/combined.h5 \
    --train-split 0.7 \
    --val-split 0.15
```

#### Generate New Simulation Data
```bash
# Run MATLAB data generation
cd src/data_generation
matlab -batch "run('data_gen_grey_scott_square.m')"
matlab -batch "run('data_gen_ginz_lan_square.m')"
```

---

## API Reference

### Core PINN Model
```python
from src.models.train_final import create_pinn_model, train_pinn

# Initialize model
model, params = create_pinn_model(
    input_dim=3,
    u_layers=[64, 64, 64, 1],
    v_layers=[128, 128, 128, 128, 1],
    fourier_features=32,
    fourier_scale=2.0,
    key=jax.random.PRNGKey(0)
)

# Train
trained_params, metrics = train_pinn(
    params,
    data_dict,
    config,
    pde_params=(ep1, ep2, b1, c1, b2, c2)
)
```

### Prediction
```python
# Make predictions
def predict(params, coords):
    """
    Args:
        params: Model parameters
        coords: (N, 3) array of (x, y, t)
    Returns:
        u_pred: (N,) predictions for u
        v_pred: (N,) predictions for v
    """
    u_pred, v_pred = model.apply(params, coords)
    return u_pred, v_pred

# Example
import numpy as np
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
t = np.array([0.5])
X, Y, T = np.meshgrid(x, y, t)
coords = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=-1)

u_pred, v_pred = predict(trained_params, coords)
u_field = u_pred.reshape(100, 100)
v_field = v_pred.reshape(100, 100)
```

### Loss Computation
```python
from src.models.train_final import compute_loss

# Compute individual loss components
loss_dict = compute_loss(
    params,
    data_dict,
    pde_params,
    weights_u,
    weights_v
)

print(f"IC Loss: {loss_dict['ic_u']:.6f}")
print(f"Residual Loss: {loss_dict['res_u']:.6f}")
print(f"Data Loss: {loss_dict['data_u']:.6f}")
```

### Data Loading
```python
import scipy.io
import jax.numpy as jnp

def load_mat_file(path):
    """Load MATLAB simulation data"""
    data = scipy.io.loadmat(path)
    return {
        'u': jnp.array(data['usol']),
        'v': jnp.array(data['vsol']),
        't': jnp.array(data['t'].flatten()),
        'x': jnp.array(data['x'].flatten()),
        'y': jnp.array(data['y'].flatten()),
        'params': {
            'b1': float(data['b1']),
            'b2': float(data['b2']),
            'ep1': float(data['ep1']),
            'ep2': float(data['ep2']),
        }
    }

# Usage
data = load_mat_file('data/input/mat/grey_scott_variation5.mat')
```

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size in config
'batch_size': 128  # Instead of 256

# Or reduce collocation points
'colloc_points': 5000  # Instead of 10000
```

#### Slow Training
```python
# Enable JIT compilation explicitly
from jax import jit

train_step = jit(train_step)

# Use XLA optimizations
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
```

#### Loss Not Converging
```python
# Try lower learning rate
'learning_rate': 5e-4  # Instead of 1e-3

# Increase gradient weighting update frequency
'update_freq': 500  # Instead of 1000

# Add more supervised data
'data_fraction': 0.10  # Instead of 0.05
```

#### NaN in Gradients
```python
# Clip gradients
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate)
)

# Check input normalization
# Ensure coordinates in [-1, 1] range
```

---

## Testing
```bash
# Run unit tests
python -m pytest tests/

# Test specific module
python -m pytest tests/test_architecture.py

# With coverage
python -m pytest --cov=src tests/
```

---

## Performance Optimization

### GPU Memory Management
```python
# Clear cache between runs
import jax
jax.clear_backends()

# Monitor memory
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
```

### Profiling
```python
# Profile training
import jax.profiler

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    trained_params = train_pinn(...)
```

---

## Development

### Adding New PDE Systems

1. Create data generation script in `src/data_generation/`
2. Add configuration in `src/config/`
3. Create training script in `src/models/`
4. Update this README

**Template:**
```python
# src/models/train_my_pde.py
def pde_residual(params, inputs, pde_params):
    """Define your PDE here"""
    u, v = predict(params, inputs)
    # Compute derivatives
    # Return residuals
    return ru, rv
```

---

## Citation

If you use this code:
```bibtex
@software{Bhavsar_Multi-Architecture_Coupled_Ensemble_2025,
author = {Bhavsar, Rushir},
license = {Apache-2.0},
month = apr,
title = {{Multi-Architecture Coupled Ensemble Physics-Informed Neural Networks (MACE-PINN)}},
url = {https://github.com/rushirb2001/thesis-mace-pinn},
version = {1.0.0},
year = {2025}
}
```

---

## Support

- **Issues:** [GitHub Issues](https://github.com/rushirb2001/thesis-mace-pinn/issues)
- **Email:** rushirbhavsar@gmail.com
- **Thesis:** [ASU Library](https://keep.lib.asu.edu/items/201211)

---

**Last Updated:** April 20, 2025
