# Physics-Informed Neural Networks for Reaction-Diffusion Systems

Multi-Architecture Coupled Ensemble (MACE) PINN for solving coupled PDEs.

## Equations Studied
- **Gray-Scott**: Reaction-diffusion system with pattern formation
- **Ginzburg-Landau**: Complex-valued pattern-forming system

## Key Features
- Parallel subnetworks for coupled variables (u and v)
- Fourier feature embeddings for spectral bias
- Gradient norm-based adaptive loss weighting
- JAX/Flax implementation for GPU acceleration

## Project Structure
```
├── data/
│   ├── input/mat/          # MATLAB simulation outputs
│   ├── output/gif/         # Generated visualizations
│   └── processed/h5/       # Processed datasets
├── src/
│   ├── models/             # Training scripts
│   ├── config/             # Experiment configurations
│   ├── visualization/      # Plotting utilities
│   ├── processing/         # Data processing
│   ├── legacy/             # Implementation evolution
│   └── data_generation/    # MATLAB scripts
├── experiments/            # Benchmark iterations
└── notebooks/              # Jupyter notebooks
```

## Requirements
- Python 3.8+
- JAX with CUDA support
- Flax, Optax
- MATLAB R2021a+ (for data generation)
- WandB (for experiment tracking)

## Usage
```bash
# Train on Gray-Scott
python src/models/train_final.py

# Train on Ginzburg-Landau
python src/models/train_gls.py

# Generate visualizations
python src/visualization/generate_gifs.py
```

## Results
Generated GIFs and comparison plots available in `data/output/`
