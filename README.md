# Physics-Informed Neural Networks for Reaction-Diffusion Systems

Research project exploring neural network approaches for solving coupled PDEs.

## Current Focus
- Gray-Scott reaction-diffusion model
- Pattern formation dynamics (spots, stripes, chaos)
- Data generation using MATLAB/Chebfun

## Structure
- `data/` - Simulation outputs (.mat files) and results
- `src/data_generation/` - MATLAB scripts for PDE simulations
- `notebooks/` - Experimental notebooks

## Data Generation
Requires MATLAB with Chebfun package for spectral methods.

## Dependencies
- Python 3.8+
- PyTorch
- NumPy, SciPy, Matplotlib
- h5py
- MATLAB R2021a+ (for data generation)
