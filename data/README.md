# Data Directory

## input/mat/
MATLAB simulation outputs for training and validation.

### Gray-Scott Variations
- `grey_scott_variation0.mat` - Default parameters (benchmark)
- `grey_scott_variation1.mat` - Mitosis (spot formation)
- `grey_scott_variation2.mat` - Stripes
- `grey_scott_variation3.mat` - Mixed patterns
- `grey_scott_variation4.mat` - Chaotic patterns
- `grey_scott_variation5.mat` - Self-replicating spots

### Ginzburg-Landau Variations
- `ginzburg_landau_square_0.mat` - Original
- `ginzburg_landau_square_1.mat` - Localized perturbation
- `ginzburg_landau_square_2.mat` - Spatial forcing
- `ginzburg_landau_square_3.mat` - Checkerboard initial
- `ginzburg_landau_square_4.mat` - Boundary-driven oscillations

Each file contains:
- `usol`, `vsol`: Solution arrays (Nt × Nx × Ny)
- `x`, `y`: Spatial coordinates
- `t`: Time vector
- `ep1`, `ep2`, `b1`, `b2`, `c1`, `c2`: PDE parameters

## output/
Generated visualizations showing:
- Temporal evolution of patterns
- PINN predictions vs ground truth
- Error distributions

## processed/
Processed datasets for benchmarking and analysis.
- `comparison_dataset.h5`: Combined dataset for all variations
