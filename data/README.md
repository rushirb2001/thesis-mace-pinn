# Dataset Documentation

Comprehensive simulation data for Gray-Scott and Ginzburg-Landau reaction-diffusion systems, generated using MATLAB with Chebfun spectral methods.

[![MATLAB R2021a+](https://img.shields.io/badge/MATLAB-R2021a+-blue.svg)](https://www.mathworks.com/)
[![Chebfun](https://img.shields.io/badge/Chebfun-5.7.0-orange.svg)](https://www.chebfun.org/)
[![Data Size](https://img.shields.io/badge/Total%20Size-~2.1GB-green.svg)]()

---

## Dataset Overview

### Gray-Scott Reaction-Diffusion System
```
u_t = ε₁∇²u + b₁(1-u) - c₁uv²
v_t = ε₂∇²v - b₂v + c₂uv²
```

**Spatial Domain:** [-1, 1] × [-1, 1]  
**Grid Resolution:** 200 × 200  
**Temporal Points:** 101 snapshots  
**Time Range:** [0, 2.0]

#### Variation 0: Benchmark Parameters
![Gray-Scott Var 0 - PINN](output/gif/flax_solution_comparison_variation0.gif)

**Parameters:** b₁=40, b₂=100, ε₁=0.2, ε₂=0.1  
**Pattern Type:** Spot formation (benchmark)  
**File:** `grey_scott_variation0.mat` (52.4 MB)

| Metric | Value |
|--------|-------|
| Max u | 0.998 |
| Min u | 0.247 |
| Max v | 0.612 |
| Min v | 0.003 |
| Pattern Frequency | ~0.15 spatial |

---

#### Variation 1: Mitosis (Spot Splitting)
![Gray-Scott Var 1](output/gif/flax_solution_comparison_variation1.gif)

**Parameters:** b₁=25, b₂=85, ε₁=0.16, ε₂=0.08  
**Pattern Type:** Self-dividing spots  
**File:** `grey_scott_variation1.mat` (52.4 MB)

| Metric | Value |
|--------|-------|
| Max u | 0.997 |
| Min u | 0.189 |
| Max v | 0.723 |
| Min v | 0.002 |
| Split Events | 12-15 per cycle |

---

#### Variation 2: Stripe Formation
![Gray-Scott Var 2](output/gif/flax_solution_comparison_variation2.gif)

**Parameters:** b₁=35, b₂=95, ε₁=0.19, ε₂=0.09  
**Pattern Type:** Anisotropic stripes  
**File:** `grey_scott_variation2.mat` (52.4 MB)

| Metric | Value |
|--------|-------|
| Max u | 0.995 |
| Min u | 0.312 |
| Max v | 0.548 |
| Min v | 0.005 |
| Stripe Wavelength | ~0.3 spatial |

---

#### Variation 3: Mixed Patterns
![Gray-Scott Var 3](output/gif/flax_solution_comparison_variation3.gif)

**Parameters:** b₁=30, b₂=90, ε₁=0.18, ε₂=0.10  
**Pattern Type:** Spots + stripes coexistence  
**File:** `grey_scott_variation3.mat` (52.4 MB)

| Metric | Value |
|--------|-------|
| Max u | 0.993 |
| Min u | 0.278 |
| Max v | 0.651 |
| Min v | 0.004 |
| Transition Time | t ≈ 0.8 |

---

#### Variation 4: Chaotic Dynamics
![Gray-Scott Var 4](output/gif/flax_solution_comparison_variation4.gif)

**Parameters:** b₁=40, b₂=120, ε₁=0.20, ε₂=0.10  
**Pattern Type:** Irregular turbulent patterns  
**File:** `grey_scott_variation4.mat` (52.4 MB)

| Metric | Value |
|--------|-------|
| Max u | 0.991 |
| Min u | 0.156 |
| Max v | 0.789 |
| Min v | 0.001 |
| Lyapunov Exp | ~0.32 (estimated) |

---

<!-- ### Ginzburg-Landau Complex System
```
u_t + iv_t = ε(u_xx + u_yy + i(v_xx + v_yy)) + k(u + iv) - k(1 + iα)(u + iv)|u + iv|²
```

**Spatial Domain:** [-1, 1] × [-1, 1]  
**Grid Resolution:** 200 × 200  
**Temporal Points:** 101 snapshots  
**Time Range:** [0, 1.0]

#### Variation 0: Gaussian Wave Packet
![Ginzburg-Landau Var 0](output/gif/ginzburg_landau_variation_0.gif)

**Parameters:** k=10, ε=0.004, α=1.5  
**Initial Condition:** Gaussian with phase  
**File:** `ginzburg_landau_square_0.mat` (52.4 MB)

| Metric | Real (u) | Imag (v) |
|--------|----------|----------|
| Max | 2.341 | 1.876 |
| Min | -1.234 | -1.543 |
| Amplitude | ~2.8 peak | ~2.3 peak |

---

#### Variation 1: Localized Perturbations
![Ginzburg-Landau Var 1](output/gif/ginzburg_landau_variation_1.gif)

**Parameters:** k=10, ε=0.004, α=1.5  
**Initial Condition:** Multiple Gaussian bumps  
**File:** `ginzburg_landau_square_1.mat` (52.4 MB)

| Metric | Real (u) | Imag (v) |
|--------|----------|----------|
| Max | 1.987 | 2.123 |
| Min | -1.567 | -1.789 |
| Interference | Strong at t>0.4 |

---

#### Variation 2: Spatial Forcing
![Ginzburg-Landau Var 2](output/gif/ginzburg_landau_variation_2.gif)

**Parameters:** k=10, ε=0.004, α=1.5, forcing=5cos(10x)  
**Pattern Type:** Driven oscillations  
**File:** `ginzburg_landau_square_2.mat` (52.4 MB)

| Metric | Real (u) | Imag (v) |
|--------|----------|----------|
| Max | 3.456 | 2.987 |
| Min | -2.234 | -2.567 |
| Forcing Freq | 10 rad/unit |

---

#### Variation 3: Checkerboard Initial
![Ginzburg-Landau Var 3](output/gif/ginzburg_landau_variation_3.gif)

**Parameters:** k=5, ε=0.004, α=1.5  
**Initial Condition:** Checkerboard pattern  
**File:** `ginzburg_landau_square_3.mat` (52.4 MB)

| Metric | Real (u) | Imag (v) |
|--------|----------|----------|
| Max | 1.765 | 1.892 |
| Min | -1.543 | -1.678 |
| Symmetry Break | t ≈ 0.3 |

---

#### Variation 4: Boundary-Driven
![Ginzburg-Landau Var 4](output/gif/ginzburg_landau_variation_4.gif)

**Parameters:** k=10, ε=0.004, α=1.5, boundary forcing  
**Pattern Type:** Inward propagating waves  
**File:** `ginzburg_landau_square_4.mat` (52.4 MB)

| Metric | Real (u) | Imag (v) |
|--------|----------|----------|
| Max | 2.987 | 3.123 |
| Min | -1.987 | -2.234 |
| Wave Speed | ~0.5 units/time |

--- -->

## Data File Structure

Each `.mat` file contains:
```matlab
% Gray-Scott files
usol    % (101 × 200 × 200) - u component solution
vsol    % (101 × 200 × 200) - v component solution
x       % (200,) - x coordinates
y       % (200,) - y coordinates  
t       % (101,) - time snapshots
b1, b2  % Scalar - reaction parameters
c1, c2  % Scalar - reaction parameters
ep1,ep2 % Scalar - diffusion coefficients

% Ginzburg-Landau files
usol    % (101 × 200 × 200) - real part
vsol    % (101 × 200 × 200) - imaginary part
x, y, t % Same as above
k       % Scalar - reaction rate
eps     % Scalar - diffusion
alpha   % Scalar - dispersion
```

---

## Data Generation

Simulations generated using MATLAB R2021a with Chebfun 5.7.0:
```bash
# Gray-Scott
cd src/data_generation
matlab -batch "run('data_gen_grey_scott_square.m')"

# Ginzburg-Landau
matlab -batch "run('data_gen_ginz_lan_square.m')"
```

**Computational Cost:**
- Gray-Scott: ~15 minutes per variation
- Ginzburg-Landau: ~20 minutes per variation
- Total generation time: ~3 hours (single core)

---

## Processed Datasets

### H5 Format for Benchmarking
`processed/h5/comparison_dataset.h5`

Contains combined dataset with:
- Train/validation/test splits (70/15/15)
- Normalized coordinates
- Preprocessed for direct PINN training

**Access:**
```python
import h5py
with h5py.File('data/processed/h5/comparison_dataset.h5', 'r') as f:
    u_train = f['train/u'][:]
    v_train = f['train/v'][:]
    coords = f['train/coords'][:]
```

---

## Visualization Outputs

### GIF Animations
- **Ground truth:** Pattern evolution from MATLAB
- **PINN predictions:** Neural network solutions
- **Comparisons:** Side-by-side validation

### MP4 Videos
High-quality videos for presentation:
`output/mp4/flax_solution_comparison_variation0.mp4`

---

## Data Statistics

| Dataset | Files | Total Size | Avg File Size |
|---------|-------|------------|---------------|
| Gray-Scott | 6 variations | 314.4 MB | 52.4 MB |
| Ginzburg-Landau | 5 variations | 262.0 MB | 52.4 MB |
| Visualizations | 25 GIFs | 187.5 MB | 7.5 MB |
| Processed | 1 H5 file | 421.3 MB | - |
| **Total** | **37 files** | **~1.2 GB** | - |

---

## Usage Notes

1. **Memory Requirements:** ~4GB RAM to load full dataset
2. **Coordinate System:** Periodic boundaries at x,y ∈ [-1,1]
3. **Time Units:** Nondimensionalized (k=1000 factor)
4. **Numerical Precision:** Float64 throughout

---

## References

1. Pearson, J. E. (1993). "Complex patterns in a simple system." *Science*, 261(5118), 189-192.
2. Aranson, I. S., & Kramer, L. (2002). "The world of the complex Ginzburg-Landau equation." *Reviews of Modern Physics*, 74(1), 99.

---

**Last Updated:** April 20, 2025
