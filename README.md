# Multi-Architecture Coupled Ensemble Physics-Informed Neural Networks (MACE-PINN)

A novel neural network architecture for solving coupled partial differential equations using parallel subnetworks with adaptive loss weighting. Successfully applied to Gray-Scott and Ginzburg-Landau reaction-diffusion systems.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-orange.svg)](https://github.com/google/jax)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Thesis](https://img.shields.io/badge/Thesis-ASU%20Library-red.svg)](https://keep.lib.asu.edu/items/201211)

---

## About This Research

This repository contains the complete implementation from my Master's thesis defended at **Arizona State University** on **April 10, 2025**. The research introduces MACE-PINN, a physics-informed neural network architecture specifically designed for coupled PDE systems.

**Author:** Rushir Bhavsar  
**Advisor:** Dr.Kookjin Lee 
**Institution:** Arizona State University  
**Defense Date:** April 10, 2025

---

## Key Findings

### Pattern Formation Capture
Our method demonstrates superior pattern formation capture compared to single-network PINNs across multiple parameter regimes.

#### Gray-Scott: Self-Replicating Spots
![Gray-Scott Animation](data/output/gif/flax_solution_comparison_variation1.gif)

---

## Research Contributions

### 1. Parallel Subnetwork Architecture
**Problem:** Traditional PINNs use single networks for coupled variables, causing gradient interference.  
**Solution:** Separate networks for each PDE variable (u and v) trained jointly.  
**Result:** 40-60% reduction in relative L2 error compared to baseline.

### 2. Fourier Feature Embeddings
**Problem:** Spectral bias prevents learning high-frequency patterns.  
**Solution:** Random Fourier features map inputs to higher-dimensional space.  
**Result:** Successful capture of fine-scale pattern structures.

### 3. Gradient Norm Adaptive Weighting
**Problem:** Loss components dominate each other, causing training instability.  
**Solution:** Dynamic weighting based on gradient magnitudes with exponential moving average.  
**Result:** Balanced gradients across IC, residual, and data loss terms.

---

## Results Summary

### Quantitative Performance

| System | Variation | Relative L2 Error | Training Time | Pattern Type |
|--------|-----------|-------------------|---------------|--------------|
| Gray-Scott | 0 (Benchmark) | 2.3% | 1.8 hrs | Spots |
| Gray-Scott | 5 (Self-Rep) | 2.7% | 2.1 hrs | Replicating |
| Ginzburg-Landau | 2 (Forcing) | 3.1% | 2.4 hrs | Waves |
| Ginzburg-Landau | 4 (Boundary) | 3.5% | 2.3 hrs | Oscillations |

*Hardware: NVIDIA H100 GPU, 80GB memory*

### Qualitative Observations

✅ **Successfully captured:** Spot formation, stripe patterns, self-replication dynamics  
✅ **Maintained:** Physical constraints and conservation laws  
✅ **Reproduced:** Complex pattern transitions and bifurcations  
⚠️ **Challenges:** Chaotic regimes require more collocation points

---

## Quick Start

See detailed installation and usage instructions in [`src/README.md`](src/README.md).
```bash
# Clone repository
git clone https://github.com/rushirb2001/thesis-mace-pinn.git
cd thesis-mace-pinn

# Install dependencies
pip install -r requirements.txt

# Train on Gray-Scott
python src/models/train_final.py --config src/config/config_greyscott.py

# Generate visualizations
python src/visualization/generate_gifs.py
```

---

## Repository Structure
```
├── src/                    # Source code and training scripts
├── data/                   # Simulation data and results
├── experiments/            # Benchmark iterations and ablations
├── notebooks/              # Jupyter analysis notebooks
├── NOTES.md               # Research journal and insights
└── README.md              # This file
```

**Detailed Documentation:**
- **[src/README.md](src/README.md)** - Installation, usage, and API reference
- **[data/README.md](data/README.md)** - Dataset descriptions and visualizations
- **[experiments/README.md](experiments/README.md)** - Experimentation process and ablation studies

---

## Architecture Overview
```
Input: (x, y, t) ∈ [-1,1]² × [0,T]
         ↓
   Fourier Embedding (64-dim)
         ↓
    ┌─────┴─────┐
    ↓           ↓
U-Network    V-Network
[64,64,64,1] [128,128,128,128,1]
    ↓           ↓
   u(x,y,t)   v(x,y,t)
         ↓
Loss = λ_ic·L_ic + λ_res·L_res + λ_data·L_data
```

**Key Features:**
- Separate networks prevent gradient interference
- Adaptive weights balance loss components
- Periodic boundary conditions enforced

---

## Citation

If you use this work, please cite:
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

## License

Licensed under Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright © 2024-2025 Rushir Bhavsar

---

## Acknowledgments

- Arizona State University faculty and research computing resources
- JAX and Flax development teams at Google
- Physics-informed neural networks research community
- MATLAB and Chebfun for simulation infrastructure

---

**Last Updated:** April 20, 2025
