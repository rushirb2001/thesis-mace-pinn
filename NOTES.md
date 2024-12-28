# Research Notes

## October 2024

### Literature Review
- Physics-informed neural networks (Raissi et al.)
- Reaction-diffusion systems
- Gray-Scott model dynamics

### Next Steps
- Generate simulation data using MATLAB
- Implement basic PINN framework

### Gray-Scott Simulations
- Successfully generated first dataset using Chebfun
- Pattern formation visible at t=2.0
- Need to explore different parameter regimes

### Pattern Formation Observations
Different parameter regimes produce distinct patterns:
- Spots: Localized structures
- Stripes: Anisotropic formations  
- Mixed: Transition regimes
- Chaotic: Sensitive dynamics

Next: Systematic parameter sweep

## December 2024

### PINN Architecture Research
- Raissi et al. 2019: Original PINN formulation
- Loss function: PDE residual + boundary conditions + initial conditions
- Automatic differentiation for computing derivatives

### Challenges
- Spectral bias in vanilla MLPs
- Stiff gradient dynamics for coupled PDEs
- Need for better input encoding

### Neural Network Framework Selection
Exploring options:
- PyTorch: Most common, good autodiff
- JAX: Functional, better for scientific computing
- Considerations: Need automatic differentiation for PDE residuals

### Progress Summary (End 2024)
- Generated Gray-Scott simulation data with various parameter regimes
- Implemented MATLAB data loading utilities
- Reviewed PINN literature and architectures
- Considering JAX/FLAX for implementation

### Next Steps (Jan 2025)
- Decide on framework (leaning toward JAX)
- Implement first working PINN
- Test on Gray-Scott benchmark data
