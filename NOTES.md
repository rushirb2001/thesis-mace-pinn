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

## January 2025

### Framework Decision: JAX
Selected JAX over PyTorch for:
- Functional programming paradigm
- Efficient automatic differentiation (jacfwd for Hessians)
- GPU acceleration via XLA
- Clean gradient computation for PDE residuals

Using jax.example_libraries.stax for network definitions

### Architecture Ideas
- Separate subnetworks for u and v (parallel learning)
- SIREN activation (periodic sine functions)
- Fourier feature embeddings for spectral bias

## January 2025

### Architecture Research
SIREN (Sitzmann et al. 2020):
- Sine activations for coordinate-based functions
- Special initialization: first layer ~U(-1/ω₀, 1/ω₀)
- Hidden layers: ~U(-√(6/fan_in)/ω₀, √(6/fan_in)/ω₀)

Fourier Features (Tancik et al. 2020):
- Random Fourier features overcome spectral bias
- Map (x,y,t) → [sin(2πBx), cos(2πBx)] where B~N(0,σ²)
- P=64 dimensional embedding tested

### Parallel Subnetwork Design
Key insight: separate networks for u and v
- Reduces gradient interference between coupled variables
- Each network specializes on one component
- Coupling enforced through PDE residual loss

### Derivative Computation
For Laplacian (u_xx + u_yy):
- Use jax.jacfwd for efficient Hessian diagonal
- Compute second derivatives per dimension
- Vectorize over batch with vmap

### Implementation Progress
Derivative computation working with jacfwd:
- First derivatives: ∇u = [u_x, u_y, u_t]
- Second derivatives: [u_xx, u_yy, u_tt] (diagonal of Hessian)
- Laplacian: Δu = u_xx + u_yy
- Vectorized over collocation points with vmap

Periodic boundary conditions:
- Domain: [-1,1] × [-1,1]
- Map: x → (x+1) mod 2 - 1
- Ensures continuity at boundaries

## February 2025

### First Working PINN Implementation
Architecture:
- 5-layer SIREN (256 units each) with ω₀=30
- Fourier embedding: P=64, d=3
- Parallel subnetworks for u and v
- Periodic boundary enforcement

Loss components:
- Initial condition loss
- Boundary condition loss (redundant with periodic BC, but kept for generality)
- PDE residual loss (Gray-Scott equations)

Adaptive weighting (Kendall et al.):
- log_sigma parameters for each loss term
- Total loss: Σ [0.5*L_i/σ_i² + log(σ_i)]

Training:
- Adam optimizer with exponential decay
- Learning rate: 1e-3 → 0.95 decay every 10k steps
- WandB integration for tracking
