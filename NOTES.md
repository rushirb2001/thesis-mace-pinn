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

### Architecture Experiments
Testing deeper SIREN: 7 layers vs 5 layers
- Hypothesis: more layers capture finer patterns
- Observation: training becomes slower
- Need to balance capacity vs computation

### Architecture Experiments
Testing deeper SIREN: 7 layers vs 5 layers
- Hypothesis: more layers capture finer patterns
- Observation: training becomes slower
- Back to 5 layers with refined initialization

### Simplification Experiments
Removed boundary condition loss:
- Periodic BC already enforced via wrapping
- BC loss was redundant and causing gradient issues
- Now: IC loss + Residual loss only

Architecture change:
- Testing vanilla MLP (tanh activations) vs SIREN
- 4-layer MLP: [128, 128, 128, 128, 1]
- Simpler might be better for debugging

Optimization insights:
- Using jax.hessian for second derivatives
- Direct computation of Laplacian components

### Gradient Norm Weighting
New approach for loss balancing:
- Compute gradient norms for each loss component
- Weight inversely proportional to gradient magnitude
- Prevents one loss term from dominating
- Exponential moving average (α=0.9) for stability

Separate weighting for u and v networks:
- Each network may need different balance
- Track gradient norms independently

Also testing LBFGS optimizer:
- Second-order method for better convergence
- Slower per iteration but fewer total iterations needed

### Fourier Embeddings Revisited
Combining gradient weighting with Fourier features:
- Random Fourier features: B~N(0, scale²)
- Embedding dimension: 32 (16 sin + 16 cos)
- Scale parameter: 2.0

Architecture adjustments:
- U network: [64, 64, 64, 1] (simpler with Fourier features)
- V network: [128, 128, 128, 128, 1] (needs more capacity)

JAX config: highest precision for matmul stability

### Benchmark Framework Setup
Starting systematic benchmarking:
- Multiple parameter configurations
- Tracking convergence metrics
- Comparing different loss weightings
- Need reproducible evaluation protocol

### Benchmark Iterations v2-v5
Testing variations:
- Different collocation point sampling strategies
- Adjusting data supervision amounts
- Hyperparameter tuning for gradient weights
- Loss component balance experiments

### v6 Architecture Exploration
Heavy experimentation with v6 variants:
- v6: baseline with new loss formulation
- v6a: adjusted residual loss weighting
- v6b: modified data sampling strategy  
- v6c: different optimizer settings
- v6d: combined best practices from a,b,c

This was a critical phase - testing many hypotheses

### Final Benchmark Refinements v7-v12
Progressive improvements:
- v7-v9: stability improvements
- v10: added better logging and metrics
- v11: refined collocation sampling  
- v12: final optimized version

Key insights gained:
- Gradient weighting needs careful tuning
- Collocation point distribution matters significantly
- Balance between IC/residual/data loss is problem-dependent

### Residual Loss Focused Experiments
Specialized experiments on residual loss:
- Iterative residual loss computation
- Different weighting schemes for residual terms
- Testing impact of residual sampling density

### February Progress Summary
Major achievements:
- Established robust benchmark framework
- Tested 12+ architecture/training variants
- Identified optimal gradient weighting strategy
- Residual loss formulation working well

Best configuration so far:
- Fourier embeddings (32 features, scale=2.0)
- Gradient norm adaptive weighting (α=0.2)
- U: [64,64,64,1], V: [128,128,128,128,1]
- 3-component loss: IC + Residual + Supervised data

Next steps (March):
- Move to Flax for better modularity
- Test on Ginzburg-Landau equation
- Prepare final implementation structure

## March 2025

### Transition to Flax
Moving from Stax to Flax for better modularity:
- Flax provides cleaner NN abstractions
- Better state management
- Easier to extend and modify architectures
- Industry-standard framework

Initial Flax implementations:
- Porting MLP architecture
- Testing compatibility with existing training loop
- Validating results match Stax version

### Production Code Structure
Restructuring for clean implementation:
- main.py: primary training script
- Separate configs for Gray-Scott and Ginzburg-Landau
- WandB integration for all experiments
- Modular architecture design

Testing both Gray-Scott and Ginzburg-Landau:
- GLS adds complexity with complex-valued solutions
- Need to handle real/imaginary components separately

### Architecture Variants
Testing different architectures on both equations:
- Standard MLP baseline
- Gated architectures for better gradient flow
- Comparing performance on Gray-Scott vs GLS

Key finding: architecture choice matters more for GLS
- Complex-valued PDEs benefit from gated units
- Gray-Scott works well with simpler architectures

### Visualization and Data Processing
Building analysis tools:
- GIF generation for temporal evolution
- Time-series plotting for quantitative analysis
- Data processing pipeline for multiple variations
- H5 dataset creation for benchmarking

Generated visualizations for:
- All 6 Gray-Scott parameter variations
- 5 Ginzburg-Landau variations
- Comparison plots: PINN vs ground truth

### MATLAB Data Generation Complete
All simulation scripts organized:
- Gray-Scott with 6 parameter variations
- Ginzburg-Landau with 5 variations
- Consistent naming and structure
- All .mat files generated successfully

Data characteristics:
- 200x200 spatial grid
- 101 time steps
- Multiple pattern types: spots, stripes, chaos, self-replicating

### Final Project Structure
Organized into clear modules:
- `src/models/` - Training scripts for different architectures
- `src/config/` - Configuration files for experiments
- `src/visualization/` - Plotting and GIF generation
- `src/processing/` - Data pipeline utilities
- `src/legacy/` - Evolution of implementations
- `experiments/` - Benchmark iterations

Key achievements:
- Stable training across both PDEs
- Multiple architecture comparisons
- Comprehensive visualization suite
- Reproducible experiment framework

## March Progress Summary

### Major Milestones
1. Successfully transitioned to Flax framework
2. Implemented and tested multiple architectures
3. Validated on both Gray-Scott and Ginzburg-Landau
4. Complete visualization pipeline
5. Production-ready codebase

### Key Technical Insights
- Fourier embeddings + gradient weighting = stable training
- Parallel subnetworks crucial for coupled PDEs
- Architecture choice matters more for complex-valued equations
- Adaptive loss weighting prevents gradient imbalance

### Thesis Preparation
Code base ready for:
- Chapter 3: Methodology (architecture diagrams)
- Chapter 4: Implementation (code structure)
- Chapter 5: Results (all visualizations generated)

### Performance Metrics
Gray-Scott (variation 5):
- Relative L2 error: ~2-3%
- Training time: ~2 hours (GPU)
- Pattern formation captured accurately

Next: Write thesis chapters, prepare final plots

### Data Files Organization
Adding generated data to repository structure:
- MATLAB simulation outputs (.mat files)
- Generated visualizations (GIFs)
- Processed datasets (H5)

Note: Large binary files tracked via git-lfs
