# Experimentation Process

Documentation of the iterative development process, benchmark iterations, and ablation studies that led to the final MACE-PINN architecture.

---

## Experimentation Timeline
```
Oct 2024 ───► Feb 2025 ───► Mar 2025 ───► Apr 2025
   │              │             │            │
Initial        Benchmark    Production   Defense
Prototypes     Iterations    Code
```

**Total Experiments:** 50+ iterations  
**Key Breakthroughs:** 5 major architectural decisions  
**Failed Approaches:** 12 documented dead-ends

---

## Phase 1: Architecture Exploration (Oct - Jan 2025)

### Initial Approaches

#### Vanilla MLP Baseline
**File:** `pinn_vanilla_mlp.py`  
**Date:** Feb 7, 2025

**Configuration:**
- Single network for both u and v
- 4 hidden layers [128, 128, 128, 128]
- Tanh activation
- No Fourier features

**Results:**
- ❌ High relative L2 error: 15-20%
- ❌ Poor high-frequency capture
- ❌ Gradient interference between u and v
- ✅ Fast training (baseline)

**Key Learning:** Single network insufficient for coupled systems

---

#### SIREN Networks
**Files:** `pinn_v1.py`, `pinn_v2_deeper.py`  
**Dates:** Feb 2-3, 2025

**Configuration:**
- Sine activations throughout
- Special weight initialization (ω₀=30)
- Tested 5-layer vs 7-layer

**Results:**
- ✅ Better high-frequency learning
- ⚠️ Initialization sensitive
- ❌ Still single-network limitations
- ⚠️ Deeper (7-layer) → slower, minimal gain

**Key Learning:** Activation function less important than architecture

---

### Breakthrough 1: Parallel Subnetworks
**Date:** Feb 8, 2025

**Insight:** Separate networks for u and v reduce gradient interference

**Implementation:**
```python
# Before: Single network
output = network(params, inputs)
u, v = output[:, 0], output[:, 1]

# After: Parallel networks
u = u_network(params['u'], inputs)
v = v_network(params['v'], inputs)
```

**Impact:** 40% error reduction immediately

---

## Phase 2: Loss Function Optimization (Feb 2025)

### Adaptive Loss Weighting Evolution

#### Attempt 1: Fixed Weights
**File:** `pinn_vanilla_mlp.py`
```python
loss = loss_ic + loss_res
```

**Problem:** Residual loss dominates, IC ignored

---

#### Attempt 2: Manual Tuning
**File:** `pinn_grad_weighting.py`
```python
loss = 1.0*loss_ic + 100.0*loss_res + 1.0*loss_data
```

**Problem:** Needs retuning for each problem

---

#### Attempt 3: Kendall-Style Uncertainty
**File:** `pinn_v1.py`
```python
loss = 0.5*L_ic/σ_ic² + log(σ_ic) + ...
```

**Problem:** σ parameters difficult to optimize

---

#### Breakthrough 2: Gradient Norm Weighting
**File:** `pinn_fourier_opt.py`  
**Date:** Feb 11, 2025
```python
# Compute gradient magnitudes
grad_ic = ||∇_θ L_ic||
grad_res = ||∇_θ L_res||

# Inverse weighting
w_ic = 1 / (grad_ic + ε)
w_res = 1 / (grad_res + ε)

# Exponential moving average
w_ic = α*w_ic_old + (1-α)*w_ic_new
```

**Results:**
- ✅ Automatic balancing
- ✅ Stable across problems
- ✅ 25% further error reduction

---

### Breakthrough 3: Fourier Embeddings
**Date:** Feb 12, 2025

**Problem:** Spectral bias prevents fine pattern learning

**Solution:**
```python
# Map (x,y,t) → 64-dim space
B = random.normal(key, (32, 3)) * scale
embedding = [sin(2πBx), cos(2πBx)]
```

**Hyperparameter Tuning:**

| Scale | Embedding Dim | Error | Pattern Quality |
|-------|---------------|-------|-----------------|
| 1.0 | 32 | 4.2% | Medium |
| 2.0 | 32 | 2.8% | **Good** ✓ |
| 2.0 | 64 | 2.7% | Good |
| 5.0 | 32 | 3.5% | Over-smoothed |

**Selected:** scale=2.0, dim=32 (best quality/speed trade-off)

---

## Phase 3: Benchmark Iterations (Feb 14-28, 2025)

### Systematic Testing Framework

#### v1-v5: Initial Explorations
**Files:** `benchmark_v1.py` → `benchmark_v5.py`

**Tested:**
- Collocation point sampling strategies
- Data supervision amounts (0%, 5%, 10%, 20%)
- Optimizer choices (Adam, AdamW, LBFGS)
- Learning rate schedules

**Best Configuration:**
- 10,000 collocation points
- 5% supervised data
- Adam with exponential decay
- Initial LR: 1e-3

---

#### v6 Variants: Critical Phase
**Files:** `benchmark_v6*.py` (5 variants)  
**Date:** Feb 18-20, 2025

This was the most intensive experimentation phase:

**v6a - Residual Loss Focus:**
- Increased residual loss weight λ_res = 100→500
- Result: Better PDE satisfaction but worse IC fit

**v6b - Data Sampling Strategy:**
- Changed from random to Latin hypercube sampling
- Result: More uniform coverage, 5% improvement

**v6c - Optimizer Tuning:**
- Tested learning rate warmup
- Result: Slightly better stability

**v6d - Combined Best:**
- Merged successful strategies from a,b,c
- Result: Best performance to date

**Key Decision:** Proceed with v6d configuration

---

#### v7-v12: Refinement
**Files:** `benchmark_v7.py` → `benchmark_v12.py`

**v7-v9:** Stability improvements  
**v10:** Added comprehensive logging  
**v11:** Refined collocation sampling  
**v12:** Final optimized version

**v12 Performance:**
- Gray-Scott: 2.3% avg error
- Ginzburg-Landau: 3.1% avg error
- Training: 1.8-2.4 hours

---

## Phase 4: Specialized Experiments (Feb 26, 2025)

### Residual Loss Studies
**Files:** `residual_loss_*.py`

**Question:** How does residual loss sampling affect accuracy?

**Tested:**
1. Uniform random sampling
2. Adaptive sampling (high error regions)
3. Iterative refinement

**Finding:** Uniform sampling works well with sufficient points (10k+)

---

## Phase 5: Production Implementation (Mar 2025)

### Framework Transition: Stax → Flax
**Date:** Mar 3, 2025

**Motivation:**
- Better state management
- Cleaner API
- Industry standard

**Migration:**
```python
# Stax (old)
init_fn, apply_fn = stax.serial(...)
_, params = init_fn(key, input_shape)
output = apply_fn(params, input)

# Flax (new)
model = MLP(features=[64, 64, 64, 1])
params = model.init(key, input)
output = model.apply(params, input)
```

**Result:** Successful migration, no performance change

---

## Failed Approaches (Lessons Learned)

### 1. Boundary Condition Loss
**Problem:** Redundant with periodic BC enforcement  
**Impact:** Slowed training, no benefit  
**Lesson:** Remove unnecessary constraints

### 2. Very Deep Networks (10+ layers)
**Problem:** Diminishing returns, much slower  
**Impact:** 3x training time, <5% improvement  
**Lesson:** 4-6 layers sufficient

### 3. Very Large Fourier Embeddings (128-dim)
**Problem:** Over-parameterization  
**Impact:** Overfitting on small datasets  
**Lesson:** 32-64 dim optimal

### 4. Second-Order Optimizers (LBFGS)
**Problem:** Memory intensive, unstable for large models  
**Impact:** OOM errors on GPU  
**Lesson:** Adam sufficient for this scale

---

## Ablation Studies

### Component Contribution Analysis

| Configuration | Rel. L2 Error | Δ from Best |
|---------------|---------------|-------------|
| **Full MACE-PINN** | **2.3%** | **-** |
| - Fourier features | 4.1% | +78% ↑ |
| - Adaptive weighting | 3.5% | +52% ↑ |
| - Parallel networks | 5.8% | +152% ↑ |
| - All (vanilla MLP) | 15.2% | +561% ↑ |

**Conclusion:** All components essential

---

## Hyperparameter Sensitivity

### Most Sensitive Parameters
1. **Collocation points:** 1k → 10k = 60% error reduction
2. **Fourier scale:** 1.0 → 2.0 = 33% error reduction  
3. **Network width:** 32 → 64 = 15% error reduction

### Robust Parameters
- Learning rate (within 5e-4 to 2e-3)
- Gradient weighting α (0.1-0.3)
- Batch size (128-512)

---

## Computational Resources

### Training Time Analysis

| Experiment Type | GPU Hours | Wall Time | Iterations |
|-----------------|-----------|-----------|----------|
| Single run | 2.0 | 2.5 hrs | 50,000 |
| Benchmark suite | 12.0 | 1 day | 6 runs |
| Full ablation | 40.0 | 3 days | 20 runs |
| **Total research** | **~200** | **3 weeks** | **50+ runs** |

**Hardware:** NVIDIA A100 40GB

---

## Key Takeaways

### What Worked ✅
1. Parallel subnetworks for coupled variables
2. Fourier features with moderate scale (2.0)
3. Gradient norm adaptive weighting
4. Latin hypercube collocation sampling
5. 5-10% supervised data for regularization

### What Didn't Work ❌
1. Single network for both variables
2. Very deep networks (>8 layers)
3. Fixed loss weights
4. Pure residual training (no data)
5. Second-order optimizers at scale

### Critical Insights 💡
1. **Architecture matters more than depth**
2. **Loss balancing is problem-dependent but automatable**
3. **Fourier features overcome spectral bias effectively**
4. **Some supervised data dramatically helps**
5. **Periodic BC enforcement better than loss term**

---

## Future Directions

### Potential Improvements
- [ ] Causal training (time-marching)
- [ ] Multi-fidelity data integration
- [ ] Transfer learning across parameters
- [ ] Adaptive collocation point selection
- [ ] Mixed precision training (FP16)

### Open Questions
- Optimal network depth for different PDE types?
- Can we eliminate supervised data entirely?
- How to handle extreme parameter regimes?

---

## Reproducibility

All experiments tracked with:
- **Git:** Version control for code
- **WandB:** Metrics and hyperparameters
- **NOTES.md:** Research journal

To reproduce any experiment:
```bash
# Example: Run benchmark v12
python experiments/benchmark_v12.py --seed 42
```

---

**Last Updated:** April 20, 2025
