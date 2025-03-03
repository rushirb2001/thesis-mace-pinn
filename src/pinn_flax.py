import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
from pathlib import Path

# Add near the top after parameters
VIS_ROOT = Path("training_progress")
VIS_ROOT.mkdir(exist_ok=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Parameters from MATLAB code (k=1000)
ep1 = 0.2  # 0.0002 * 1000
ep2 = 0.1  # 0.0001 * 1000
b1 = 40.0  # 0.04 * 1000
c1 = 1000.0
b2 = 100.0  # 0.1 * 1000
c2 = 1000.0

# Load MATLAB data
mat_data = loadmat('grey_scott.mat')
usol_true = mat_data['usol']
vsol_true = mat_data['vsol']
x_data = mat_data['x'].flatten()
y_data = mat_data['y'].flatten()
t_data = mat_data['t'].flatten()

# Define the MLP model
class GrayScottMLP(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(2)(x)  # Outputs [u, v]
        return x

# Initial condition functions
def u0_initial(x, y):
    return 1 - jnp.exp(-10 * ((x + 0.05)**2 + (y + 0.02)**2))

def v0_initial(x, y):
    return jnp.exp(-10 * ((x - 0.05)**2 + (y - 0.02)**2))

# Compute PDE residuals
def compute_residuals(model, params, xyt):
    x, y, t = xyt
    inputs = jnp.array([x, y, t])
    
    # Compute u and v
    uv = model.apply(params, inputs)
    u, v = uv[0], uv[1]
    
    # First derivatives
    du = grad(lambda inp: model.apply(params, inp)[0])(inputs)
    dv = grad(lambda inp: model.apply(params, inp)[1])(inputs)
    du_dt = du[2]
    dv_dt = dv[2]
    
    # Second derivatives for Laplacian
    def laplacian(f):
        hessian_fn = jax.hessian(lambda inp: f(model.apply(params, inp)))
        hessian = hessian_fn(inputs)
        return hessian[0][0] + hessian[1][1]
    
    laplacian_u = laplacian(lambda uv: uv[0])
    laplacian_v = laplacian(lambda uv: uv[1])
    
    # Gray-Scott equations
    residual_u = du_dt - (ep1 * laplacian_u + b1 * (1 - u) - c1 * u * v**2)
    residual_v = dv_dt - (ep2 * laplacian_v - b2 * v + c2 * u * v**2)
    
    return residual_u, residual_v

# Loss functions
# Updated loss functions section
def loss_fn(params, model, ic_batch, res_batch):
    # Initial condition loss
    def ic_loss(batch):
        x = batch[:, 0]  # Extract columns instead of unpacking
        y = batch[:, 1]
        t = batch[:, 2]
        uv_pred = vmap(model.apply, in_axes=(None, 0))(params, batch)
        u_pred, v_pred = uv_pred[:, 0], uv_pred[:, 1]
        u_true = vmap(u0_initial)(x, y)
        v_true = vmap(v0_initial)(x, y)
        return jnp.mean((u_pred - u_true)**2 + (v_pred - v_true)**2)
    
    # Residual loss
    def res_loss(batch):
        residuals = vmap(compute_residuals, in_axes=(None, None, 0))(model, params, batch)
        res_u, res_v = residuals
        return jnp.mean(res_u**2 + res_v**2)
    
    loss_ic = ic_loss(ic_batch)
    loss_res = res_loss(res_batch)
    total_loss = loss_ic + loss_res
    return total_loss, (loss_ic, loss_res)

# Curriculum training setup
def create_train_state(rng, learning_rate):
    model = GrayScottMLP()
    params = model.init(rng, jnp.ones(3))  # Input is [x, y, t]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def visualize_predictions(model, params, stage, epoch, time_idx=-1):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    X, Y = jnp.meshgrid(x_data, y_data)
    
    # Ground truth
    u_true = usol_true[time_idx]
    v_true = vsol_true[time_idx]
    
    # Model prediction
    T = jnp.ones_like(X) * t_data[time_idx]
    inputs = jnp.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
    uv_pred = vmap(model.apply, in_axes=(None, 0))(params, inputs)
    u_pred = uv_pred[:, 0].reshape(X.shape)
    v_pred = uv_pred[:, 1].reshape(X.shape)

    # Plot U component
    im1 = axs[0,0].imshow(u_true, extent=(-1,1,-1,1), origin='lower', cmap='viridis')
    axs[0,0].set_title(f'Ground Truth U (t={t_data[time_idx]:.2f})')
    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im1, cax=cax)

    im2 = axs[0,1].imshow(u_pred, extent=(-1,1,-1,1), origin='lower', cmap='viridis')
    axs[0,1].set_title(f'Predicted U (t={t_data[time_idx]:.2f})')
    divider = make_axes_locatable(axs[0,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im2, cax=cax)

    # Plot V component
    im3 = axs[1,0].imshow(v_true, extent=(-1,1,-1,1), origin='lower', cmap='viridis')
    axs[1,0].set_title(f'Ground Truth V (t={t_data[time_idx]:.2f})')
    divider = make_axes_locatable(axs[1,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im3, cax=cax)

    im4 = axs[1,1].imshow(v_pred, extent=(-1,1,-1,1), origin='lower', cmap='viridis')
    axs[1,1].set_title(f'Predicted V (t={t_data[time_idx]:.2f})')
    divider = make_axes_locatable(axs[1,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im4, cax=cax)

    stage_dir = VIS_ROOT / f"stage_{stage:02d}"
    stage_dir.mkdir(exist_ok=True)
    fname = stage_dir / f"epoch_{epoch:04d}_t_{t_data[time_idx]:.2f}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def train(rng, num_stages=5, epochs_per_stage=1000, batch_size=256):
    # Curriculum stages: incrementally increase max time
    final_time = 2.0  # From MATLAB code (2000/k = 2.0)
    time_stages = jnp.linspace(0.2, final_time, num_stages)
    
    # Initialize model and state
    init_rng = rng
    model = GrayScottMLP()
    state = create_train_state(init_rng, 1e-3)

    loss_history = []
    
    for stage in range(num_stages):
        current_max_time = time_stages[stage]
        print(f"Curriculum stage {stage+1}/{num_stages}, t_max={current_max_time:.2f}")
        
        # Training loop for current stage
        for epoch in tqdm(range(epochs_per_stage), desc="PINN training (2D)"):
            
            # Sample initial condition and collocation points
            rng, ic_rng, res_rng = random.split(rng, 3)
            
            # Initial condition batch (t=0)
            ic_batch = random.uniform(ic_rng, (batch_size, 3), minval=-1.0, maxval=1.0)
            ic_batch = ic_batch.at[:, 2].set(0.0)  # t=0
            
            # Collocation batch (random x, y, t)
            res_batch = random.uniform(res_rng, (batch_size, 3), minval=jnp.array([-1.0, -1.0, 0.0]),
                               maxval=jnp.array([1.0, 1.0, current_max_time]))
            
            # Compute loss and gradients
            (total_loss, (loss_ic, loss_res)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, model, ic_batch, res_batch)
            
            # Update parameters
            state = state.apply_gradients(grads=grads)
            
            # Store losses
            loss_history.append((total_loss, loss_ic, loss_res))
            
            # Visualization every 100 epochs
            if epoch % 100 == 0:
                # Get closest time index
                time_idx = int((current_max_time / final_time) * (len(t_data)-1))
                visualize_predictions(model, state.params, stage, epoch, time_idx)
                
                # Save loss plot
                fig, ax = plt.subplots(figsize=(10, 6))
                losses = np.array(loss_history)
                ax.semilogy(losses[:, 0], label='Total Loss')
                ax.semilogy(losses[:, 1], label='IC Loss')
                ax.semilogy(losses[:, 2], label='Residual Loss')
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss Value")
                ax.legend()
                plt.savefig(VIS_ROOT / "loss_progression.png", bbox_inches='tight')
                plt.close()
        
        # Final visualization for the stage
        time_idx = int((current_max_time / final_time) * (len(t_data)-1))
        visualize_predictions(model, state.params, stage, epochs_per_stage, time_idx)
    
    return state

# Main execution
if __name__ == "__main__":
    rng = random.PRNGKey(42)
    trained_state = train(rng, epochs_per_stage=5000, batch_size=4096)
    print("Training completed.")