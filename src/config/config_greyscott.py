import os
import shutil
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax import serialization
import optax
import numpy as np
from flax.core import freeze
import scipy.io
import imageio.v3 as iio
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from flax.training import train_state
from functools import partial
from jax import random, jit, grad, vmap, hessian
from flax.training import checkpoints
import wandb
import logging

os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_default_matmul_precision", "bfloat16")
logging.basicConfig(level=logging.INFO)
# ====================================================
# 1. Data Loading & Preprocessing
# ====================================================
def load_grey_scott_mat(file_path):
    data = scipy.io.loadmat(file_path)
    return {
        'b1': float(data['b1'].item()),
        'b2': float(data['b2'].item()),
        'c1': float(data['c1'].item()),
        'c2': float(data['c2'].item()),
        'ep1': float(data['ep1'].item()),
        'ep2': float(data['ep2'].item()),
        'usol': jnp.array(data['usol']),
        'vsol': jnp.array(data['vsol']),
        't': jnp.array(data['t'].flatten()),
        'x': jnp.array(data['x'].flatten()),
        'y': jnp.array(data['y'].flatten())
    }

def get_all_points_at_time(data, t_idx):
    X, Y = jnp.meshgrid(data['x'], data['y'], indexing='ij')
    return jnp.stack([X.flatten(), Y.flatten(), jnp.full_like(X.flatten(), data['t'][t_idx])], axis=-1)

@jit
def get_apt_at_time(usol, vsol, x_data, y_data, t_idx, t_val):
    X, Y = jnp.meshgrid(x_data, y_data, indexing='ij')
    Xf, Yf = X.flatten(), Y.flatten()
    inputs = jnp.stack([Xf, Yf, jnp.full_like(Xf, t_val)], axis=-1)
    targets = jnp.stack([usol[t_idx].flatten(), vsol[t_idx].flatten()], axis=-1)
    return inputs, targets


@partial(jit, static_argnums=(1, 2))
def gather_window_data(data, start_idx, end_idx):
    usol, vsol, t_data, x_data, y_data = data['usol'], data['vsol'], data['t'], data['x'], data['y']

    def get_at_time(ti):
        return get_apt_at_time(usol, vsol, x_data, y_data, ti, jnp.array(t_data[ti], dtype=jnp.float32))

    inputs, targets = vmap(get_at_time)(jnp.arange(start_idx, end_idx))
    return inputs.reshape(-1, 3), targets.reshape(-1, 2)

# ====================================================
# 2. Neural Network Architecture
# ====================================================
class FourierFeatureLayer(nn.Module):
    num_features: int
    input_dim: int = 3  # Explicit input dimension
    scale: float = 6.0
    trainable: bool = False

    @nn.compact
    def __call__(self, inputs):
        # Verify input shape matches expected dimension
        assert inputs.shape[-1] == self.input_dim, \
            f"Expected input dimension {self.input_dim}, got {inputs.shape[-1]}"
        
        # Fixed parameter shape based on declared input_dim
        B = self.param('B', 
                      nn.initializers.normal(self.scale),
                      (self.num_features // 2, self.input_dim)) # Gaussian initialization
        
        proj = inputs @ B.T
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)

class MLP(nn.Module):
    hidden_dims: list
    name: str  # Add name parameter
    
    @nn.compact
    def __call__(self, x):
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, 
                        name=f'{self.name}_dense_{i}',
                        kernel_init=nn.initializers.he_normal())(x)
            x = nn.tanh(x)
        x = nn.Dense(1, 
                       name=f'{self.name}_output',
                       kernel_init=nn.initializers.he_normal())(x)
        x = nn.tanh(x)
        return x.squeeze()

class GrayScottPINN(nn.Module):
    hidden_dims: list = struct.field(default_factory=lambda: [256, 256, 128])
    fourier_features: int = struct.field(default=64)
    fourier_scale: float = struct.field(default=6.0)

    @nn.compact
    def __call__(self, x):
        # Check that x has shape (batch, 3)
        assert x.shape[-1] == 3, f"Input must have 3 features (x,y,t), got {x.shape[-1]}"

        # Use slicing to preserve the batch dimension (shape (N,1))
        x_wrap = jnp.concatenate([
            (x[..., 0:1] + 1) % 2 - 1,
            (x[..., 1:2] + 1) % 2 - 1,
            x[..., 2:3]
        ], axis=-1)

        # Explicitly set input_dim=3 so that the Fourier layer expects 3 features
        ff = FourierFeatureLayer(
            self.fourier_features,
            input_dim=3,
            scale=self.fourier_scale
        )(x_wrap)

        # Create submodules for u and v predictions
        u = MLP(self.hidden_dims, name='u')(ff)
        v = MLP(self.hidden_dims, name='v')(ff)

        return u, v

# ====================================================
# 3. Training Utilities & Loss Calculation
# ====================================================
def create_train_state(rng, config, data):
    model = GrayScottPINN(
        hidden_dims=config['hidden_dims'],
        fourier_features=config['fourier_features'],
        fourier_scale=config['fourier_scale']
    )
    
    # model = GrayScottPINN(
    #     hidden_dims=config['hidden_dims'],
    #     fourier_features=config['fourier_features']
    # )

    dummy_input = jnp.ones((1, 3))
    params = model.init(rng, dummy_input)
    
    tx = optax.chain(
        optax.clip_by_global_norm(config['max_grad_norm']),
        optax.lamb(optax.cosine_decay_schedule( # Switched from adamw to optimistic_adam
        config['lr'], 
        config['epochs']
        ))
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@partial(jit)
def compute_loss(state, params, batch, colloc_points, constants):
    # Get predictions
    u_pred, v_pred = state.apply_fn(params, batch['inputs'])
    
    # Split into IC and data components
    n_ic = 40000
    targets_ic = batch['targets'][:n_ic]
    targets_data = batch['targets'][n_ic:]
    
    # IC loss (first half of batch)
    loss_ic_u = jnp.mean((u_pred[:n_ic] - targets_ic[:, 0])**2)
    loss_ic_v = jnp.mean((v_pred[:n_ic] - targets_ic[:, 1])**2)
    
    # Data loss (second half of batch)
    loss_data_u = jnp.mean((u_pred[n_ic:] - targets_data[:, 0])**2)
    loss_data_v = jnp.mean((v_pred[n_ic:] - targets_data[:, 1])**2)
    
    # Residual loss
    ru, rv = compute_residuals(state, params, colloc_points, constants)
    loss_res_u = jnp.mean(ru**2)
    loss_res_v = jnp.mean(rv**2)
    
    # Weighted total loss
    total_loss = (
        constants['lambda_ic_u'] * loss_ic_u +
        constants['lambda_ic_v'] * loss_ic_v +
        constants['lambda_data_u'] * loss_data_u +
        constants['lambda_data_v'] * loss_data_v +
        constants['lambda_res_u'] * loss_res_u +
        constants['lambda_res_v'] * loss_res_v
    )
    
    return total_loss, {
        'total': total_loss,
        'ic_u': loss_ic_u,
        'ic_v': loss_ic_v,
        'data_u': loss_data_u,
        'data_v': loss_data_v,
        'res_u': loss_res_u,
        'res_v': loss_res_v
    }

@partial(jit)
def train_step(state, batch, colloc_points, constants):
    grad_fn = jax.value_and_grad(compute_loss, argnums=1, has_aux=True)
    (loss, metrics), grads = grad_fn(state, state.params, batch, colloc_points, constants)
    state = state.apply_gradients(grads=grads)
    return state, metrics

# ====================================================
# 3. Enhanced Training Utilities with Adaptive Weighting
# ====================================================
@jax.jit
def compute_component_gradients(state, batch, colloc_points, constants):
    """Compute gradients for each loss component"""
    # Split batch into IC and data parts
    n_ic = 40000
    inputs_ic = batch['inputs'][:n_ic]
    targets_ic = batch['targets'][:n_ic]
    inputs_data = batch['inputs'][n_ic:]
    targets_data = batch['targets'][n_ic:]

    # Define component loss functions
    def ic_loss(params):
        u, v = state.apply_fn(params, inputs_ic)
        return jnp.mean((u - targets_ic[:, 0])**2) + jnp.mean((v - targets_ic[:, 1])**2)

    def data_loss(params):
        u, v = state.apply_fn(params, inputs_data)
        return jnp.mean((u - targets_data[:, 0])**2) + jnp.mean((v - targets_data[:, 1])**2)

    def res_loss(params):
        ru, rv = compute_residuals(state, params, colloc_points, constants)
        return jnp.mean(ru**2) + jnp.mean(rv**2)

    # Compute gradients
    grad_ic = grad(ic_loss)(state.params)
    grad_data = grad(data_loss)(state.params)
    grad_res = grad(res_loss)(state.params)

    # Calculate gradient norms per network
    def network_norm(grads, network):
        # Access parameters through proper hierarchy
        leaves = jax.tree_util.tree_leaves(grads['params'][network])
        return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
    
    return {
        'ic_u': network_norm(grad_ic, 'u'),
        'ic_v': network_norm(grad_ic, 'v'),
        'data_u': network_norm(grad_data, 'u'),
        'data_v': network_norm(grad_data, 'v'),
        'res_u': network_norm(grad_res, 'u'),
        'res_v': network_norm(grad_res, 'v')
    }

# def compute_component_gradients(state, batch, colloc_points, constants):
#     """Compute gradients for each loss component"""
#     # Split batch into IC and data parts
#     n_ic = 40000
#     inputs_ic = batch['inputs'][:n_ic]
#     targets_ic = batch['targets'][:n_ic]
#     inputs_data = batch['inputs'][n_ic:]
#     targets_data = batch['targets'][n_ic:]

#     # Define component loss functions
#     def ic_loss(params):
#         u, v = state.apply_fn(params, inputs_ic)
#         return jnp.mean((u - targets_ic[:, 0])**2) + jnp.mean((v - targets_ic[:, 1])**2)

#     def data_loss(params):
#         u, v = state.apply_fn(params, inputs_data)
#         return jnp.mean((u - targets_data[:, 0])**2) + jnp.mean((v - targets_data[:, 1])**2)

#     def res_loss(params):
#         ru, rv = compute_residuals(state, params, colloc_points, constants)
#         return jnp.mean(ru**2) + jnp.mean(rv**2)

#     # Compute gradients
#     grad_ic = grad(ic_loss)(state.params)
#     grad_data = grad(data_loss)(state.params)
#     grad_res = grad(res_loss)(state.params)

#     # Function to calculate gradient norms for u and v subnetworks
#     def network_norm(grads, prefix):
#         # Filter parameters by prefix (e.g., 'u_dense' or 'v_dense')
#         leaves = []
#         def collect_leaves(path, value):
#             if any(p.startswith(prefix) for p in path):
#                 leaves.append(value)
#             return value

#         # Traverse the parameter tree and collect leaves matching the prefix
#         jax.tree_util.tree_map_with_path(collect_leaves, grads['params'])

#         # Calculate the norm of the collected gradients
#         return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))

#     return {
#         'ic_u': network_norm(grad_ic, 'u_'),
#         'ic_v': network_norm(grad_ic, 'v_'),
#         'data_u': network_norm(grad_data, 'u_'),
#         'data_v': network_norm(grad_data, 'v_'),
#         'res_u': network_norm(grad_res, 'u_'),
#         'res_v': network_norm(grad_res, 'v_')
#     }
# ====================================================
# 3C. Residual Computation Function
# ====================================================
@jax.jit
def compute_residuals(state, params, colloc_points, constants):
    def compute_grad_and_laplacian(fn, x):
            grad_fn = jax.grad(fn)
            grad_val = grad_fn(x)
            hess = jax.hessian(fn)(x)  # Hessian is (3,3)
            laplacian = hess[0,0] + hess[1,1]  # Spatial Laplacian: d²u/dx² + d²u/dy²
            return grad_val, laplacian

    def residual_fn(x):
        u, v = state.apply_fn(params, x)

        u_grad, laplacian_u = compute_grad_and_laplacian(lambda x: state.apply_fn(params, x)[0], x)
        v_grad, laplacian_v = compute_grad_and_laplacian(lambda x: state.apply_fn(params, x)[1], x)

        residual_u = (u_grad[2] - (constants['ep1'] * laplacian_u) -
                        (constants['b1'] * (1 - u)) + (constants['c1'] * u * v**2))
        residual_v = (v_grad[2] - (constants['ep2'] * laplacian_v) +
                        (constants['b2'] * v) - (constants['c2'] * u * v**2))

        return residual_u, residual_v

    return jax.vmap(residual_fn)(colloc_points)  # Vectorize over collocation points

# ====================================================
# 4. Training Loop with Curriculum & Visualization
# ====================================================
def train(config, data):
    rng = jax.random.PRNGKey(config['seed'])
    state = create_train_state(rng, config, data)
    
    # Initialize adaptive weights
    lambda_ic_u = 1.0
    lambda_ic_v = 1.0
    lambda_data_u = 1.0
    lambda_data_v = 1.0
    lambda_res_u = 1.0
    lambda_res_v = 1.0
    alpha = 0.3  # EMA Factor
    
    window_size = config['min_window_size']
    t_start = 0
    best_loss = float('inf')

    window_end = t_start + window_size
    inputs, targets = gather_window_data(
        data,
        t_start, window_end
    )
    batch = {
        'inputs': inputs,
        'targets': targets
    }

    with tqdm(total=config['epochs'], ncols=50) as pbar:
        for epoch in range(config['epochs']):

            # Collocation points for entire window timeframe
            colloc_points = sample_collocation_points(
                config['num_colloc'],
                data['x'],
                data['y'],
                float(data['t'][t_start]),
                float(data['t'][window_end-1])
            )

            # Update constants with current weights
            constants = freeze({
                'ep1': data['ep1'],
                'ep2': data['ep2'],
                'b1': data['b1'],
                'b2': data['b2'],
                'c1': data['c1'],
                'c2': data['c2'],
                'lambda_ic_u': lambda_ic_u,
                'lambda_ic_v': lambda_ic_v,
                'lambda_data_u': lambda_data_u,
                'lambda_data_v': lambda_data_v,
                'lambda_res_u': lambda_res_u,
                'lambda_res_v': lambda_res_v
            })

            # Training step
            state, metrics = train_step(state, batch, colloc_points, constants)

            # Adaptive weighting every 100 epochs
            if epoch % config['log_freq'] == 0 and epoch > 0:
                grad_norms = compute_component_gradients(state, batch, colloc_points, constants)
                
                # Calculate total norms
                total_norm_u = grad_norms['ic_u'] + grad_norms['data_u'] + grad_norms['res_u']
                total_norm_v = grad_norms['ic_v'] + grad_norms['data_v'] + grad_norms['res_v']

                if metrics['total'] > best_loss:
                    # Update weights for U network
                    lambda_ic_u_new = alpha*lambda_ic_u + (1-alpha)*(total_norm_u/(grad_norms['ic_u'] + 1e-8))
                    lambda_data_u_new = alpha*lambda_data_u + (1-alpha)*(total_norm_u/(grad_norms['data_u'] + 1e-8))
                    lambda_res_u_new = alpha*lambda_res_u + (1-alpha)*(total_norm_u/(grad_norms['res_u'] + 1e-8))

                    # Update weights for V network
                    lambda_ic_v_new = alpha*lambda_ic_v + (1-alpha)*(total_norm_v/(grad_norms['ic_v'] + 1e-8))
                    lambda_data_v_new = alpha*lambda_data_v + (1-alpha)*(total_norm_v/(grad_norms['data_v'] + 1e-8))
                    lambda_res_v_new = alpha*lambda_res_v + (1-alpha)*(total_norm_v/(grad_norms['res_v'] + 1e-8))

                    # Normalise Lambda
                    total_lambda_u = lambda_data_u_new + lambda_ic_u_new + lambda_res_u_new
                    total_lambda_v = lambda_data_v_new + lambda_ic_v_new + lambda_res_v_new
                    lambda_data_u = lambda_data_u_new / total_lambda_u
                    lambda_data_v = lambda_data_v_new / total_lambda_v
                    lambda_ic_u = lambda_ic_u_new / total_lambda_u
                    lambda_ic_v = lambda_ic_v_new / total_lambda_v
                    lambda_res_u = lambda_res_u_new / total_lambda_u
                    lambda_res_v = lambda_res_v_new / total_lambda_v

                loss_val_u = metrics['data_u'] + metrics['ic_u'] + metrics['res_u']
                loss_val_v = metrics['data_v'] + metrics['ic_v'] + metrics['res_v']

                best_loss = metrics['total']

                l2_u, l2_v, mae_u, mae_v = compute_metrics(state, data)
                wandb.log({
                    "epoch": epoch,
                    "total_loss": metrics['total'],
                    "ic_u_loss": metrics['ic_u'], "ic_v_loss": metrics['ic_v'],
                    "data_u_loss": metrics['data_u'], "data_v_loss": metrics['data_v'],
                    "res_u_loss": metrics['res_u'], "res_v_loss": metrics['res_v'],
                    "l2_u": l2_u, "l2_v": l2_u,
                    "mae_u": mae_u, "mae_v": mae_v
                })

            pbar.update(1)

        save_checkpoint(state, epoch)
    return state

# ====================================================
# 5. Visualization & Utilities
# ====================================================
def visualize_solution(state, data, epoch, t_idx):
    Nx, Ny = 200, 200 
    x_np = np.array(data['x'])
    y_np = np.array(data['y'])
    t_val = float(data['t'][t_idx].item())
    
    X, Y = np.meshgrid(x_np, y_np, indexing='ij')
    y_np = np.array(data['y'])
    os.makedirs(f"figures/epoch_{epoch}", exist_ok=True)
    
    inputs = get_all_points_at_time(data, t_idx)
    u_ref = data['usol'][t_idx]
    v_ref = data['vsol'][t_idx]
    
    u_pred, v_pred = state.apply_fn(state.params, inputs)
    u_pred = np.array(u_pred).reshape(Nx, Ny)
    v_pred = np.array(v_pred).reshape(Nx, Ny)

    # Compute errors
    l2_u = np.linalg.norm(u_ref - u_pred) / np.linalg.norm(u_ref)
    l2_v = np.linalg.norm(v_ref - v_pred) / np.linalg.norm(v_ref)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, (ref, pred, title) in enumerate(zip(
        [u_ref, v_ref], [u_pred, v_pred], ['U', 'V']
    )):
        axes[i, 0].imshow(ref, cmap='jet', origin='lower')
        axes[i, 0].set_title(f"{title} Reference")
        axes[i, 1].imshow(pred, cmap='jet', origin='lower')
        axes[i, 1].set_title(f"{title} Predicted")
        axes[i, 2].imshow(np.abs(ref - pred), cmap='jet', origin='lower')
        axes[i, 2].set_title("Absolute Error")
    
    print(f"\n\n[VIZ]==[Visualisation Saved to figures/epoch_{epoch}/solution_{t_idx}.png]==[VIZ]")
    plt.savefig(f"figures/epoch_{epoch}/solution_{t_idx}.png")
    plt.close()

    return l2_u, l2_v

def compute_metrics(state, data):
    Nx, Ny = 200, 200
    inputs = get_all_points_at_time(data, -1)  # Last time step
    u_ref, v_ref = data['usol'][-1], data['vsol'][-1]

    u_pred, v_pred = state.apply_fn(state.params, inputs)
    u_pred, v_pred = np.array(u_pred).reshape(Nx, Ny), np.array(v_pred).reshape(Nx, Ny)

    # Compute errors
    l2_u = np.linalg.norm(u_ref - u_pred) / np.linalg.norm(u_ref)
    l2_v = np.linalg.norm(v_ref - v_pred) / np.linalg.norm(v_ref)

    mae_u = np.mean(np.abs(u_ref - u_pred))
    mae_v = np.mean(np.abs(v_ref - v_pred))

    return l2_u, l2_v, mae_u, mae_v

def auto_version_filename(base_filename):
    """
    Given a base filename (e.g. "trained_params_sliding_optimisation_uv_value_100.pkl"),
    returns a versioned filename by appending/incrementing _vXX before the extension.
    """
    base, ext = os.path.splitext(base_filename)
    version = 1
    new_filename = f"{base}_v{version}{ext}"
    # Loop until we find a filename that does not exist
    # Create a candidate with version appended
    while os.path.exists(new_filename):
        version += 1
        new_filename = f"{base}_v{version}{ext}"
    return new_filename

def create_gif(filenames, output_gif="flax_solution_comparison.gif", fps=60):
    """
    Assemble a list of image filenames into a single GIF.
    """
    output_gif = auto_version_filename(output_gif)
    with imageio.get_writer(output_gif, mode='I', fps=fps) as writer:
        for filename in tqdm(filenames, desc="Creating GIF", ncols=50):
            image = iio.imread(filename)
            writer.append_data(image)

@partial(jit, static_argnums=(0,))
def sample_collocation_points(num_points, x, y, t_start, t_end):
    key = random.PRNGKey(jax.random.randint(random.PRNGKey(0), (), 0, int(1e6)))
    
    # Convert inputs to JAX arrays
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    t_start = jnp.asarray(t_start)
    t_end = jnp.asarray(t_end)

    # Generate uniform random samples
    x_pts = random.uniform(key, (num_points,), minval=x.min(), maxval=x.max())
    y_pts = random.uniform(key, (num_points,), minval=y.min(), maxval=y.max())
    t_pts = random.uniform(key, (num_points,), minval=t_start, maxval=t_end)
    
    return jnp.stack([x_pts, y_pts, t_pts], axis=-1)  # Returns a single JAX array

def save_checkpoint(state, epoch):
    checkpoint = serialization.to_state_dict(state)
    with open(f"params_epoch_{epoch}.pkl", 'wb') as f:
        pickle.dump(checkpoint, f)

def generate_gif_frames_flax(state, data_dict, output_dir="flax_gif_frames"):
    """Generate animation frames using Flax model predictions"""
    os.makedirs(output_dir, exist_ok=True)
    filenames = []
    Nx, Ny = 200, 200
    usol, vsol, t_data, x_data, y_data = data_dict['usol'], data_dict['vsol'], data_dict['t'], data_dict['x'], data_dict['y']
    for t_idx in tqdm(range(len(t_data)), desc="Generating frames", ncols=50):
        # Get reference solution
        U_ref = usol[t_idx]
        V_ref = vsol[t_idx]
        t_val = t_data[t_idx]

        # Create input grid
        X, Y = np.meshgrid(x_data, y_data, indexing='ij')
        inputs = jnp.stack([
            X.flatten(), 
            Y.flatten(), 
            jnp.full(Nx*Ny, t_val)
        ], axis=-1)

        # Flax model prediction
        U_pred, V_pred = state.apply_fn(state.params, inputs)
        U_pred = np.array(U_pred).reshape(Nx, Ny)
        V_pred = np.array(V_pred).reshape(Nx, Ny)

        # Compute errors
        l2_u = np.linalg.norm(U_ref - U_pred) / np.linalg.norm(U_ref)
        l2_v = np.linalg.norm(V_ref - V_pred) / np.linalg.norm(V_ref)

        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        plt.suptitle(f"Time = {t_val:.2f}\nL2 Errors: U={l2_u:.2e}, V={l2_v:.2e}")

        # U plots
        axes[0,0].imshow(U_ref, cmap='jet', origin='lower')
        axes[0,0].set_title('Reference U')
        axes[0,1].imshow(U_pred, cmap='jet', origin='lower')
        axes[0,1].set_title('Predicted U')
        axes[0,2].imshow(np.abs(U_ref - U_pred), cmap='jet', origin='lower')
        axes[0,2].set_title('U Error')

        # V plots
        axes[1,0].imshow(V_ref, cmap='jet', origin='lower')
        axes[1,0].set_title('Reference V')
        axes[1,1].imshow(V_pred, cmap='jet', origin='lower')
        axes[1,1].set_title('Predicted V')
        axes[1,2].imshow(np.abs(V_ref - V_pred), cmap='jet', origin='lower')
        axes[1,2].set_title('V Error')

        # Save frame
        filename = os.path.join(output_dir, f"frame_{t_idx:04d}.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        filenames.append(filename)

    return filenames

# ====================================================
# 6. Configuration & Execution
# ====================================================
config = {
    'seed': 42,
    'hidden_dims': [64, 64, 32]*1,
    'fourier_features': 64,
    'fourier_scale': 3,
    'lr': 1e-3,
    'max_grad_norm': 1.0,
    'epochs': 40000,
    'log_freq': 100,
    'num_colloc': 4096*64,
    'min_window_size': 101,
    'max_window_size': 101,
    'window_update_freq': 40000
}

if __name__ == "__main__":


    for i in tqdm(range(0,5)):
        variation = i
        run = wandb.init(project="gray-scott-pinn-thesis", config=config, reinit=True)
        data = load_grey_scott_mat(f"grey_scott_variation{variation}.mat")
        trained_state = train(config, data)
        frames = generate_gif_frames_flax(
            trained_state, 
            data
        )
        create_gif(frames, "flax_solution_comparison.gif")
        run.finish()