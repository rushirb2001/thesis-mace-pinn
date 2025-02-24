# --- Configure Logging First ---
import logging
import warnings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("Loggings and Warnings have been configured.")
logging.getLogger('orbax').setLevel(logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('orbax.checkpoint._src.handlers').setLevel(logging.ERROR)
logging.getLogger('orbax.checkpoint.test_utils').setLevel(logging.ERROR)

class TensorFlowWarningFilter(logging.Filter):
    def filter(self, record):
        return "Tensorflow library not found" not in record.getMessage()

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(TensorFlowWarningFilter())

# --- Standard Library Imports ---
import os
import shutil
import pickle
from functools import partial
logging.info("Standard library imports loaded successfully.")

# --- Third-Party Imports: NumPy, SciPy, Matplotlib, and TQDM ---
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm
logging.info("NumPy, SciPy, Matplotlib, and tqdm imported successfully.")

# --- Third-Party Imports: JAX ---
import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap, hessian, jacfwd
logging.info("JAX and related modules imported successfully.")

# --- Third-Party Imports: Flax ---
import flax.linen as nn
from flax import struct
from flax import serialization
from flax.core import freeze
from flax.training import train_state, checkpoints
logging.info("Flax libraries imported successfully.")

# --- Third-Party Imports: Optax ---
import optax
logging.info("Optax imported successfully.")

logging.info("All imports have been loaded successfully.")

os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_default_matmul_precision", "bfloat16")

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
    Nx, Ny = data['usol'].shape[1], data['usol'].shape[2]
    X, Y = np.meshgrid(data['x'], data['y'], indexing='ij')
    return jnp.stack([X.flatten(), Y.flatten(), jnp.full(Nx * Ny, data['t'][t_idx])], axis=-1)

def get_apt_at_time(usol, vsol, x_data, y_data, t_idx, t_val):
    Nx, Ny = usol.shape[1], usol.shape[2]
    X, Y = np.meshgrid(x_data, y_data, indexing='ij')
    Xf = X.flatten()
    Yf = Y.flatten()
    Tvals = np.full_like(Xf, t_val)
    U_slice = usol[t_idx].flatten()
    V_slice = vsol[t_idx].flatten()
    inputs = np.stack([Xf, Yf, Tvals], axis=-1)  # (Nx*Ny, 3)
    targets = np.stack([U_slice, V_slice], axis=-1)  # (Nx*Ny, 2)
    return jnp.array(inputs), jnp.array(targets)

def gather_window_data(data, start_idx, end_idx):
    inps_list = []
    tars_list = []
    usol, vsol, t_data, x_data, y_data = data['usol'], data['vsol'], data['t'], data['x'], data['y']
    for ti in range(start_idx, end_idx):
        t_val = float(t_data[ti])
        inp_t, tar_t = get_apt_at_time(usol, vsol, x_data, y_data, ti, t_val)
        inps_list.append(inp_t)
        tars_list.append(tar_t)
    if len(inps_list) == 0:
        return jnp.zeros((0,3)), jnp.zeros((0,2))
    inps = jnp.concatenate(inps_list, axis=0)
    tars = jnp.concatenate(tars_list, axis=0)
    return inps, tars

# ====================================================
# 2. Neural Network Architecture
# ====================================================
class FourierFeatureLayer(nn.Module):
    num_features: int
    input_dim: int = 3
    scale: float = 6.0
    trainable: bool = False

    @nn.compact
    def __call__(self, inputs):
        assert inputs.shape[-1] == self.input_dim, f"Expected input dimension {self.input_dim}, got {inputs.shape[-1]}"
        B = self.param('B', nn.initializers.normal(self.scale), (self.num_features // 2, self.input_dim))
        proj = inputs @ B.T
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)

class MLP(nn.Module):
    hidden_dims: list
    name: str
    
    @nn.compact
    def __call__(self, x):
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f'{self.name}_dense_{i}', kernel_init=nn.initializers.he_normal())(x)
            x = nn.BatchNorm(dim, name=f'{self.name}_batchnorm_{i}', momentum=0.9, epsilon=1e-5)(x)
            x = nn.swish(x)
        return nn.Dense(1, name=f'{self.name}_output', kernel_init=nn.initializers.he_normal())(x).squeeze()

class GrayScottPINN(nn.Module):
    hidden_dims: list = struct.field(default_factory=lambda: [64, 64, 32])
    fourier_features: int = 128
    fourier_scale: float = 6.0

    def initialize_causally(self, params, data, config, rng):
        N, M = config['init_N'], config['init_M']
        t_data, x_data, y_data = data['t'], data['x'], data['y']
        usol, vsol = data['usol'], data['vsol']
        t_max = float(t_data[-1])
        subwindow_size = t_max / N
        inputs, targets = [], []
        
        for i in range(N):
            t_min = i * subwindow_size
            t_max_sub = (i + 1) * subwindow_size
            key, t_key, x_key, y_key = random.split(rng, 4)
            t_samples = random.uniform(t_key, (M,), minval=t_min, maxval=t_max_sub)
            x_samples = random.uniform(x_key, (M,), minval=x_data[0], maxval=x_data[-1])
            y_samples = random.uniform(y_key, (M,), minval=y_data[0], maxval=y_data[-1])
            inp = jnp.stack([x_samples, y_samples, t_samples], axis=-1)
            t_idx = jnp.searchsorted(t_data, t_samples)
            x_idx = jnp.searchsorted(x_data, x_samples)
            y_idx = jnp.searchsorted(y_data, y_samples)
            u_samples = usol[t_idx, x_idx, y_idx]
            v_samples = vsol[t_idx, x_idx, y_idx]
            tar = jnp.stack([u_samples, v_samples], axis=-1)
            inputs.append(inp)
            targets.append(tar)
        
        inputs = jnp.vstack(inputs)
        targets = jnp.vstack(targets)
        ff = self.fourier_feature_layer(params, inputs)
        
        t_norm = inputs[:, 2] / t_max
        w = jnp.exp(-config['init_alpha'] * t_norm)
        W = jnp.diag(w)
        
        u_target, v_target = targets[:, 0], targets[:, 1]
        u_coeffs = jnp.linalg.lstsq(W @ ff, W @ u_target, rcond=None)[0]
        v_coeffs = jnp.linalg.lstsq(W @ ff, W @ v_target, rcond=None)[0]
        coeffs = jnp.stack([u_coeffs, v_coeffs], axis=-1)
        
        new_params = params.copy()
        new_params['params']['u']['Dense_2']['kernel'] = coeffs[:, 0:1]
        new_params['params']['v']['Dense_2']['kernel'] = coeffs[:, 1:2]
        return new_params

    def fourier_feature_layer(self, params, x):
        B = params['params']['FourierFeatureLayer_0']['B']
        proj = x @ B.T
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)

    @nn.compact
    def __call__(self, x):
        assert x.shape[-1] == 3, f"Input must have 3 features (x,y,t), got {x.shape[-1]}"
        x_wrap = jnp.concatenate([(x[..., 0:1] + 1) % 2 - 1, (x[..., 1:2] + 1) % 2 - 1, x[..., 2:3]], axis=-1)
        ff = FourierFeatureLayer(self.fourier_features, input_dim=3, scale=self.fourier_scale)(x_wrap)
        u = MLP(self.hidden_dims, name='u')(ff)
        v = MLP(self.hidden_dims, name='v')(ff)
        return u, v

# ====================================================
# 3. Training Utilities & Loss Calculation
# ====================================================
def create_train_state(rng, config, data):
    model = GrayScottPINN(hidden_dims=config['hidden_dims'], fourier_features=config['fourier_features'])
    dummy_input = jnp.ones((1, 3))
    params = model.init(rng, dummy_input)
    params = model.initialize_causally(params, data, config, rng)  # Physics-informed initialization
    schedule = optax.cosine_decay_schedule(config['lr'], config['epochs'], 1e-5)
    tx = optax.chain(optax.clip_by_global_norm(config['max_grad_norm']), 
                     optax.optimistic_adam(schedule, 7.84e-03))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def compute_loss(state, params, batch, colloc_points, constants, epoch):
    u_pred, v_pred = state.apply_fn(params, batch['inputs'])
    t_batch = batch['inputs'][:, 2]
    t_max = constants['t_max']
    
    stages = [(8000, 25, 5.0), (16000, 50, 4.0), (24000, 75, 3.0), (32000, 100, 2.0), (40000, 100, 1.0)]
    for end_epoch, t_idx, alpha_val in stages:
        if epoch <= end_epoch:
            alpha, t_end = alpha_val, constants['t_data'][t_idx]
            break
    
    # IC loss
    loss_ic_u = jnp.mean((u_pred[:1] - batch['targets'][:1, 0])**2)
    loss_ic_v = jnp.mean((v_pred[:1] - batch['targets'][:1, 1])**2)
    
    # Supervised LSE per time step
    mask = t_batch[1:] <= t_end
    u_pred_data = u_pred[1:][mask]
    v_pred_data = v_pred[1:][mask]
    targets_data = batch['targets'][1:][mask]
    t_batch_data = t_batch[1:][mask]
    points_per_step = 200 * 200
    unique_t = jnp.unique(t_batch_data)
    lse_u = jnp.array([jnp.sum((u_pred_data[t_batch_data == t] - targets_data[t_batch_data == t, 0])**2) 
                       for t in unique_t]) / points_per_step
    lse_v = jnp.array([jnp.sum((v_pred_data[t_batch_data == t] - targets_data[t_batch_data == t, 1])**2) 
                       for t in unique_t]) / points_per_step
    gamma_data = jnp.exp(-alpha * unique_t / t_max)
    loss_data_u = jnp.mean(lse_u * gamma_data)
    loss_data_v = jnp.mean(lse_v * gamma_data)
    
    # Residual loss
    ru, rv = compute_residuals(state, params, colloc_points, constants)
    t_colloc = colloc_points[:, 2]
    gamma_res = jnp.exp(-alpha * t_colloc / t_max)
    loss_res_u = jnp.mean((ru**2) * gamma_res)
    loss_res_v = jnp.mean((rv**2) * gamma_res)
    
    total_loss = (
        constants['lambda_ic_u'] * loss_ic_u + constants['lambda_ic_v'] * loss_ic_v +
        constants['lambda_data_u'] * loss_data_u + constants['lambda_data_v'] * loss_data_v +
        constants['lambda_res_u'] * loss_res_u + constants['lambda_res_v'] * loss_res_v
    )
    return total_loss, {
        'total': total_loss, 'ic_u': loss_ic_u, 'ic_v': loss_ic_v, 
        'data_u': loss_data_u, 'data_v': loss_data_v, 'res_u': loss_res_u, 'res_v': loss_res_v
    }

@partial(jit)
def train_step(state, batch, colloc_points, constants, epoch):
    grad_fn = jax.value_and_grad(compute_loss, argnums=1, has_aux=True)
    (loss, metrics), grads = grad_fn(state, state.params, batch, colloc_points, constants, epoch)
    state = state.apply_gradients(grads=grads)
    return state, metrics

def compute_component_gradients(state, batch, colloc_points, constants):
    n_ic = batch['inputs'].shape[0] // 2
    inputs_ic = batch['inputs'][:n_ic]
    targets_ic = batch['targets'][:n_ic]
    inputs_data = batch['inputs'][n_ic:]
    targets_data = batch['targets'][n_ic:]

    def ic_loss(params):
        u, v = state.apply_fn(params, inputs_ic)
        return jnp.mean((u - targets_ic[:, 0])**2) + jnp.mean((v - targets_ic[:, 1])**2)

    def data_loss(params):
        u, v = state.apply_fn(params, inputs_data)
        return jnp.mean((u - targets_data[:, 0])**2) + jnp.mean((v - targets_data[:, 1])**2)

    def res_loss(params):
        ru, rv = compute_residuals(state, params, colloc_points, constants)
        return jnp.mean(ru**2) + jnp.mean(rv**2)

    grad_ic = grad(ic_loss)(state.params)
    grad_data = grad(data_loss)(state.params)
    grad_res = grad(res_loss)(state.params)

    def network_norm(grads, network):
        leaves = jax.tree_util.tree_leaves(grads['params'][network])
        return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
    
    return {
        'ic_u': network_norm(grad_ic, 'u'), 'ic_v': network_norm(grad_ic, 'v'),
        'data_u': network_norm(grad_data, 'u'), 'data_v': network_norm(grad_data, 'v'),
        'res_u': network_norm(grad_res, 'u'), 'res_v': network_norm(grad_res, 'v')
    }

def compute_residuals(state, params, colloc_points, constants):
    def residual_fn(x):
        u, v = state.apply_fn(params, x)
        u_grad = grad(lambda x: state.apply_fn(params, x[None])[0].squeeze())(x)
        v_grad = grad(lambda x: state.apply_fn(params, x[None])[1].squeeze())(x)
        u_hess = hessian(lambda x: state.apply_fn(params, x[None])[0].squeeze())(x)
        v_hess = hessian(lambda x: state.apply_fn(params, x[None])[1].squeeze())(x)
        ru = u_grad[2] - constants['ep1'] * (u_hess[0,0] + u_hess[1,1]) - \
             constants['b1'] * (1 - u) + constants['c1'] * u * v**2
        rv = v_grad[2] - constants['ep2'] * (v_hess[0,0] + v_hess[1,1]) + \
             constants['b2'] * v - constants['c2'] * u * v**2
        return ru, rv
    return vmap(residual_fn)(colloc_points)

def ntk_fn(apply_fn, params, inputs):
    def u_fn(p, x): return apply_fn(p, x[None])[0].squeeze()
    def v_fn(p, x): return apply_fn(p, x[None])[1].squeeze()
    J_u = vmap(jacfwd(u_fn, argnums=0), in_axes=(None, 0))(params, inputs)
    J_v = vmap(jacfwd(v_fn, argnums=0), in_axes=(None, 0))(params, inputs)
    J_u_flat = vmap(lambda j: jax.tree_util.tree_leaves(j))(J_u)
    J_v_flat = vmap(lambda j: jax.tree_util.tree_leaves(j))(J_v)
    K_u = jnp.mean(jnp.sum(J_u_flat**2, axis=-1))
    K_v = jnp.mean(jnp.sum(J_v_flat**2, axis=-1))
    return K_u, K_v

def update_lambdas(lambda_w, metrics, grad_norms, alpha, epoch, state, batch, colloc_points, constants):
    total_grad = sum(grad_norms.values())
    total_loss = metrics['total']
    loss_ratios = {k: metrics[k]/total_loss for k in ['ic_u', 'ic_v', 'data_u', 'data_v', 'res_u', 'res_v']}
    
    if epoch <= 24000:  # NTK in Stages 1-3
        def apply_fn(params, x): return state.apply_fn(params, x)
        ntk_ic_u, ntk_ic_v = ntk_fn(apply_fn, state.params, batch['inputs'][:1])
        ntk_data_u, ntk_data_v = ntk_fn(apply_fn, state.params, batch['inputs'][1:])
        ntk_res_u, ntk_res_v = ntk_fn(apply_fn, state.params, colloc_points)
        ntk_norms = {
            'ic_u': ntk_ic_u, 'ic_v': ntk_ic_v, 'data_u': ntk_data_u, 
            'data_v': ntk_data_v, 'res_u': ntk_res_u, 'res_v': ntk_res_v
        }
        total_ntk = sum(ntk_norms.values())
        ratios = {k: (1/ntk_norms[k]) * total_ntk / total_loss for k in lambda_w}
    else:  # Grad norm in Stages 4-5
        grad_ratios = {k: v/total_grad for k,v in grad_norms.items()}
        ratios = {k: 0.5 * (grad_ratios[k] + loss_ratios[k]) for k in lambda_w}
    
    new_lambda = {k: (1 - alpha) * lambda_w[k] + alpha * ratios[k] for k in lambda_w}
    return {k: jnp.clip(v, 0.1, 100.0) for k,v in new_lambda.items()}

def log_training_metrics(epoch, lambda_w, grad_norms, metrics, loss_val, alpha):
    log_str = "\n" + "-"*50 + "\n"
    log_str += f"Epoch {epoch} | Total Loss: {metrics['total']:.3e}\n"
    log_str += "-"*50 + "\n"
    log_str += f"Component Losses:\n"
    log_str += f"  IC: U={metrics['ic_u']:.2e} | V={metrics['ic_v']:.2e}\n"
    log_str += f"  Data: U={metrics['data_u']:.2e} | V={metrics['data_v']:.2e}\n"
    log_str += f"  Residual: U={metrics['res_u']:.2e} | V={metrics['res_v']:.2e}\n\n"
    log_str += "-"*50 + "\n"
    log_str += f"Gradient Norms:\n"
    log_str += f"  IC: U={grad_norms['ic_u']:.2e} | V={grad_norms['ic_v']:.2e}\n"
    log_str += f"  Data: U={grad_norms['data_u']:.2e} | V={grad_norms['data_v']:.2e}\n"
    log_str += f"  Residual: U={grad_norms['res_u']:.2e} | V={grad_norms['res_v']:.2e}\n\n"
    log_str += "-"*50 + "\n"
    log_str += f"Adaptive Weights (α={alpha:.3e}):\n"
    log_str += f"  IC: U={lambda_w['ic_u']:.2e} | V={lambda_w['ic_v']:.2e}\n"
    log_str += f"  Data: U={lambda_w['data_u']:.2e} | V={lambda_w['data_v']:.2e}\n"
    log_str += f"  Residual: U={lambda_w['res_u']:.2e} | V={lambda_w['res_v']:.2e}"
    log_str += "-"*50 + "\n"
    logging.info(log_str)

# ====================================================
# 4. Training Loop with Curriculum & Visualization
# ====================================================
def train(config, data):
    rng = jax.random.PRNGKey(config['seed'])
    state = create_train_state(rng, config, data)
    lambda_w = {'ic_u': 1.0, 'ic_v': 1.0, 'data_u': 0.1, 'data_v': 0.1, 'res_u': 100.0, 'res_v': 100.0}
    training_state = {'best_total': float('inf'), 'best_u': float('inf'), 'best_v': float('inf'), 
                      'window_size': config['min_window_size'], 'alpha': 0.4, 'beta': 1.00005, 'gamma': 0.99997}
    
    inputs, targets = gather_window_data(data, 0, len(data['t']))
    batch = {'inputs': inputs, 'targets': targets}
    
    with tqdm(total=config['epochs']) as pbar:
        for epoch in range(config['epochs']):
            stage = 1 if epoch <= 8000 else 2 if epoch <= 16000 else 3 if epoch <= 24000 else 4 if epoch <= 32000 else 5
            t_end_idx = [25, 50, 75, 100, 100][stage-1]
            t_end = data['t'][t_end_idx]
            colloc_points = sample_collocation_points(data['x'], data['y'], data['t'][0], t_end, stage)
            constants = freeze({
                'ep1': data['ep1'], 'ep2': data['ep2'], 'b1': data['b1'], 'b2': data['b2'],
                'c1': data['c1'], 'c2': data['c2'], 't_max': data['t'][-1], 'epochs': config['epochs'],
                't_data': data['t'], 'lambda_ic_u': 1.0, 'lambda_ic_v': 1.0,
                'lambda_data_u': 0.1 + 0.4 * (epoch / config['epochs']),
                'lambda_data_v': 0.1 + 0.4 * (epoch / config['epochs']),
                'lambda_res_u': 100.0 - 50.0 * (epoch / config['epochs']),
                'lambda_res_v': 100.0 - 50.0 * (epoch / config['epochs'])
            })
            state, metrics = train_step(state, batch, colloc_points, constants, epoch)
            
            if epoch % config['log_freq'] == 0:
                grad_norms = compute_component_gradients(state, batch, colloc_points, constants)
                lambda_w = update_lambdas(lambda_w, metrics, grad_norms, training_state['alpha'], 
                                         epoch, state, batch, colloc_points, constants)
                if metrics['total'] < training_state['best_total']:
                    training_state['alpha'] *= training_state['gamma']
                else:
                    training_state['alpha'] = min(training_state['alpha'] * training_state['beta'], 0.5)
                training_state['best_total'] = min(training_state['best_total'], metrics['total'])
                log_training_metrics(epoch, lambda_w, grad_norms, metrics, 
                                    {'u': metrics['ic_u'] + metrics['data_u'] + metrics['res_u'], 
                                     'v': metrics['ic_v'] + metrics['data_v'] + metrics['res_v']}, 
                                    training_state['alpha'])
                visualize_solution(state, data, epoch, t_end_idx)
            
            pbar.update(1)
            pbar.set_postfix({'total_loss': f"{metrics['total']:.2e}", 'stage': stage})
    
    save_checkpoint(state, epoch)
    return state

def sample_collocation_points(x, y, t_start, t_end, stage):
    x_min, x_max = float(x[0]), float(x[-1])
    y_min, y_max = float(y[0]), float(y[-1])
    t_min, t_max = float(t_start), float(t_end)
    res_0, r = 20, (200 / 20)**(1/4)
    resolutions = [int(res_0 * r**(i-1)) for i in range(5)]
    nx = ny = resolutions[stage - 1]
    x_grid = jnp.linspace(x_min, x_max, nx)
    y_grid = jnp.linspace(y_min, y_max, ny)
    t_grid = jnp.linspace(t_min, t_max, max(nx, 10))
    X, Y, T = jnp.meshgrid(x_grid, y_grid, t_grid, indexing='ij')
    return jnp.stack([X.flatten(), Y.flatten(), T.flatten()], axis=-1)

def visualize_solution(state, data, epoch, t_idx):
    x_np = np.array(data['x'])
    y_np = np.array(data['y'])
    t_val = float(data['t'][t_idx].item())
    
    X, Y = np.meshgrid(x_np, y_np, indexing='ij')
    os.makedirs(f"figures/epoch_{epoch}", exist_ok=True)
    
    inputs = get_all_points_at_time(data, t_idx)
    u_ref = data['usol'][t_idx]
    v_ref = data['vsol'][t_idx]
    
    u_pred, v_pred = state.apply_fn(state.params, inputs)
    u_pred = u_pred.reshape(u_ref.shape)
    v_pred = v_pred.reshape(v_ref.shape)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, (ref, pred, title) in enumerate(zip([u_ref, v_ref], [u_pred, v_pred], ['U', 'V'])):
        axes[i, 0].imshow(ref, cmap='jet', origin='lower')
        axes[i, 0].set_title(f"{title} Reference")
        axes[i, 1].imshow(pred, cmap='jet', origin='lower')
        axes[i, 1].set_title(f"{title} Predicted")
        axes[i, 2].imshow(np.abs(ref - pred), cmap='jet', origin='lower')
        axes[i, 2].set_title("Absolute Error")
    
    logging.info(f"[VIZ]==[Visualisation Saved to figures/epoch_{epoch}/solution_{t_idx}.png]==[VIZ]")
    plt.savefig(f"figures/epoch_{epoch}/solution_{t_idx}.png")
    plt.close()

def save_checkpoint(state, epoch):
    checkpoint = serialization.to_state_dict(state)
    with open(f"params_epoch_{epoch}.pkl", 'wb') as f:
        pickle.dump(checkpoint, f)

# ====================================================
# 6. Configuration & Execution
# ====================================================
config = {
    'seed': 42,
    'hidden_dims': [64, 64, 32],
    'fourier_features': 128,
    'lr': 1e-3,
    'max_grad_norm': 0.1,
    'epochs': 40000,
    'log_freq': 100,
    'num_colloc': 4096*16,
    'min_window_size': 101,
    'init_N': 10,
    'init_M': 100,
    'init_alpha': 5.0
}

if __name__ == "__main__":
    figures_dir = "/home/rbhavsa4/figures"
    try:
        if os.path.exists(figures_dir):
            shutil.rmtree(figures_dir)
            logging.info("Removed figures directory: %s", figures_dir)
    except Exception as e:
        logging.warning("Could not remove figures directory %s: %s", figures_dir, e)
    
    dataset_file = "grey_scott.mat"
    try:
        data = load_grey_scott_mat(dataset_file)
        logging.info("Successfully loaded dataset from '%s'.", dataset_file)
    except Exception as e:
        logging.error("Failed to load dataset from '%s': %s", dataset_file, e)

    logging.info("Starting training loop.")
    try:
        trained_state = train(config, data)
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error("Training failed: %s", e)

    ckpt_dir = '/home/rbhavsa4/checkpoints'
    try:
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=trained_state, step=trained_state.step, overwrite=True)
        logging.info("Model checkpoint saved at '%s' (step: %s).", ckpt_dir, trained_state.step)
    except Exception as e:
        logging.error("Failed to save model checkpoint: %s", e)