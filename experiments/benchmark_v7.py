import os
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import optax
import numpy as np
from flax.core import freeze
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm
from flax.training import train_state
from functools import partial
from jax import random, jit, grad, vmap, hessian
from jax.flatten_util import ravel_pytree

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
    return jnp.stack([
        X.flatten(), 
        Y.flatten(), 
        jnp.full(Nx * Ny, data['t'][t_idx])
    ], axis=-1)

# ====================================================
# 2. Neural Network Architecture
# ====================================================
class FourierFeatureLayer(nn.Module):
    num_features: int
    scale: float = 6.0
    learnable: bool = True

    @nn.compact
    def __call__(self, inputs):
        if self.learnable:
            B = self.param('B', nn.initializers.normal(self.scale),
                           (self.num_features // 2, inputs.shape[-1]))
        else:
            key = self.make_rng('params')
            B = self.variable('non_trainable', 'B', 
                              lambda: random.normal(key, (self.num_features // 2, inputs.shape[-1])) * self.scale)
            B = B.value
        proj = inputs @ B.T
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)

class GrayScottPINN(nn.Module):
    hidden_dims: list = struct.field(default_factory=lambda: [128, 128, 64])
    fourier_features: int = struct.field(default=128)
    fourier_scale: float = struct.field(default=6.0)
    
    @nn.compact
    def __call__(self, x):
        x_wrap = jnp.concatenate([
            (x[..., :1] + 1) % 2 - 1,
            (x[..., 1:2] + 1) % 2 - 1,
            x[..., 2:]
        ], axis=-1)
        
        ff = FourierFeatureLayer(self.fourier_features, self.fourier_scale)(x_wrap)
        
        u = self._build_mlp(ff, name='u')
        v = self._build_mlp(ff, name='v')
        return u, v
    
    def _build_mlp(self, x, name):
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f'{name}_dense_{i}')(x)
            x = nn.gelu(x) + 0.1 * jnp.sin(4 * jnp.pi * x)
        return nn.Dense(1, name=f'{name}_output')(x).squeeze()

# ====================================================
# 3. Training Utilities & Loss Calculation
# ====================================================
def create_train_state(rng, config, data):
    model = GrayScottPINN(
        hidden_dims=config['hidden_dims'],
        fourier_features=config['fourier_features']
    )
    
    dummy_input = jnp.ones((1, 3))
    params = model.init(rng, dummy_input)
    
    tx = optax.chain(
        optax.clip_by_global_norm(config['max_grad_norm']),
        optax.adamw(config['lr'], weight_decay=1e-4)
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def compute_loss(state, params, batch, colloc_points, ic_points, ic_targets, constants):
    # Data loss
    (u_pred, v_pred) = state.apply_fn(params, batch['inputs'])
    data_loss_u = jnp.mean((u_pred - batch['targets'][:, 0])**2)
    data_loss_v = jnp.mean((v_pred - batch['targets'][:, 1])**2)
    
    # PDE residual loss
    def residual_fn(x):
        u, v = state.apply_fn(params, x)
        u_grad = grad(lambda x: state.apply_fn(params, x[None])[0])(x)
        v_grad = grad(lambda x: state.apply_fn(params, x[None])[1])(x)
        u_hess = hessian(lambda x: state.apply_fn(params, x[None])[0])(x)
        v_hess = hessian(lambda x: state.apply_fn(params, x[None])[1])(x)
        ru = u_grad[2] - constants['ep1'] * (u_hess[0, 0] + u_hess[1, 1]) - \
             constants['b1'] * (1 - u) + constants['c1'] * u * v**2
        rv = v_grad[2] - constants['ep2'] * (v_hess[0, 0] + v_hess[1, 1]) + \
             constants['b2'] * v - constants['c2'] * u * v**2
        return ru**2, rv**2

    ru, rv = vmap(residual_fn)(colloc_points)
    res_loss_u = jnp.mean(ru)
    res_loss_v = jnp.mean(rv)
    
    # Initial Condition (IC) loss
    u_ic_pred, v_ic_pred = state.apply_fn(params, ic_points)
    ic_loss_u = jnp.mean((u_ic_pred - ic_targets[:, 0])**2)
    ic_loss_v = jnp.mean((v_ic_pred - ic_targets[:, 1])**2)
    ic_loss = ic_loss_u + ic_loss_v
    
    total_loss = (
        constants['lambda_data'] * (data_loss_u + data_loss_v) +
        constants['lambda_res'] * (res_loss_u + res_loss_v) +
        constants['lambda_ic'] * ic_loss
    )
    
    return total_loss, {
        'total': total_loss,
        'data_u': data_loss_u,
        'data_v': data_loss_v,
        'res_u': res_loss_u,
        'res_v': res_loss_v,
        'ic': ic_loss
    }

@partial(jit)
def train_step(state, batch, colloc_points, ic_points, ic_targets, constants):
    grad_fn = jax.value_and_grad(compute_loss, argnums=1, has_aux=True)
    (loss, metrics), grads = grad_fn(state, state.params, batch, colloc_points, ic_points, ic_targets, constants)
    state = state.apply_gradients(grads=grads)
    return state, metrics

# ----------------------------------------------------
# Helper functions for advanced logging
# ----------------------------------------------------
def tree_l2_norm(tree):
    return jnp.sqrt(sum([jnp.sum(x**2) for x in jax.tree_util.tree_leaves(tree)]))

def filter_tree(tree, key_filter):
    # Returns a filtered subtree containing only keys that include key_filter.
    return {k: v for k, v in tree.items() if key_filter in k}

def compute_gradient_norms(state, batch, colloc_points, ic_points, ic_targets, constants):
    # Compute gradients for the three loss components separately.
    ic_loss_fn = lambda params: jnp.mean((state.apply_fn(params, ic_points)[0] - ic_targets[:, 0])**2) + \
                                 jnp.mean((state.apply_fn(params, ic_points)[1] - ic_targets[:, 1])**2)
    grad_ic = jax.grad(ic_loss_fn)(state.params)
    
    data_loss_fn = lambda params: jnp.mean((state.apply_fn(params, batch['inputs'])[0] - batch['targets'][:, 0])**2) + \
                                   jnp.mean((state.apply_fn(params, batch['inputs'])[1] - batch['targets'][:, 1])**2)
    grad_data = jax.grad(data_loss_fn)(state.params)
    
    def res_loss_fn(params):
        def residual_fn(x):
            u, v = state.apply_fn(params, x)
            u_grad = grad(lambda x: state.apply_fn(params, x[None])[0])(x)
            v_grad = grad(lambda x: state.apply_fn(params, x[None])[1])(x)
            u_hess = hessian(lambda x: state.apply_fn(params, x[None])[0])(x)
            v_hess = hessian(lambda x: state.apply_fn(params, x[None])[1])(x)
            ru = u_grad[2] - constants['ep1'] * (u_hess[0, 0] + u_hess[1, 1]) - \
                 constants['b1'] * (1 - state.apply_fn(params, x)[0]) + constants['c1'] * state.apply_fn(params, x)[0] * state.apply_fn(params, x)[1]**2
            rv = v_grad[2] - constants['ep2'] * (v_hess[0, 0] + v_hess[1, 1]) + constants['b2'] * state.apply_fn(params, x)[1] - constants['c2'] * state.apply_fn(params, x)[0] * state.apply_fn(params, x)[1]**2
            return ru**2 + rv**2
        return jnp.mean(vmap(residual_fn)(colloc_points))
    grad_res = jax.grad(res_loss_fn)(state.params)
    
    # Separate the gradients for the U- and V-branches.
    grad_ic_u = tree_l2_norm(filter_tree(grad_ic['params'], 'u'))
    grad_ic_v = tree_l2_norm(filter_tree(grad_ic['params'], 'v'))
    grad_data_u = tree_l2_norm(filter_tree(grad_data['params'], 'u'))
    grad_data_v = tree_l2_norm(filter_tree(grad_data['params'], 'v'))
    grad_res_u = tree_l2_norm(filter_tree(grad_res['params'], 'u'))
    grad_res_v = tree_l2_norm(filter_tree(grad_res['params'], 'v'))
    
    return {'ic': {'u': grad_ic_u, 'v': grad_ic_v},
            'data': {'u': grad_data_u, 'v': grad_data_v},
            'res': {'u': grad_res_u, 'v': grad_res_v}}

def compute_ntk(state, params, subset):
    # Compute an approximate NTK for U and V outputs on a small subset.
    def grad_u(x):
        grad_fn = jax.grad(lambda p: state.apply_fn(p, x)[0])
        g = grad_fn(params)
        flat, _ = ravel_pytree(g)
        return flat
    def grad_v(x):
        grad_fn = jax.grad(lambda p: state.apply_fn(p, x)[1])
        g = grad_fn(params)
        flat, _ = ravel_pytree(g)
        return flat
    grads_u = jnp.stack([grad_u(x) for x in subset])
    grads_v = jnp.stack([grad_v(x) for x in subset])
    ntk_u = grads_u @ grads_u.T
    ntk_v = grads_v @ grads_v.T
    return ntk_u, ntk_v

def compute_l2_error(state, data, t_idx):
    inputs = get_all_points_at_time(data, t_idx)
    u_pred, v_pred = state.apply_fn(state.params, inputs)
    u_ref = data['usol'][t_idx].flatten()
    v_ref = data['vsol'][t_idx].flatten()
    l2_u = jnp.sqrt(jnp.mean((u_pred.flatten() - u_ref)**2))
    l2_v = jnp.sqrt(jnp.mean((v_pred.flatten() - v_ref)**2))
    return l2_u, l2_v

# ====================================================
# 4. Training Loop with Advanced Logging & Visualization
# ====================================================
def train(config, data):
    rng = jax.random.PRNGKey(config['seed'])
    state = create_train_state(rng, config, data)
    
    window_size = config['min_window_size']
    t_start = 0
    best_loss = float('inf')
    
    # Precompute initial condition points and targets (at t=0)
    ic_points = get_all_points_at_time(data, 0)
    ic_targets = jnp.stack([data['usol'][0].flatten(), data['vsol'][0].flatten()], axis=1)
    
    # Initialize adaptive loss weight variables and smoothing factors
    alpha_u = 0.9
    alpha_v = 0.9
    lambda_ic_u = 0.1
    lambda_res_u = 0.1
    lambda_data_u = 0.1
    lambda_ic_v = 0.1
    lambda_res_v = 0.1
    lambda_data_v = 0.1
    
    with tqdm(total=config['epochs']) as pbar:
        for epoch in range(config['epochs']):
            if epoch % config['window_update_freq'] == 0:
                window_size = min(window_size + 2, config['max_window_size'])
                t_start = min(t_start + 1, len(data['t']) - window_size - 1)
            
            inputs_t0 = get_all_points_at_time(data, t_start)
            inputs_t1 = get_all_points_at_time(data, t_start + window_size - 1)
            batch_inputs = jnp.concatenate([inputs_t0, inputs_t1], axis=0)
            
            usol_t0 = data['usol'][t_start].flatten()
            vsol_t0 = data['vsol'][t_start].flatten()
            usol_t1 = data['usol'][t_start + window_size - 1].flatten()
            vsol_t1 = data['vsol'][t_start + window_size - 1].flatten()
            
            targets_t0 = jnp.stack([usol_t0, vsol_t0], axis=1)
            targets_t1 = jnp.stack([usol_t1, vsol_t1], axis=1)
            batch_targets = jnp.concatenate([targets_t0, targets_t1], axis=0)
            
            batch = {
                'inputs': batch_inputs,
                'targets': batch_targets
            }
            
            t_min = float(data['t'][t_start])
            t_max = float(data['t'][t_start + window_size - 1])
            
            colloc_points = sample_collocation_points(
                config['num_colloc'],
                data['x'],
                data['y'],
                t_min,
                t_max
            )
            
            constants = freeze({
                'ep1': data['ep1'],
                'ep2': data['ep2'],
                'b1': data['b1'],
                'b2': data['b2'],
                'c1': data['c1'],
                'c2': data['c2'],
                'lambda_data': config['lambda_data'],
                'lambda_res': config['lambda_res'],
                'lambda_ic': config['lambda_ic']
            })
            
            state, metrics = train_step(state, batch, colloc_points, ic_points, ic_targets, constants)
            
            # Simple progress bar update
            pbar.set_postfix({
                'loss': metrics['total'],
                'win': f"{t_start}-{t_start + window_size - 1}",
                'data_u': metrics['data_u'],
                'res_v': metrics['res_v'],
                'ic': metrics['ic']
            })
            
            # Every 1000 epochs (except the first), compute advanced logging metrics
            if epoch % 1000 == 0 and epoch != 0:
                # Compute gradient norms
                grad_norms = compute_gradient_norms(state, batch, colloc_points, ic_points, ic_targets, constants)
                gnorms_u = grad_norms['ic']['u'], grad_norms['res']['u'], grad_norms['data']['u']
                gnorms_v = grad_norms['ic']['v'], grad_norms['res']['v'], grad_norms['data']['v']
                total_norm_u = grad_norms['ic']['u'] + grad_norms['res']['u'] + grad_norms['data']['u']
                total_norm_v = grad_norms['ic']['v'] + grad_norms['res']['v'] + grad_norms['data']['v']
                total_norm = total_norm_u + total_norm_v

                # Use a subset (up to 10 points) from the collocation points to compute NTK
                subset = colloc_points[:10] if colloc_points.shape[0] > 10 else colloc_points
                mat_u, mat_v = compute_ntk(state, state.params, subset)
                cond_ntk_u = jnp.linalg.cond(jnp.array(mat_u)) if subset.shape[0] > 0 else jnp.nan
                cond_ntk_v = jnp.linalg.cond(jnp.array(mat_v)) if subset.shape[0] > 0 else jnp.nan

                # Update adaptive loss weights using exponential moving averages.
                lambda_ic_u = alpha_u * lambda_ic_u + (1 - alpha_u) * (total_norm / (grad_norms['ic']['u'] + 1e-8))
                lambda_res_u = alpha_u * lambda_res_u + (1 - alpha_u) * (total_norm / (grad_norms['res']['u'] + 1e-8))
                lambda_data_u = alpha_u * lambda_data_u + (1 - alpha_u) * (total_norm / (grad_norms['data']['u'] + 1e-8))

                lambda_ic_v = alpha_v * lambda_ic_v + (1 - alpha_v) * (total_norm / (grad_norms['ic']['v'] + 1e-8))
                lambda_res_v = alpha_v * lambda_res_v + (1 - alpha_v) * (total_norm / (grad_norms['res']['v'] + 1e-8))
                lambda_data_v = alpha_v * lambda_data_v + (1 - alpha_v) * (total_norm / (grad_norms['data']['v'] + 1e-8))

                # Clip the adaptive weights to avoid extreme values.
                lambda_ic_u = jnp.clip(lambda_ic_u, 1e-2, 5e-1)
                lambda_ic_v = jnp.clip(lambda_ic_v, 1e-2, 5e-1)
                lambda_data_u = jnp.clip(lambda_data_u, 1e-5, 5e-4)
                lambda_data_v = jnp.clip(lambda_data_v, 1e-5, 5e-4)

                # Compute L2 error at the window end.
                l2_u, l2_v = compute_l2_error(state, data, t_start + window_size - 1)

                # Formatting for logging
                sep = "│ "
                grad_fmt = lambda x: f"{x:.2e}".ljust(8)
                weight_fmt = lambda x: f"{x:.2e}".rjust(8)
                
                # Detailed formatted log output.
                log_str = (
                    f"\n╭──────── Window {t_start:2.0f} | Epoch {epoch:5.0f} ────────╮\n"
                    f"│   U-Loss: {metrics['data_u']:.4e} │  NTK U: {cond_ntk_u:.2e} │\n"
                    f"│   V-Loss: {metrics['data_v']:.4e} │  NTK V: {cond_ntk_v:.2e} │\n"
                    f"├──────────── L2 Error ────────────────│\n"
                    f"│   U-L2: {l2_u:.3e}    │   V-L2: {l2_v:.3e}   │\n"
                    f"├──────── Gradient Norms ──────────────│\n"
                    f"│    Component       │  U-Net   │  V-Net   │\n"
                    f"│ {sep}IC Loss         │ {grad_fmt(grad_norms['ic']['u'])} │ {grad_fmt(grad_norms['ic']['v'])} │\n"
                    f"│ {sep}Residual Loss   │ {grad_fmt(grad_norms['res']['u'])} │ {grad_fmt(grad_norms['res']['v'])} │\n"
                    f"│ {sep}Data Loss       │ {grad_fmt(grad_norms['data']['u'])} │ {grad_fmt(grad_norms['data']['v'])} │\n"
                    f"├────── Loss Weights ────────────────│\n"
                    f"│    Component       │  U-Net   │  V-Net   │\n"
                    f"│ {sep}IC Weight       │ {weight_fmt(lambda_ic_u)} │ {weight_fmt(lambda_ic_v)} │\n"
                    f"│ {sep}Residual Weight │ {weight_fmt(lambda_res_u)} │ {weight_fmt(lambda_res_v)} │\n"
                    f"│ {sep}Data Weight     │ {weight_fmt(lambda_data_u)} │ {weight_fmt(lambda_data_v)} │\n"
                    f"╰────────────────────────────────────────╯"
                )
                print(log_str)
            
            # Save a checkpoint and visualize if improvement
            if metrics['total'] < best_loss:
                visualize_solution(state, data, epoch, t_start + window_size - 1)
                best_loss = metrics['total']
                save_checkpoint(state, epoch)
            
            pbar.update(1)
    
    return state

# ====================================================
# 5. Visualization & Utilities
# ====================================================
def visualize_solution(state, data, epoch, t_idx):
    x_np = np.array(data['x'])
    y_np = np.array(data['y'])
    os.makedirs(f"figures/epoch_{epoch}", exist_ok=True)
    
    inputs = get_all_points_at_time(data, t_idx)
    u_ref = data['usol'][t_idx]
    v_ref = data['vsol'][t_idx]
    
    u_pred, v_pred = state.apply_fn(state.params, inputs)
    u_pred = u_pred.reshape(u_ref.shape)
    v_pred = v_pred.reshape(v_ref.shape)
    
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
    
    plt.savefig(f"figures/epoch_{epoch}/solution_{t_idx}.png")
    plt.close()

def sample_collocation_points(num_points, x, y, t_start, t_end):
    key = random.PRNGKey(np.random.randint(int(1e6)))
    key, x_key, y_key, t_key = random.split(key, 4)
    
    x_min = float(x[0].item())
    x_max = float(x[-1].item())
    y_min = float(y[0].item())
    y_max = float(y[-1].item())
    
    x_pts = random.uniform(x_key, (num_points,), minval=x_min, maxval=x_max)
    y_pts = random.uniform(y_key, (num_points,), minval=y_min, maxval=y_max)
    t_pts = random.uniform(t_key, (num_points,), minval=t_start, maxval=t_end)
    
    return jnp.stack([x_pts, y_pts, t_pts], axis=-1)

def save_checkpoint(state, epoch):
    with open(f"params_epoch_{epoch}.pkl", 'wb') as f:
        pickle.dump(state.params, f)

# ====================================================
# 6. Configuration & Execution
# ====================================================
config = {
    'seed': 42,
    'hidden_dims': [256, 256, 128],
    'fourier_features': 256,
    'lr': 1e-3,
    'max_grad_norm': 1.0,
    'epochs': 10000,
    'log_freq': 100,        # basic logging frequency via pbar
    'num_colloc': 8192,
    'min_window_size': 5,
    'max_window_size': 50,
    'window_update_freq': 250,
    'lambda_data': 1.0,
    'lambda_res': 0.1,
    'lambda_ic': 1.0  # weight for the initial condition loss
}

if __name__ == "__main__":
    data = load_grey_scott_mat("grey_scott.mat")
    trained_state = train(config, data)
    save_checkpoint(trained_state, "final")