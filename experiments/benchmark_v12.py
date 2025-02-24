# --- Configure Logging First ---
import logging
import warnings

class SuppressMessagesFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Tensorflow library not found" in msg or "already exists in the registry" in msg:
            return False
        return True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)
logging.info("Loggings and Warnings have been configured.")
logging.getLogger('orbax').setLevel(logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('orbax.checkpoint._src.handlers').setLevel(logging.WARNING)
logging.getLogger('orbax.checkpoint.test_utils').setLevel(logging.WARNING)

root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(SuppressMessagesFilter())

# --- Standard Library Imports ---
import os
import shutil
import pickle
from functools import partial
import traceback
logging.info("Standard library imports loaded successfully.")

# --- Third-Party Imports: NumPy, SciPy, Matplotlib, and TQDM ---
try:
    import numpy as np
    import scipy.io
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    logging.info("NumPy, SciPy, Matplotlib, and tqdm imported successfully.")
except Exception as e:
    logging.error("Failed to import NumPy/SciPy/Matplotlib/tqdm: %s\n%s", str(e), traceback.format_exc())

# --- Third-Party Imports: JAX ---
try:
    import jax
    import jax.numpy as jnp
    from jax import random, jit, grad, vmap, hessian, jacfwd
    logging.info("JAX and related modules imported successfully.")
except Exception as e:
    logging.error("Failed to import JAX: %s\n%s", str(e), traceback.format_exc())

# --- Third-Party Imports: Flax ---
try:
    import flax.linen as nn
    from flax import struct
    from flax import serialization
    from flax.core import freeze
    from flax.training import train_state, checkpoints
    logging.info("Flax libraries imported successfully.")
except Exception as e:
    logging.error("Failed to import Flax: %s\n%s", str(e), traceback.format_exc())

# --- Third-Party Imports: Optax ---
try:
    import optax
    logging.info("Optax imported successfully.")
except Exception as e:
    logging.error("Failed to import Optax: %s\n%s", str(e), traceback.format_exc())

logging.info("All imports have been loaded successfully.")

os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_default_matmul_precision", "bfloat16")

# ====================================================
# 1. Data Loading & Preprocessing
# ====================================================
def load_grey_scott_mat(file_path):
    try:
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
    except Exception as e:
        logging.error("Failed in load_grey_scott_mat: %s\n%s", str(e), traceback.format_exc())
        raise

def get_all_points_at_time(data, t_idx):
    try:
        X, Y = jnp.meshgrid(data['x'], data['y'], indexing='ij')
        return jnp.stack([X.flatten(), Y.flatten(), jnp.full_like(X.flatten(), data['t'][t_idx])], axis=-1)
    except Exception as e:
        logging.error("Failed in get_all_points_at_time: %s\n%s", str(e), traceback.format_exc())
        raise

@jit
def get_apt_at_time(usol, vsol, x_data, y_data, t_idx, t_val):
    try:
        X, Y = jnp.meshgrid(x_data, y_data, indexing='ij')
        Xf, Yf = X.flatten(), Y.flatten()
        inputs = jnp.stack([Xf, Yf, jnp.full_like(Xf, t_val)], axis=-1)
        targets = jnp.stack([usol[t_idx].flatten(), vsol[t_idx].flatten()], axis=-1)
        return inputs, targets
    except Exception as e:
        logging.error("Failed in get_apt_at_time: %s\n%s", str(e), traceback.format_exc())
        raise


@partial(jit, static_argnums=(1, 2))
def gather_window_data(data, start_idx, end_idx):
    try:
        usol, vsol, t_data, x_data, y_data = data['usol'], data['vsol'], data['t'], data['x'], data['y']

        def get_at_time(ti):
            return get_apt_at_time(usol, vsol, x_data, y_data, ti, jnp.array(t_data[ti], dtype=jnp.float32))

        inputs, targets = vmap(get_at_time)(jnp.arange(start_idx, end_idx))
        return inputs.reshape(-1, 3), targets.reshape(-1, 2)
    except Exception as e:
        logging.error("Failed in gather_window_data: %s\n%s", str(e), traceback.format_exc())
        raise

# ====================================================
# 2. Neural Network Architecture
# ====================================================
class FourierFeatureLayer(nn.Module):
    num_features: int
    input_dim: int = 3
    scale: float = 6.0

    @nn.compact
    def __call__(self, inputs):
        try:
            assert inputs.shape[-1] == self.input_dim, f"Expected input dimension {self.input_dim}, got {inputs.shape[-1]}"
            B = self.param('B', nn.initializers.normal(self.scale), (self.num_features // 2, self.input_dim))
            
            proj = jnp.dot(inputs, B.T)  # More efficient than `@`
            return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        except Exception as e:
            logging.error("Failed in FourierFeatureLayer: %s\n%s", str(e), traceback.format_exc())
            raise

class MLP(nn.Module):
    hidden_dims: list
    name: str
    
    @nn.compact
    def __call__(self, x):
        try:
            for i, dim in enumerate(self.hidden_dims):
                x = nn.Dense(dim, name=f'{self.name}_dense_{i}', kernel_init=nn.initializers.he_normal())(x)
                # x = nn.BatchNorm(dim, name=f'{self.name}_batchnorm_{i}', momentum=0.9, epsilon=1e-5)(x)
                x = nn.tanh(x)
            return nn.Dense(1, name=f'{self.name}_output', kernel_init=nn.initializers.he_normal())(x).squeeze()
        except Exception as e:
            logging.error("Failed in MLP: %s\n%s", str(e), traceback.format_exc())
            raise

class GrayScottPINN(nn.Module):
    hidden_dims: list = struct.field(default_factory=lambda: [256, 256, 128])
    fourier_features: int = struct.field(default=64)
    fourier_scale: float = struct.field(default=6.0)

    def fourier_feature_layer(self, params, x):
        try:
            B = params['params']['FourierFeatureLayer_0']['B']
            proj = x @ B.T
            return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        except Exception as e:
            logging.error("Failed in fourier_feature_layer: %s\n%s", str(e), traceback.format_exc())
            raise

    @nn.compact
    def __call__(self, x):
        try:
            assert x.shape[-1] == 3, f"Input must have 3 features (x,y,t), got {x.shape[-1]}"
            ff = FourierFeatureLayer(self.fourier_features, input_dim=3, scale=self.fourier_scale)(x)
            u = MLP(self.hidden_dims, name='u')(ff)
            v = MLP(self.hidden_dims, name='v')(ff)
            return u, v
        except Exception as e:
            logging.error("Failed in GrayScottPINN: %s\n%s", str(e), traceback.format_exc())
            raise

# ====================================================
# 3. Training Utilities & Loss Calculation
# ====================================================
def create_train_state(rng, config, data):
    try:
        model = GrayScottPINN(hidden_dims=config['hidden_dims'], fourier_features=config['fourier_features'])
        dummy_input = jnp.ones((1, 3))
        params = model.init(rng, dummy_input)
        # params = model.initialize_causally(params, data, config, rng)
        tx = optax.chain(
            optax.clip_by_global_norm(config['max_grad_norm']),
            optax.lamb(optax.warmup_cosine_decay_schedule(
                        init_value=0.0,
                        peak_value=config['lr'],
                        warmup_steps=1000,
                        decay_steps=config['epochs']
                    ))
            )
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    except Exception as e:
        logging.error("Failed in create_train_state: %s\n%s", str(e), traceback.format_exc())
        raise

@partial(jit)
def compute_loss(state, params, batch, colloc_points, constants):
    try:
        u_pred, v_pred = state.apply_fn(params, batch['inputs'])
        targets_ic = batch['targets'][:1]
        targets_data = batch['targets'][1:]
        
        loss_ic_u = jnp.mean((u_pred[:1] - targets_ic[:, 0])**2)
        loss_ic_v = jnp.mean((v_pred[:1] - targets_ic[:, 1])**2)

        gamma_data = batch['gamma_data'][1:]
        loss_data_u = jnp.mean((u_pred[1:] - targets_data[:, 0])**2 * gamma_data)
        loss_data_v = jnp.mean((v_pred[1:] - targets_data[:, 1])**2 * gamma_data)
        
        ru, rv = compute_residuals(state, params, colloc_points, constants)
        gamma_res = colloc_points['gamma_res']
        loss_res_u = jnp.mean((ru**2) * gamma_res) 
        loss_res_v = jnp.mean((rv**2) * gamma_res)
        
        total_loss = (
            (constants['lambda_ic_u'] * loss_ic_u) +
            (constants['lambda_ic_v'] * loss_ic_v) +
            (constants['lambda_data_u'] * loss_data_u) +
            (constants['lambda_data_v'] * loss_data_v) +
            (constants['lambda_res_u'] * loss_res_u) +
            (constants['lambda_res_v'] * loss_res_v)
        )
        return total_loss, {
            'total': total_loss, 'ic_u': loss_ic_u, 'ic_v': loss_ic_v, 
            'data_u': loss_data_u, 'data_v': loss_data_v, 'res_u': loss_res_u, 'res_v': loss_res_v
        }
    except Exception as e:
        logging.error("Failed in compute_loss: %s\n%s", str(e), traceback.format_exc())
        raise

@partial(jit)
def train_step(state, batch, colloc_points, constants):
    try:
        grad_fn = jax.value_and_grad(compute_loss, argnums=1, has_aux=True)
        (loss, metrics), grads = grad_fn(state, state.params, batch, colloc_points, constants)
        state = state.apply_gradients(grads=grads)
        return state, metrics
    except Exception as e:
        logging.error("Failed in train_step: %s\n%s", str(e), traceback.format_exc())
        raise

@jax.jit
def compute_residuals(state, params, colloc_points, constants):
    try:
        def compute_grad_and_laplacian(fn, x):
            grad_fn = jax.grad(fn)
            grad_val = grad_fn(x)
            laplacian = jnp.trace(jax.jacfwd(grad_fn)(x))  # Directly trace the Hessian
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

        return jax.vmap(residual_fn)(colloc_points['points'])  # Vectorize over collocation points
    except Exception as e:
        logging.error("Failed in compute_residuals: %s\n%s", str(e), traceback.format_exc())
        raise

@jax.jit
def compute_approx_grad_norms(params, batch, colloc_points, constants, state):
    try:
        # Define separate loss functions for u and v
        def ic_loss_u(params):
            u, _ = state.apply_fn(params, batch['inputs'][:1])
            return jnp.mean((u - batch['targets'][:1, 0])**2)

        def ic_loss_v(params):
            _, v = state.apply_fn(params, batch['inputs'][:1])
            return jnp.mean((v - batch['targets'][:1, 1])**2)

        def data_loss_u(params):
            u, _ = state.apply_fn(params, batch['inputs'][1:])
            return jnp.mean((u - batch['targets'][1:, 0])**2 * batch['gamma_data'][1:])

        def data_loss_v(params):
            _, v = state.apply_fn(params, batch['inputs'][1:])
            return jnp.mean((v - batch['targets'][1:, 1])**2 * batch['gamma_data'][1:])

        def res_loss_u(params):
            ru, _ = compute_residuals(state, params, colloc_points, constants)
            return jnp.mean(ru**2 * colloc_points['gamma_res'])

        def res_loss_v(params):
            _, rv = compute_residuals(state, params, colloc_points, constants)
            return jnp.mean(rv**2 * colloc_points['gamma_res'])

        # Compute gradients for each separate loss
        grad_ic_u = grad(ic_loss_u)(params)
        grad_ic_v = grad(ic_loss_v)(params)
        grad_data_u = grad(data_loss_u)(params)
        grad_data_v = grad(data_loss_v)(params)
        grad_res_u = grad(res_loss_u)(params)
        grad_res_v = grad(res_loss_v)(params)

        def network_norm(grads, network):
            leaves = jax.tree_util.tree_leaves(grads['params'][network])
            return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))

        # Compute norms for u-network and v-network
        grad_norms = {
            'ic_u': network_norm(grad_ic_u, 'u'),
            'ic_v': network_norm(grad_ic_v, 'v'),
            'data_u': network_norm(grad_data_u, 'u'),
            'data_v': network_norm(grad_data_v, 'v'),
            'res_u': network_norm(grad_res_u, 'u'),
            'res_v': network_norm(grad_res_v, 'v')
        }
        return grad_norms
    except Exception as e:
        logging.error("Failed in compute_approx_grad_norms: %s\n%s", str(e), traceback.format_exc())
        raise

@partial(jit)
def update_lambdas(lambda_w, metrics, grad_norms, initial_losses, epoch, state, batch, colloc_points, constants, beta=0.9):
    try:
        # Convert lambda_w and grad_norms to JAX arrays for vectorization
        lambda_w_array = jnp.array([lambda_w[c] for c in ['ic_u', 'data_u', 'res_u', 'ic_v', 'data_v', 'res_v']])
        grad_norms_array = jnp.array([grad_norms[c] for c in ['ic_u', 'data_u', 'res_u', 'ic_v', 'data_v', 'res_v']])

        # Identify u and v components (indices 0-2 for u, 3-5 for v)
        is_u = jnp.array([1, 1, 1, 0, 0, 0], dtype=jnp.bool_)
        total_norms = jnp.where(is_u, 
                                jnp.sum(grad_norms_array[:3]),  # Sum of u components
                                jnp.sum(grad_norms_array[3:]))  # Sum of v components)

        # Update lambda_w using EMA
        new_lambda_w = beta * lambda_w_array + (1 - beta) * (total_norms / (grad_norms_array + 1e-8))

        # Normalize u and v components separately
        sum_u = jnp.sum(new_lambda_w[:3])
        sum_v = jnp.sum(new_lambda_w[3:])
        normed_lambda_w = jnp.where(
            is_u,
            new_lambda_w / jnp.where(sum_u > 0, sum_u, 1.0),  # Avoid division by zero
            new_lambda_w / jnp.where(sum_v > 0, sum_v, 1.0)
        )

        # Clip weights
        clipped_lambda_w = jnp.clip(normed_lambda_w, 0.01, 1.0)

        # Update the dictionary
        updated_lambda_w = dict(zip(['ic_u', 'data_u', 'res_u', 'ic_v', 'data_v', 'res_v'], clipped_lambda_w))
        return updated_lambda_w

    except Exception as e:
        logging.error("Failed in update_lambdas: %s\n%s", str(e), traceback.format_exc())
        raise

# ====================================================
# 4. Training Loop with Curriculum & Visualization
# ====================================================
def logging_metrics(lambda_w, metrics, loss_val, epoch):
    try:
        sep = "│ "
        grad_fmt = lambda x: f"{x:.2e}".ljust(8) if x else "N/A".ljust(8)
        weight_fmt = lambda x: f"{x:.2e}".rjust(8)
        print(
            f"\n╭──────── Window {float(epoch):5.0} | Epoch {float(epoch):5.0} ────────╮\n"
            f"│   U-Loss: {loss_val['u']:.4e} │                     │\n"
            f"│   V-Loss: {loss_val['v']:.4e} │                     │\n"
            f"├────────────── Error Values ────────────────│\n"
            f"│       Component      │  U-Net   │  V-Net   │\n"
            f"│ {sep}Initial Condition  │ {weight_fmt(metrics['ic_u'])} │ {weight_fmt(metrics['ic_v'])} │\n"
            f"│ {sep}Residual           │ {weight_fmt(metrics['res_u'])} │ {weight_fmt(metrics['res_v'])} │\n"
            f"│ {sep}Data               │ {weight_fmt(metrics['data_u'])} │ {weight_fmt(metrics['data_v'])} │\n"
            f"├────────────── Loss Weights ────────────────│\n"
            f"│       Component      │  U-Net   │  V-Net   │\n"
            f"│ {sep}Initial Condition  │ {weight_fmt(lambda_w['ic_u'])} │ {weight_fmt(lambda_w['ic_v'])} │\n"
            f"│ {sep}Residual           │ {weight_fmt(lambda_w['res_u'])} │ {weight_fmt(lambda_w['res_v'])} │\n"
            f"│ {sep}Data               │ {weight_fmt(lambda_w['data_u'])} │ {weight_fmt(lambda_w['data_v'])} │\n"
            f"╰────────────────────────────────────────────╯"
        )
    except Exception as e:
        logging.error("Failed in logging_metrics: %s\n%s", str(e), traceback.format_exc())
        raise

def train(config, data):
    try:
        rng = jax.random.PRNGKey(config['seed'])
        state = create_train_state(rng, config, data)
        
        lambda_w = {'ic_u': 1.0, 'ic_v': 1.0, 'data_u': 1.0, 'data_v': 1.0, 'res_u': 1.0, 'res_v': 1.0}
        loss_val = {'u': 0.0, 'v': 0.0}
        
        stages = [
            (int(config['epochs']*0.2), 20, 30, 3.0),   # Stage 1:     0-  8000, t=[0,  20],  30x30, alpha= 3.0, collocation size: 18000
            (int(config['epochs']*0.4), 40, 30, 3.0),   # Stage 2:  8001- 16000, t=[0,  40],  30x30, alpha= 3.0, collocation size: 
            (int(config['epochs']*0.6), 60, 30, 2.0),   # Stage 3: 16001- 24000, t=[0,  60],  30x30, alpha= 2.0
            (int(config['epochs']*0.8), 80, 30, 1.0),   # Stage 4: 24001- 32000, t=[0,  80],  30x30, alpha= 1.0
            (int(config['epochs']*1.0), 100, 30, 1.0)   # Stage 5: 32001- 40000, t=[0, 100],  30x30, alpha= 1.0
        ]

        dinp, dtar = gather_window_data(data, 0, 101)

        
        with tqdm(total=config['epochs'], ncols=150) as pbar:
            for epoch in range(config['epochs']):
                for stage_end, t_idx, res, alpha in stages:
                    if epoch <= stage_end:
                        window_end = t_idx + 1
                        colloc_nx = res
                        alpha = 0
                        break
                
                inputs, targets = dinp[:window_end], dtar[:window_end]
                t_batch = inputs[:, 2]
                gamma_data = jnp.exp(-alpha * t_batch / data['t'][-1])
                batch = {'inputs': inputs, 'targets': targets, 'gamma_data': gamma_data}
                
                colloc_points_raw = sample_collocation_points(colloc_nx, data['x'], data['y'], data['t'][0:window_end])

                gamma_res = jnp.exp(-alpha * colloc_points_raw[:, 2] / data['t'][-1])
                colloc_points = {'points': colloc_points_raw, 'gamma_res': gamma_res}
                
                constants = freeze({
                    'ep1': data['ep1'], 'ep2': data['ep2'], 'b1': data['b1'], 'b2': data['b2'],
                    'c1': data['c1'], 'c2': data['c2'], 'lambda_ic_u': lambda_w['ic_u'],
                    'lambda_ic_v': lambda_w['ic_v'], 'lambda_data_u': lambda_w['data_u'],
                    'lambda_data_v': lambda_w['data_v'], 'lambda_res_u': lambda_w['res_u'],
                    'lambda_res_v': lambda_w['res_v'], 'lambda_lr': config['lambda_lr']
                })
                
                state, metrics = train_step(state, batch, colloc_points, constants)
                
                total_loss = metrics['total']
                loss_val['u'] = metrics['ic_u'] + metrics['res_u'] + metrics['data_u']
                loss_val['v'] = metrics['ic_v'] + metrics['res_v'] + metrics['data_v']

                
                if epoch % config['wu_freq'] == 0 and epoch > 0:
                    grad_norms = compute_approx_grad_norms(state.params, batch, colloc_points, constants, state)
                    lambda_w = update_lambdas(lambda_w, metrics, grad_norms, initial_losses, epoch, state, batch, colloc_points, constants, beta=0.9)

                if epoch % config['log_freq'] == 0:
                    initial_losses = metrics
                    logging_metrics(lambda_w, metrics, loss_val, epoch)
                    u_l2, v_l2 = visualize_solution(state, data, epoch, window_end - 1)
                
                pbar.update(1)
                pbar.set_postfix({'total_loss': f"{total_loss:.2e}", 'window': window_end})
        
        save_checkpoint(state, epoch)
        return state
    except Exception as e:
        logging.error("Failed in train: %s\n%s", str(e), traceback.format_exc())
        raise

# ====================================================
# 5. Visualization & Utilities
# ====================================================
def visualize_solution(state, data, epoch, t_idx):
    """
    Visualizes the current PINN prediction vs. reference solution at time index t_idx.
    Returns the L2 errors (relative) for U and V.

    L2 Error (relative) is computed as:
        L2(u) = ||u_pred - u_ref||_2 / (||u_ref||_2 + 1e-12)

    This helps measure how close the predicted solution is to the reference.
    """
    try:
        x_np = np.array(data['x'])
        y_np = np.array(data['y'])
        t_val = float(data['t'][t_idx].item())
        
        X, Y = np.meshgrid(x_np, y_np, indexing='ij')
        os.makedirs(f"figures/epoch_{epoch}", exist_ok=True)
        
        # Reference solution
        inputs = get_all_points_at_time(data, t_idx)
        u_ref = data['usol'][t_idx]  # shape (nx, ny)
        v_ref = data['vsol'][t_idx]  # shape (nx, ny)

        # PINN predictions
        u_pred, v_pred = state.apply_fn(state.params, inputs)
        u_pred = u_pred.reshape(u_ref.shape)
        v_pred = v_pred.reshape(v_ref.shape)

        # Compute relative L2 errors
        # --------------------------------
        u_diff = (u_pred - u_ref).ravel()
        v_diff = (v_pred - v_ref).ravel()

        u_ref_norm = np.linalg.norm(u_ref.ravel())
        v_ref_norm = np.linalg.norm(v_ref.ravel())

        u_pred_norm = np.linalg.norm(u_diff)
        v_pred_norm = np.linalg.norm(v_diff)

        # Add small epsilon to avoid zero-division if reference is near 0
        eps = 1e-12
        u_l2_error = u_pred_norm / (u_ref_norm + eps)
        v_l2_error = v_pred_norm / (v_ref_norm + eps)
        
        # Plot reference, prediction, absolute error
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for i, (ref, pred, title) in enumerate(zip([u_ref, v_ref], [u_pred, v_pred], ['U', 'V'])):
            axes[i, 0].imshow(ref, cmap='jet', origin='lower')
            axes[i, 0].set_title(f"{title} Reference")

            axes[i, 1].imshow(pred, cmap='jet', origin='lower')
            axes[i, 1].set_title(f"{title} Predicted")

            axes[i, 2].imshow(np.abs(ref - pred), cmap='jet', origin='lower')
            axes[i, 2].set_title("Absolute Error")
        
        # Save plot
        logging.info(f"[VIZ]==[Visualisation Saved to figures/epoch_{epoch}/solution_{t_idx}.png]==[VIZ]")
        plt.savefig(f"figures/epoch_{epoch}/solution_{t_idx}.png")
        plt.close()

        # Optionally log or print the L2 Errors
        logging.info(f"L2 Errors @ epoch={epoch}, t_idx={t_idx}: U={u_l2_error:.4e}, V={v_l2_error:.4e}")

        # Return them for external usage (e.g., weighting updates)
        return float(u_l2_error), float(v_l2_error)
    
    except Exception as e:
        logging.error("Failed in visualize_solution: %s\n%s", str(e), traceback.format_exc())
        raise

@partial(jit, static_argnums=(0))
def sample_collocation_points(colloc_nx, x_data, y_data, t_vals):
    try:
        def create_grid(t_val):
            X, Y = jnp.meshgrid(
                jnp.linspace(x_data[0], x_data[-1], colloc_nx),
                jnp.linspace(y_data[0], y_data[-1], colloc_nx),
                indexing='ij'
            )
            return jnp.stack([X.flatten(), Y.flatten(), jnp.full_like(X.flatten(), t_val)], axis=-1)

        return jnp.vstack(vmap(create_grid)(t_vals))
    except Exception as e:
        logging.error("Failed in sample_collocation_points: %s\n%s", str(e), traceback.format_exc())
        raise

def save_checkpoint(state, epoch):
    try:
        checkpoint = serialization.to_state_dict(state)
        with open(f"params_epoch_{epoch}.pkl", 'wb') as f:
            pickle.dump(checkpoint, f)
    except Exception as e:
        logging.error("Failed in save_checkpoint: %s\n%s", str(e), traceback.format_exc())
        raise

# ... [generate_gif_frames_flax and create_gif unchanged, require imageio] ...

# ====================================================
# 6. Configuration & Execution
# ====================================================
config = {
    'seed': 42,
    'hidden_dims': [64, 64, 32],
    'fourier_features': 64,
    'lr': 1e-3,
    'lambda_lr': 1e-2,
    'max_grad_norm': 1.0,
    'epochs': 50000,
    'log_freq': 500,
    'wu_freq': 100,
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
        logging.warning("Could not remove figures directory %s: %s", figures_dir, str(e))
    
    dataset_file = "grey_scott.mat"
    try:
        data = load_grey_scott_mat(dataset_file)
        logging.info("Successfully loaded dataset from '%s'.", dataset_file)
    except Exception as e:
        logging.error("Failed to load dataset from '%s': %s\n%s", dataset_file, str(e), traceback.format_exc())
        raise

    logging.info("Starting training loop.")
    try:
        trained_state = train(config, data)
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error("Training failed: %s\n%s", str(e), traceback.format_exc())
        raise

    ckpt_dir = '/home/rbhavsa4/checkpoints'
    try:
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=trained_state, step=trained_state.step, overwrite=True)
        logging.info("Model checkpoint saved at '%s' (step: %s).", ckpt_dir, trained_state.step)
    except Exception as e:
        logging.error("Failed to save model checkpoint: %s\n%s", str(e), traceback.format_exc())
        raise