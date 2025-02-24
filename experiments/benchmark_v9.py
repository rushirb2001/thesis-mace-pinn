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

# Optional: Filter specific messages if they still appear
class TensorFlowWarningFilter(logging.Filter):
    def filter(self, record):
        return "Tensorflow library not found" not in record.getMessage()

# Apply the filter to all handlers to suppress specific warnings
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(TensorFlowWarningFilter())

# --- Standard Library Imports ---
import os, shutil
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
from jax import random, jit, grad, vmap, hessian
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
    return jnp.stack([
        X.flatten(), 
        Y.flatten(), 
        jnp.full(Nx * Ny, data['t'][t_idx])
    ], axis=-1)

def get_apt_at_time(usol, vsol, x_data, y_data, t_idx, t_val):
    """
    Nx, Ny grid => shape (Nx*Ny, 3), and targets => shape (Nx*Ny, 2)
    """
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
    """
    Nx x Ny for each time step in [start_idx, end_idx), stacked
    """
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
                      (self.num_features // 2, self.input_dim))
        
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
            x = nn.BatchNorm(dim, 
                        name=f'{self.name}_batchnorm_{i}', momentum=0.9, epsilon=1e-5)(x)
            # x = x * nn.tanh(jnp.exp(x))  # Changed from nn.swish to custom activation: x * tanh(exp(x))
            x = nn.swish(x)
        return nn.Dense(1, 
                       name=f'{self.name}_output',
                       kernel_init=nn.initializers.he_normal())(x).squeeze()

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
    """
        Lamb Optimiser has stability and with additional tol based loss weight update 
        but it works great overall. Lets attempt at optimistic adam with 
        paper based optimiser ablation study and experimentation
    """

    model = GrayScottPINN(
        hidden_dims=config['hidden_dims'],
        fourier_features=config['fourier_features']
    )
    
    dummy_input = jnp.ones((1, 3))
    params = model.init(rng, dummy_input)
    
    tx = optax.chain(
        optax.clip_by_global_norm(config['max_grad_norm']),
        optax.optimistic_adam(config['lr'], 7.84e-03)
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def compute_loss(state, params, batch, colloc_points, constants):
    # Get predictions
    u_pred, v_pred = state.apply_fn(params, batch['inputs'])
    
    # Split into IC and data components
    targets_ic = batch['targets'][:1]
    targets_data = batch['targets'][1:]
    
    # IC loss (first half of batch)
    loss_ic_u = jnp.mean((u_pred[:1] - targets_ic[:, 0])**2)
    loss_ic_v = jnp.mean((v_pred[:1] - targets_ic[:, 1])**2)
    
    # Data loss (second half of batch)
    loss_data_u = jnp.mean((u_pred[1:] - targets_data[:, 0])**2)
    loss_data_v = jnp.mean((v_pred[1:] - targets_data[:, 1])**2)
    
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
def compute_component_gradients(state, batch, colloc_points, constants):
    """Compute gradients for each loss component"""
    # Split batch into IC and data parts
    n_ic = batch['inputs'].shape[0] // 2
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

# ====================================================
# 3C. Residual Computation Function
# ====================================================
def compute_residuals(state, params, colloc_points, constants):
    """Compute PDE residuals for collocation points"""
    def residual_fn(x):
        # Get predictions and derivatives
        u, v = state.apply_fn(params, x)
        
        # First derivatives
        u_grad = grad(lambda x: state.apply_fn(params, x[None])[0].squeeze())(x)
        v_grad = grad(lambda x: state.apply_fn(params, x[None])[1].squeeze())(x)
        
        # Second derivatives
        u_hess = hessian(lambda x: state.apply_fn(params, x[None])[0].squeeze())(x)
        v_hess = hessian(lambda x: state.apply_fn(params, x[None])[1].squeeze())(x)
        
        # Gray-Scott equations
        residual_u = u_grad[2] - constants['ep1'] * (u_hess[0,0] + u_hess[1,1]) - \
                     constants['b1'] * (1 - u) + constants['c1'] * u * v**2
        residual_v = v_grad[2] - constants['ep2'] * (v_hess[0,0] + v_hess[1,1]) + \
                     constants['b2'] * v - constants['c2'] * u * v**2
        
        return residual_u**2, residual_v**2

    # Vectorize over all collocation points
    ru, rv = vmap(residual_fn)(colloc_points)
    return jnp.mean(ru), jnp.mean(rv)

# ====================================================
# 4. Training Loop with Curriculum & Visualization
# ====================================================
# ====================================================
# 3C. Enhanced Adaptive Weighting Implementation
# ====================================================
def update_lambdas(lambda_w, metrics, grad_norms, alpha, 
                  min_lambda=0.2, max_lambda=5.0):
    """Improved lambda update rule with gradient/loss balancing"""
    # Calculate total gradient norm
    total_grad = sum(grad_norms.values())
    if total_grad < 1e-8:  # Prevent division by zero
        return lambda_w
    
    # Normalize gradients and losses
    grad_ratios = {k: v/total_grad for k,v in grad_norms.items()}
    total_loss = metrics['total']
    loss_ratios = {
        k: metrics[k]/total_loss for k in ['ic_u', 'ic_v', 
                                         'data_u', 'data_v',
                                         'res_u', 'res_v']
    }
    
    # Update rule with momentum
    new_lambda = {}
    for key in lambda_w:
        combined = 0.5*(grad_ratios[key] + loss_ratios[key])
        new_lambda[key] = (1 - alpha)*lambda_w[key] + alpha*combined
        
    # Apply soft constraints
    return {k: jnp.clip(v, min_lambda, max_lambda) 
            for k,v in new_lambda.items()}

def log_training_metrics(epoch, lambda_w, grad_norms, metrics, loss_val, alpha):
    """Enhanced metrics logging with component analysis"""
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
# 4. Enhanced Training Loop
# ====================================================
def train(config, data):
    rng = jax.random.PRNGKey(config['seed'])
    state = create_train_state(rng, config, data)
    
    # Initialize adaptive weights with better scaling
    lambda_w = {   
        'ic_u' : 1.0,
        'ic_v' : 1.0,
        'data_u' : 1.0,
        'data_v' : 1.0,
        'res_u' : 10.0,  # Higher initial residual weight
        'res_v' : 10.0
    }

    # Training state tracking
    training_state = {
        'best_total': float('inf'),
        'best_u': float('inf'),
        'best_v': float('inf'),
        'window_size': config['min_window_size'],
        'alpha': 0.4,
        'beta': 1.00005,
        'gamma': 0.99997
    }

    # Initialize first window
    window_end = config['min_window_size']
    inputs, targets = gather_window_data(data, 0, window_end)
    batch = {'inputs': inputs, 'targets': targets}

    with tqdm(total=config['epochs']) as pbar:
        for epoch in range(config['epochs']):
            # Collocation sampling
            colloc_points = sample_collocation_points(
                config['num_colloc'],
                data['x'],
                data['y'],
                float(data['t'][0]),
                float(data['t'][window_end-1])
            )

            constants = freeze({
                'ep1': data['ep1'],
                'ep2': data['ep2'],
                'b1': data['b1'],
                'b2': data['b2'],
                'c1': data['c1'],
                'c2': data['c2'],
                'lambda_ic_u': lambda_w['ic_u'],
                'lambda_ic_v': lambda_w['ic_v'],
                'lambda_data_u': lambda_w['data_u'],
                'lambda_data_v': lambda_w['data_v'],
                'lambda_res_u': lambda_w['res_u'],
                'lambda_res_v': lambda_w['res_v']
            })
            # Training step
            state, metrics = train_step(state, batch, colloc_points, constants)
            
            # Update loss tracking
            loss_u = metrics['ic_u'] + metrics['data_u'] + metrics['res_u']
            loss_v = metrics['ic_v'] + metrics['data_v'] + metrics['res_v']
            total_loss = loss_u + loss_v

            # Adaptive weighting and logging
            if epoch % config['log_freq'] == 0:
                grad_norms = compute_component_gradients(state, batch, 
                                                       colloc_points, 
                                                       freeze({**data, **lambda_w}))
                
                # Update lambda weights
                lambda_w = update_lambdas(lambda_w, metrics, grad_norms,
                                         training_state['alpha'])
                
                # Dynamic alpha adjustment
                if total_loss < training_state['best_total']:
                    training_state['alpha'] *= training_state['gamma']
                else:
                    training_state['alpha'] = min(
                        training_state['alpha'] * training_state['beta'],
                        0.5
                    )
                
                # Update best losses
                training_state['best_total'] = min(training_state['best_total'], 
                                                  total_loss)
                training_state['best_u'] = min(training_state['best_u'], loss_u)
                training_state['best_v'] = min(training_state['best_v'], loss_v)
                
                # Logging and visualization
                log_training_metrics(epoch, lambda_w, grad_norms, metrics,
                                    {'u': loss_u, 'v': loss_v},
                                    training_state['alpha'])
                visualize_solution(state, data, epoch, window_end-1)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'total_loss': f"{total_loss:.2e}",
                'window': training_state['window_size']
            })

    save_checkpoint(state, epoch)
    return state

# ====================================================
# 5. Visualization & Utilities
# ====================================================
def visualize_solution(state, data, epoch, t_idx):
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
    
    logging.info(f"[VIZ]==[Visualisation Saved to figures/epoch_{epoch}/solution_{t_idx}.png]==[VIZ]")
    plt.savefig(f"figures/epoch_{epoch}/solution_{t_idx}.png")
    plt.close()

def sample_collocation_points(num_points, x, y, t_start, t_end):
    key = random.PRNGKey(np.random.randint(int(1e6)))
    key, x_key, y_key, t_key = random.split(key, 4)
    
    x_min = float(x[0].item())
    x_max = float(x[-1].item())
    y_min = float(y[0].item())
    y_max = float(y[-1].item())
    t_min = float(t_start)
    t_max = float(t_end)
    
    x_pts = random.uniform(x_key, (num_points,), minval=x_min, maxval=x_max)
    y_pts = random.uniform(y_key, (num_points,), minval=y_min, maxval=y_max)
    t_pts = random.uniform(t_key, (num_points,), minval=t_min, maxval=t_max)
    
    return jnp.stack([x_pts, y_pts, t_pts], axis=-1)

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
    for t_idx in tqdm(range(len(t_data)), desc="Generating frames"):
        # Get reference solution
        U_ref = usol[t_idx]
        V_ref = vsol[t_idx]
        print(t_data[t_idx])
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

def create_gif(filenames, output_gif="flax_solution.gif", fps=60):
    """Create GIF from frames"""
    with imageio.get_writer(output_gif, mode='I', fps=fps) as writer:
        for filename in tqdm(filenames, desc="Creating GIF"):
            image = imageio.imread(filename)
            writer.append_data(image)

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
    'min_window_size': 100,
    'max_window_size': 100,
    'window_update_freq': 40000
}

if __name__ == "__main__":
    
    figures_dir = "/home/rbhavsa4/figures"
    try:
        shutil.rmtree(figures_dir)
        logging.info("Removed figures directory: %s", figures_dir)
    except Exception as e:
        logging.warning("Could not remove figures directory %s: %s", figures_dir, e)
    
    # Load dataset
    dataset_file = "grey_scott.mat"
    try:
        data = load_grey_scott_mat(dataset_file)
        logging.info("Successfully loaded dataset from '%s'.", dataset_file)
    except Exception as e:
        logging.error("Failed to load dataset from '%s': %s", dataset_file, e)

    # Run training loop
    logging.info("Starting training loop.")
    try:
        trained_state = train(config, data)
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error("Training failed: %s", e)

    # Save model checkpoint
    ckpt_dir = '/home/rbhavsa4/checkpoints'
    try:
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=trained_state, step=trained_state.step, overwrite=True)
        logging.info("Model checkpoint saved at '%s' (step: %s).", ckpt_dir, trained_state.step)
    except Exception as e:
        logging.error("Failed to save model checkpoint: %s", e)