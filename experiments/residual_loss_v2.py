import os
import pickle
import jax
import jax.numpy as jnp
from jax import random, jit, grad
import optax
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ["JAX_PLATFORM_NAME"] = "gpu"
jax.config.update("jax_default_matmul_precision", "highest") # Switch ot bfloat16 for Low Prec and Faster Training.

# ====================================================
# 1) LOAD & SAVE
# ====================================================
def load_grey_scott_mat(file_path):
    data = scipy.io.loadmat(file_path)
    b1 = float(data['b1'].item())
    b2 = float(data['b2'].item())
    c1 = float(data['c1'].item())
    c2 = float(data['c2'].item())
    ep1 = float(data['ep1'].item())
    ep2 = float(data['ep2'].item())
    usol = jnp.array(data['usol'])  # (Ntime, Nx, Ny)
    vsol = jnp.array(data['vsol'])
    t_data = jnp.array(data['t'].flatten())
    x_data = jnp.array(data['x'].flatten())
    y_data = jnp.array(data['y'].flatten())
    return b1, b2, c1, c2, ep1, ep2, usol, vsol, t_data, x_data, y_data

def save_trained_params(params, filename):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)

def load_trained_params(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# ====================================================
# 2) TWO BASIC MLPs
# ====================================================
from jax.example_libraries import stax

def create_mlp(layer_dims):
    """
    A standard MLP with tanh activations.
    layer_dims:
    """
    layers = []
    for dim in layer_dims[:-1]:
        layers += [
            stax.Dense(dim),
            stax.Tanh,
            stax.FanOut(2),
            stax.parallel(stax.Identity, stax.Dense(dim)),  # Residual branch
            stax.FanInSum  # x + 0.1*Dense(x)
        ]
    # Final layer
    layers.append(stax.Dense(layer_dims[-1]))
    init_fn, apply_fn = stax.serial(*layers)
    return init_fn, apply_fn

shared_dims = [256, 256, 128]  # Symmetrical core for both networks
init_u, apply_u = create_mlp([*shared_dims, 1])
init_v, apply_v = create_mlp([*shared_dims, 1])

def create_fourier_embedding(key, input_dim=3, num_features=64, scale=6.0):
    assert num_features % 2 == 0
    B = random.normal(key, shape=(num_features//2, input_dim)) * scale
    return B

B_u = create_fourier_embedding(random.PRNGKey(0), 3, 64)
B_v = create_fourier_embedding(random.PRNGKey(0), 3, 64)

def fourier_embed(x, B):
    proj = jnp.dot(x, B.T)  # (N, num_features//2)
    return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)

def pinn_predict_2d(params, inputs):
    """
    inputs: shape (N,3) => (x, y, t)
    We'll wrap x,y in [-1,1] to keep periodic BC, then feed directly to MLPs.
    """
    global B_u, B_v
    x, y, t = inputs[:,0], inputs[:,1], inputs[:,2]
    # Periodic wrapping
    x_wrap = (x + 1.0) % 2.0 - 1.0
    y_wrap = (y + 1.0) % 2.0 - 1.0
    mlp_input = jnp.stack([x_wrap, y_wrap, t], axis=-1)  # shape (N,3)

    mlp_input_u = fourier_embed(mlp_input, B_u)
    mlp_input_v = fourier_embed(mlp_input, B_v)

    u_out = apply_u(params["u"], mlp_input_u).squeeze()
    v_out = apply_v(params["v"], mlp_input_v).squeeze()

    # Ensure shape (N,) if needed
    u_out = jnp.reshape(u_out, (-1,))
    v_out = jnp.reshape(v_out, (-1,))
    return u_out, v_out

# ====================================================
# 3) PDE Residual (Hessian with jax)
# ====================================================
def compute_residual(params, inputs, ep1, ep2, b1, c1, b2, c2):
    """
    Gray-Scott PDE:
      u_t = ep1(u_xx + u_yy) + b1*(1 - u) - c1*u*v^2
      v_t = ep2(v_xx + v_yy) - b2*v + c2*u*v^2
    We'll compute first & second derivatives using jax.hessian.
    """
    def u_fn(pt):
        u, _ = pinn_predict_2d(params, pt[None,:])
        return jnp.squeeze(u)
    
    def v_fn(pt):
        _, v = pinn_predict_2d(params, pt[None,:])
        return jnp.squeeze(v)

    def compute_terms(fn, pt):
        grad = jax.grad(fn)(pt)
        hess = jax.hessian(fn)(pt)
        return {
            'val': fn(pt) + 1e-6,
            'grad_t': grad[2] + 1e-8,  # Time derivative
            'lap': hess[0,0] + hess[1,1]  # Spatial Laplacian
        }

    def point_residual(pt):
        u_terms = compute_terms(u_fn, pt)
        v_terms = compute_terms(v_fn, pt)
        
        # Gray-Scott residuals
        ru = (u_terms['grad_t'] 
              - ep1 * u_terms['lap'] 
              - b1 * (1 - u_terms['val']) 
              + c1 * u_terms['val'] * v_terms['val']**2)
        
        rv = (v_terms['grad_t'] 
              - ep2 * v_terms['lap'] 
              + b2 * v_terms['val'] 
              - c2 * u_terms['val'] * v_terms['val']**2)
        
        return ru, rv

    return jax.vmap(point_residual)(inputs)

# ====================================================
# 4) BUILDING DATA
# ====================================================
def get_all_points_at_time(usol, vsol, x_data, y_data, t_idx, t_val):
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

def gather_window_data(usol, vsol, t_data, x_data, y_data, start_idx, end_idx):
    """
    Nx x Ny for each time step in [start_idx, end_idx), stacked
    """
    inps_list = []
    tars_list = []
    for ti in range(start_idx, end_idx):
        t_val = float(t_data[ti])
        inp_t, tar_t = get_all_points_at_time(usol, vsol, x_data, y_data, ti, t_val)
        inps_list.append(inp_t)
        tars_list.append(tar_t)
    if len(inps_list) == 0:
        return jnp.zeros((0,3)), jnp.zeros((0,2))
    inps = jnp.concatenate(inps_list, axis=0)
    tars = jnp.concatenate(tars_list, axis=0)
    return inps, tars


def sample_collocation_points_2d(num_points, x_data, y_data, tmin, tmax):
    key = random.PRNGKey(np.random.randint(1e6))
    x_rand = random.uniform(key, (num_points,), minval=x_data[0], maxval=x_data[-1])
    y_rand = random.uniform(key, (num_points,), minval=y_data[0], maxval=y_data[-1])
    t_rand = random.uniform(key, (num_points,), minval=tmin, maxval=tmax)
    return jnp.stack([x_rand, y_rand, t_rand], axis=-1)

# ====================================================
# 5) PARTIAL LOSS & GRAD NORM & NTK
# ====================================================
def partial_ic_loss(params, inp, tar):
    if inp.shape[0] == 0:
        return 0.0
    u_ic, v_ic = pinn_predict_2d(params, inp)
    return jnp.mean((u_ic - tar[:,0])**2), jnp.mean((v_ic - tar[:,1])**2)

def partial_res_loss(params, inp, ep1, ep2, b1, c1, b2, c2):
    if inp.shape[0] == 0:
        return 0.0
    ru, rv = compute_residual(params, inp, ep1, ep2, b1, c1, b2, c2)
    return jnp.mean(ru**2), jnp.mean(rv**2)

def compute_gradient_norms_u(params, data_dict, ep1, ep2, b1, c1, b2, c2):
    """Gradient norms for U-Net only"""
    def ic_fn(pp):
        return partial_ic_loss(pp, data_dict["ic"][0], data_dict["ic"][1])[0]
    def res_fn(pp):
        return partial_res_loss(pp, data_dict["colloc"], ep1, ep2, b1, c1, b2, c2)[0]

    g_ic  = jax.grad(ic_fn)(params)["u"]
    g_res = jax.grad(res_fn)(params)["u"]

    def tree_l2(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.linalg.norm(jnp.concatenate([l.ravel() for l in leaves])) if leaves else 0.0

    return {
        "ic":   tree_l2(g_ic),
        "res":  tree_l2(g_res),
    }

def compute_gradient_norms_v(params, data_dict, ep1, ep2, b1, c1, b2, c2):
    """Gradient norms for V-Net only"""
    def ic_fn(pp):
        return partial_ic_loss(pp, data_dict["ic"][0], data_dict["ic"][1])[1]
    def res_fn(pp):
        return partial_res_loss(pp, data_dict["colloc"], ep1, ep2, b1, c1, b2, c2)[1]

    g_ic  = jax.grad(ic_fn)(params)["v"]
    g_res = jax.grad(res_fn)(params)["v"]

    def tree_l2(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.linalg.norm(jnp.concatenate([l.ravel() for l in leaves])) if leaves else 0.0

    return {
        "ic":   tree_l2(g_ic),
        "res":  tree_l2(g_res),
    }

def compute_ntk(params, inputs_subset):
    """
    Computes the NTK for both subnetworks (u and v). 
    Returns two kernel matrices: u_ntk and v_ntk.
    """
    def single_grad(x):
        # Function to compute gradients for u and v at a single input point x
        def u_fun(p):
            u_val, _ = pinn_predict_2d(p, x[None, :])
            return u_val[0]
        
        def v_fun(p):
            _, v_val = pinn_predict_2d(p, x[None, :])
            return v_val[0]
        
        u_grad = jax.grad(u_fun)(params)
        v_grad = jax.grad(v_fun)(params)
        return u_grad, v_grad

    def flatten(tree):
        # Flattens a PyTree into a single vector
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.concatenate([l.ravel() for l in leaves])

    N = inputs_subset.shape[0]
    u_grads, v_grads = [], []
    
    # Precompute all flattened gradients
    for x in inputs_subset:
        u_grad, v_grad = single_grad(x)
        u_grads.append(flatten(u_grad))
        v_grads.append(flatten(v_grad))
    
    # Convert lists to matrices (N x D) where D is the parameter dimension
    u_grads_stacked = jnp.stack(u_grads)  # Shape (N, D_u)
    v_grads_stacked = jnp.stack(v_grads)  # Shape (N, D_v)

    # Compute NTK matrices using matrix multiplication
    u_ntk = jnp.dot(u_grads_stacked, u_grads_stacked.T)
    v_ntk = jnp.dot(v_grads_stacked, v_grads_stacked.T)

    return u_ntk, v_ntk
# ====================================================
# 6) FULL LOSS
# ====================================================
# Hyperparameters (now separate for U and V)
alpha = 0.2
update_freq = 100  

# Initialize weights for U
lambda_ic_u = 1.0
lambda_res_u = 1.0

# Initialize weights for V
lambda_ic_v = 1.0
lambda_res_v = 1.0

def tree_l2_norm(grad_tree):
    leaves = jax.tree_util.tree_leaves(grad_tree)
    return jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))

def clip_gradients(grads, max_norm=1.0):
    """
    Clips the gradient tree 'grads' so that its total norm does not exceed 'max_norm'.
    Returns a new tree of clipped gradients.
    """
    # Flatten all gradient leaves
    leaves = jax.tree_util.tree_leaves(grads)
    flat_g = jnp.concatenate([g.ravel() for g in leaves if g is not None])

    # Compute total norm
    grad_norm = jnp.linalg.norm(flat_g)
    clip_ratio = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))

    # Scale all leaves by clip_ratio
    clipped_grads = jax.tree_util.tree_map(lambda g: clip_ratio * g if g is not None else None, grads)
    return clipped_grads

def loss_fn(params, data_dict, ep1, ep2, b1, c1, b2, c2):
    # Compute all components
    L_ic_u, L_ic_v = partial_ic_loss(params, data_dict["ic"][0], data_dict["ic"][1])
    L_res_u, L_res_v = partial_res_loss(params, data_dict["colloc"], ep1, ep2, b1, c1, b2, c2)
    
    # Total losses for U and V
    total_loss_u = (
        lambda_ic_u * L_ic_u + 
        lambda_res_u * L_res_u
    )
    
    total_loss_v = (
        lambda_ic_v * L_ic_v + 
        lambda_res_v * L_res_v
    )
    total_loss_u = jnp.nan_to_num(total_loss_u, nan=1e6)
    total_loss_v = jnp.nan_to_num(total_loss_v, nan=1e6)
    
    return total_loss_u, total_loss_v

# def create_optimizers(lr_u=1e-4, lr_v=1e-4, decay_steps=100, decay_rate=0.5):
#     # sched_u = optax.exponential_decay(lr_u, 5e3, decay_rate*0.7)
#     sched_u = optax.exponential_decay(lr_u, 5e3, decay_rate)
#     sched_v = optax.exponential_decay(lr_v, 5e4, decay_rate)
#     # sched_v = optax.exponential_decay(lr_v, 5e4, decay_rate*1.9)
#     return optax.adamax(sched_u), optax.adamax(sched_v)

def create_onecycle_optimizer(
    peak_lr: float,
    total_steps: int,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4
):
    """Create LAMB optimizer with onecycle schedule"""
    schedule = optax.schedules.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=peak_lr,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )
    return optax.chain(
        optax.lamb(learning_rate=schedule)
    )

def train_step(params, opt_state_u, opt_state_v, data_dict, optim_u, optim_v, ep1, ep2, b1, c1, b2, c2):
    # Separate loss calculations
    total_loss_u, total_loss_v = loss_fn(params, data_dict, ep1, ep2, b1, c1, b2, c2)
    
    # U-Net gradients
    grad_u = jax.grad(lambda pu: loss_fn(
        {"u": pu, "v": params["v"]}, data_dict, ep1, ep2, b1, c1, b2, c2
    )[0])(params["u"])
    
    # V-Net gradients
    grad_v = jax.grad(lambda pv: loss_fn(
        {"u": params["u"], "v": pv}, data_dict, ep1, ep2, b1, c1, b2, c2
    )[1])(params["v"])
    
    # Clip gradients separately
    grad_u = clip_gradients(grad_u, max_norm=1e-1)
    grad_v = clip_gradients(grad_v, max_norm=1e-1)
    
    # Update U network
    updates_u, opt_state_u = optim_u.update(grad_u, opt_state_u, params["u"])
    new_params_u = optax.apply_updates(params["u"], updates_u)
    
    # Update V network
    updates_v, opt_state_v = optim_v.update(grad_v, opt_state_v, params["v"])
    new_params_v = optax.apply_updates(params["v"], updates_v)
    
    # Combine parameters and track total loss
    new_params = {"u": new_params_u, "v": new_params_v}
    
    return new_params, opt_state_u, opt_state_v, total_loss_u, total_loss_v

train_step = jit(train_step, static_argnums=(4, 5))

# ====================================================
# 7) SLIDING WINDOW: FIRST SLICE = IC
# ====================================================
def train_pinn_2d(
    file_path='grey_scott.mat',
    num_windows=19,
    window_slide=5,
    window_size=10,
    epochs_per_window=500,
    batch_colloc=1000,
    init_lr=1e-3
):
    """
    Outer loop => `num_windows`
    Each window => [time_start, time_start+window_size)
    The "IC" is the entire Nx×Ny at `time_start`.
    The rest => supervised data from [time_start+1, time_start+window_size).
    We do `epochs_per_window` on that local sub-problem.
    """
    print("Loading dataset:", file_path)
    global lambda_ic_u, lambda_ic_v, lambda_res_u, lambda_res_v
    b1, b2, c1, c2, ep1, ep2, usol, vsol, t_data, x_data, y_data = load_grey_scott_mat(file_path)
    total_steps = usol.shape[0]

    # Build net
    key = random.PRNGKey(0)
    # init for MLP-u
    _, init_u_params = init_u(key, (-1, 64))  # input: x,y,t => shape(3)
    key, subkey = random.split(key)
    # init for MLP-v
    _, init_v_params = init_v(subkey, (-1, 64))

    # Keep them in a single dictionary
    params = {"u": init_u_params, "v": init_v_params}

    optim_u = create_onecycle_optimizer(
        peak_lr=1e-3,
        total_steps=20000,
        pct_start=0.3,
        div_factor=75.0
    )

    optim_v = create_onecycle_optimizer(
        peak_lr=5e-3,
        total_steps=10000,
        pct_start=0.4,
        div_factor=50.0
    )

    opt_state_u = optim_u.init(params["u"])
    opt_state_v = optim_v.init(params["v"])

    all_losses = []
    global_epoch_count = 0

    # Outer loop => windows
    for w_idx in range(num_windows):

        # Disjoint windows => time_start = w_idx*window_size
        time_start = w_idx * window_slide
        time_end   = time_start + window_size
        if time_end > total_steps:
            time_end = total_steps

        print(f"Window {w_idx}: times [{time_start}, {time_end})")

        if time_start >= total_steps:
            print("No valid times left => done.")
            break

        # IC = Nx×Ny at time_start
        t_val_start = float(t_data[time_start])
        ic_inp, ic_tar = get_all_points_at_time(usol, vsol, x_data, y_data, time_start, t_val_start)

        # Inner loop => epochs
        for epoch in tqdm(range(epochs_per_window), desc="PINN training (2D)"):
            global_epoch_count += 1
            # PDE collocation & BC => random
            if time_end> time_start:
                tmax = float(t_data[time_end-1])
            else:
                tmax = 0.0
            
            tmin = float(t_data[time_start])

            # bc_inp, bc_tar = sample_bc_points_2d(batch_bc, x_data, y_data, tmax)
            colloc_inp = sample_collocation_points_2d(batch_colloc, x_data, y_data, tmin, tmax)

            data_dict = {
                "ic": (ic_inp, ic_tar),
                "colloc": colloc_inp,
            }

            # Single training step
            params, opt_state_u, opt_state_v, loss_val_u, loss_val_v = train_step(
                                                            params, opt_state_u, opt_state_v, data_dict, 
                                                            optim_u, optim_v, ep1, ep2, b1, c1, b2, c2
                                                        )
            loss_val = loss_val_u + loss_val_v
            all_losses.append(loss_val)

            
            if epoch % 10 == 0 and epoch != 0:
                # Compute gradient norms for both networks
                gnorms_u = compute_gradient_norms_u(params, data_dict, ep1, ep2, b1, c1, b2, c2)
                gnorms_v = compute_gradient_norms_v(params, data_dict, ep1, ep2, b1, c1, b2, c2)
                subset = colloc_inp[:10] if colloc_inp.shape[0]>10 else colloc_inp
                mat_u, mat_v = compute_ntk(params, subset)
                cond_ntk_u = jnp.linalg.cond(jnp.array(mat_u)) if subset.shape[0] > 0 else jnp.nan
                cond_ntk_v = jnp.linalg.cond(jnp.array(mat_v)) if subset.shape[0] > 0 else jnp.nan

                lambda_ic_u = alpha*lambda_ic_u + (1-alpha)*(gnorms_u['res']/(gnorms_u['ic'] + 1e-8))
                lambda_res_u = 1.0/lambda_ic_u
                
                # Similarly for v-network
                lambda_ic_v = alpha*lambda_ic_v + (1-alpha)*(gnorms_v['res']/(gnorms_v['ic'] + 1e-8))
                lambda_res_v = 1.0/lambda_ic_v
                
                if epoch % 10  == 0:
                    # Formatting constants
                    sep = "│ "
                    grad_fmt = lambda x: f"{x:.2e}".ljust(8)
                    weight_fmt = lambda x: f"{x:.2e}".rjust(8)
                    l2_u, l2_v = visualize_window_end(params, file_path, time_end, w_idx, global_epoch_count)

                    print(
                        f"\n╭──────── Window {float(w_idx):2.0} | Epoch {float(epoch):5.0} ────────╮\n"
                        f"│   U-Loss: {loss_val_u:.4e} │  NTK Unet: {cond_ntk_u:.2e} │\n"
                        f"│   V-Loss: {loss_val_v:.4e} │  NTK Vnet: {cond_ntk_v:.2e} │\n"
                        f"├──────────── L2 Error - Norm ───────────────│\n"
                        f"│   U-L2: {l2_u:.3e}    │   V-L2: {l2_v:.3e}   │\n"
                        f"├──────────── Gradient Norms ────────────────│\n"
                        f"│       Component      │  U-Net   │  V-Net   │\n"
                        f"│ {sep}Initial Condition  │ {grad_fmt(gnorms_u['ic'])} │ {grad_fmt(gnorms_v['ic'])} │\n"
                        f"│ {sep}Residual           │ {grad_fmt(gnorms_u['res'])} │ {grad_fmt(gnorms_v['res'])} │\n"
                        # f"├────────────── Loss Weights ────────────────│\n"
                        # f"│       Component      │  U-Net   │  V-Net   │\n"
                        # f"│ {sep}Initial Condition  │ {weight_fmt(lambda_ic_u)} │ {weight_fmt(lambda_ic_v)} │\n"
                        # f"│ {sep}Residual           │ {weight_fmt(lambda_res_u)} │ {weight_fmt(lambda_res_v)} │\n"
                        f"╰────────────────────────────────────────────╯"
                    )

                    visualize_window_end(params, file_path, time_end, w_idx, global_epoch_count)

    return params, all_losses

def visualize_window_end(
    params, file_path, 
    time_end,  # integer index for the window's end
    w_idx,     # which window
    epoch=0    # global epoch count
):
    """
    Visualize the PINN at 'time_end - 1' (the last slice in the current window),
    comparing ground truth (u,v) vs predicted, plus absolute error.

    Saves figures to: figures/window_{w_idx}/epoch_{epoch}/
    """

    if time_end <= 0:
        print("No valid time slice to visualize (time_end<=0). Skipping.")
        return

    # We'll visualize the last valid slice: time_end-1
    t_slice = time_end - 1

    # 1) Load .mat data
    b1, b2, c1, c2, ep1, ep2, usol, vsol, t_data, x_data, y_data = load_grey_scott_mat(file_path)
    Nx, Ny = usol.shape[1], usol.shape[2]

    if t_slice < 0 or t_slice >= usol.shape[0]:
        print(f"Time slice {t_slice} is out of range [0..{usol.shape[0]-1}]. Skipping visualization.")
        return

    # 2) Reference Data (u,v) at that slice
    U_ref = usol[t_slice]  # shape (Nx,Ny)
    V_ref = vsol[t_slice]
    t_val = t_data[t_slice]

    # 3) Build (x,y) grid
    X, Y = np.meshgrid(x_data, y_data, indexing='ij')
    Xf = X.flatten()
    Yf = Y.flatten()
    T_arr = np.full_like(Xf, t_val)
    inputs_np = np.stack([Xf, Yf, T_arr], axis=-1)
    inputs_jax = jnp.array(inputs_np)

    # 4) PINN Predictions
    U_pred, V_pred = pinn_predict_2d(params, inputs_jax)  # shapes => (Nx*Ny,) each
    U_pred = np.array(U_pred).reshape(Nx, Ny)
    V_pred = np.array(V_pred).reshape(Nx, Ny)

    err_U = np.abs(U_ref - U_pred)
    err_V = np.abs(V_ref - V_pred)

    # 5) Output directory
    out_dir = f"figures_opt_resloss/window_{w_idx}/epoch_{epoch}"
    os.makedirs(out_dir, exist_ok=True)

    # 6) Plot U
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    im0 = axes[0].imshow(U_ref, origin="lower", extent=[x_data[0], x_data[-1], y_data[0], y_data[-1]], cmap="jet")
    axes[0].set_title("Ref U")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(U_pred, origin="lower", extent=[x_data[0], x_data[-1], y_data[0], y_data[-1]], cmap="jet")
    axes[1].set_title(f"Pred U @ slice={t_slice}")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(err_U, origin="lower", extent=[x_data[0], x_data[-1], y_data[0], y_data[-1]], cmap="jet")
    axes[2].set_title("Error |U_ref - U_pred|")
    plt.colorbar(im2, ax=axes[2])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"U_epoch_{epoch:05d}.png"), dpi=300)
    plt.close(fig)

    # 7) Plot V
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    im0 = axes[0].imshow(V_ref, origin="lower", extent=[x_data[0], x_data[-1], y_data[0], y_data[-1]], cmap="jet")
    axes[0].set_title("Ref V")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(V_pred, origin="lower", extent=[x_data[0], x_data[-1], y_data[0], y_data[-1]], cmap="jet")
    axes[1].set_title(f"Pred V @ slice={t_slice}")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(err_V, origin="lower", extent=[x_data[0], x_data[-1], y_data[0], y_data[-1]], cmap="jet")
    axes[2].set_title("Error |V_ref - V_pred|")
    plt.colorbar(im2, ax=axes[2])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"V_epoch_{epoch:05d}.png"), dpi=300)
    plt.close(fig)

    print(f"[Viz] Saved end-of-window slice {t_slice} at epoch={epoch} to {out_dir}")

    return np.linalg.norm(U_ref - U_pred), np.linalg.norm(V_ref - V_pred)

# =========== MAIN ============
if __name__=="__main__":
    trained_params, losses = train_pinn_2d(
        file_path='grey_scott.mat',
        num_windows=19,
        window_slide=5,
        window_size=10,
        epochs_per_window=50000,
        batch_colloc=int(4096),
        init_lr=1e-4
    )
    save_trained_params(trained_params, "trained_params_sliding_optimisation_v_value.pkl")