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

# Optional stubs for testing/visualization
def test_pinn_vs_mat(params, file_path='grey_scott.mat', time_index=9):
    pass

def visualize_predictions(file_path='grey_scott.mat', trained_params=None, time_index=9, workdir='results'):
    pass

# ====================================================
# 2) TWO BASIC MLPs
# ====================================================
from jax.example_libraries import stax

def create_mlp(layer_dims):
    """
    A standard MLP with tanh activations.
    layer_dims: e.g. [128,128,128,128,1]
    """
    layers = []
    for dim in layer_dims[:-1]:
        layers.append(stax.Dense(dim))
        layers.append(stax.Tanh)
    # Final layer
    layers.append(stax.Dense(layer_dims[-1]))
    init_fn, apply_fn = stax.serial(*layers)
    return init_fn, apply_fn

# Build separate MLPs for u and v
init_u, apply_u = create_mlp([128, 128, 128, 128, 1])
init_v, apply_v = create_mlp([128, 128, 128, 128, 1])

def pinn_predict_2d(params, inputs):
    """
    inputs: shape (N,3) => (x, y, t)
    We'll wrap x,y in [-1,1] to keep periodic BC, then feed directly to MLPs.
    """
    x, y, t = inputs[:,0], inputs[:,1], inputs[:,2]
    # Periodic wrapping
    x_wrap = (x + 1.0) % 2.0 - 1.0
    y_wrap = (y + 1.0) % 2.0 - 1.0
    mlp_input = jnp.stack([x_wrap, y_wrap, t], axis=-1)  # shape (N,3)

    u_out = apply_u(params["u"], mlp_input).squeeze()
    v_out = apply_v(params["v"], mlp_input).squeeze()

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
    def single_pt_res(pt):
        # Make two scalar functions
        def u_fun(x):
            u_val, _ = pinn_predict_2d(params, x[None, :])
            return jnp.squeeze(u_val)

        def v_fun(x):
            _, v_val = pinn_predict_2d(params, x[None, :])
            return jnp.squeeze(v_val)

        grad_u = jax.grad(u_fun)(pt)   # shape (3,) => (u_x, u_y, u_t)
        grad_v = jax.grad(v_fun)(pt)

        hess_u = jax.hessian(u_fun)(pt)  # shape (3,3)
        hess_v = jax.hessian(v_fun)(pt)

        lap_u = hess_u[0, 0] + hess_u[1, 1]  # u_xx + u_yy
        lap_v = hess_v[0, 0] + hess_v[1, 1]

        # PDE
        u_val = jnp.squeeze(u_fun(pt))
        v_val = jnp.squeeze(v_fun(pt))
        u_t = grad_u[2]
        v_t = grad_v[2]

        ru = u_t - ep1*lap_u - b1*(1.0 - u_val) + c1*u_val*(v_val**2)
        rv = v_t - ep2*lap_v + b2*v_val - c2*u_val*(v_val**2)
        return ru, rv

    ru_vec, rv_vec = jax.vmap(single_pt_res)(inputs)
    return ru_vec, rv_vec

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

# def sample_bc_points_2d(num_points, x_data, y_data, tmax):
#     key = random.PRNGKey(np.random.randint(1e6))
#     quarter = num_points // 4
#     t_bc = random.uniform(key, (num_points,), minval=0.0, maxval=tmax)

#     x_left = jnp.full((quarter,), x_data[0])
#     x_right= jnp.full((quarter,), x_data[-1])
#     y_rand = random.uniform(key, (quarter,), minval=y_data[0], maxval=y_data[-1])

#     x_rand = random.uniform(key, (quarter,), minval=x_data[0], maxval=x_data[-1])
#     y_bottom = jnp.full((quarter,), y_data[0])
#     y_top    = jnp.full((quarter,), y_data[-1])

#     bc_pts = []
#     bc_pts.append(jnp.stack([x_left,  y_rand,   t_bc[:quarter]], axis=-1))
#     bc_pts.append(jnp.stack([x_right, y_rand,   t_bc[quarter:2*quarter]], axis=-1))
#     bc_pts.append(jnp.stack([x_rand,  y_bottom, t_bc[2*quarter:3*quarter]], axis=-1))
#     bc_pts.append(jnp.stack([x_rand,  y_top,    t_bc[3*quarter:4*quarter]], axis=-1))
#     bc_pts = jnp.concatenate(bc_pts, axis=0)

#     bc_tar = jnp.tile(jnp.array([1.0, 0.0]), (num_points,1))  # Dirichlet BC => (1,0)
#     return bc_pts, bc_tar

def sample_collocation_points_2d(num_points, x_data, y_data, tmax):
    key = random.PRNGKey(np.random.randint(1e6))
    x_rand = random.uniform(key, (num_points,), minval=x_data[0], maxval=x_data[-1])
    y_rand = random.uniform(key, (num_points,), minval=y_data[0], maxval=y_data[-1])
    t_rand = random.uniform(key, (num_points,), minval=0.0, maxval=tmax)
    return jnp.stack([x_rand, y_rand, t_rand], axis=-1)

# ====================================================
# 5) PARTIAL LOSS & GRAD NORM & NTK
# ====================================================
def partial_ic_loss(params, inp, tar):
    if inp.shape[0] == 0:
        return 0.0
    u_ic, v_ic = pinn_predict_2d(params, inp)
    return jnp.mean((u_ic - tar[:,0])**2) + jnp.mean((v_ic - tar[:,1])**2)

# def partial_bc_loss(params, inp, tar):
#     if inp.shape[0] == 0:
#         return 0.0
#     u_bc, v_bc = pinn_predict_2d(params, inp)
#     return jnp.mean((u_bc - tar[:,0])**2) + jnp.mean((v_bc - tar[:,1])**2)

def partial_res_loss(params, inp, ep1, ep2, b1, c1, b2, c2):
    if inp.shape[0] == 0:
        return 0.0
    ru, rv = compute_residual(params, inp, ep1, ep2, b1, c1, b2, c2)
    return jnp.mean(ru**2) + jnp.mean(rv**2)

def partial_data_loss(params, inp, tar):
    if inp.shape[0] == 0:
        return 0.0
    u_d, v_d = pinn_predict_2d(params, inp)
    return jnp.mean((u_d - tar[:,0])**2) + jnp.mean((v_d - tar[:,1])**2)

def compute_gradient_norms(params, data_dict, ep1, ep2, b1, c1, b2, c2):
    # four partial losses
    def ic_fn(pp):
        return partial_ic_loss(pp, data_dict["ic"][0], data_dict["ic"][1])
    # def bc_fn(pp):
    #     return partial_bc_loss(pp, data_dict["bc"][0], data_dict["bc"][1])
    def res_fn(pp):
        return partial_res_loss(pp, data_dict["colloc"], ep1, ep2, b1, c1, b2, c2)
    def data_fn(pp):
        return partial_data_loss(pp, data_dict["sup"][0], data_dict["sup"][1])

    g_ic  = jax.grad(ic_fn)(params)
    # g_bc  = jax.grad(bc_fn)(params)
    g_res = jax.grad(res_fn)(params)
    g_dat = jax.grad(data_fn)(params)

    def tree_l2(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.linalg.norm(jnp.concatenate([l.ravel() for l in leaves])) if leaves else 0.0

    return {
        "ic":   tree_l2(g_ic),
        # "bc":   tree_l2(g_bc),
        "res":  tree_l2(g_res),
        "data": tree_l2(g_dat)
    }

def compute_ntk(params, inputs_subset):
    """
    Minimal NTK on the 'u' subnetwork. Flatten param grads for each input, build kernel matrix.
    """
    def single_grad(x):
        # We'll define a small function for u
        def u_fun(p):
            # directly feed x into pinn_predict_2d
            u_val, _ = pinn_predict_2d(p, x[None,:])
            return u_val[0]  # shape(1,) => scalar

        return jax.grad(u_fun)(params)

    def flatten(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.concatenate([l.ravel() for l in leaves])

    N = inputs_subset.shape[0]
    mat = []
    for i in range(N):
        gi = flatten(single_grad(inputs_subset[i]))
        row_i = []
        for j in range(N):
            gj = flatten(single_grad(inputs_subset[j]))
            row_i.append(jnp.dot(gi, gj))
        mat.append(jnp.stack(row_i))
    return jnp.stack(mat)

# ====================================================
# 6) FULL LOSS
# ====================================================
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
    L_ic = partial_ic_loss(params, data_dict["ic"][0], data_dict["ic"][1])
    # L_bc = partial_bc_loss(params, data_dict["bc"][0], data_dict["bc"][1])
    L_res= partial_res_loss(params, data_dict["colloc"], ep1, ep2, b1, c1, b2, c2)
    L_dat= partial_data_loss(params, data_dict["sup"][0], data_dict["sup"][1])
    # return L_ic + L_bc + L_res + L_dat
    return L_ic + L_res + L_dat

def create_optimizers(lr_u=1e-3, lr_v=1e-3, decay_steps=10000, decay_rate=1.0):
    sched_u = optax.exponential_decay(lr_u, decay_steps, decay_rate)
    sched_v = optax.exponential_decay(lr_v, decay_steps, decay_rate)
    return optax.adam(schedule=sched_u), optax.adam(schedule=sched_v)

def train_step(params, opt_state_u, opt_state_v, data_dict, optim_u, optim_v, ep1, ep2, b1, c1, b2, c2):
    loss_val, grads = jax.value_and_grad(loss_fn)(params, data_dict, ep1, ep2, b1, c1, b2, c2)

    # Split gradients into U and V components
    grads_u = {"u": grads["u"]}  # Only keep U network gradients
    grads_v = {"v": grads["v"]}  # Only keep V network gradients
    
    # Clip gradients separately
    grads_u_clipped = clip_gradients(grads_u, max_norm=10.0)
    grads_v_clipped = clip_gradients(grads_v, max_norm=10.0)
    
    # Update U network
    updates_u, opt_state_u = optim_u.update(grads_u_clipped, opt_state_u, params["u"])
    new_params_u = optax.apply_updates(params["u"], updates_u)
    
    # Update V network
    updates_v, opt_state_v = optim_v.update(grads_v_clipped, opt_state_v, params["v"])
    new_params_v = optax.apply_updates(params["v"], updates_v)
    
    # Combine updated parameters
    new_params = {"u": new_params_u, "v": new_params_v}
    
    return new_params, opt_state_u, opt_state_v, loss_val

train_step = jit(train_step, static_argnums=(3,))

# ====================================================
# 7) SLIDING WINDOW: FIRST SLICE = IC
# ====================================================
def train_pinn_2d(
    file_path='grey_scott.mat',
    num_windows=10,
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
    b1, b2, c1, c2, ep1, ep2, usol, vsol, t_data, x_data, y_data = load_grey_scott_mat(file_path)
    total_steps = usol.shape[0]

    # Build net
    key = random.PRNGKey(0)
    # init for MLP-u
    _, init_u_params = init_u(key, (-1, 3))  # input: x,y,t => shape(3)
    key, subkey = random.split(key)
    # init for MLP-v
    _, init_v_params = init_v(subkey, (-1, 3))

    # Keep them in a single dictionary
    params = {"u": init_u_params, "v": init_v_params}

    # Optimizer
    optim_u, optim_v = create_optimizers(lr_u=1e-3, lr_v=1e-2)  # Higher LR for V
    opt_state_u = optim_u.init(params["u"])
    opt_state_v = optim_v.init(params["v"])

    all_losses = []
    global_epoch_count = 0

    # Outer loop => windows
    for w_idx in range(num_windows):
        # Disjoint windows => time_start = w_idx*window_size
        time_start = w_idx * window_size
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

        # Supervised data => [time_start+1, time_end)
        sup_start = time_start+1
        if sup_start>= time_end:
            sup_inp = jnp.zeros((0,3))
            sup_tar = jnp.zeros((0,2))
        else:
            sup_inp, sup_tar = gather_window_data(usol, vsol, t_data, x_data, y_data, sup_start, time_end)

        # Inner loop => epochs
        for epoch in tqdm(range(epochs_per_window), desc="PINN training (2D)"):
            global_epoch_count += 1
            # PDE collocation & BC => random
            if time_end> time_start:
                tmax = float(t_data[time_end-1])
            else:
                tmax = 0.0
            # bc_inp, bc_tar = sample_bc_points_2d(batch_bc, x_data, y_data, tmax)
            colloc_inp = sample_collocation_points_2d(batch_colloc, x_data, y_data, tmax)

            data_dict = {
                "ic": (ic_inp, ic_tar),
                # "bc": (bc_inp, bc_tar),
                "colloc": colloc_inp,
                "sup": (sup_inp, sup_tar)
            }

            # Single training step
            params, opt_state_u, opt_state_v, loss_val = train_step(
                                                            params, opt_state_u, opt_state_v, data_dict, 
                                                            optim_u, optim_v, ep1, ep2, b1, c1, b2, c2
                                                        )
            all_losses.append(loss_val)

            # Print & grad norms / NTK every 200 epochs
            if epoch % 1000 == 0:
                gnorms = compute_gradient_norms(params, data_dict, ep1, ep2, b1, c1, b2, c2)
                subset = colloc_inp[:10] if colloc_inp.shape[0]>10 else colloc_inp
                if subset.shape[0]>0:
                    # Potentially do CPU fallback for cond => np.linalg.cond(...)
                    K = jnp.array(compute_ntk(params, subset))  # might fail on Metal
                    cond_ntk = jnp.linalg.cond(K)
                else:
                    cond_ntk = jnp.nan

                print(
                        f"\n=== Window {w_idx} | Epoch {epoch} ===\n"
                        f"• Loss:       {loss_val:.4e}\n"
                        f"├── Gradient Norms\n"
                        f"│   - IC:     {gnorms['ic']:.2e}\n"
                        # f"│   - BC:     {gnorms['bc']:.2e}\n"
                        f"│   - Res:    {gnorms['res']:.2e}\n"
                        f"│   - Data:   {gnorms['data']:.2e}\n"
                        f"└── NTK Cond: {cond_ntk:.2e}\n"
                        f"{'-'*40}"
                    )
                # print(f"[Window={w_idx}, Epoch={epoch}] Loss={loss_val:.4e}")

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

    Saves figures to: figures/window_{w_idx}_epoch_{epoch}/
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
    out_dir = f"figures/window_{w_idx}_epoch_{epoch}"
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

# =========== MAIN ============
if __name__=="__main__":
    trained_params, losses = train_pinn_2d(
        file_path='grey_scott.mat',
        num_windows=10,
        window_size=10,
        epochs_per_window=50000,
        # batch_bc=200,
        batch_colloc=4096,
        init_lr=1e-3
    )
    save_trained_params(trained_params, "trained_params_sliding_IC_each_window_noFourier.pkl")
    # test_pinn_vs_mat(...)
    # visualize_predictions(...)