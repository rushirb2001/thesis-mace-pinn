import jax
import jax.numpy as jnp
from jax import random, jit, vmap, jacfwd
from jax.example_libraries import stax
import optax
import numpy as np
import scipy.io
from tqdm import tqdm
import wandb
import os

os.environ["JAX_PLATFORM_NAME"] = "gpu"

# ====================================================
# SIREN Architecture - DEEPER VERSION (7 layers)
# ====================================================
def create_siren_subnetwork(omega_0=30.0):
    """
    Create a SIREN-style subnetwork using sine activations.
    This network has SEVEN hidden layers with sine activations.
    """
    def first_layer_init(key, shape, dtype=jnp.float32):
        return random.uniform(key, shape, dtype, minval=-1/omega_0, maxval=1/omega_0)
    
    def hidden_layer_init(key, shape, dtype=jnp.float32):
        fan_in = shape[0]
        scale = jnp.sqrt(6.0 / fan_in)
        return random.uniform(key, shape, dtype, minval=-scale, maxval=scale)
    
    init_fn = stax.serial(
        stax.Dense(256, W_init=first_layer_init), stax.elementwise(jnp.sin),
        stax.Dense(256, W_init=hidden_layer_init), stax.elementwise(jnp.sin),
        stax.Dense(256, W_init=hidden_layer_init), stax.elementwise(jnp.sin),
        stax.Dense(256, W_init=hidden_layer_init), stax.elementwise(jnp.sin),
        stax.Dense(256, W_init=hidden_layer_init), stax.elementwise(jnp.sin),
        stax.Dense(256, W_init=hidden_layer_init), stax.elementwise(jnp.sin),
        stax.Dense(256, W_init=hidden_layer_init), stax.elementwise(jnp.sin),
        stax.Dense(1, W_init=hidden_layer_init)
    )
    return init_fn


# Initialize networks
init_fn_u, u_net = create_siren_subnetwork(omega_0=30.0)
init_fn_v, v_net = create_siren_subnetwork(omega_0=30.0)

u_key = random.PRNGKey(0)
v_key = random.PRNGKey(1)
_, u_params = init_fn_u(u_key, (-1, 64))
_, v_params = init_fn_v(v_key, (-1, 64))

loss_params = {
    'log_sigma_ic': jnp.array(0.0),
    'log_sigma_bc': jnp.array(0.0),
    'log_sigma_res': jnp.array(0.0)
}

params_global = {'u': u_params, 'v': v_params, 'loss': loss_params}


# ====================================================
# Fourier Embedding
# ====================================================
def create_embedding_matrix(P, d):
    key = random.PRNGKey(42)
    return random.normal(key, shape=(P // 2, d))


def fourier_embed(x, B):
    proj = jnp.dot(x, B.T)
    return jnp.concatenate([jnp.sin(2 * jnp.pi * proj), 
                            jnp.cos(2 * jnp.pi * proj)], axis=-1)


B2 = create_embedding_matrix(64, 3)


# ====================================================
# Prediction with Periodic BC
# ====================================================
def enforce_periodic_bc(inputs):
    x, y, t = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    x_periodic = (x + 1) % 2 - 1
    y_periodic = (y + 1) % 2 - 1
    return jnp.stack([x_periodic, y_periodic, t], axis=-1)


def pinn_predict_2d(params, inputs):
    periodic_inputs = enforce_periodic_bc(inputs)
    embedded = fourier_embed(periodic_inputs, B2)
    u_pred = u_net(params['u'], embedded).squeeze()
    v_pred = v_net(params['v'], embedded).squeeze()
    return u_pred, v_pred


# ====================================================
# Derivative Computation
# ====================================================
def compute_first_and_second_derivatives(fun, inp):
    first = jacfwd(fun)(inp)
    second = jnp.array([
        jacfwd(lambda x: jacfwd(fun)(x)[i])(inp)[i] 
        for i in range(len(inp))
    ])
    return first, second


# ====================================================
# PDE Residual
# ====================================================
def compute_residual(params, inputs, ep1, ep2, b1, c1, b2, c2):
    def residual(inp):
        u_val, v_val = pinn_predict_2d(params, inp[None, :])
        
        def u_fun(x):
            return pinn_predict_2d(params, x[None, :])[0]
        def v_fun(x):
            return pinn_predict_2d(params, x[None, :])[1]
        
        u_first, u_second = compute_first_and_second_derivatives(u_fun, inp)
        v_first, v_second = compute_first_and_second_derivatives(v_fun, inp)
        
        u_t = u_first[2]
        v_t = v_first[2]
        lap_u = u_second[0] + u_second[1]
        lap_v = v_second[0] + v_second[1]
        
        ru = u_t - ep1 * lap_u - b1*(1 - u_val) + c1*u_val*v_val**2
        rv = v_t - ep2 * lap_v + b2*v_val - c2*u_val*v_val**2
        return ru, rv
    
    ru, rv = vmap(residual)(inputs)
    return ru, rv


def loss_fn(params, data, ep1, ep2, b1, c1, b2, c2):
    log_sigma_ic = params['loss']['log_sigma_ic']
    log_sigma_bc = params['loss']['log_sigma_bc']
    log_sigma_res = params['loss']['log_sigma_res']
    sigma_ic = jnp.exp(log_sigma_ic)
    sigma_bc = jnp.exp(log_sigma_bc)
    sigma_res = jnp.exp(log_sigma_res)
    
    inputs_ic, targets_ic = data['ic']
    u_pred_ic, v_pred_ic = pinn_predict_2d(params, inputs_ic)
    loss_ic = jnp.mean((u_pred_ic - targets_ic[:, 0])**2) + \
              jnp.mean((v_pred_ic - targets_ic[:, 1])**2)
    
    inputs_bc, targets_bc = data['bc']
    u_pred_bc, v_pred_bc = pinn_predict_2d(params, inputs_bc)
    loss_bc = jnp.mean((u_pred_bc - targets_bc[:, 0])**2) + \
              jnp.mean((v_pred_bc - targets_bc[:, 1])**2)
    
    inputs_colloc = data['colloc']
    ru, rv = compute_residual(params, inputs_colloc, ep1, ep2, b1, c1, b2, c2)
    loss_res = jnp.mean(ru**2) + jnp.mean(rv**2)
    
    total_loss = (
        0.5 * loss_ic / (sigma_ic ** 2) + jnp.log(sigma_ic) +
        0.5 * loss_bc / (sigma_bc ** 2) + jnp.log(sigma_bc) +
        0.5 * loss_res / (sigma_res ** 2) + jnp.log(sigma_res)
    )
    
    return total_loss


def load_grey_scott_mat(file_path):
    data = scipy.io.loadmat(file_path)
    b1 = float(data['b1'].item())
    b2 = float(data['b2'].item())
    c1 = float(data['c1'].item())
    c2 = float(data['c2'].item())
    ep1 = float(data['ep1'].item())
    ep2 = float(data['ep2'].item())
    usol = jnp.array(data['usol'])
    vsol = jnp.array(data['vsol'])
    t = jnp.array(data['t'].flatten())
    x = jnp.array(data['x'].flatten())
    y = jnp.array(data['y'].flatten())
    return b1, b2, c1, c2, ep1, ep2, usol, vsol, t, x, y
