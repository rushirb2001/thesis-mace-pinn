"""
Efficient derivative computation for PDEs using JAX
"""
import jax
import jax.numpy as jnp
from jax import jacfwd, vmap


def compute_derivatives(fun, inp):
    """
    Compute first and second derivatives of scalar function
    Returns:
        first: [∂f/∂x, ∂f/∂y, ∂f/∂t]
        second: [∂²f/∂x², ∂²f/∂y², ∂²f/∂t²]
    """
    first = jacfwd(fun)(inp)
    second = jnp.array([
        jacfwd(lambda x: jacfwd(fun)(x)[i])(inp)[i] 
        for i in range(len(inp))
    ])
    return first, second


def enforce_periodic_bc(inputs):
    """Enforce periodic boundary conditions on [-1,1]²"""
    x, y, t = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    x_periodic = (x + 1) % 2 - 1
    y_periodic = (y + 1) % 2 - 1
    return jnp.stack([x_periodic, y_periodic, t], axis=-1)


# Test on simple function
def test_func(x):
    return jnp.sin(x[0]) * jnp.cos(x[1])

test_point = jnp.array([0.5, 0.5, 0.0])
first, second = compute_derivatives(test_func, test_point)
print("First derivatives:", first[:2])
print("Second derivatives (Laplacian components):", second[:2])
