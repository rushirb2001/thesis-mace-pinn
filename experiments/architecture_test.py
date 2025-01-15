"""
Testing SIREN architecture with Fourier embeddings
"""
import jax
import jax.numpy as jnp
from jax import random
from jax.example_libraries import stax


def create_siren_network(omega_0=30.0):
    """SIREN network with sine activations"""
    def first_init(key, shape, dtype=jnp.float32):
        return random.uniform(key, shape, dtype, minval=-1/omega_0, maxval=1/omega_0)
    
    def hidden_init(key, shape, dtype=jnp.float32):
        fan_in = shape[0]
        scale = jnp.sqrt(6.0 / fan_in) / omega_0
        return random.uniform(key, shape, dtype, minval=-scale, maxval=scale)
    
    return stax.serial(
        stax.Dense(256, W_init=first_init), stax.elementwise(jnp.sin),
        stax.Dense(256, W_init=hidden_init), stax.elementwise(jnp.sin),
        stax.Dense(256, W_init=hidden_init), stax.elementwise(jnp.sin),
        stax.Dense(1, W_init=hidden_init)
    )


def create_fourier_embedding(P, d, key):
    """Random Fourier feature matrix"""
    return random.normal(key, shape=(P // 2, d))


def fourier_embed(x, B):
    """Apply Fourier embedding"""
    proj = jnp.dot(x, B.T)
    return jnp.concatenate([jnp.sin(2 * jnp.pi * proj), 
                            jnp.cos(2 * jnp.pi * proj)], axis=-1)


# Test initialization
key = random.PRNGKey(42)
B = create_fourier_embedding(64, 3, key)
init_fn, net = create_siren_network()

print("Fourier embedding matrix shape:", B.shape)
print("Expected output embedding dim:", 64)
