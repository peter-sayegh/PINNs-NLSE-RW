import jax, jax.numpy as jnp
from jax import random

#-functions to initialize weights and biases-------------------------

def _uniform_glorot_init(key, shape):
    
    n_in, n_out = shape

    W = random.uniform(
        key, shape=(n_in,n_out), 
        minval=-jnp.sqrt(6)/jnp.sqrt(n_in+n_out), 
        maxval= jnp.sqrt(6)/jnp.sqrt(n_in+n_out)
    )
    
    return W

def _factorized_glorot_init(key, shape, mean=1.0, std=0.1):
    
    _, n_out = shape

    key_W, key_s = random.split(key, num=2)

    W = _uniform_glorot_init(key_W, shape)

    s = mean + std * random.normal(key_s, shape=(n_out,))
    s = jnp.exp(s)

    V = W / s

    return V, s