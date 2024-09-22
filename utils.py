import jax.numpy as jnp

beta_min = 0.1
beta_max = 20

def beta_t(t):
    return beta_min + t*(beta_max - beta_min)

def alpha_t(t):
    return t * beta_min + 0.5 * t**2 * (beta_max - beta_min)

def drift(x, t):
    return -0.5 * beta_t(t)[:,None,None] * x

def dispersion(t):
    return jnp.sqrt(beta_t(t))

def mean_factor(t):
    return jnp.exp(-0.5 * alpha_t(t))

def var(t):
    return 1 - jnp.exp(-alpha_t(t))