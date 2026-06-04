"""
CDU network infrastructure: Glorot-init of the primal (π_θ) and dual (δ_φ)
networks, packed into a single JAX-pytree dict.
"""
import jax
import jax.numpy as jnp

from .config import HIDDEN


def init_primal_params(key, n_vars: int, n_cons: int):
    """
    Glorot-init weights for the primal network π_θ.
    Input  : [z (n_vars)  ‖  λ (n_cons)]  →  hidden  →  x (n_vars)
    Output : x mapped through sigmoid to [lb, ub] inside CDU loss.
    """
    n_in = n_vars + n_cons
    k = jax.random.split(key, 6)
    return [
        jax.random.normal(k[0], (n_in,   HIDDEN)) * jnp.sqrt(2.0 / n_in),   jnp.zeros(HIDDEN),
        jax.random.normal(k[1], (HIDDEN, HIDDEN)) * jnp.sqrt(2.0 / HIDDEN), jnp.zeros(HIDDEN),
        jax.random.normal(k[2], (HIDDEN, n_vars)) * jnp.sqrt(2.0 / HIDDEN), jnp.zeros(n_vars),
    ]


def init_dual_params(key, n_vars: int, n_cons: int):
    """
    Glorot-init weights for the dual network δ_φ.
    Input  : [x (n_vars)  ‖  λ (n_cons)]  →  hidden  →  λ' (n_cons)
    Output : softplus-activated to enforce λ' ≥ 0 (dual feasibility).
    """
    n_in = n_vars + n_cons
    k = jax.random.split(key, 6)
    return [
        jax.random.normal(k[0], (n_in,   HIDDEN)) * jnp.sqrt(2.0 / n_in),   jnp.zeros(HIDDEN),
        jax.random.normal(k[1], (HIDDEN, HIDDEN)) * jnp.sqrt(2.0 / HIDDEN), jnp.zeros(HIDDEN),
        jax.random.normal(k[2], (HIDDEN, n_cons)) * jnp.sqrt(2.0 / HIDDEN), jnp.zeros(n_cons),
    ]


def init_cdu_params(key, n_vars: int, n_cons: int) -> dict:
    """Return a JAX-pytree dict holding both primal and dual parameters."""
    k1, k2 = jax.random.split(key)
    return {
        'primal': init_primal_params(k1, n_vars, n_cons),
        'dual':   init_dual_params(k2,   n_vars, n_cons),
    }
