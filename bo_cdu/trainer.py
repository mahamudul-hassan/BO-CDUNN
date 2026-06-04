"""
JIT trainer infrastructure: a cached scan-based Adam trainer and a warmup
routine that pre-compiles all (problem × K × ρ × lr) graphs before timing.
"""
import jax
import jax.numpy as jnp
from jax import jit
import optax

from .config import (CDU_K_CHOICES, RHO_CHOICES, LR_CHOICES, N_STEPS)
from .problems import PROBLEMS, PROBLEM_ORDER, DISPLAY_NAMES
from .networks import init_cdu_params
from .losses import get_cdu_loss


# ─── Trainer cache ─────────────────────────────────────────────────────────────
_trainer_cache: dict = {}


def get_scan_trainer(loss_fn, lr: float, n_steps: int):
    """
    Return (train_fn, optimizer) for (loss_fn, lr); compile once.
    params is a pytree dict {'primal': [...], 'dual': [...]}.
    z is the fixed latent input passed per-trial by Optuna.
    """
    cache_key = (id(loss_fn), lr, n_steps)
    if cache_key not in _trainer_cache:
        optimizer = optax.adam(lr)

        @jit
        def train_fn(params, z, opt_state):
            """
            Run n_steps of Adam on (primal + dual) params for fixed latent z.
            Returns updated params and per-step history of (loss, f, pen, x).
            """
            @jit
            def scan_body(carry, _):
                params, opt_state = carry
                (loss, (f_val, pen, x)), grads = jax.value_and_grad(
                    loss_fn, argnums=0, has_aux=True)(params, z)
                updates, new_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return (new_params, new_state), (loss, f_val, pen, x)

            (final_params, final_opt_state), hist = jax.lax.scan(
                scan_body, (params, opt_state), None, length=n_steps)
            return (final_params, final_opt_state), hist

        _trainer_cache[cache_key] = (train_fn, optimizer)
    return _trainer_cache[cache_key]


def warmup_jit():
    """
    Pre-compile all (problem × K × ρ × lr) JIT graphs before timing starts.
    CDU requires one compilation per (K, ρ, lr) triple per problem since
    JAX statically unrolls the Python for-loop over K.
    """
    total = len(PROBLEM_ORDER) * len(CDU_K_CHOICES) * len(RHO_CHOICES) * len(LR_CHOICES)
    print(f"  Pre-compiling {total} JIT graphs (problem × K × ρ × lr) ...")
    for pk in PROBLEM_ORDER:
        p = PROBLEMS[pk]
        key = jax.random.PRNGKey(0)
        params = init_cdu_params(key, p['n_vars'], p['n_cons'])
        z = jnp.zeros(p['n_vars'])
        for K in CDU_K_CHOICES:
            for rho in RHO_CHOICES:
                loss_fn = get_cdu_loss(pk, K, rho)
                for lr in LR_CHOICES:
                    tf, opt = get_scan_trainer(loss_fn, lr, N_STEPS)
                    out = tf(params, z, opt.init(params))
                    jax.block_until_ready(out[0])
        print(f"    OK  {DISPLAY_NAMES[pk]}", flush=True)
    print()
