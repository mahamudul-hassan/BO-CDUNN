"""
CDU loss construction.

Provides the log-compressed objective helper and the augmented-Lagrangian
CDU loss factory, plus a cache keyed by (problem, K, ρ) so the same loss
object id is reused and the JIT trainer cache hits correctly.
"""
import jax
import jax.numpy as jnp

from .config import LOG_EPS
from .problems import PROBLEMS


# ══════════════════════════════════════════════════════════════════════════════
#  LOG-LOSS HELPER
#  loss_obj = log((f + M)² + ε)
#
#  Gradient:  2(f+M) / ((f+M)² + ε)  ≈  2/(f+M)  for large |f+M|
#  => objective gradient shrinks as |f+M| grows, so the Lagrangian dual
#     term (λ_K · relu(g)) always dominates near any violated constraint,
#     regardless of how negative f is.
#
#  REQUIREMENT: f(x) + M > 0  for all x in the search box.
# ══════════════════════════════════════════════════════════════════════════════
def log_loss(f, M):
    """log((f+M)² + ε)  — numerically safe, gradient-friendly."""
    return jnp.log((f + M) ** 2 + LOG_EPS)


def _make_cdu_loss(obj_fn, cons_fn, lb_j, ub_j, n_vars, n_cons, M, K, rho):
    """
    Factory: builds a CDU loss closure for fixed (K, ρ).

    The returned loss_fn(params, z) runs K unrolling steps:
      x_k      = sigmoid_scaled(primal_net(z ‖ λ_k))   — primal descent
      λ_{k+1}  = softplus(dual_net(x_k ‖ λ_k))         — dual ascent

    Then computes:
      ℒ = log_loss(f(x_K), M)
        + λ_K · relu(g(x_K))
        + (ρ/2) ‖relu(g(x_K))‖²

    Box constraints [lb, ub] are satisfied BY CONSTRUCTION via sigmoid
    scaling inside the primal network — no additional box penalty needed.
    """
    def loss_fn(params, z):
        pp = params['primal']   # primal network weights
        dp = params['dual']     # dual   network weights

        # ── Initialise multipliers ─────────────────────────────────────────
        lam = jnp.zeros(n_cons)

        # ── K primal-dual unrolling steps ──────────────────────────────────
        # Python for-loop: JAX traces & unrolls statically at JIT time.
        x = lb_j  # placeholder; overwritten in first iteration
        for _ in range(K):
            # Primal step: π_θ(z ‖ λ) → x ∈ [lb, ub]
            inp_p = jnp.concatenate([z, lam])
            h = jax.nn.relu(jnp.dot(inp_p, pp[0]) + pp[1])
            h = jax.nn.relu(jnp.dot(h,     pp[2]) + pp[3])
            x = lb_j + jax.nn.sigmoid(jnp.dot(h, pp[4]) + pp[5]) * (ub_j - lb_j)

            # Dual step: δ_φ(x ‖ λ) → λ' ≥ 0  (softplus enforces non-negativity)
            inp_d = jnp.concatenate([x, lam])
            h = jax.nn.relu(jnp.dot(inp_d, dp[0]) + dp[1])
            h = jax.nn.relu(jnp.dot(h,     dp[2]) + dp[3])
            lam = jax.nn.softplus(jnp.dot(h, dp[4]) + dp[5])

        # ── Augmented Lagrangian loss ───────────────────────────────────────
        f  = obj_fn(x)
        g  = cons_fn(x)
        rv = jax.nn.relu(g)                            # relu(g): violations only

        aug_lag = (log_loss(f, M)                      # log-compressed objective
                   + jnp.dot(lam, rv)                  # Lagrangian dual term
                   + 0.5 * rho * jnp.sum(rv ** 2))     # quadratic augmentation

        pen = jnp.sum(rv)   # total constraint violation (for logging)
        return aug_lag, (f, pen, x)

    return loss_fn


# ─── CDU loss cache: maps (prob_key, K, rho) → compiled loss_fn ───────────────
_CDU_LOSS_CACHE: dict = {}


def get_cdu_loss(prob_key: str, K: int, rho: float):
    """
    Return (and cache) a CDU loss function for (problem, K, rho).
    Caching ensures the same Python object id is reused, so the JIT
    trainer cache hits correctly without recompilation.
    """
    cache_key = (prob_key, K, rho)
    if cache_key not in _CDU_LOSS_CACHE:
        p = PROBLEMS[prob_key]
        _CDU_LOSS_CACHE[cache_key] = _make_cdu_loss(
            obj_fn  = p['obj_fn'],
            cons_fn = p['cons_fn'],
            lb_j    = p['lb_j'],
            ub_j    = p['ub_j'],
            n_vars  = p['n_vars'],
            n_cons  = p['n_cons'],
            M       = p['M'],
            K       = K,
            rho     = rho,
        )
    return _CDU_LOSS_CACHE[cache_key]
