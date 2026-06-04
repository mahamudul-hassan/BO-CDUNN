"""
CDU study runner.

Optuna jointly tunes K (unrolling depth), ρ (augmented Lagrangian penalty),
lr (Adam learning rate), and the latent input z. Also provides the relative
optimality gap helper.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optuna
from tqdm import tqdm

from .config import LR_CHOICES, CDU_K_CHOICES, RHO_CHOICES, _EPS
from .networks import init_cdu_params
from .losses import get_cdu_loss
from .trainer import get_scan_trainer


# ══════════════════════════════════════════════════════════════════════════════
#  RELATIVE OPTIMALITY GAP
# ══════════════════════════════════════════════════════════════════════════════
def relative_gap(f_a, f_b):
    if any(np.isinf(v) or np.isnan(v) for v in [f_a, f_b]):
        return np.nan
    return abs(f_a - f_b) / max(abs(f_b), _EPS)


# ══════════════════════════════════════════════════════════════════════════════
#  CDU STUDY RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_study(prob_key: str, obj_fn, cons_fn, n_vars, n_cons,
              n_trials, n_steps,
              jax_seed=42, problem_name="", optuna_seed=42, show_pbar=True):
    """
    CDU version of run_study.
    Returns: study, record, fmask, best_x, best_f, best_fhist
    """
    record       = dict(bx=[], bl=[])
    best_f_live  = [np.inf]
    best_fhist   = [None]
    n_feas_count = [0]

    if show_pbar:
        pbar = tqdm(total=n_trials, desc=f"  {problem_name:<32}",
                    unit="trial", colour="blue",
                    bar_format="{l_bar}{bar:30}{r_bar}", dynamic_ncols=True)

    def objective(trial):
        # ── Optuna suggests CDU hyperparameters ───────────────────────────
        lr  = trial.suggest_categorical('lr',  LR_CHOICES)
        K   = trial.suggest_categorical('K',   CDU_K_CHOICES)
        rho = trial.suggest_categorical('rho', RHO_CHOICES)
        z   = jnp.array([trial.suggest_float(f"z{i}", -1., 1.)
                         for i in range(n_vars)])

        # ── Build / retrieve CDU loss for this (K, rho) ───────────────────
        loss_fn = get_cdu_loss(prob_key, K, rho)

        key       = jax.random.PRNGKey(jax_seed * 10000 + trial.number)
        params    = init_cdu_params(key, n_vars, n_cons)
        train_fn, opt = get_scan_trainer(loss_fn, lr, n_steps)
        opt_state = opt.init(params)

        # ── Train: gradient descent on primal + dual params jointly ───────
        (final_params, _), hist = train_fn(params, z, opt_state)
        losses, f_vals, penalties, xs = hist
        jax.block_until_ready(losses)

        best_idx  = int(jnp.argmin(losses))
        best_x    = np.asarray(xs[best_idx])
        best_loss = float(losses[best_idx])

        # ── Strict feasibility check ──────────────────────────────────────
        raw_cv  = np.array(cons_fn(jnp.array(best_x)))
        is_feas = bool(np.all(raw_cv <= 0.0))

        record['bx'].append(best_x)
        record['bl'].append(best_loss)

        if is_feas:
            fv = float(obj_fn(jnp.array(best_x)))
            if fv < best_f_live[0]:
                best_f_live[0] = fv
                best_fhist[0]  = np.array(f_vals)
            n_feas_count[0] += 1

        if show_pbar:
            bs = f"{best_f_live[0]:.4f}" if not np.isinf(best_f_live[0]) else "---"
            pbar.set_postfix(
                feasible=f"{n_feas_count[0]}/{trial.number+1}",
                best_f=bs, K=K, rho=f"{rho:.0e}", refresh=False)
            pbar.update(1)
        return best_loss

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=optuna_seed))
    study.optimize(objective, n_trials=n_trials)
    if show_pbar:
        pbar.close()

    record['bx']  = np.array(record['bx'])
    record['blc'] = np.minimum.accumulate(record['bl'])

    # Best feasible solution (strict: all g_i ≤ 0)
    sols   = record['bx']
    cv_all = jax.vmap(cons_fn)(jnp.array(sols))
    fmask  = np.array(jnp.all(cv_all <= 0.0, axis=-1))
    feas_s = sols[fmask]

    if len(feas_s):
        fobjs  = np.array(jax.vmap(obj_fn)(jnp.array(feas_s)))
        best_x = feas_s[int(np.argmin(fobjs))]
        best_f = float(np.min(fobjs))
    else:
        best_x = sols[int(np.argmin(record['bl']))]
        best_f = np.inf

    return study, record, fmask, best_x, best_f, best_fhist[0]
