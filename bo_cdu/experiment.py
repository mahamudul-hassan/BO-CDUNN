"""
Main experiment loop: for each problem, runs the main BO+CDU study, the IPOPT
baseline, and the statistical robustness sweep. Returns the populated
main_results and stat_results dicts.
"""
import time

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .config import N_TRIALS_MAIN, N_TRIALS_STAT, N_STAT_RUNS, N_STEPS, \
    CDU_K_CHOICES, RHO_CHOICES, LR_CHOICES
from .problems import PROBLEMS, PROBLEM_ORDER, DISPLAY_NAMES
from .trainer import warmup_jit
from .study import run_study, relative_gap
from .ipopt_solver import solve_ipopt_fixed


def run_experiments():
    """Run the full benchmark suite and return (main_results, stat_results)."""
    print("=" * 70)
    print("  BO+NN+CDU vs IPOPT (fixed init) — Journal Benchmark Suite")
    print("  CDU Loss: log((f+M)²+ε) + λ_K·relu(g) + (ρ/2)‖relu(g)‖²")
    print("  Optuna tunes: K (unrolling depth), ρ (aug. Lagrangian), lr, z")
    print("=" * 70)
    print(f"  N_TRIALS_MAIN={N_TRIALS_MAIN}  N_STEPS={N_STEPS}")
    print(f"  CDU_K_CHOICES={CDU_K_CHOICES}  RHO_CHOICES={RHO_CHOICES}")
    print(f"  LR_CHOICES={LR_CHOICES}")
    print(f"  Feasibility: strict g_i(x) <= 0.0 for all i\n")

    warmup_jit()

    main_results = {}
    stat_results = {}

    for prob_key in PROBLEM_ORDER:
        p     = PROBLEMS[prob_key]
        dname = DISPLAY_NAMES[prob_key]
        print(f"\n{'---'*23}\n  [{prob_key}]  {dname}\n{'---'*23}")

        # ── A. Main BO+CDU run ────────────────────────────────────────────────
        print(f"  >  Main BO+CDU ({N_TRIALS_MAIN} trials) ...")
        t0 = time.perf_counter()
        study, record, fmask, bo_x, bo_f, bo_fhist = run_study(
            prob_key=prob_key,
            obj_fn=p['obj_fn'], cons_fn=p['cons_fn'],
            n_vars=p['n_vars'], n_cons=p['n_cons'],
            n_trials=N_TRIALS_MAIN, n_steps=N_STEPS,
            jax_seed=42, problem_name=dname, optuna_seed=42)
        bo_time = (time.perf_counter() - t0) * 1000
        bo_cv   = np.array(p['cons_fn'](jnp.array(bo_x)))
        bo_feas = bool(np.all(bo_cv <= 0.0))
        bo_fr   = fmask.mean() * 100
        bo_tcv  = float(np.sum(np.maximum(bo_cv, 0.)))
        print(f"  OK  f={bo_f:.5f}  feasible={bo_feas}  "
              f"feas_rate={bo_fr:.0f}%  time={bo_time:.0f}ms")

        # Best CDU trial hyperparameters (for reporting)
        best_trial = study.best_trial
        best_K   = best_trial.params.get('K',   '?')
        best_rho = best_trial.params.get('rho', '?')
        best_lr  = best_trial.params.get('lr',  '?')
        print(f"       Best CDU trial: K={best_K}  ρ={best_rho:.0e}  lr={best_lr}")

        # ── B. IPOPT fixed init ───────────────────────────────────────────────
        print(f"  >  IPOPT fixed [1,...,1] ...")
        fi_f, fi_x, fi_cv, fi_time, fi_feas = solve_ipopt_fixed(prob_key)
        fi_tcv = (float(np.sum(np.maximum(fi_cv, 0.)))
                  if not np.all(np.isnan(fi_cv)) else np.nan)
        rel_gap_bo = relative_gap(bo_f, fi_f)
        rel_gap_fi = relative_gap(fi_f, bo_f)
        if not np.isinf(fi_f):
            print(f"  OK  f={fi_f:.5f}  feasible={fi_feas}  time={fi_time:.0f}ms")
            print(f"      Rel gaps: BO+CDU={rel_gap_bo:.4f}  IPOPT={rel_gap_fi:.4f}")
        else:
            print("  XX  IPOPT: CasADi not available or solver failed.")

        main_results[prob_key] = dict(
            bo=dict(best_f=bo_f, best_x=bo_x, cv=bo_cv, feasible=bo_feas,
                    feas_rate=bo_fr, total_cv=bo_tcv, time_ms=bo_time,
                    fmask=fmask, record=record, fhist=bo_fhist,
                    rel_gap=rel_gap_bo,
                    best_K=best_K, best_rho=best_rho, best_lr=best_lr),
            ipopt=dict(best_f=fi_f, best_x=fi_x, cv=fi_cv, feasible=fi_feas,
                       total_cv=fi_tcv, time_ms=fi_time, rel_gap=rel_gap_fi),
        )

        # ── C. Statistical robustness ─────────────────────────────────────────
        print(f"  >  Statistical analysis ({N_STAT_RUNS} × {N_TRIALS_STAT} trials) ...")
        stat_rows = []
        for seed in tqdm(range(N_STAT_RUNS),
                         desc=f"  {'Stats: '+dname:<32}",
                         unit="run", colour="green",
                         bar_format="{l_bar}{bar:28}{r_bar}", dynamic_ncols=True):
            ts = time.perf_counter()
            _, srec, sfmask, sbx, sbf, _ = run_study(
                prob_key=prob_key,
                obj_fn=p['obj_fn'], cons_fn=p['cons_fn'],
                n_vars=p['n_vars'], n_cons=p['n_cons'],
                n_trials=N_TRIALS_STAT, n_steps=N_STEPS,
                jax_seed=seed, problem_name="", optuna_seed=42, show_pbar=False)
            run_t = (time.perf_counter() - ts) * 1000
            scv   = np.array(p['cons_fn'](jnp.array(sbx)))
            sfeas = bool(np.all(scv <= 0.0))
            stat_rows.append(dict(
                problem=prob_key, seed=seed,
                best_f=sbf, best_x=sbx.tolist(),
                is_feasible=int(sfeas),
                feas_rate=sfmask.mean() * 100,
                total_cv=float(np.sum(np.maximum(scv, 0.))),
                time_ms=run_t,
            ))

        stat_results[prob_key] = stat_rows
        valid_fs = [r['best_f'] for r in stat_rows
                    if r['is_feasible'] and not np.isinf(r['best_f'])]
        print(f"  OK  feasible={sum(r['is_feasible'] for r in stat_rows)}/{N_STAT_RUNS}"
              + (f"  mean_f={np.mean(valid_fs):.5f}  std_f={np.std(valid_fs):.5f}"
                 if valid_fs else "  (no feasible)"))

    return main_results, stat_results
