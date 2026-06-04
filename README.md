# BO + NN + CDU Benchmark Suite

> Constrained Dual Unrolling (CDU) for constrained optimization, with Optuna (TPE)
> hyperparameter search, benchmarked against an IPOPT fixed-initialization baseline
> across seven classic constrained problems.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![JAX](https://img.shields.io/badge/JAX-x64-orange)
![License](https://img.shields.io/badge/license-MIT-green)

This is a modular refactor of a single-cell research notebook (`Experiments.ipynb`)
into an importable Python package.

---

## Overview

CDU replaces a single soft-penalty MLP with **two coupled networks** whose forward
pass embeds primal-descent / dual-ascent dynamics as an inductive bias:

| Network | Map | Output activation |
|---------|-----|-------------------|
| Primal `π_θ` | `(z ∈ ℝⁿ, λ ∈ ℝᵐ) → x ∈ [lb, ub]` | sigmoid-scaled to the box |
| Dual `δ_φ`   | `(x ∈ ℝⁿ, λ ∈ ℝᵐ) → λ' ∈ ℝᵐ₊`     | softplus (λ ≥ 0)          |

**Unrolling (K learned steps):**

```
λ₀ = 0
for k = 0 … K-1:
    xₖ   = π_θ(z, λₖ)      # primal step
    λₖ₊₁ = δ_φ(xₖ, λₖ)     # dual step
```

**Augmented-Lagrangian loss:**

```
ℒ = log((f(x_K)+M)² + ε)            # log-compressed objective
  + λ_K · relu(g(x_K))               # Lagrangian dual term
  + (ρ/2) ‖relu(g(x_K))‖²            # quadratic augmentation
```

Box constraints are satisfied **by construction** via sigmoid scaling, so no box
penalty is needed. Optuna jointly tunes the unrolling depth `K`, penalty `ρ`,
learning rate `lr`, and the latent input `z` per trial.

---

## Benchmark problems

| Key | Display name | n_vars | n_cons |
|-----|--------------|:------:|:------:|
| `Branin`         | Branin                 | 2 | 1 |
| `Himmelblau`     | Himmelblau             | 2 | 2 |
| `Rosenbrock`     | Constrained Rosenbrock | 2 | 2 |
| `PressureVessel` | Pressure Vessel        | 4 | 3 |
| `GearTrain`      | G04 Gear Train         | 5 | 6 |
| `G06`            | G06 Global Opt.        | 2 | 2 |
| `WeldedBeam`     | Welded Beam            | 4 | 5 |

Feasibility is **strict**: a solution is feasible only if `gᵢ(x) ≤ 0` for all `i`.

---

## Repository layout

```
bo_cdu_benchmark/
├── main.py                  # entry point: run -> save CSVs -> figures -> summary
├── requirements.txt
├── README.md
└── bo_cdu/
    ├── __init__.py          # public API
    ├── config.py            # global config, search spaces, matplotlib style
    ├── problems.py          # objective/constraint fns + PROBLEMS registry
    ├── networks.py          # CDU primal/dual parameter initialization
    ├── losses.py            # log-loss helper + CDU augmented-Lagrangian loss factory
    ├── trainer.py           # cached scan-based Adam trainer + JIT warmup
    ├── study.py             # Optuna CDU study runner + relative gap
    ├── ipopt_solver.py      # CasADi/IPOPT baseline (fixed init)
    ├── experiment.py        # main loop -> main_results, stat_results
    ├── stats.py             # statistical summary helper
    ├── reporting.py         # CSV writers + terminal summary
    └── plotting.py          # the 5 publication figures
```

---

## Installation

```bash
git clone https://github.com/<your-username>/bo_cdu_benchmark.git
cd bo_cdu_benchmark
pip install -r requirements.txt
```

`casadi` is optional. If it is not installed, the IPOPT baseline is skipped
gracefully and the CDU pipeline still runs end to end.

---

## Usage

```bash
python main.py
```

Or drive it from your own script:

```python
from bo_cdu import run_experiments, save_csv_results, generate_figures, print_summary

main_results, stat_results = run_experiments()
save_csv_results(main_results, stat_results)
generate_figures(main_results, stat_results)
print_summary(main_results, stat_results)
```

---

## Outputs

Written to the working directory:

**`figures/`**
| File | Content |
|------|---------|
| `01_convergence_curves.png`   | Best-trial `f(x_K)` per step vs IPOPT solution |
| `02_relative_gap.png`         | Relative optimality gap (main + statistical) |
| `03_feasibility_analysis.png` | Feasibility rate and total constraint violation |
| `04_comparison_summary.png`   | Objective, gap, and wall-clock comparison |
| `05_variable_distribution.png`| Per-variable mean ± std across runs |

**`results/`**
| File | Content |
|------|---------|
| `01_main_results.csv`            | Main BO+CDU vs IPOPT, with best `K`, `ρ`, `lr` |
| `02_feasibility_analysis.csv`    | Per-constraint values and violations |
| `03_relative_gap.csv`            | Relative gaps + statistical summary |
| `04_statistical_robustness.csv`  | Per-seed runs with decision variables |
| `04b_variable_distribution.csv`  | Variable distribution statistics |
| `05_convergence_data.csv`        | Per-step objective history |
| `06_initialization_comparison.csv`| Initialization strategy comparison |

---

## Configuration

All tunables live in `bo_cdu/config.py`:

| Setting | Meaning | Default |
|---------|---------|---------|
| `N_TRIALS_MAIN` | Optuna trials in the main run | 1000 |
| `N_TRIALS_STAT` | trials per statistical run    | 1000 |
| `N_STAT_RUNS`   | number of statistical seeds   | 30 |
| `N_STEPS`       | Adam steps per trial          | 150 |
| `HIDDEN`        | hidden width of each network  | 32 |
| `CDU_K_CHOICES` | unrolling depths searched     | `[3, 5, 8]` |
| `RHO_CHOICES`   | penalty coefficients searched | `[1e2, 1e3, 1e4]` |
| `LR_CHOICES`    | learning rates searched       | `[1e-4 … 5e-2]` |

> Note: the defaults are heavy (the full sweep compiles many JIT graphs and runs
> thousands of trials). Lower `N_TRIALS_*` and `N_STAT_RUNS` for a quick trial run.

---

## License

Released under the MIT License. See `LICENSE` for details.
