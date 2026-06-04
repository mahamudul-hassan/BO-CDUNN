"""
Global configuration, hyperparameter search spaces, and matplotlib style.
"""
import warnings
import os

import jax
import matplotlib
import matplotlib.pyplot as plt
import optuna

# ─── Suppress warnings & prepare output dirs ──────────────────────────────────
warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)
jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
HIDDEN         = 32
N_STEPS        = 150
N_TRIALS_MAIN  = 1000
N_TRIALS_STAT  = 1000
N_STAT_RUNS    = 30
BO_FEAS_TOL    = 0.0   # strict: any g_i(x) > 0 => infeasible

# ── CDU-specific hyperparameter search spaces ─────────────────────────────────
# K  : number of primal-dual unrolling steps (Optuna tunes per trial)
# ρ  : augmented Lagrangian quadratic penalty coefficient
CDU_K_CHOICES  = [3, 5, 8]           # unrolling depth choices
RHO_CHOICES    = [1e2, 1e3, 1e4]     # augmented Lagrangian ρ choices
LR_CHOICES     = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

# log((f+M)² + ε) numerical safety
LOG_EPS        = 1e-30

# relative gap epsilon
_EPS           = 1e-12

# ─── Publication plot style ───────────────────────────────────────────────────
GRID_KW = dict(alpha=0.30, linestyle='--', color='#999999')
plt.rcParams.update({
    'figure.facecolor': 'white',   'axes.facecolor':   'white',
    'axes.edgecolor':   '#444444', 'axes.linewidth':   0.8,
    'text.color':       '#111111', 'axes.labelcolor':  '#111111',
    'xtick.color':      '#333333', 'ytick.color':      '#333333',
    'grid.color':       '#999999', 'grid.linestyle':   '--',
    'grid.alpha':       0.30,
    'font.family':      'serif',   'font.size':        10,
    'axes.titlesize':   11,        'axes.labelsize':   10,
    'xtick.labelsize':  9,         'ytick.labelsize':  9,
    'legend.fontsize':  9,         'legend.framealpha': 0.92,
    'legend.edgecolor': '#888888', 'figure.dpi':       150,
    'savefig.dpi':      300,       'savefig.bbox':     'tight',
    'lines.linewidth':  1.6,
})
C_BO    = '#2166ac'   # BO+CDU  — blue
C_FIXED = '#d6604d'   # IPOPT   — red-orange
