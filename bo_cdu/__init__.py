"""
BO+NN + CDU Benchmark Suite — Q1 Journal Version (CDU Extension)
================================================================
Constrained Dual Unrolling (CDU) replaces the soft-penalty MLP with two
coupled networks that embed primal-descent / dual-ascent dynamics as
inductive biases, following 2025-2026 CDU literature.

Architecture
------------
  Primal net  π_θ : (z ∈ ℝⁿ, λ ∈ ℝᵐ)  →  x ∈ [lb, ub]   (sigmoid-scaled)
  Dual   net  δ_φ : (x ∈ ℝⁿ, λ ∈ ℝᵐ)  →  λ' ∈ ℝᵐ₊        (softplus output)

Loss (Augmented Lagrangian form)
---------------------------------
  ℒ = log((f(x_K)+M)²+ε) + λ_K · relu(g(x_K)) + (ρ/2) ‖relu(g(x_K))‖²

Public API
----------
  run_experiments() -> (main_results, stat_results)
  save_csv_results, print_summary, generate_figures
"""
from .config import *  # noqa: F401,F403
from .problems import PROBLEMS, PROBLEM_ORDER, DISPLAY_NAMES
from .experiment import run_experiments
from .reporting import save_csv_results, print_summary
from .plotting import generate_figures
from .stats import stat_summary

__all__ = [
    "PROBLEMS", "PROBLEM_ORDER", "DISPLAY_NAMES",
    "run_experiments", "save_csv_results", "print_summary",
    "generate_figures", "stat_summary",
]
