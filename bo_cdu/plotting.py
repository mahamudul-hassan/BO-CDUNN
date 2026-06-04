"""
Figure generation: produces the five publication figures into figures/.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np

from .config import N_STEPS, N_STAT_RUNS, GRID_KW, C_BO, C_FIXED
from .problems import PROBLEMS, PROBLEM_ORDER, DISPLAY_NAMES
from .study import relative_gap
from .stats import stat_summary


def _sv(v):
    return float(v) if (not np.isinf(v) and not np.isnan(v)) else 0.


def _style_ax(ax):
    ax.tick_params(colors='#333333', labelcolor='#333333')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.title.set_color('#111111')
    ax.xaxis.label.set_color('#111111')
    ax.yaxis.label.set_color('#111111')


def generate_figures(main_results, stat_results):
    labels = [DISPLAY_NAMES[pk] for pk in PROBLEM_ORDER]
    x_pos  = np.arange(len(PROBLEM_ORDER))
    bw     = 0.35
    steps  = np.arange(1, N_STEPS + 1)

    print("\n  >  Generating figures ...")

    # ─── FIGURE 01 — CONVERGENCE CURVES ──────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(
        "CDU Convergence: BO+CDU Best Trial f(x_K) vs IPOPT Solution",
        fontsize=12, fontweight='bold', y=1.01)
    axes_flat = axes.flatten()

    for idx, pk in enumerate(PROBLEM_ORDER):
        ax    = axes_flat[idx]
        fhist = main_results[pk]['bo']['fhist']
        ip_f  = main_results[pk]['ipopt']['best_f']
        bK    = main_results[pk]['bo'].get('best_K', '?')
        br    = main_results[pk]['bo'].get('best_rho', '?')

        if fhist is not None:
            ax.plot(steps, fhist, color=C_BO, lw=1.8,
                    label=f'BO+CDU (K={bK}, ρ={br:.0e})')
            ax.scatter([steps[-1]], [fhist[-1]], color=C_BO, s=45, zorder=5)

        if not np.isinf(ip_f):
            ax.axhline(ip_f, color=C_FIXED, lw=1.4, ls='--',
                       label=f'IPOPT fixed ({ip_f:.4g})')

        ax.set_title(DISPLAY_NAMES[pk], pad=5)
        ax.set_xlabel("Gradient Step")
        ax.set_ylabel("Objective f(x_K)")
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, **GRID_KW)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
        _style_ax(ax)

    axes_flat[-1].set_visible(False)
    plt.tight_layout(h_pad=2.0, w_pad=1.2)
    plt.savefig("figures/01_convergence_curves.png")
    plt.close()
    print("      -> figures/01_convergence_curves.png")

    # ─── FIGURE 02 — RELATIVE OPTIMALITY GAP ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Relative Optimality Gap: BO+CDU vs IPOPT",
                 fontsize=12, fontweight='bold')

    rel_gap_bo_main = [_sv(main_results[pk]['bo']['rel_gap'])    for pk in PROBLEM_ORDER]
    rel_gap_fi_main = [_sv(main_results[pk]['ipopt']['rel_gap']) for pk in PROBLEM_ORDER]

    stat_gap_means, stat_gap_stds = [], []
    for pk in PROBLEM_ORDER:
        ipf  = main_results[pk]['ipopt']['best_f']
        gaps = [relative_gap(r['best_f'], ipf) for r in stat_results[pk]]
        gaps = [g for g in gaps if not np.isnan(g)]
        stat_gap_means.append(np.mean(gaps) if gaps else 0.)
        stat_gap_stds.append(np.std(gaps)   if gaps else 0.)

    ax1.bar(x_pos - bw/2, rel_gap_bo_main, bw, color=C_BO, alpha=0.85,
            label='BO+CDU', edgecolor='#333333', linewidth=0.5)
    ax1.bar(x_pos + bw/2, rel_gap_fi_main, bw, color=C_FIXED, alpha=0.85,
            label='IPOPT fixed [1,...,1]', edgecolor='#333333', linewidth=0.5)
    ax1.set_yscale('symlog', linthresh=1e-4)
    ax1.set_xticks(x_pos);  ax1.set_xticklabels(labels, rotation=30, ha='right')
    ax1.set_title("|f_a - f_b| / |f_b|  (each vs the other)", pad=6)
    ax1.set_ylabel("Relative Optimality Gap (symlog)")
    ax1.legend();  ax1.grid(True, axis='y', **GRID_KW);  _style_ax(ax1)

    ax2.bar(x_pos, stat_gap_means, bw*1.2, color=C_BO, alpha=0.85,
            label=f'BO+CDU mean ({N_STAT_RUNS} runs)', edgecolor='#333333', linewidth=0.5)
    ax2.errorbar(x_pos, stat_gap_means, yerr=stat_gap_stds,
                 fmt='none', color='#333333', capsize=5, lw=1.3, label='std dev')
    ax2.set_yscale('symlog', linthresh=1e-4)
    ax2.set_xticks(x_pos);  ax2.set_xticklabels(labels, rotation=30, ha='right')
    ax2.set_title(f"BO+CDU Rel. Gap: Mean +/- Std ({N_STAT_RUNS} runs vs IPOPT)", pad=6)
    ax2.set_ylabel("Relative Gap (mean +/- std, symlog)")
    ax2.legend();  ax2.grid(True, axis='y', **GRID_KW);  _style_ax(ax2)

    plt.tight_layout()
    plt.savefig("figures/02_relative_gap.png")
    plt.close()
    print("      -> figures/02_relative_gap.png")

    # ─── FIGURE 03 — FEASIBILITY RELIABILITY ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CDU Feasibility Reliability Analysis", fontsize=12, fontweight='bold')

    feas_rates = [stat_summary(stat_results, pk)['feas_rate'] for pk in PROBLEM_ORDER]
    tcv_bo     = [_sv(main_results[pk]['bo']['total_cv'])    for pk in PROBLEM_ORDER]
    tcv_fi     = [_sv(main_results[pk]['ipopt']['total_cv']) for pk in PROBLEM_ORDER]

    bar_colors = ['#2166ac' if r >= 80 else '#fc8d59' if r >= 50 else '#d73027'
                  for r in feas_rates]
    bars = ax1.bar(x_pos, feas_rates, color=bar_colors, alpha=0.85, width=0.55,
                   edgecolor='#333333', linewidth=0.6)
    ax1.axhline(100, color='#444444', lw=0.8, ls=':', alpha=0.6)
    for bar, val in zip(bars, feas_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1.2,
                 f"{val:.0f}%", ha='center', va='bottom',
                 fontsize=8, fontweight='bold', color='#111111')
    ax1.set_xticks(x_pos);  ax1.set_xticklabels(labels, rotation=30, ha='right')
    ax1.set_ylim(0, 115)
    ax1.set_title(f"BO+CDU Feasibility Rate ({N_STAT_RUNS} runs)", pad=6)
    ax1.set_ylabel("Feasibility Rate (%)")
    ax1.grid(True, axis='y', **GRID_KW)
    ax1.legend(handles=[
        Patch(facecolor='#2166ac', label='Rate >= 80%'),
        Patch(facecolor='#fc8d59', label='50% <= Rate < 80%'),
        Patch(facecolor='#d73027', label='Rate < 50%')], fontsize=8, loc='lower right')
    _style_ax(ax1)

    ax2.bar(x_pos - bw/2, tcv_bo, bw, color=C_BO, alpha=0.85,
            label='BO+CDU', edgecolor='#333333', linewidth=0.5)
    ax2.bar(x_pos + bw/2, tcv_fi, bw, color=C_FIXED, alpha=0.85,
            label='IPOPT fixed [1,...,1]', edgecolor='#333333', linewidth=0.5)
    ax2.set_yscale('symlog', linthresh=1e-8)
    ax2.set_xticks(x_pos);  ax2.set_xticklabels(labels, rotation=30, ha='right')
    ax2.set_title("Total Constraint Violation  Σ max(gᵢ, 0)", pad=6)
    ax2.set_ylabel("Total CV (symlog scale)")
    ax2.legend();  ax2.grid(True, axis='y', **GRID_KW);  _style_ax(ax2)

    plt.tight_layout()
    plt.savefig("figures/03_feasibility_analysis.png")
    plt.close()
    print("      -> figures/03_feasibility_analysis.png")

    # ─── FIGURE 04 — COMPARISON SUMMARY ──────────────────────────────────────
    fig, axes4 = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("BO+CDU vs IPOPT Fixed Init — Overall Comparison",
                 fontsize=12, fontweight='bold')

    bo_f_vals = [_sv(main_results[pk]['bo']['best_f'])    for pk in PROBLEM_ORDER]
    fi_f_vals = [_sv(main_results[pk]['ipopt']['best_f']) for pk in PROBLEM_ORDER]
    bo_t_vals = [main_results[pk]['bo']['time_ms']        for pk in PROBLEM_ORDER]
    fi_t_vals = [main_results[pk]['ipopt']['time_ms']     for pk in PROBLEM_ORDER]

    for ax, ybo, yfi, title, ylabel, yscale in [
        (axes4[0], bo_f_vals,       fi_f_vals,       "Best Objective Value",    "f(x)",         None),
        (axes4[1], rel_gap_bo_main, rel_gap_fi_main, "Relative Optimality Gap", "Relative Gap", 'symlog'),
        (axes4[2], bo_t_vals,       fi_t_vals,       "Wall-Clock Time (ms)",    "Time (ms)",    None),
    ]:
        ax.bar(x_pos - bw/2, ybo, bw, color=C_BO, alpha=0.85,
               label='BO+CDU', edgecolor='#333333', linewidth=0.5)
        ax.bar(x_pos + bw/2, yfi, bw, color=C_FIXED, alpha=0.85,
               label='IPOPT fixed [1,...,1]', edgecolor='#333333', linewidth=0.5)
        if yscale:
            ax.set_yscale(yscale, linthresh=1e-4)
        ax.set_xticks(x_pos);  ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_title(title, pad=6);  ax.set_ylabel(ylabel)
        ax.legend();  ax.grid(True, axis='y', **GRID_KW);  _style_ax(ax)

    plt.tight_layout()
    plt.savefig("figures/04_comparison_summary.png")
    plt.close()
    print("      -> figures/04_comparison_summary.png")

    # ─── FIGURE 05 — DECISION VARIABLE DISTRIBUTION ──────────────────────────
    fig, axes5 = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        f"Decision Variable Distribution across {N_STAT_RUNS} CDU Runs (BO+CDU)",
        fontsize=12, fontweight='bold', y=1.01)
    axes5_flat = axes5.flatten()

    for idx, pk in enumerate(PROBLEM_ORDER):
        ax     = axes5_flat[idx]
        p      = PROBLEMS[pk]
        ss     = stat_summary(stat_results, pk)
        vnames = p['var_names']
        nv     = p['n_vars']
        vx     = np.arange(nv)

        ax.errorbar(vx - 0.12, ss['var_mean_all'], yerr=ss['var_std_all'],
                    fmt='o', color=C_BO, capsize=5, lw=1.4, ms=5,
                    label='All runs (mean +/- std)')
        if not np.all(np.isnan(ss['var_mean_feas'])):
            ax.errorbar(vx + 0.12, ss['var_mean_feas'], yerr=ss['var_std_feas'],
                        fmt='s', color=C_FIXED, capsize=5, lw=1.4, ms=5,
                        label='Feasible only (mean +/- std)')

        ax.set_xticks(vx);  ax.set_xticklabels(vnames)
        ax.set_title(DISPLAY_NAMES[pk], pad=5)
        ax.set_ylabel("Variable Value");  ax.set_xlabel("Decision Variable")
        ax.legend(fontsize=8);  ax.grid(True, axis='y', **GRID_KW);  _style_ax(ax)

        mf = ss['mean_f'];  sf = ss['std_f']
        label_txt = (f"f: {mf:.4g} +/- {sf:.3g}" if not np.isnan(mf)
                     else "No feasible solutions")
        ax.text(0.97, 0.97, label_txt, transform=ax.transAxes,
                fontsize=7.5, ha='right', va='top', color='#111111',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#999999', alpha=0.9))

    axes5_flat[-1].set_visible(False)
    plt.tight_layout(h_pad=2.5, w_pad=1.5)
    plt.savefig("figures/05_variable_distribution.png")
    plt.close()
    print("      -> figures/05_variable_distribution.png")
