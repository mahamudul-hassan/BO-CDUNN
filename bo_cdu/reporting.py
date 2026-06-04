"""
Result reporting: writes all CSV files to results/ and prints the terminal
summary tables.
"""
import csv

import numpy as np

from .config import N_STEPS, N_STAT_RUNS
from .problems import PROBLEMS, PROBLEM_ORDER, DISPLAY_NAMES
from .stats import stat_summary


def _fstr(v):
    return f"{v:.6f}" if (not np.isinf(v) and not np.isnan(v)) else "N/A"


def _xstr(x):
    return ";".join(f"{v:.5f}" for v in x) if not np.all(np.isnan(x)) else "N/A"


def _sv(v):
    return float(v) if (not np.isinf(v) and not np.isnan(v)) else 0.


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE CSV RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def save_csv_results(main_results, stat_results):
    print("\n  >  Saving CSV results ...")

    # 01. Main results  (CDU-specific columns: Best_K, Best_rho, Best_lr)
    with open("results/01_main_results.csv", "w", newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["Problem", "Method", "Best_f",
                    "Feasible", "Total_CV", "Feas_Rate_pct",
                    "Rel_Gap_wrt_other", "Time_ms",
                    "Best_K", "Best_rho", "Best_lr",
                    "Best_x"])
        for pk in PROBLEM_ORDER:
            bo = main_results[pk]['bo']
            ip = main_results[pk]['ipopt']
            for tag, d, k_, r_, l_ in [
                ("BO+CDU",      bo, bo.get('best_K', 'N/A'),
                 bo.get('best_rho', 'N/A'), bo.get('best_lr', 'N/A')),
                ("IPOPT_fixed", ip, 'N/A', 'N/A', 'N/A'),
            ]:
                fr = d.get('feas_rate', 'N/A')
                w.writerow([pk, tag, _fstr(d['best_f']),
                            d['feasible'],
                            _fstr(d['total_cv']) if not np.isnan(d['total_cv']) else "N/A",
                            f"{fr:.1f}" if isinstance(fr, float) else fr,
                            _fstr(d['rel_gap']),
                            f"{d['time_ms']:.0f}",
                            k_, r_, l_,
                            _xstr(d['best_x'])])

    # 02. Feasibility analysis
    with open("results/02_feasibility_analysis.csv", "w", newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["Problem", "Method", "Constraint_Index", "Constraint_Label",
                    "g_i_value", "Violation_max_gi_0",
                    "Constraint_Feasible", "Total_CV", "Solution_Feasible"])
        for pk in PROBLEM_ORDER:
            p = PROBLEMS[pk]
            for tag, d in [("BO+CDU",      main_results[pk]['bo']),
                           ("IPOPT_fixed", main_results[pk]['ipopt'])]:
                cv_arr = np.atleast_1d(d['cv'])
                is_nan = np.all(np.isnan(cv_arr))
                total  = d['total_cv']
                for idx, lbl in enumerate(p['g_labels']):
                    gv   = float(cv_arr[idx]) if not is_nan else np.nan
                    viol = max(gv, 0.) if not np.isnan(gv) else np.nan
                    fi   = ("yes" if gv <= 0.0 else "VIOLATED") if not np.isnan(gv) else "N/A"
                    w.writerow([pk, tag, idx+1, lbl,
                                f"{gv:.10g}"  if not np.isnan(gv)   else "N/A",
                                f"{viol:.6e}" if not np.isnan(viol) else "N/A",
                                fi,
                                f"{total:.6e}" if not np.isnan(total) else "N/A",
                                "yes" if d['feasible'] else "VIOLATED"])

    # 03. Relative gap
    with open("results/03_relative_gap.csv", "w", newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["Problem",
                    "BO_CDU_f", "IPOPT_f",
                    "Rel_gap_BO_CDU_wrt_IPOPT",
                    "Rel_gap_IPOPT_wrt_BO_CDU",
                    "BO_CDU_feasible", "IPOPT_feasible",
                    "Stat_mean_f_BO_CDU", "Stat_std_f_BO_CDU", "Stat_var_f_BO_CDU",
                    "Stat_feas_rate_pct"])
        for pk in PROBLEM_ORDER:
            bo = main_results[pk]['bo'];  ip = main_results[pk]['ipopt']
            ss = stat_summary(stat_results, pk)
            w.writerow([pk,
                        _fstr(bo['best_f']), _fstr(ip['best_f']),
                        _fstr(bo['rel_gap']), _fstr(ip['rel_gap']),
                        bo['feasible'], ip['feasible'],
                        _fstr(ss['mean_f']), _fstr(ss['std_f']), _fstr(ss['var_f']),
                        f"{ss['feas_rate']:.1f}"])

    # 04. Statistical robustness
    with open("results/04_statistical_robustness.csv", "w", newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        max_vars = max(PROBLEMS[pk]['n_vars'] for pk in PROBLEM_ORDER)
        var_cols  = [f"x{i+1}" for i in range(max_vars)]
        w.writerow(["Problem", "Seed", "Best_f",
                    "Is_Feasible", "Feas_Rate_pct", "Total_CV", "Time_ms"]
                   + var_cols)
        for pk in PROBLEM_ORDER:
            n = PROBLEMS[pk]['n_vars']
            for r in stat_results[pk]:
                xrow = [f"{v:.6f}" for v in r['best_x']] + ["N/A"]*(max_vars - n)
                w.writerow([r['problem'], r['seed'],
                            _fstr(r['best_f']),
                            r['is_feasible'],
                            f"{r['feas_rate']:.1f}",
                            f"{r['total_cv']:.6e}",
                            f"{r['time_ms']:.0f}"] + xrow)

    # 04b. Variable distribution
    with open("results/04b_variable_distribution.csv", "w", newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["Problem", "Variable",
                    "Mean_x_all_runs", "Std_x_all_runs",
                    "Mean_x_feasible", "Std_x_feasible",
                    "Mean_f_feasible", "Std_f_feasible",
                    "Var_f_feasible",  "Feas_Rate_pct"])
        for pk in PROBLEM_ORDER:
            p  = PROBLEMS[pk]
            ss = stat_summary(stat_results, pk)
            for vi, vname in enumerate(p['var_names']):
                mf = ss['var_mean_feas'][vi];  sf = ss['var_std_feas'][vi]
                w.writerow([pk, vname,
                            f"{ss['var_mean_all'][vi]:.6f}",
                            f"{ss['var_std_all'][vi]:.6f}",
                            f"{mf:.6f}" if not np.isnan(mf) else "N/A",
                            f"{sf:.6f}" if not np.isnan(sf) else "N/A",
                            _fstr(ss['mean_f']), _fstr(ss['std_f']),
                            _fstr(ss['var_f']),  f"{ss['feas_rate']:.1f}"])

    # 05. Convergence data
    with open("results/05_convergence_data.csv", "w", newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["Step"] + PROBLEM_ORDER)
        for step in range(N_STEPS):
            row = [step + 1]
            for pk in PROBLEM_ORDER:
                fh = main_results[pk]['bo']['fhist']
                row.append(f"{fh[step]:.6f}" if fh is not None else "N/A")
            w.writerow(row)

    # 06. Initialization comparison
    with open("results/06_initialization_comparison.csv", "w", newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["Problem", "Method", "Initialization",
                    "Best_K", "Best_rho", "Best_lr",
                    "Best_f", "Feasible", "Total_CV",
                    "Rel_Gap_wrt_other", "Time_ms"])
        for pk in PROBLEM_ORDER:
            bo = main_results[pk]['bo'];  ip = main_results[pk]['ipopt']
            for tag, init_desc, d, k_, r_, l_ in [
                ("BO+CDU",      "Bayesian TPE + CDU (Optuna)", bo,
                 bo.get('best_K', 'N/A'), bo.get('best_rho', 'N/A'), bo.get('best_lr', 'N/A')),
                ("IPOPT_fixed", "[1, 1, ..., 1] fixed", ip, 'N/A', 'N/A', 'N/A'),
            ]:
                w.writerow([pk, tag, init_desc, k_, r_, l_,
                            _fstr(d['best_f']), d['feasible'],
                            _fstr(d['total_cv']) if not np.isnan(d['total_cv']) else "N/A",
                            _fstr(d['rel_gap']),
                            f"{d['time_ms']:.0f}"])

    print("  OK  All CSVs saved to results/")


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(main_results, stat_results):
    print("\n" + "="*85)
    print("  MAIN RESULTS — BO+CDU vs IPOPT Fixed Init")
    print("  CDU Loss: log((f+M)²+ε) + λ_K·relu(g) + (ρ/2)‖relu(g)‖²")
    print("  Feasibility: g_i(x) ≤ 0 strict for all i")
    print("="*85)
    print(f"  {'Problem':<22} {'BO+CDU f':>12} {'IPOPT f':>12}  "
          f"{'Rel gap CDU':>12}  {'Rel gap IP':>12}  {'CDU':>5}  {'IP':>5}  "
          f"{'K':>4}  {'rho':>8}")
    print("-"*85)
    for pk in PROBLEM_ORDER:
        bo = main_results[pk]['bo'];  ip = main_results[pk]['ipopt']
        print(f"  {DISPLAY_NAMES[pk]:<22}"
              f" {_sv(bo['best_f']):>12.4f}"
              f" {_sv(ip['best_f']):>12.4f}"
              f"  {_sv(bo['rel_gap']):>12.4f}"
              f"  {_sv(ip['rel_gap']):>12.4f}"
              f"  {'Y' if bo['feasible'] else 'N':>5}"
              f"  {'Y' if ip['feasible'] else 'N':>5}"
              f"  {str(bo.get('best_K', '?')):>4}"
              f"  {str(bo.get('best_rho', '?')):>8}")
    print("="*85)

    print(f"\n  STATISTICAL ROBUSTNESS  (BO+CDU, N={N_STAT_RUNS} runs)")
    print(f"  {'Problem':<22} {'mean f':>12} {'std f':>10} {'var f':>12}  {'Feas%':>7}")
    print("-"*68)
    for pk in PROBLEM_ORDER:
        ss = stat_summary(stat_results, pk)
        print(f"  {DISPLAY_NAMES[pk]:<22}"
              f" {_sv(ss['mean_f']):>12.5g}"
              f" {_sv(ss['std_f']):>10.4g}"
              f" {_sv(ss['var_f']):>12.4g}"
              f"  {ss['feas_rate']:>6.1f}%")
    print("="*85)

    print("\n  DECISION VARIABLE DISTRIBUTION  (BO+CDU, feasible runs)")
    print(f"  {'Problem':<22} {'Var':>5}  {'mean (all)':>12}  {'std (all)':>10}  "
          f"{'mean (feas)':>12}  {'std (feas)':>10}")
    print("-"*80)
    for pk in PROBLEM_ORDER:
        ss = stat_summary(stat_results, pk);  p = PROBLEMS[pk]
        for vi, vname in enumerate(p['var_names']):
            mf = ss['var_mean_feas'][vi];  sf = ss['var_std_feas'][vi]
            print(f"  {DISPLAY_NAMES[pk]:<22} {vname:>5}  "
                  f"{ss['var_mean_all'][vi]:>12.5g}  "
                  f"{ss['var_std_all'][vi]:>10.4g}  "
                  f"{(mf if not np.isnan(mf) else float('nan')):>12.5g}  "
                  f"{(sf if not np.isnan(sf) else float('nan')):>10.4g}")
    print("="*85)
    print("\n  All figures  -> figures/")
    print("  All CSVs     -> results/")
    print("\n  Done")
