"""
Statistical summary helper over the per-seed robustness rows.
"""
import numpy as np


def stat_summary(stat_results, pk):
    """Summarise feasibility, objective stats, and per-variable distribution."""
    rows   = stat_results[pk]
    feas   = [r for r in rows if r['is_feasible'] and not np.isinf(r['best_f'])]
    fs     = [r['best_f'] for r in feas]
    all_x  = np.array([r['best_x'] for r in rows])
    feas_x = np.array([r['best_x'] for r in feas])

    var_mean_all  = all_x.mean(axis=0)
    var_std_all   = all_x.std(axis=0)
    var_mean_feas = (feas_x.mean(axis=0) if len(feas_x)
                     else np.full(all_x.shape[1], np.nan))
    var_std_feas  = (feas_x.std(axis=0) if len(feas_x)
                     else np.full(all_x.shape[1], np.nan))

    return dict(
        n_feas        = len(feas),
        feas_rate     = np.mean([r['is_feasible'] for r in rows]) * 100,
        mean_f        = np.mean(fs) if fs else np.nan,
        std_f         = np.std(fs)  if fs else np.nan,
        var_f         = np.var(fs)  if fs else np.nan,
        mean_tcv      = np.mean([r['total_cv'] for r in rows]),
        var_mean_all  = var_mean_all,
        var_std_all   = var_std_all,
        var_mean_feas = var_mean_feas,
        var_std_feas  = var_std_feas,
    )
