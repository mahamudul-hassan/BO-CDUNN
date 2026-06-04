"""
IPOPT baseline solver via CasADi, using a fixed initialisation [1, ..., 1].
Strict feasibility: all g_i ≤ 0.
"""
import time

import numpy as np

from .problems import PROBLEMS


def _build_casadi_nlp(prob_key):
    import casadi as ca
    xv = ca.SX.sym('x', PROBLEMS[prob_key]['n_vars'])

    if prob_key == "Branin":
        a, b, c, r, s, t = 1., 5.1/(4*np.pi**2), 5/np.pi, 6., 10., 1/(8*np.pi)
        f  = a*(xv[1]-b*xv[0]**2+c*xv[0]-r)**2+s*(1-t)*ca.cos(xv[0])+s
        g  = [0.5 - ca.sin(xv[0]+xv[1])**2]
        lb = [-ca.inf];  ub = [0]
    elif prob_key == "Himmelblau":
        f  = (xv[0]**2+xv[1]-11)**2+(xv[0]+xv[1]**2-7)**2
        g  = [xv[0]**2+xv[1]-4, xv[0]+xv[1]**2-3]
        lb = [-ca.inf]*2;  ub = [0]*2
    elif prob_key == "Rosenbrock":
        f  = 100*(xv[1]-xv[0]**2)**2+(1-xv[0])**2
        g  = [xv[0]**2+xv[1]**2-2, xv[0]+xv[1]-1]
        lb = [-ca.inf]*2;  ub = [0]*2
    elif prob_key == "PressureVessel":
        x1, x2, x3, x4 = xv[0], xv[1], xv[2], xv[3]
        f  = (0.6224*x1*x3*x4+1.7781*x2*x3**2+3.1661*x1**2*x4+19.84*x1**2*x3)
        g  = [-x1+0.0193*x3, -x2+0.00954*x3,
              ca.pi*x3**2*x4+(4/3)*ca.pi*x3**3-1296000]
        lb = [-ca.inf]*3;  ub = [0]*3
    elif prob_key == "GearTrain":
        x1, x2, x3, x4, x5 = xv[0], xv[1], xv[2], xv[3], xv[4]
        f  = 5.3578457*x3**2+0.8356891*x1*x5+37.293239*x1-40792.141
        C1 = 85.334407+0.0056858*x2*x5+0.0006262*x1*x4-0.0022053*x3*x5
        C2 = 80.51249 +0.0071317*x2*x3+0.002995*x1*x2 +0.0021813*x3**2
        C3 = 9.300961 +0.0047026*x3*x5+0.0012547*x1*x3+0.0019085*x3*x4
        g  = [C1-92, 85-C1, C2-110, 90-C2, C3-25, 20-C3]
        lb = [-ca.inf]*6;  ub = [0]*6
    elif prob_key == "G06":
        f  = (xv[0]-10)**3+(xv[1]-20)**3
        g  = [-(xv[0]-5)**2-(xv[1]-5)**2+100, (xv[0]-6)**2+(xv[1]-5)**2-82.81]
        lb = [-ca.inf]*2;  ub = [0]*2
    elif prob_key == "WeldedBeam":
        h, l, t, b = xv[0], xv[1], xv[2], xv[3]
        P = 6000; L = 14; E = 30e6; G = 12e6
        tp  = P/(ca.sqrt(2)*h*l)
        R   = ca.sqrt(l**2/4+((h+t)/2)**2)
        J   = 2*ca.sqrt(2)*h*l*(l**2/12+((h+t)/2)**2)
        tpp = P*L*R/J
        tau = ca.sqrt(tp**2+tpp**2)
        sig = 6*P*L/(b*t**2)
        dlt = 4*P*L**3/(E*b*t**3)
        Pc  = 4.013*E*ca.sqrt(t**2*b**6/36)/L**2*(1-t/(2*L)*ca.sqrt(E/(4*G)))
        f   = 1.10471*h**2*l+0.04811*t*b*(14+l)
        g   = [tau-13600, sig-30000, dlt-0.25, P-Pc, h-b]
        lb  = [-ca.inf]*5;  ub = [0]*5
    else:
        raise ValueError(f"Unknown problem: {prob_key}")

    return xv, f, g, lb, ub


def solve_ipopt_fixed(prob_key: str):
    """Single IPOPT solve from [1,...,1].  Strict feasibility: all g_i ≤ 0."""
    try:
        import casadi as ca
        p  = PROBLEMS[prob_key]
        n  = p['n_vars']
        xv, f_ca, g_ca, lbg, ubg = _build_casadi_nlp(prob_key)
        opts   = {'ipopt': {'print_level': 0, 'tol': 1e-10, 'max_iter': 3000},
                  'print_time': 0}
        solver = ca.nlpsol('S', 'ipopt',
                           {'x': xv, 'f': f_ca, 'g': ca.vertcat(*g_ca)}, opts)
        g_fn   = ca.Function('g', [xv], [ca.vertcat(*g_ca)])

        x0 = np.ones(n).tolist()
        t0 = time.perf_counter()
        try:
            sol  = solver(x0=x0,
                          lbx=p['lb'].tolist(), ubx=p['ub'].tolist(),
                          lbg=lbg, ubg=ubg)
            xs   = np.array(sol['x']).flatten()
            fs   = float(sol['f'])
            cv   = np.array(g_fn(xs)).flatten()
            feas = bool(np.all(cv <= 0.0))
        except Exception:
            xs = np.full(n, np.nan);  fs = np.inf
            cv = np.full(len(g_ca), np.nan);  feas = False

        wall_ms = (time.perf_counter() - t0) * 1000
        return fs, xs, cv, wall_ms, feas

    except ImportError:
        n = PROBLEMS[prob_key]['n_vars']
        return np.inf, np.full(n, np.nan), np.array([np.nan]), 0., False
