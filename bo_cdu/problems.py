"""
Problem definitions: objective + constraint functions and the PROBLEMS registry.

The CDU loss is built generically (see `losses._make_cdu_loss`), so no
per-problem loss function is needed here — only objectives, constraints,
box bounds, the log-loss offset M, and metadata.
"""
import jax.numpy as jnp
from jax import jit
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEM DEFINITIONS  (objective + constraint functions)
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Branin ──────────────────────────────────────────────────────────────────
_BR_LB = jnp.array([-5.,  0.]);  _BR_UB = jnp.array([10., 15.])

@jit
def branin_obj(x):
    a, b, c, r, s, t = 1., 5.1/(4*jnp.pi**2), 5./jnp.pi, 6., 10., 1./(8*jnp.pi)
    return a*(x[1]-b*x[0]**2+c*x[0]-r)**2 + s*(1-t)*jnp.cos(x[0]) + s

@jit
def branin_cons(x):
    return jnp.stack([0.5 - jnp.sin(x[0]+x[1])**2])

# ── 2. Himmelblau ──────────────────────────────────────────────────────────────
_HB_LB = jnp.array([-6., -6.]);  _HB_UB = jnp.array([6., 6.])

@jit
def himmelblau_obj(x):
    return (x[0]**2 + x[1] - 11.)**2 + (x[0] + x[1]**2 - 7.)**2

@jit
def himmelblau_cons(x):
    return jnp.stack([x[0]**2 + x[1] - 4., x[0] + x[1]**2 - 3.])

# ── 3. Constrained Rosenbrock ──────────────────────────────────────────────────
_RB_LB = jnp.array([-5., -5.]);  _RB_UB = jnp.array([5., 5.])

@jit
def rosenbrock_obj(x):
    return 100.*(x[1] - x[0]**2)**2 + (1. - x[0])**2

@jit
def rosenbrock_cons(x):
    return jnp.stack([x[0]**2 + x[1]**2 - 2., x[0] + x[1] - 1.])

# ── 4. Pressure Vessel ─────────────────────────────────────────────────────────
_PV_LB = jnp.array([0.0625, 0.0625, 10., 10.])
_PV_UB = jnp.array([99.,    99.,   200., 200.])

@jit
def pressure_vessel_obj(x):
    return (0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]**2
            + 3.1661*x[0]**2*x[3] + 19.84*x[0]**2*x[2])

@jit
def pressure_vessel_cons(x):
    return jnp.stack([-x[0]+0.0193*x[2],
                       -x[1]+0.00954*x[2],
                       jnp.pi*x[2]**2*x[3]+(4./3.)*jnp.pi*x[2]**3-1_296_000.])

# ── 5. G04 – Gear Train ────────────────────────────────────────────────────────
_G4_LB = jnp.array([78., 33., 27., 27., 27.])
_G4_UB = jnp.array([102., 45., 45., 45., 45.])

@jit
def g04_obj(x):
    return 5.3578457*x[2]**2 + 0.8356891*x[0]*x[4] + 37.293239*x[0] - 40792.141

@jit
def g04_cons(x):
    C1 = 85.334407+0.0056858*x[1]*x[4]+0.0006262*x[0]*x[3]-0.0022053*x[2]*x[4]
    C2 = 80.51249 +0.0071317*x[1]*x[2]+0.002995*x[0]*x[1] +0.0021813*x[2]**2
    C3 = 9.300961 +0.0047026*x[2]*x[4]+0.0012547*x[0]*x[2]+0.0019085*x[2]*x[3]
    return jnp.stack([C1-92., 85.-C1, C2-110., 90.-C2, C3-25., 20.-C3])

# ── 6. G06 ─────────────────────────────────────────────────────────────────────
_G6_LB = jnp.array([13.,  0.]);  _G6_UB = jnp.array([100., 100.])

@jit
def g06_obj(x):
    return (x[0] - 10.)**3 + (x[1] - 20.)**3

@jit
def g06_cons(x):
    return jnp.stack([-(x[0]-5.)**2-(x[1]-5.)**2+100.,
                       (x[0]-6.)**2+(x[1]-5.)**2-82.81])

# ── 7. Welded Beam ─────────────────────────────────────────────────────────────
_WB_LB = jnp.array([0.1, 0.1, 0.1, 0.1])
_WB_UB = jnp.array([2., 10., 10., 2.])

@jit
def welded_beam_obj(x):
    return 1.10471*x[0]**2*x[1] + 0.04811*x[2]*x[3]*(14. + x[1])

@jit
def welded_beam_cons(x):
    h, l, t, b = x[0], x[1], x[2], x[3]
    P = 6000.; L = 14.; E = 30.e6; G = 12.e6
    tp  = P/(jnp.sqrt(2.)*h*l)
    R   = jnp.sqrt(l**2/4.+((h+t)/2.)**2)
    J   = 2.*jnp.sqrt(2.)*h*l*(l**2/12.+((h+t)/2.)**2)
    tpp = P*L*R/J
    tau = jnp.sqrt(tp**2+tpp**2)
    sig = 6.*P*L/(b*t**2)
    dlt = 4.*P*L**3/(E*b*t**3)
    Pc  = (4.013*E*jnp.sqrt(t**2*b**6/36.)/L**2
           *(1.-(t/(2.*L))*jnp.sqrt(E/(4.*G))))
    return jnp.stack([tau-13600., sig-30000., dlt-0.25, P-Pc, h-b])


# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEM REGISTRY
#    n_cons   — number of inequality constraints (dual net input dim)
#    M        — log-loss offset; must satisfy f(x)+M > 0 everywhere
# ══════════════════════════════════════════════════════════════════════════════
PROBLEMS = {
    "Branin": dict(
        n_vars=2, n_cons=1,
        lb=np.array([-5., 0.]), ub=np.array([10., 15.]),
        lb_j=_BR_LB, ub_j=_BR_UB,
        obj_fn=branin_obj, cons_fn=branin_cons,
        M=50.,      # f ∈ [0.4, 308]  => f+M > 0 ✓
        g_labels=['0.5-sin²(x₁+x₂)'], var_names=['x1', 'x2'],
    ),
    "Himmelblau": dict(
        n_vars=2, n_cons=2,
        lb=np.array([-6., -6.]), ub=np.array([6., 6.]),
        lb_j=_HB_LB, ub_j=_HB_UB,
        obj_fn=himmelblau_obj, cons_fn=himmelblau_cons,
        M=10.,      # f ∈ [0, 890]    => f+M > 0 ✓
        g_labels=['x1²+x2-4', 'x1+x2²-3'], var_names=['x1', 'x2'],
    ),
    "Rosenbrock": dict(
        n_vars=2, n_cons=2,
        lb=np.array([-5., -5.]), ub=np.array([5., 5.]),
        lb_j=_RB_LB, ub_j=_RB_UB,
        obj_fn=rosenbrock_obj, cons_fn=rosenbrock_cons,
        M=5.,       # f ≥ 0           => f+M > 0 ✓
        g_labels=['x1²+x2²-2', 'x1+x2-1'], var_names=['x1', 'x2'],
    ),
    "PressureVessel": dict(
        n_vars=4, n_cons=3,
        lb=np.array([0.0625, 0.0625, 10., 10.]),
        ub=np.array([99., 99., 200., 200.]),
        lb_j=_PV_LB, ub_j=_PV_UB,
        obj_fn=pressure_vessel_obj, cons_fn=pressure_vessel_cons,
        M=100.,     # f ∈ [180, 1.2e7] => f+M > 0 ✓
        g_labels=['-x1+0.0193x3', '-x2+0.00954x3',
                  'πx3²x4+(4/3)πx3³-1296000'],
        var_names=['x1', 'x2', 'x3', 'x4'],
    ),
    "GearTrain": dict(
        n_vars=5, n_cons=6,
        lb=np.array([78., 33., 27., 27., 27.]),
        ub=np.array([102., 45., 45., 45., 45.]),
        lb_j=_G4_LB, ub_j=_G4_UB,
        obj_fn=g04_obj, cons_fn=g04_cons,
        M=40000.,   # f ≥ -32218      => f+M ≥ 7782 > 0 ✓
        g_labels=['C1-92', '85-C1', 'C2-110', '90-C2', 'C3-25', '20-C3'],
        var_names=['x1', 'x2', 'x3', 'x4', 'x5'],
    ),
    "G06": dict(
        n_vars=2, n_cons=2,
        lb=np.array([13., 0.]), ub=np.array([100., 100.]),
        lb_j=_G6_LB, ub_j=_G6_UB,
        obj_fn=g06_obj, cons_fn=g06_cons,
        M=10000.,   # f ≥ -7973       => f+M ≥ 2027 > 0 ✓
        g_labels=['-(x1-5)²-(x2-5)²+100', '(x1-6)²+(x2-5)²-82.81'],
        var_names=['x1', 'x2'],
    ),
    "WeldedBeam": dict(
        n_vars=4, n_cons=5,
        lb=np.array([0.1, 0.1, 0.1, 0.1]),
        ub=np.array([2., 10., 10., 2.]),
        lb_j=_WB_LB, ub_j=_WB_UB,
        obj_fn=welded_beam_obj, cons_fn=welded_beam_cons,
        M=5.,       # f ≥ 1.4         => f+M > 0 ✓
        g_labels=['τ-13600', 'σ-30000', 'δ-0.25', 'P-Pc', 'h-b'],
        var_names=['h', 'l', 't', 'b'],
    ),
}

PROBLEM_ORDER = ["Branin", "Himmelblau", "Rosenbrock",
                 "PressureVessel", "GearTrain", "G06", "WeldedBeam"]
DISPLAY_NAMES = {
    "Branin":         "Branin",
    "Himmelblau":     "Himmelblau",
    "Rosenbrock":     "Constrained Rosenbrock",
    "PressureVessel": "Pressure Vessel",
    "GearTrain":      "G04 Gear Train",
    "G06":            "G06 Global Opt.",
    "WeldedBeam":     "Welded Beam",
}
