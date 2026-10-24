"""Closed-form theory for the two readers, checked against the archived simulations.

Two-copy Bell (C/2 shots): the product of swap signs over A is +-1 with mean p = tr(rho_A^2), so
    Var_B(C) = (1 - p^2) / (C/2).

Single-copy shadows, one shot per random local Pauli basis, M = C snapshots, unbiased U-statistic
    U = (1 / (M(M-1))) sum_{i != j} h(x_i, x_j),   h(x, y) = tr(sigma_x sigma_y) = prod_q T6[x_q, y_q],
where x = (basis, bit) per qubit has probability P(x) = 3^-k <x_bits| U_b rho U_b^dag |x_bits>. Standard
U-statistic variance (order 2):
    Var_S(M) = 4 (M-2) / (M (M-1)) * zeta1 + 2 / (M (M-1)) * zeta2,
    zeta1 = Var_x( g(x) ),  g(x) = E_y h(x, y) = tr(sigma_x rho)          (= (T6^{(x)k} P)(x))
    zeta2 = Var_{x,y}( h(x, y) ) = P^T (T6^2)^{(x)k} P - p^2.
Both are exact tensor contractions of the 6^k probability table, applying T6 (or T6 squared
element-wise) along every axis.

Exchange rate: the copies m*C that shadows need to match Bell at C solve Var_S(mC) = Var_B(C) after
averaging over the dataset (RMSE^2 = mean variance for unbiased estimators). As C grows,
    m_inf = 2 * mean(zeta1) / mean(1 - p^2),
so the asymptotic exchange rate is set by the variance of a single snapshot's overlap with the state.

Outputs results/bell_qnn/paper_theory.json:
  * check: predicted RMSE of the UNCLIPPED S=1 reader and of Bell at C=400 vs the measured values in
    paper_scale.json (same seeds and states);
  * multipliers: predicted (unclipped) exchange rate at C=400 and its C -> inf limit, for |A| = 2..8,
    next to the measured (clipped) multipliers from paper_copies_*.json.

    PY=~/proyectos/active/q-taskfisher/.venv/bin/python
    $PY theory_bell.py [--ks 2 3 4 5 6 7 8] [--n_est 200]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import brentq

import exp_bell_paper as P

T6 = P.T6
T6SQ = T6 ** 2


def prob_table(rho: np.ndarray, k: int) -> np.ndarray:
    """P(x) as a (6,)*k tensor; see exp_bell_paper.prob_tensor (single implementation)."""
    return P.prob_tensor(rho, k)


def apply_all_axes(m: np.ndarray, t: np.ndarray) -> np.ndarray:
    for ax in range(t.ndim):
        t = np.moveaxis(np.tensordot(m, t, axes=([1], [ax])), 0, ax)
    return t


def zetas(rho: np.ndarray, k: int) -> tuple[float, float, float]:
    Pt = prob_table(rho, k)
    g = apply_all_axes(T6, Pt)                 # g(x) = tr(sigma_x rho)
    p = float(np.sum(Pt * g))                  # E g = tr(rho^2)
    z1 = float(np.sum(Pt * g * g) - p * p)
    z2 = float(np.sum(Pt * apply_all_axes(T6SQ, Pt)) - p * p)
    return p, z1, z2


def var_s(m: float, z1: float, z2: float) -> float:
    return 4 * (m - 2) / (m * (m - 1)) * z1 + 2 / (m * (m - 1)) * z2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--n_est", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(P.SEEDS))
    a = ap.parse_args()
    P.oracle()
    # exactness check of the zeta formulas on one small state (Monte Carlo of the estimator itself)
    rng = np.random.default_rng(9401)
    rho = P.rdms(9401, 3, 1)[0]
    p, z1, z2 = zetas(rho, 3)
    reps = np.array([P.shadow_purity(P.shadow_symbols(rho, 3, 200, 1, rng), 3, 1) for _ in range(4000)])
    pred = var_s(200, z1, z2)
    rel = abs(reps.var(ddof=1) - pred) / pred
    assert rel < 0.08, f"U-statistic variance formula off by {rel:.3f}"
    assert abs(reps.mean() - p) < 5 * reps.std() / np.sqrt(len(reps))
    print(f"THEORY ORACLE OK: MC var {reps.var(ddof=1):.5f} vs formula {pred:.5f} (rel {rel:.3f})", flush=True)

    rows = []
    for seed in a.seeds:
        for k in a.ks:
            rh = P.rdms(seed, k, a.n_est)
            st = np.array([zetas(r, k) for r in rh])
            p, z1, z2 = st[:, 0], st[:, 1], st[:, 2]
            vb = float(np.mean(1 - p ** 2) / 200)                 # Bell at C = 400
            vs = float(np.mean([var_s(400, x, y) for x, y in zip(z1, z2)]))
            f = lambda m: np.mean([var_s(m, x, y) for x, y in zip(z1, z2)]) - vb
            m_star = brentq(f, 4, 1e9) / 400
            m_inf = 2 * float(np.mean(z1)) / float(np.mean(1 - p ** 2))
            rows.append({"seed": seed, "k": k, "rmse_bell_pred": vb ** 0.5, "rmse_s1_pred": vs ** 0.5,
                         "mult_pred": m_star, "mult_inf": m_inf,
                         "zeta1": float(np.mean(z1)), "zeta2": float(np.mean(z2)), "one_minus_p2": float(np.mean(1 - p ** 2))})
            print(json.dumps(rows[-1]), flush=True)
            json.dump({"rows": rows}, open(P.OUT.format("theory"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
