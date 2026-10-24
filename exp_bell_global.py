"""Global single-copy rival and the published variance bound (prior-art check, 21-Oct).

Two questions a reviewer will ask, answered on the official seeds and the SAME states as the scale/copies
sweeps (exp_bell_paper.rdms):

 1. Global rival. Single-copy readers with one Haar-random unitary on all of A per shot (one shot each),
    snapshot rho_hat = (d+1) U^dag|b><b|U - I, U-statistic purity. With v_i = U_i^dag|b_i>,
        tr(rho_hat_i rho_hat_j) = (d+1)^2 |<v_i|v_j>|^2 - (d+2),
    so the estimator is a Gram-matrix average. This is the non-adaptive single-copy reader with optimal
    dimension scaling for purity (cost ~ d = 2^|A| per unit error at small M, against ~2.7^|A| for local
    Pauli bases); it needs an entangling circuit across A, whereas both the local reader and the
    transversal Bell reader use depth-one measurement circuits.
 2. Bound vs exact. Elben et al. (arXiv:2007.06305, Eq. D16) bound the local-shadow purity variance by
    4*2^k*p2/M + 4*(2^{1.5k}/M)^2. We compute the exchange rate implied by that bound and compare it with
    the exact one from paper_theory.json.

Output: results/bell_qnn/paper_global.json  (rows per seed and |A|: RMSE of the global reader at C=400,
clipped; copies multiplier to match Bell at C=400 on the grid; bound-implied local multiplier).

    PY=~/proyectos/active/q-taskfisher/.venv/bin/python
    $PY exp_bell_global.py [--ks 2 3 4 5 6] [--seeds ...] [--tag X]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import brentq

import exp_bell_paper as P

C_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600)


def haar_vectors(rho: np.ndarray, m: int, rng: np.random.Generator, chunk: int = 1024) -> np.ndarray:
    """m one-shot global Haar snapshots of rho: returns v_i = U_i^dag |b_i>, shape (m, d).

    Exact, O(m d): each basis vector of a Haar unitary is uniform on the sphere and outcome b has probability
    <v|rho|v>, so the observed v has density d<v|rho|v> = sum_i lambda_i d|<v|e_i>|^2 w.r.t. the uniform
    measure: a mixture. Pick eigenvector e_i with probability lambda_i; under d|<v|e>|^2 the overlap
    t = |<v|e>|^2 ~ Beta(2, d-1) with uniform phase, and the orthogonal part is a uniform direction.
    Checked against the explicit QR sampler (oracle)."""
    d = rho.shape[0]
    lam, vec = np.linalg.eigh(rho)
    lam = np.clip(lam, 0, None)
    lam /= lam.sum()
    idx = rng.choice(d, size=m, p=lam)
    e = vec[:, idx].T                                      # (m, d) chosen eigenvectors
    t = rng.beta(2.0, d - 1.0, size=m)
    ph = np.exp(2j * np.pi * rng.random(m))
    g = rng.standard_normal((m, d)) + 1j * rng.standard_normal((m, d))
    g -= e * np.sum(e.conj() * g, axis=1, keepdims=True)    # project out e
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return np.sqrt(t)[:, None] * ph[:, None] * e + np.sqrt(1 - t)[:, None] * g


def haar_vectors_rej(rho: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Rejection sampler (uniform v accepted w.p. <v|rho|v>/lambda_max); reference for the oracle."""
    d = rho.shape[0]
    lmax = float(np.linalg.eigvalsh(rho)[-1])
    out, got = np.empty((m, d), dtype=complex), 0
    while got < m:
        n = max(256, int(1.3 * (m - got) * d * lmax) + 16)
        g = rng.standard_normal((n, d)) + 1j * rng.standard_normal((n, d))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        acc = np.real(np.einsum("na,ab,nb->n", g.conj(), rho, g)) / lmax > rng.random(n)
        v = g[acc][: m - got]
        out[got:got + len(v)] = v
        got += len(v)
    return out


def haar_vectors_qr(rho: np.ndarray, m: int, rng: np.random.Generator, chunk: int = 1024) -> np.ndarray:
    """Reference sampler with explicit Haar unitaries (slow; used only by the oracle)."""
    d = rho.shape[0]
    out = np.empty((m, d), dtype=complex)
    for s in range(0, m, chunk):
        n = min(chunk, m - s)
        z = (rng.standard_normal((n, d, d)) + 1j * rng.standard_normal((n, d, d))) / np.sqrt(2)
        q, r = np.linalg.qr(z)
        ph = np.diagonal(r, axis1=1, axis2=2)
        q = q * (ph / np.abs(ph))[:, None, :]
        # q[i] is Haar; measuring U rho U^dag in the computational basis: p(b) = <b|U rho U^dag|b>
        pr = np.real(np.einsum("mba,ac,mbc->mb", q, rho, q.conj()))
        pr = np.clip(pr, 0, None)
        pr /= pr.sum(axis=1, keepdims=True)
        b = (pr.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
        out[s:s + n] = q[np.arange(n), b, :].conj()   # row b of U, conjugated = U^dag|b>
    return out


def global_purity(v: np.ndarray) -> float:
    """U-statistic purity from one-shot global snapshots. Uses sum_{i!=j}|<v_i|v_j>|^2 = ||sum_i v_i v_i^dag||_F^2 - m
    (O(m d^2) instead of the O(m^2 d) Gram matrix)."""
    m, d = v.shape
    s_mat = v.T @ v.conj()                             # sum_i |v_i><v_i|
    s = (float(np.real(np.vdot(s_mat, s_mat))) - m) / (m * (m - 1))
    return float((d + 1) ** 2 * s - (d + 2))


def oracle(rng: np.random.Generator) -> None:
    """Both samplers unbiased on a fixed 2-qubit mixed state, and their estimator variances agree."""
    rho = np.diag([0.6, 0.2, 0.15, 0.05]).astype(complex)
    p = float(np.real(np.trace(rho @ rho)))
    for name, fn in (("mixture", haar_vectors), ("rejection", haar_vectors_rej), ("qr", haar_vectors_qr)):
        e = np.array([global_purity(fn(rho, 60, rng)) for _ in range(3000)])
        se = e.std(ddof=1) / np.sqrt(len(e))
        assert abs(e.mean() - p) < 5 * se, f"{name}: biased {e.mean():.4f} vs {p:.4f} (se {se:.4f})"
        if name == "mixture":
            v_ref = e.var(ddof=1)
        else:
            assert abs(e.var(ddof=1) / v_ref - 1) < 0.12, f"sampler variances differ: {v_ref:.4f} vs {e.var(ddof=1):.4f}"
        print(f"GLOBAL ORACLE OK ({name}): mean {e.mean():.4f} vs {p:.4f} (se {se:.4f}), var {e.var(ddof=1):.4f}", flush=True)


def interp_copies(cs, rm, target):
    cs, rm = np.log(np.array(cs, float)), np.log(np.array(rm))
    if rm[0] <= np.log(target):
        return float(np.exp(cs[0]))
    for i in range(1, len(cs)):
        if rm[i] <= np.log(target):
            f = (np.log(target) - rm[i - 1]) / (rm[i] - rm[i - 1])
            return float(np.exp(cs[i - 1] + f * (cs[i] - cs[i - 1])))
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(P.SEEDS))
    ap.add_argument("--n_est", type=int, default=200)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    P.oracle()
    oracle(np.random.default_rng(9701))
    th = {(r["seed"], r["k"]): r for r in json.load(open(P.OUT.format("theory")))["rows"]}
    rows = []
    for seed in a.seeds:
        for k in a.ks:
            rng = np.random.default_rng(seed * 1009 + k + 77)
            rh = P.rdms(seed, k, a.n_est)
            p = np.array([float(np.real(np.trace(r @ r))) for r in rh])
            bell = P.rmse(np.array([P.bell_purity(x, 400, rng) for x in p]), p)
            rm = {}
            for c in C_GRID:
                e = np.array([np.clip(global_purity(haar_vectors(r, c, rng)), 2.0 ** -k, 1) for r in rh])
                rm[c] = P.rmse(e, p)
                if rm[c] < bell:
                    break
            cs = sorted(rm)
            m_glob = interp_copies(cs, [rm[c] for c in cs], bell) / 400
            # multiplier implied by the published local bound (Elben et al. 2020, Eq. D16)
            vb = float(np.mean(1 - p ** 2) / 200)
            f = lambda mm: float(np.mean(4 * 2 ** k * p / mm + 4 * (2 ** (1.5 * k) / mm) ** 2)) - vb
            m_bound = brentq(f, 2, 1e12) / 400
            t = th.get((seed, k), {})
            rows.append({"seed": seed, "k": k, "bell_C400": bell, "global": {str(c): v for c, v in rm.items()},
                         "global_C400": rm.get(400), "mult_global": m_glob, "mult_bound_local": m_bound,
                         "mult_exact_local": t.get("mult_pred"), "mult_inf_exact_local": t.get("mult_inf"),
                         "mult_inf_bound_local": float(2 * 2 ** k * np.mean(p) / np.mean(1 - p ** 2))})
            print(json.dumps(rows[-1]), flush=True)
            json.dump({"rows": rows}, open(P.OUT.format("global" + a.tag), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
