"""Paper experiments for idea 4 (bell_qnn), after the 21-Oct audit. Design frozen; official seeds.

Subcommands (each writes results/bell_qnn/paper_<cmd>.json):

  scale     RMSE of tr(rho_A^2) at C = 400 for |A| = 2..6 (n = 2|A|): Bell vs shadows S=1 clipped vs S=10.
  copies    Copies the strongest shadow reader needs to match Bell at C = 400, measured on a grid of C
            (no extrapolation), for |A| = 2..6; plus the C-sweep of both readers at |A| = 4.
  noise     Exact per-qubit depolarising and readout noise on raw measured bits (validated against the
            density matrix in audit_noise_model.py / audit_deep.py), |A| = 4, n = 8, C = 400.
  ntangled  NTangled (4 qubits, HWE depth 3, classes CE 0.05 vs 0.25): RMSE of the concentratable
            entanglement and binary classification, Bell vs shadows S=1 clipped.

Exactness of the reduced-state shortcut used by `scale` and `copies`:
  * Single-copy Pauli snapshots restricted to A are distributed exactly as local-basis measurements of
    rho_A, so they are sampled from rho_A (2^k x 2^k) instead of the n-qubit state.
  * The product of Bell swap signs over A is the eigenvalue of SWAP_A, a +-1 observable with mean
    tr(rho_A^2); C/2 Bell shots therefore give exactly 2 Binomial(C/2, (1+p)/2)/(C/2) - 1.
  Both are checked against the full simulators of qdlib at start-up (oracle; asserts on failure).

    PY=~/proyectos/active/q-taskfisher/.venv/bin/python
    $PY exp_bell_paper.py scale|copies|noise|ntangled [--seeds ...]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy import stats

import qdlib as Q

SEEDS = (0, 1, 7, 42, 123, 456, 789, 1234, 2024, 3407)
PROF = (1, 2, 3, 4)
OUT = "results/bell_qnn/paper_{}.json"
T6 = Q._T6  # tr(snapshot_i snapshot_j) per qubit, indexed by symbol 2*basis + bit


# ------------------------------------------------------------------ states
def brickwork(n: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for l in range(depth):
        for q in range(l % 2, n - 1, 2):
            psi = Q.apply_2q(psi, Q.haar_unitary(4, rng), q, q + 1, n)
    return psi


def rdms(seed: int, k: int, n_est: int) -> list[np.ndarray]:
    n = 2 * k
    rng = np.random.default_rng(seed * 101 + k)
    return [Q.rdm(brickwork(n, int(rng.choice(PROF)), rng), list(range(k)), n) for _ in range(n_est)]


# ------------------------------------------------------------------ readers on rho_A
def _unitaries(bases: np.ndarray) -> np.ndarray:
    u = Q.BASIS_ROT[bases[:, 0]]
    for q in range(1, bases.shape[1]):
        r = Q.BASIS_ROT[bases[:, q]]
        c, a, _ = u.shape
        u = np.einsum("cab,cde->cadbe", u, r).reshape(c, 2 * a, 2 * a)
    return u


_ALL_BASES: dict = {}


def prob_tensor(rho: np.ndarray, k: int) -> np.ndarray:
    """P(x) over x = (basis_q, bit_q)_q as a (6,)*k tensor, symbol = 2*basis + bit, basis drawn
    uniformly. Built qubit by qubit (contract ket/bra axis q with <o|U_b|i> and keep the (b, o)
    diagonal): exact, and ~6^q 4^(k-q) per step instead of 3^k 8^k for full unitaries."""
    R = Q.BASIS_ROT
    t = rho.reshape((2,) * (2 * k)).astype(complex)
    L = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for q in range(k):
        done, kets, bras = L[:q], L[20:20 + (k - q)], L[30:30 + (k - q)]
        expr = f"YZ{kets[0]},YZ{bras[0]},{done}{kets}{bras}->{done}YZ{kets[1:]}{bras[1:]}"
        t = np.einsum(expr, R, R.conj(), t, optimize=True)
        sh = t.shape
        t = t.reshape(sh[:q] + (6,) + sh[q + 2:])
    return np.real(t) / 3 ** k


def _basis_table(rho: np.ndarray, k: int) -> np.ndarray:
    """Outcome distributions of rho for all 3^k local Pauli bases: (3^k, 2^k). Exact; computed once."""
    if k not in _ALL_BASES:
        _ALL_BASES[k] = np.array(np.unravel_index(np.arange(3 ** k), (3,) * k)).T
    t = prob_tensor(rho, k).reshape((3, 2) * k)
    t = np.transpose(t, list(range(0, 2 * k, 2)) + list(range(1, 2 * k, 2))).reshape(3 ** k, 2 ** k)
    pr = np.clip(t * 3 ** k, 0, None)
    return pr / pr.sum(axis=1, keepdims=True)


def shadow_symbols(rho: np.ndarray, k: int, m: int, s: int, rng: np.random.Generator,
                   table: np.ndarray | None = None) -> np.ndarray:
    """m snapshots of rho (k qubits) in B = m/s random Pauli bases, s shots each. Returns symbols
    2*basis + bit, shape (m, k), rows grouped by basis. Uses the exact per-basis outcome table."""
    b = m // s
    if table is None:
        table = _basis_table(rho, k)
    bidx = rng.integers(0, 3 ** k, size=b)
    bases = _ALL_BASES[k][bidx]
    acum = table.cumsum(axis=1)[np.repeat(bidx, s)]
    idx = (acum > rng.random((b * s, 1))).argmax(axis=1)
    bits = (idx[:, None] >> (k - 1 - np.arange(k))) & 1
    return 2 * np.repeat(bases, s, axis=0) + bits


def _pair_sum(codes: np.ndarray, k: int) -> float:
    """sum_{i,j} prod_q T6[sym_iq, sym_jq] (all ordered pairs, i = j included) via a 6^k histogram."""
    flat = np.ravel_multi_index(codes.T, (6,) * k)
    h = np.bincount(flat, minlength=6 ** k).astype(float).reshape((6,) * k)
    t = h
    for ax in range(k):
        t = np.moveaxis(np.tensordot(T6, t, axes=([1], [ax])), 0, ax)
    return float(np.sum(h * t))


def shadow_purity(sym: np.ndarray, k: int, s: int) -> float:
    """Unbiased U-statistic of tr(rho^2) summing only pairs from different bases (groups of s)."""
    m = sym.shape[0]
    b = m // s
    tot = _pair_sum(sym, k)
    within = sum(_pair_sum(sym[g * s:(g + 1) * s], k) for g in range(b)) if s > 1 else m * 5.0 ** k
    return (tot - within) / (b * (b - 1) * s * s)


def bell_purity(p: float, copies: int, rng: np.random.Generator) -> float:
    shots = copies // 2
    return 2.0 * rng.binomial(shots, min(1.0, max(0.0, (1.0 + p) / 2.0))) / shots - 1.0


def oracle() -> None:
    """Fast readers == qdlib readers (same estimator), and unbiased against exact purity."""
    rng = np.random.default_rng(9301)
    k = 3
    psi = brickwork(6, 3, rng)
    rho = Q.rdm(psi, [0, 1, 2], 6)
    p = float(np.real(np.trace(rho @ rho)))
    # 0) fast outcome table == explicit unitaries for every basis
    allb = np.array(np.unravel_index(np.arange(27), (3,) * 3)).T
    u = _unitaries(allb)
    direct = np.real(np.einsum("cia,ab,cib->ci", u, rho, u.conj()))
    assert np.abs(_basis_table(rho, k) - direct / direct.sum(1, keepdims=True)).max() < 1e-12, "basis table"
    # 1) identical estimator: feed the same symbols to qdlib.shadow_traces
    sym = shadow_symbols(rho, k, 400, 10, rng)
    bases, bits = sym // 2, sym % 2
    mine = shadow_purity(sym, k, 10)
    Q.SHOTS_POR_BASE = 10
    ref, _ = Q.shadow_traces(bases, bits, list(range(k)))
    assert abs(mine - ref) < 1e-9, f"shadow estimator mismatch {mine} vs {ref}"
    # 2) unbiased, S = 1 and S = 10
    for s in (1, 10):
        reps = np.array([shadow_purity(shadow_symbols(rho, k, 400, s, rng), k, s) for _ in range(300)])
        z = abs(reps.mean() - p) / (reps.std(ddof=1) / np.sqrt(len(reps)))
        assert z < 5, f"shadow S={s} biased: z={z:.2f}"
    # 3) Bell binomial == full Bell sampler of qdlib on the same state (means within 5 sigma)
    full = np.array([float(np.prod(Q.sample_bell(psi, 6, 200, rng)[:, :k], axis=1).mean()) for _ in range(300)])
    fast = np.array([bell_purity(p, 400, rng) for _ in range(300)])
    z = abs(full.mean() - fast.mean()) / np.sqrt(full.var(ddof=1) / 300 + fast.var(ddof=1) / 300)
    assert z < 5, f"Bell shortcut differs from full sampler: z={z:.2f}"
    assert abs(full.std() - fast.std()) / fast.std() < 0.2, "Bell shortcut variance mismatch"
    Q.SHOTS_POR_BASE = 10
    print("ORACLE OK: fast shadow estimator == qdlib.shadow_traces; unbiased S=1,10; Bell binomial == full",
          flush=True)


def rmse(est: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((est - p) ** 2)))


# ------------------------------------------------------------------ scale
def cmd_scale(seeds, n_est=200, copies=400, ks=(2, 3, 4, 5, 6)):
    rows = []
    for seed in seeds:
        for k in ks:
            t0 = time.time()
            rh = rdms(seed, k, n_est)
            p = np.array([float(np.real(np.trace(r @ r))) for r in rh])
            rng = np.random.default_rng(seed * 7919 + k)
            eb = np.array([bell_purity(x, copies, rng) for x in p])
            e1 = np.array([shadow_purity(shadow_symbols(r, k, copies, 1, rng), k, 1) for r in rh])
            e10 = np.array([shadow_purity(shadow_symbols(r, k, copies, 10, rng), k, 10) for r in rh])
            lo = 2.0 ** -k
            rows.append({"seed": seed, "k": k, "bell": rmse(eb, p), "bell_clip": rmse(np.clip(eb, lo, 1), p),
                         "shadow_S1": rmse(e1, p), "shadow_S1_clip": rmse(np.clip(e1, lo, 1), p),
                         "shadow_S10": rmse(e10, p), "shadow_S10_clip": rmse(np.clip(e10, lo, 1), p),
                         "out_of_range_S1": float(np.mean((e1 < lo) | (e1 > 1))),
                         "sd_purity": float(p.std()), "sec": round(time.time() - t0, 1)})
            print(json.dumps(rows[-1]), flush=True)
            json.dump({"copies": copies, "n_est": n_est, "rows": rows}, open(OUT.format("scale"), "w"), indent=1)


# ------------------------------------------------------------------ copies
C_GRID = (100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600)


def cmd_copies(seeds, n_est=100, ks=(2, 3, 4, 5, 6)):
    rows = []
    for seed in seeds:
        for k in ks:
            rh = rdms(seed, k, n_est)
            p = np.array([float(np.real(np.trace(r @ r))) for r in rh])
            rng = np.random.default_rng(seed * 104729 + k)
            lo = 2.0 ** -k
            target = rmse(np.array([bell_purity(x, 400, rng) for x in p]), p)
            row = {"seed": seed, "k": k, "bell_C400": target, "shadow": {}, "bell": {}}
            tables = [_basis_table(r, k) for r in rh]
            for c in C_GRID:
                if k == 4 or c <= 3200:
                    row["bell"][c] = rmse(np.array([bell_purity(x, c, rng) for x in p]), p)
                e = np.array([shadow_purity(shadow_symbols(r, k, c, 1, rng, t), k, 1) for r, t in zip(rh, tables)])
                row["shadow"][c] = rmse(np.clip(e, lo, 1), p)
                if row["shadow"][c] < 0.8 * target:
                    break
            print(json.dumps(row), flush=True)
            rows.append(row)
            json.dump({"rows": rows}, open(OUT.format("copies"), "w"), indent=1)


def copies_to_match(row: dict) -> float:
    """log-log interpolation of the shadow RMSE curve at the Bell RMSE (C = 400)."""
    cs = sorted(int(c) for c in row["shadow"])
    r = [row["shadow"][c] if c in row["shadow"] else row["shadow"][str(c)] for c in cs]
    t = row["bell_C400"]
    for i in range(1, len(cs)):
        if r[i] <= t <= r[i - 1]:
            x0, x1, y0, y1 = np.log(cs[i - 1]), np.log(cs[i]), np.log(r[i - 1]), np.log(r[i])
            return float(np.exp(x0 + (np.log(t) - y0) * (x1 - x0) / (y1 - y0)))
    return float("nan")


# ------------------------------------------------------------------ noise (full n-qubit, raw bits)
def _bell_bits(psi, n, shots, rng):
    pr = Q.bell_probs_two_copies(psi, n)
    idx = rng.choice(pr.size, size=shots, p=pr)
    bits = (idx[:, None] >> (2 * n - 1 - np.arange(2 * n))) & 1
    return bits[:, :n].copy(), bits[:, n:].copy()


def _noisy_bell_signs(a, b, p, eps, rng):
    a, b = a.copy(), b.copy()
    if p > 0:
        for _ in range(2):  # both copies
            u = rng.random(a.shape)
            fx, fz, fy = u < p / 4, (u >= p / 4) & (u < p / 2), (u >= p / 2) & (u < 3 * p / 4)
            b ^= (fx | fy).astype(b.dtype)
            a ^= (fz | fy).astype(a.dtype)
    if eps > 0:
        a ^= (rng.random(a.shape) < eps).astype(a.dtype)
        b ^= (rng.random(b.shape) < eps).astype(b.dtype)
    return 1 - 2 * (a & b)


def _noisy_symbols(sym, p, eps, rng):
    bits, bases = sym % 2, sym // 2
    if p > 0:
        m = rng.random(bits.shape) < p
        bits = np.where(m, rng.integers(0, 2, size=bits.shape), bits)
    if eps > 0:
        bits = bits ^ (rng.random(bits.shape) < eps)
    return 2 * bases + bits


NOISE_GRID = [("depol", v) for v in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)] + \
             [("readout", v) for v in (0.01, 0.02, 0.05, 0.10, 0.20)]


def cmd_noise(seeds, n_est=200, copies=400, k=4):
    n = 2 * k
    lo = 2.0 ** -k
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed * 31 + 5)
        est = [brickwork(n, int(rng.choice(PROF)), rng) for _ in range(n_est)]
        rh = [Q.rdm(s, list(range(k)), n) for s in est]
        p = np.array([float(np.real(np.trace(r @ r))) for r in rh])
        clean = [(_bell_bits(s, n, copies // 2, rng), shadow_symbols(r, k, copies, 1, rng))
                 for s, r in zip(est, rh)]
        for kind, v in NOISE_GRID:
            pd, eps = (v, 0.0) if kind == "depol" else (0.0, v)
            eb, pf, es = [], [], []
            for (a, b), sym in clean:
                sg = _noisy_bell_signs(a, b, pd, eps, rng)
                eb.append(float(np.prod(sg[:, :k], axis=1).mean()))
                pf.append(float(np.prod(sg, axis=1).mean()))
                es.append(shadow_purity(_noisy_symbols(sym, pd, eps, rng), k, 1))
            eb, es = np.array(eb), np.clip(np.array(es), lo, 1)
            lam = float(np.clip(((np.mean(pf) - 2.0 ** -n) / (1 - 2.0 ** -n)) ** (1 / n), 0.02, 1))
            lk = lam ** k
            cb = np.clip((eb - (1 - lk) / 2 ** k) / lk, lo, 1)
            rows.append({"seed": seed, "kind": kind, "value": v, "lam": lam,
                         "bell": rmse(eb, p), "bell_corr": rmse(cb, p), "shadow": rmse(es, p),
                         "bias_bell": float(np.mean(eb - p)), "bias_shadow": float(np.mean(es - p)),
                         "sd_bell": float(np.std(eb - p)), "sd_shadow": float(np.std(es - p))})
        print(json.dumps({"seed": seed, "done": len(rows)}), flush=True)
    json.dump({"copies": copies, "n_est": n_est, "k": k, "rows": rows}, open(OUT.format("noise"), "w"), indent=1)


# ------------------------------------------------------------------ NTangled
NT = "/tmp/ntangled/Hardware_Efficient/4_Qubits/Depth_3/hwe_4q_ps_{}_3_weights.npy"


def cmd_ntangled(seeds, per_class=100, copies=400):
    import dev_ntangled as D
    n = 4
    subs = D.subsets(n)[1:]
    thetas = {0: np.load(NT.format(5)), 1: np.load(NT.format(25))}
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed * 13 + 1)
        X, y = [], []
        for cls in (0, 1):
            for _ in range(per_class):
                X.append(D.hwe(thetas[cls], D.product_state(n, rng, True)))
                y.append(cls)
        perm = rng.permutation(len(X))
        X, y = [X[i] for i in perm], np.array([y[i] for i in perm])
        ce = np.array([D.ce_exacto(s, n, [()] + subs) for s in X])
        eb, es = [], []
        for s in X:
            sg = Q.sample_bell(s, n, copies // 2, rng)
            eb.append(1.0 - (1.0 + sum(float(np.prod(sg[:, list(S)], axis=1).mean()) for S in subs)) / 2 ** n)
            Q.SHOTS_POR_BASE = 1
            b, o = Q.sample_shadows(s, n, copies, rng)
            Q.SHOTS_POR_BASE = 10
            tot = 1.0
            for S in subs:
                p2, _ = Q.shadow_traces(b, o, list(S))
                tot += float(np.clip(p2, 2.0 ** -len(S), 1.0))
            es.append(1.0 - tot / 2 ** n)
        eb, es = np.array(eb), np.array(es)
        ntr = int(0.6 * len(X))
        tr, te = np.arange(ntr), np.arange(ntr, len(X))
        rows.append({"seed": seed, "rmse_bell": rmse(eb, ce), "rmse_shadow": rmse(es, ce),
                     "err_bell": Q.logreg_error(eb[tr, None], y[tr], eb[te, None], y[te]),
                     "err_shadow": Q.logreg_error(es[tr, None], y[tr], es[te, None], y[te]),
                     "err_exact": Q.logreg_error(ce[tr, None], y[tr], ce[te, None], y[te]),
                     "ce_class_means": [float(ce[y == 0].mean()), float(ce[y == 1].mean())]})
        print(json.dumps(rows[-1]), flush=True)
    d = np.array([r["rmse_bell"] - r["rmse_shadow"] for r in rows])
    w = stats.wilcoxon(d, alternative="less", method="exact")
    json.dump({"copies": copies, "per_class": per_class, "rows": rows,
               "wilcoxon_rmse": float(w.pvalue)}, open(OUT.format("ntangled"), "w"), indent=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scale", "copies", "noise", "ntangled", "oracle"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--ks", type=int, nargs="+", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global OUT
    if a.tag:
        OUT = "results/bell_qnn/paper_{}_" + a.tag + ".json"
    kw = {"ks": tuple(a.ks)} if a.ks and a.cmd in ("scale", "copies") else {}
    Q.set_backend("pennylane")
    oracle()
    {"scale": cmd_scale, "copies": cmd_copies, "noise": cmd_noise, "ntangled": cmd_ntangled,
     "oracle": lambda s: None}[a.cmd](a.seeds, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
