"""Deep audit of idea 4 (bell_qnn). Development seeds only (9101+); confirmatory seeds untouched.

A. Exact noise model on raw Bell bits, validated against density matrices, then the noise sweep redone.
   Depolarising p per qubit per copy == random Pauli X/Y/Z each w.p. p/4. Through the Bell circuit
   (CNOT q->n+q, H q) a Pauli before measurement becomes a bit flip:
       copy-1 X -> b,  copy-1 Z -> a,  copy-2 X -> b,  copy-2 Z -> a,  Y -> a and b.
   Readout: every measured bit flips w.p. eps. Shadows: depolarised qubit -> uniform bit (exact, I/2 is
   a product state), readout -> bit flip.
B. Baseline strength: shots per basis S in {1, 2, 5, 10, 20} at C = 400 (the campaign used S = 10), and
   physical clipping of both estimators to [2^-k, 1].
C. Copies needed by shadows to match Bell at C = 400, measured directly instead of extrapolated.

    PY=~/proyectos/active/q-taskfisher/.venv/bin/python
    $PY audit_deep.py --part A|B|C
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import qdlib as Q

PROF = (1, 2, 3, 4)
K = 4
N = 2 * K


def brickwork(n: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for l in range(depth):
        for q in range(l % 2, n - 1, 2):
            psi = Q.apply_2q(psi, Q.haar_unitary(4, rng), q, q + 1, n)
    return psi


def states(seed: int, n_est: int, n: int = N) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    est = np.array([brickwork(n, int(rng.choice(PROF)), rng) for _ in range(n_est)])
    A = list(range(n // 2))
    p = np.array([np.real(np.trace((r := Q.rdm(s, A, n)) @ r)) for s in est])
    return est, p


def bell_bits(psi: np.ndarray, n: int, shots: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pr = Q.bell_probs_two_copies(psi, n)
    idx = rng.choice(pr.size, size=shots, p=pr)
    bits = (idx[:, None] >> (2 * n - 1 - np.arange(2 * n))) & 1
    return bits[:, :n].copy(), bits[:, n:].copy()


def noisy_bell(a, b, p, eps, rng):
    a, b = a.copy(), b.copy()
    if p > 0:
        for _copy in range(2):
            u = rng.random(a.shape)
            # X w.p. p/4 -> flip b ; Z w.p. p/4 -> flip a ; Y w.p. p/4 -> flip both
            fx = u < p / 4
            fz = (u >= p / 4) & (u < p / 2)
            fy = (u >= p / 2) & (u < 3 * p / 4)
            b ^= (fx | fy).astype(a.dtype)
            a ^= (fz | fy).astype(a.dtype)
    if eps > 0:
        a ^= (rng.random(a.shape) < eps).astype(a.dtype)
        b ^= (rng.random(b.shape) < eps).astype(b.dtype)
    return 1 - 2 * (a & b)


def noisy_shadow(bases, bits, p, eps, rng):
    bits = bits.copy()
    if p > 0:
        m = rng.random(bits.shape) < p
        bits = np.where(m, rng.integers(0, 2, size=bits.shape), bits)
    if eps > 0:
        bits = bits ^ (rng.random(bits.shape) < eps)
    return bases, bits


def validate_against_dm() -> dict:
    """Pauli-bit model vs exact density matrix on n = 2 (from audit_noise_model.py)."""
    import audit_noise_model as M
    rng = np.random.default_rng(9150)
    n = 2
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    rho2 = np.outer(np.kron(psi, psi), np.kron(psi, psi).conj())
    out = {}
    for p, e in ((0.1, 0.0), (0.2, 0.0), (0.0, 0.05), (0.1, 0.05)):
        rho = rho2.copy()
        for q in range(2 * n):
            rho = M.depol_qubit(rho, q, 2 * n, p)
        exact = M.sign_expect(M.bell_probs_from_rho(rho, n), n, [0, 1], eps=e)
        a, b = bell_bits(psi, n, 400000, rng)
        mc = float(np.prod(noisy_bell(a, b, p, e, rng), axis=1).mean())
        out[f"p={p},eps={e}"] = {"exact": exact, "pauli_model": mc, "diff": mc - exact}
    return out


def part_a(seeds, n_est, copies):
    val = validate_against_dm()
    print("validation vs density matrix:", json.dumps(val, indent=1))
    grid = [("depol", v) for v in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30)] + \
           [("readout", v) for v in (0.02, 0.05, 0.10, 0.20)]
    ia = sum(N - j + 1 for j in range(1, K))
    rows = []
    for seed in seeds:
        est, p_ex = states(seed, n_est)
        rng = np.random.default_rng(seed + 1)
        clean = []
        for s in est:
            a, b = bell_bits(s, N, copies // 2, rng)
            clean.append((a, b, Q.sample_shadows(s, N, copies, rng)))
        for kind, v in grid:
            p, e = (v, 0.0) if kind == "depol" else (0.0, v)
            eb, es, pf = [], [], []
            for a, b, (ba, bi) in clean:
                sg = noisy_bell(a, b, p, e, rng)
                eb.append(float(np.prod(sg[:, :K], axis=1).mean()))
                pf.append(float(np.prod(sg, axis=1).mean()))
                ba2, bi2 = noisy_shadow(ba, bi, p, e, rng)
                es.append(Q.shadow_window_purities(ba2, bi2, K)[ia])
            eb, es = np.array(eb), np.array(es)
            lam = float(np.clip(((np.mean(pf) - 2.0 ** -N) / (1 - 2.0 ** -N)) ** (1 / N), 0.02, 1))
            lk = lam ** K
            cb = (eb - (1 - lk) / 2 ** K) / lk
            rmse = lambda x: float(np.sqrt(np.mean((x - p_ex) ** 2)))
            rows.append({"seed": seed, "kind": kind, "value": v, "lam": lam,
                         "rmse_bell": rmse(eb), "rmse_bell_corr": rmse(cb), "rmse_shadow": rmse(es),
                         "bias_bell": float(np.mean(eb - p_ex)), "bias_shadow": float(np.mean(es - p_ex)),
                         "sd_bell": float(np.std(eb - p_ex)), "sd_shadow": float(np.std(es - p_ex))})
    json.dump({"validation": val, "rows": rows}, open("results/bell_qnn/audit_noise_exact.json", "w"), indent=1)
    print("\n| channel | value | RMSE Bell | Bell corr | RMSE shadows | bias Bell | sd Bell | bias shadows |")
    print("|---|---|---|---|---|---|---|---|")
    for kind, v in grid:
        f = [r for r in rows if r["kind"] == kind and r["value"] == v]
        g = lambda k: float(np.mean([r[k] for r in f]))
        print(f"| {kind} | {v} | {g('rmse_bell'):.3f} | {g('rmse_bell_corr'):.3f} | {g('rmse_shadow'):.3f} | "
              f"{g('bias_bell'):+.3f} | {g('sd_bell'):.3f} | {g('bias_shadow'):+.3f} |")


def part_b(seeds, n_est, copies):
    ia = sum(N - j + 1 for j in range(1, K))
    lo = 2.0 ** -K
    rows = []
    for seed in seeds:
        est, p_ex = states(seed, n_est)
        rng = np.random.default_rng(seed + 2)
        eb = np.array([float(np.prod(Q.sample_bell(s, N, copies // 2, rng)[:, :K], axis=1).mean()) for s in est])
        row = {"seed": seed, "bell": float(np.sqrt(np.mean((eb - p_ex) ** 2))),
               "bell_clip": float(np.sqrt(np.mean((np.clip(eb, lo, 1) - p_ex) ** 2)))}
        for S in (1, 2, 5, 10, 20):
            Q.SHOTS_POR_BASE = S
            es = np.array([Q.shadow_window_purities(*Q.sample_shadows(s, N, copies, rng), K)[ia] for s in est])
            row[f"S{S}"] = float(np.sqrt(np.mean((es - p_ex) ** 2)))
            row[f"S{S}_clip"] = float(np.sqrt(np.mean((np.clip(es, lo, 1) - p_ex) ** 2)))
        Q.SHOTS_POR_BASE = 10
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open("results/bell_qnn/audit_baseline_strength.json", "w"), indent=1)
    g = lambda k: float(np.mean([r[k] for r in rows]))
    print("\n| reader | RMSE | RMSE clipped to [2^-k,1] |\n|---|---|---|")
    print(f"| Bell (C/2 shots) | {g('bell'):.3f} | {g('bell_clip'):.3f} |")
    for S in (1, 2, 5, 10, 20):
        print(f"| shadows S={S} (B={copies // S}) | {g(f'S{S}'):.3f} | {g(f'S{S}_clip'):.3f} |")


def part_c(seeds, n_est):
    ia = sum(N - j + 1 for j in range(1, K))
    rows = []
    for seed in seeds:
        est, p_ex = states(seed, n_est)
        rng = np.random.default_rng(seed + 3)
        eb = np.array([float(np.prod(Q.sample_bell(s, N, 200, rng)[:, :K], axis=1).mean()) for s in est])
        row = {"seed": seed, "bell_C400": float(np.sqrt(np.mean((eb - p_ex) ** 2)))}
        for S in (1, 10):
            Q.SHOTS_POR_BASE = S
            for C in (400, 1600, 6400, 12800):
                es = np.array([Q.shadow_window_purities(*Q.sample_shadows(s, N, C, rng), K)[ia] for s in est])
                row[f"S{S}_C{C}"] = float(np.sqrt(np.mean((es - p_ex) ** 2)))
        Q.SHOTS_POR_BASE = 10
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open("results/bell_qnn/audit_copies_to_match.json", "w"), indent=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "C"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[9101, 9102, 9103, 9104])
    ap.add_argument("--n_est", type=int, default=200)
    ap.add_argument("--copies", type=int, default=400)
    a = ap.parse_args()
    Q.set_backend("pennylane")
    {"A": lambda: part_a(a.seeds, a.n_est, a.copies), "B": lambda: part_b(a.seeds, a.n_est, a.copies),
     "C": lambda: part_c(a.seeds, a.n_est)}[a.part]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
