"""Learning tasks where the label is NOT a closed-form functional estimated by the reader.

Pre-registered here, before any run (21-Oct, after the audit). Official seeds, n = 8, C = 400 copies per
state for every reader, 400 states per seed (240 train / 160 test), same learner for every reader.

  L1  Preparation depth (4 classes, brickwork depth 1..4 drawn uniformly). No single purity determines
      the depth; the classifier must combine all 26 contiguous-window purities. Learner: multinomial
      logistic regression on standardised features, C = 1. Metric: test error.
  L2  Von Neumann entropy S(rho_A), |A| = 4 (left half). Not a polynomial of the state, so the map from
      purities to the target must be learned. Learner: ridge regression (alpha = 1) on standardised
      window purities and their logs. Metric: test RMSE.

Readers (same 400 copies per state):
  bell        all window purities from one set of C/2 Bell shots (full 2n-qubit sampler)
  shadow_s1   window purities, U-statistics, one shot per random Pauli basis, clipped to [2^-|W|, 1]
  shadow_s10  the same with 10 shots per basis (the configuration common on hardware)
  huang       local Pauli expectations (weight <= 2) from the S = 1 snapshots (shadow-ML features)
  exact       exact window purities (ceiling; not a reader)

Primary comparison: bell vs the best single-copy reader per seed (min over shadow_s1, shadow_s10, huang),
exact one-sided Wilcoxon over the ten seeds. Exploratory family (not in the Holm family of the campaign).

    PY=~/proyectos/active/q-taskfisher/.venv/bin/python
    $PY exp_bell_learn.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats
from scipy.linalg import logm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import qdlib as Q

SEEDS = (0, 1, 7, 42, 123, 456, 789, 1234, 2024, 3407)
N, K, COPIES, N_EST, N_TRAIN = 8, 4, 400, 400, 240
PROF = (1, 2, 3, 4)


def brickwork(depth, rng):
    psi = np.zeros(2 ** N, dtype=complex)
    psi[0] = 1.0
    for l in range(depth):
        for q in range(l % 2, N - 1, 2):
            psi = Q.apply_2q(psi, Q.haar_unitary(4, rng), q, q + 1, N)
    return psi


def widths() -> np.ndarray:
    return np.array([k for k in range(1, K + 1) for _ in range(N - k + 1)])


def vn_entropy(rho: np.ndarray) -> float:
    ev = np.clip(np.linalg.eigvalsh(rho), 1e-15, None)
    return float(-np.sum(ev * np.log2(ev)))


def features(f: np.ndarray, w: np.ndarray) -> np.ndarray:
    f = np.clip(f, 2.0 ** -w, 1.0)
    return np.hstack([f, np.log2(f)])


def run(seed: int) -> dict:
    rng = np.random.default_rng(seed * 17 + 3)
    depth = rng.choice(PROF, size=N_EST)
    est = [brickwork(int(d), rng) for d in depth]
    A = list(range(K))
    s_vn = np.array([vn_entropy(Q.rdm(s, A, N)) for s in est])
    w = widths()
    X = {"exact": np.array([Q.exact_window_purities(s, N, K) for s in est])}
    X["bell"] = np.array([Q.bell_window_purities(Q.sample_bell(s, N, COPIES // 2, rng), K) for s in est])
    Q.SHOTS_POR_BASE = 1
    s1, hu = [], []
    for s in est:
        b, o = Q.sample_shadows(s, N, COPIES, rng)
        s1.append(Q.shadow_window_purities(b, o, K))
        hu.append(Q.local_pauli_features(b, o, 2))
    Q.SHOTS_POR_BASE = 10
    X["shadow_s1"], hub = np.array(s1), np.array(hu)
    X["shadow_s10"] = np.array([Q.shadow_window_purities(*Q.sample_shadows(s, N, COPIES, rng), K) for s in est])

    tr, te = np.arange(N_TRAIN), np.arange(N_TRAIN, N_EST)
    out = {"seed": seed, "L1_error": {}, "L2_rmse": {}}
    for name in ("exact", "bell", "shadow_s1", "shadow_s10", "huang"):
        F = hub if name == "huang" else features(X[name], w)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(F[tr], depth[tr])
        out["L1_error"][name] = float(np.mean(clf.predict(F[te]) != depth[te]))
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        reg.fit(F[tr], s_vn[tr])
        out["L2_rmse"][name] = float(np.sqrt(np.mean((reg.predict(F[te]) - s_vn[te]) ** 2)))
    out["L2_sd_target"] = float(np.std(s_vn[te]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args()
    Q.set_backend("pennylane")
    rows = []
    for s in a.seeds:
        rows.append(run(s))
        print(json.dumps(rows[-1]), flush=True)
    summary = {}
    for task in ("L1_error", "L2_rmse"):
        b = np.array([r[task]["bell"] for r in rows])
        best = np.array([min(r[task][k] for k in ("shadow_s1", "shadow_s10", "huang")) for r in rows])
        d = b - best
        p = float(stats.wilcoxon(d, alternative="less", method="exact").pvalue) if np.any(d != 0) else 1.0
        summary[task] = {k: [float(np.mean([r[task][k] for r in rows])), float(np.std([r[task][k] for r in rows], ddof=1))]
                         for k in rows[0][task]}
        summary[task]["delta_vs_best_single"] = [float(d.mean()), float(d.std(ddof=1))]
        summary[task]["negative"] = int((d < 0).sum())
        summary[task]["p_wilcoxon"] = p
    json.dump({"rows": rows, "summary": summary}, open("results/bell_qnn/paper_learn.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
