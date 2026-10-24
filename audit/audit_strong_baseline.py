"""Audit: confirmatory seeds re-scored against the strongest single-copy baseline found in the audit
(shadows with one shot per basis, S = 1, clipped to [2^-k, 1]). Same 400 states per seed as the campaign
(identical RNG draws for state preparation); fresh measurement RNG. Hardening the rival is conservative.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import stats

sys.argv = ["exp_bell_qnn.py", "--mode", "campaign"]
import exp_bell_qnn as E  # noqa: E402
import qdlib as Q  # noqa: E402

SEEDS = (0, 1, 7, 42, 123, 456, 789, 1234, 2024, 3407)


def main() -> int:
    Q.set_backend("pennylane")
    Q.SHOTS_POR_BASE = 1
    k, lo = len(E.A), 2.0 ** -len(E.A)
    ia = E.N + (E.N - 1) + (E.N - 2)
    rows = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        est = [E.brickwork(int(rng.choice(E.PROF)), rng) for _ in range(E.N_TRAIN_CAMP + E.N_TEST_CAMP)]
        p = np.array([np.real(np.trace(np.linalg.matrix_power(Q.rdm(x, E.A, E.N), 2))) for x in est])
        m = np.random.default_rng(s + 777)
        eb = np.array([float(np.prod(Q.sample_bell(x, E.N, E.COPIES // 2, m)[:, :k], axis=1).mean()) for x in est])
        es = np.array([Q.shadow_window_purities(*Q.sample_shadows(x, E.N, E.COPIES, m), k)[ia] for x in est])
        rb = float(np.sqrt(np.mean((eb - p) ** 2)))
        rs = float(np.sqrt(np.mean((np.clip(es, lo, 1) - p) ** 2)))
        rows.append({"seed": s, "bell": rb, "shadow_S1_clip": rs, "delta": rb - rs, "ratio": rs / rb})
        print(rows[-1], flush=True)
    d = np.array([r["delta"] for r in rows])
    w = stats.wilcoxon(d, alternative="less", method="exact")
    res = {"rows": rows, "mean_delta": float(d.mean()), "sd": float(d.std(ddof=1)), "neg": int((d < 0).sum()),
           "p_wilcoxon": float(w.pvalue), "p_holm_m2": min(1.0, 2 * float(w.pvalue)),
           "mean_ratio": float(np.mean([r["ratio"] for r in rows]))}
    json.dump(res, open("results/bell_qnn/audit_strong_baseline.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
