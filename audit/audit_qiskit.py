"""Audit: campaign on the Qiskit backend + backend parity + gate/campaign overlap at seed 42.

Runs experiment() in-process so the gate's Qiskit checkpoint is not overwritten (run_cli would reset the
checkpoint suffix to `_qiskit`). Writes results/bell_qnn/audit_qiskit.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.argv = ["exp_bell_qnn.py", "--mode", "campaign"]
import exp_bell_qnn as E  # noqa: E402
import qdlib as Q  # noqa: E402

SEEDS = (0, 1, 7, 42, 123, 456, 789, 1234, 2024, 3407)


def main() -> int:
    out = {"qiskit": {}, "parity": {}, "overlap_seed42": None}
    Q.set_backend("qiskit")
    Q.CHECKPOINT_SUFIJO = "_campaign_qiskit"
    for s in SEEDS:
        m, status, _ = E.experiment(s)
        out["qiskit"][s] = {"ours": m["ours_test_loss"], "q_base": m["q_baseline_test_loss"], "status": status}
        print(s, out["qiskit"][s], flush=True)
    Q.set_backend("pennylane")
    for s in SEEDS:
        Q.CHECKPOINT_SUFIJO = "_campaign"
        ck = Q.load_checkpoint(E.IDEA_ID, s)
        est = ck["X_test"].numpy()
        d = max(np.abs(Q.window_purities_from_bell_probs(Q.bell_probs_pennylane(x, E.N), E.N, E.K_WIN)
                       - Q.window_purities_from_bell_probs(Q.bell_probs_qiskit(x, E.N), E.N, E.K_WIN)).max()
                for x in est)
        out["parity"][s] = float(d)
    # overlap: are the first 200 campaign states at seed 42 the gate's 200 states?
    rng_g, rng_c = np.random.default_rng(42), np.random.default_rng(42)
    g = [E.brickwork(int(rng_g.choice(E.PROF)), rng_g) for _ in range(200)]
    c = [E.brickwork(int(rng_c.choice(E.PROF)), rng_c) for _ in range(400)]
    out["overlap_seed42"] = int(sum(np.allclose(g[i], c[i]) for i in range(200)))
    json.dump(out, open("results/bell_qnn/audit_qiskit.json", "w"), indent=1)
    print(json.dumps({"parity_max": max(out["parity"].values()), "overlap_seed42": out["overlap_seed42"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
