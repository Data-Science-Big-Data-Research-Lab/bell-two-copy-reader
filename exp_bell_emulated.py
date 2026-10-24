"""Emulated IBM environment for idea 4 (protocol step 'Emulated'): real transpilation to the heavy-hex
FakeFez (Heron r2) device and its calibrated noise model in Aer, at equal copies.

Per state (n = 2|A| qubits, A = left half, brickwork depth 1..4, Haar two-qubit gates):
  Bell     two copies on 2n physical qubits: prep (x) prep, CNOT(q -> n+q), H(q), measure. C/2 shots.
  shadows  one copy: prep, random Pauli basis on A only, measure A. One shot per basis (S = 1). Because
           shots are i.i.d., drawing C basis labels and running each distinct basis (3^|A| of them) with
           its multinomial count is exactly the S = 1 protocol.
Readers: Bell raw; Bell self-calibrated (anchor: full-register purity, which is 1 for the ideal pure
state); shadows S = 1 clipped. Truth: exact tr(rho_A^2) of the ideal state. Also reports two-qubit gate
counts and depth after transpilation, and an ideal (noise-free) Aer run of the same pipeline as a check.

    PY=~/proyectos/active/q-taskfisher/.venv/bin/python
    $PY exp_bell_emulated.py --k 2 --n_est 100 [--seeds ...] [--ideal]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeFez

import qdlib as Q
import exp_bell_paper as P

SEEDS = (0, 1, 7, 42, 123, 456, 789, 1234, 2024, 3407)
PROF = (1, 2, 3, 4)
COPIES = 400


def gates(n: int, rng: np.random.Generator) -> list[tuple[int, int, np.ndarray]]:
    depth = int(rng.choice(PROF))
    return [(q, q + 1, Q.haar_unitary(4, rng)) for l in range(depth) for q in range(l % 2, n - 1, 2)]


def state_from(n: int, gl) -> np.ndarray:
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for q1, q2, u in gl:
        psi = Q.apply_2q(psi, u, q1, q2, n)
    return psi


def add_prep(qc: QuantumCircuit, gl, offset: int) -> None:
    # qdlib applies u on (q1, q2) with q1 the most significant index; Qiskit's UnitaryGate takes
    # qubits little-endian, so the operand order is reversed.
    for q1, q2, u in gl:
        qc.append(UnitaryGate(u), [offset + q2, offset + q1])


def bell_circuit(n: int, gl) -> QuantumCircuit:
    qc = QuantumCircuit(2 * n, 2 * n)
    add_prep(qc, gl, 0)
    add_prep(qc, gl, n)
    for q in range(n):
        qc.cx(q, n + q)
        qc.h(q)
    qc.measure(range(2 * n), range(2 * n))
    return qc


def shadow_circuit(n: int, k: int, gl, basis: tuple[int, ...]) -> QuantumCircuit:
    qc = QuantumCircuit(n, k)
    add_prep(qc, gl, 0)
    for q, b in enumerate(basis):
        if b == 0:
            qc.h(q)
        elif b == 1:
            qc.sdg(q)
            qc.h(q)
    qc.measure(range(k), range(k))
    return qc


def bits_from_counts(counts: dict, nbits: int) -> np.ndarray:
    rows = []
    for key, c in counts.items():
        b = [int(ch) for ch in key.replace(" ", "")[::-1]]   # clbit i = position i
        rows += [b[:nbits]] * c
    return np.array(rows, dtype=np.int64)


_NOISE: dict = {}


def run(seed: int, k: int, n_est: int, ideal: bool) -> dict:
    """All circuits of a seed are transpiled in one call and simulated in one batched Aer run with
    memory=True; each circuit takes its first `shots` i.i.d. outcomes, which is exactly equivalent to
    separate runs. The noise model is built once per process."""
    n = 2 * k
    backend = FakeFez()
    if ideal:
        sim = AerSimulator()
    else:
        if "nm" not in _NOISE:
            _NOISE["nm"] = NoiseModel.from_backend(backend)
        sim = AerSimulator(noise_model=_NOISE["nm"])
    rng = np.random.default_rng(seed * 7 + k)
    all_bases = [tuple(int(x) for x in np.unravel_index(i, (3,) * k)) for i in range(3 ** k)]
    circuits, plan, pex = [], [], []
    for i in range(n_est):
        gl = gates(n, rng)
        psi = state_from(n, gl)
        r = Q.rdm(psi, list(range(k)), n)
        pex.append(float(np.real(np.trace(r @ r))))
        circuits.append(bell_circuit(n, gl))
        plan.append((i, "bell", None, COPIES // 2))
        counts = rng.multinomial(COPIES, [1 / 3 ** k] * 3 ** k)
        for bi, cnt in enumerate(counts):
            if cnt:
                circuits.append(shadow_circuit(n, k, gl, all_bases[bi]))
                plan.append((i, "shadow", bi, int(cnt)))
    tc = transpile(circuits, backend, optimization_level=3, seed_transpiler=seed)
    # Bell and shadow circuits are simulated in SEPARATE batches with an explicit method: a single
    # batch mixing them (Aer picks statevector for one kind, density_matrix for the other) returned
    # near-constant '00' outcomes for the shadow circuits (verified 21-Oct; each circuit alone is fine).
    mem_of: dict = {}
    for kind in ("bell", "shadow"):
        idx = [j for j, p in enumerate(plan) if p[1] == kind]
        shots = max(plan[j][3] for j in idx)
        sim.set_options(method="density_matrix" if (kind == "shadow" or n <= 4) else "statevector")
        rk = sim.run([tc[j] for j in idx], shots=shots, memory=True,
                     seed_simulator=int(rng.integers(1 << 30))).result()
        for t, j in enumerate(idx):
            mem_of[j] = rk.get_memory(t)
    eb, pf, syms = np.zeros(n_est), np.zeros(n_est), [[] for _ in range(n_est)]
    n2q_b, n2q_s, dep_b = [], [], []
    for j, (i, kind, bi, cnt) in enumerate(plan):
        mem = mem_of[j][:cnt]
        nb = 2 * n if kind == "bell" else k
        bits = np.array([[int(ch) for ch in m.replace(" ", "")[::-1][:nb]] for m in mem], dtype=np.int64)
        n2 = int(sum(v for g, v in tc[j].count_ops().items() if g in ("cz", "cx", "ecr")))
        if kind == "bell":
            sg = 1 - 2 * (bits[:, :n] & bits[:, n:])
            eb[i] = float(np.prod(sg[:, :k], axis=1).mean())
            pf[i] = float(np.prod(sg, axis=1).mean())
            n2q_b.append(n2)
            dep_b.append(tc[j].depth())
        else:
            syms[i].append(2 * np.array(all_bases[bi])[None, :] + bits)
            n2q_s.append(n2)
    es = np.array([float(np.clip(P.shadow_purity(np.concatenate(s), k, 1), 2.0 ** -k, 1)) for s in syms])
    pex = np.array(pex)
    lam = float(np.clip(((pf.mean() - 2.0 ** -n) / (1 - 2.0 ** -n)) ** (1 / n), 0.02, 1)) if pf.mean() > 2.0 ** -n else 0.02
    lk = lam ** k
    cb = np.clip((eb - (1 - lk) / 2 ** k) / lk, 2.0 ** -k, 1)
    r = lambda x: float(np.sqrt(np.mean((x - pex) ** 2)))
    return {"seed": seed, "k": k, "ideal": ideal, "n_est": n_est, "lam": lam,
            "rmse_bell": r(eb), "rmse_bell_cal": r(cb), "rmse_shadow": r(es),
            "bias_bell": float(np.mean(eb - pex)), "bias_shadow": float(np.mean(es - pex)),
            "n2q_bell_mean": float(np.mean(n2q_b)), "n2q_shadow_mean": float(np.mean(n2q_s)),
            "depth_bell_mean": float(np.mean(dep_b)), "sd_purity": float(pex.std())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n_est", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--ideal", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = f"results/bell_qnn/paper_emulated_k{a.k}{'_ideal' if a.ideal else ''}{a.tag}.json"
    rows = []
    for s in a.seeds:
        t0 = time.time()
        rows.append(run(s, a.k, a.n_est, a.ideal))
        rows[-1]["sec"] = round(time.time() - t0, 1)
        print(json.dumps(rows[-1]), flush=True)
        json.dump({"backend": "FakeFez", "copies": COPIES, "rows": rows}, open(out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
