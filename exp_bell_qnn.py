"""Gate and campaign for idea 4 - Bell-QNN: two copies versus shadows for nonlinear labels.

Question: is the label nonlinear in the state (entanglement of the left half: 1 if
tr(rho_A^2) < train median, A = qubits 0..3, n = 8)? States come from random brickwork circuits of
depth 1..4. At an equal total number of copies C per state, does a reader with Bell measurements on
two copies beat classifiers on single-copy shadows?

Two endpoints, both pre-registered in SPEC.md:
- `--mode gate` (historical, already run, kept intact): classification error, N = 120/80,
  margin 0.05. It is a filter, not evidence.
- `--mode campaign` (amendment of 21 Oct): RMSE of the tr(rho_A^2) estimator at C = 400 copies, with
  a global-lambda bias correction, N = 400 states (240/160). Classification remains a secondary
  readout in DETAIL and does not decide the verdict.

The gate path does NOT change: the sampling loop consumes the RNG in the same order and the added
quantities (full-register purity, shadow RDM) are deterministic given what was sampled, so
`--mode gate --seed 42` reproduces the archived result bit for bit.

Campaign:
- ours: RMSE of the Bell purity (the correction is the identity without noise: lambda = 1 exactly).
- q_baseline: minimum over {U-statistic RMSE of grouped shadows S = 10, RMSE of the shadow RDM,
  RMSE of S = 1 shadows clipped to [2^-k, 1]}. The third was added by the 21 Oct audit (it is the
  strongest classical rival found) and is sampled with a separate RNG.
- classical_rival: RMSE of the shadow RDM (alternative classical estimator, biased upwards).
- PASS per seed: ours < q_baseline. The real verdict comes from the ten-seed audit (Holm m = 2).
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import qdlib as Q

IDEA_ID = "bell_qnn"

# ==========================================
# HYPERPARAMETERS FIXED IN ADVANCE
# ==========================================
N = 8
A = [0, 1, 2, 3]
N_TRAIN, N_TEST = 120, 80              # gate
N_TRAIN_CAMP, N_TEST_CAMP = 240, 160   # campaign (N = 400 states)
PROF = (1, 2, 3, 4)          # profundidades brickwork muestreadas uniformemente
COPIES = 400                 # copies per state for every method
K_WIN = 4                   # maximum window size for purities
K_PAULI = 2
MARGEN = 0.05
PARITY_N = 2

MODO = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "gate"


def brickwork(depth: int, rng: np.random.Generator) -> np.ndarray:
    psi = np.zeros(2 ** N, dtype=complex)
    psi[0] = 1.0
    for l in range(depth):
        for q in range(l % 2, N - 1, 2):
            psi = Q.apply_2q(psi, Q.haar_unitary(4, rng), q, q + 1, N)
    return psi


def experiment(seed: int) -> tuple[dict, str, str]:
    rng = np.random.default_rng(seed)
    n_train, n_test = (N_TRAIN_CAMP, N_TEST_CAMP) if MODO == "campaign" else (N_TRAIN, N_TEST)
    est = np.array([brickwork(int(rng.choice(PROF)), rng) for _ in range(n_train + n_test)])
    pur_a = np.array([np.real(np.trace(np.linalg.matrix_power(Q.rdm(s, A, N), 2))) for s in est])
    umbral = np.median(pur_a[:n_train])
    y = (pur_a < umbral).astype(int)
    tr, te = np.arange(n_train), np.arange(n_train, n_train + n_test)
    idx_a = N + (N - 1) + (N - 2) + 0  # window k = 4 starting at 0 (= A)

    f_bell, f_spur, f_pau, p_full, e_rdm = [], [], [], [], []
    for s in est:
        sg = Q.sample_bell(s, N, COPIES // 2, rng)
        f_bell.append(Q.bell_window_purities(sg, K_WIN))
        p_full.append(float(np.prod(sg, axis=1).mean()))  # determinista dado sg: no consume RNG
        b, o = Q.sample_shadows(s, N, COPIES, rng)
        f_spur.append(Q.shadow_window_purities(b, o, K_WIN))
        f_pau.append(Q.local_pauli_features(b, o, K_PAULI))
        rdm = Q.shadow_rdm(b, o, A)
        e_rdm.append(float(np.real(np.trace(rdm @ rdm))))
    f_bell, f_spur, f_pau = map(np.array, (f_bell, f_spur, f_pau))
    e_bell, e_spur = f_bell[:, idx_a], f_spur[:, idx_a]
    e_rdm = np.array(e_rdm)

    # Hardened baseline (21 Oct audit): shadows with S = 1 shot per basis, clipped to the physical
    # range [2^-k, 1]. Sampled with a SEPARATE generator so that not one Bell sample nor one grouped
    # shadow sample changes: the archived campaign numbers stay reproducible.
    e_s1 = None
    if MODO == "campaign":
        rng_s1 = np.random.default_rng(seed + 1_000_003)
        s_prev = Q.SHOTS_POR_BASE
        Q.SHOTS_POR_BASE = 1
        e_s1 = np.array([Q.shadow_window_purities(*Q.sample_shadows(s, N, COPIES, rng_s1), K_WIN)[idx_a]
                         for s in est])
        Q.SHOTS_POR_BASE = s_prev
        e_s1 = np.clip(e_s1, 2.0 ** -len(A), 1.0)

    # Bias correction (Bell only): anchored on the full-register purity, which is 1 because the states
    # states son puros. lam = ((E[p_full] - 2^-N)/(1 - 2^-N))^(1/N); invierte E = lam^k p + (1-lam^k)/2^k.
    lam = float(np.clip(((np.mean(p_full) - 2.0 ** -N) / (1.0 - 2.0 ** -N)) ** (1.0 / N), 0.02, 1.0))
    lamk = lam ** K_WIN
    c_bell = (e_bell - (1.0 - lamk) / 2 ** K_WIN) / lamk

    rmse_bell = float(np.sqrt(np.mean((c_bell - pur_a) ** 2)))
    rmse_sombras = float(np.sqrt(np.mean((e_spur - pur_a) ** 2)))
    rmse_rdm = float(np.sqrt(np.mean((e_rdm - pur_a) ** 2)))
    e_clas_bell = Q.logreg_error(c_bell[tr, None], y[tr], c_bell[te, None], y[te])
    e_clas_sombra = Q.logreg_error(f_spur[tr], y[tr], f_spur[te], y[te])
    rival_clas = Q.svm_rbf_error(f_pau[tr], y[tr], f_pau[te], y[te])

    Q.detail({"seed": seed, "modo": MODO, "lam": lam, "umbral_pureza": float(umbral),
              "rmse": {"bell_corregido": rmse_bell, "sombras": rmse_sombras, "rdm": rmse_rdm,
                       "sombras_S1_recortadas": (None if e_s1 is None
                                                 else float(np.sqrt(np.mean((e_s1 - pur_a) ** 2))))},
              "clasificacion": {"bell": e_clas_bell, "sombras": e_clas_sombra, "rival_svm": rival_clas},
              "copias": COPIES, "n_est": n_train + n_test})

    if MODO == "campaign":
        Q.save_checkpoint(IDEA_ID, seed, {"rmse_bell": torch.tensor(rmse_bell)},
                          torch.tensor(est[te][:PARITY_N]), y[te][:PARITY_N],
                          {"seed": seed, "copias": COPIES, "n_est": n_train + n_test})
        rmse_s1 = float(np.sqrt(np.mean((e_s1 - pur_a) ** 2)))
        q_base = min(rmse_sombras, rmse_rdm, rmse_s1)
        ok = rmse_bell < q_base
        razon = "" if ok else f"bell={rmse_bell:.4f} >= q_baseline={q_base:.4f}"
        return Q.metricas(rmse_bell, q_base, rmse_rdm), "PASS" if ok else "KILL", razon

    # --- gate path, untouched ---
    ours = Q.logreg_error(f_bell[tr], y[tr], f_bell[te], y[te])
    e_pau = Q.logreg_error(f_pau[tr], y[tr], f_pau[te], y[te])
    q_base = min(e_clas_sombra, e_pau)
    Q.save_checkpoint(IDEA_ID, seed, {"err_ours": torch.tensor(ours)}, torch.tensor(est[te][:PARITY_N]),
                      y[te][:PARITY_N], {"seed": seed, "copias": COPIES})
    ok = ours <= q_base - MARGEN
    razon = "" if ok else (f"bell={ours:.4f} > q_baseline-{MARGEN}={q_base - MARGEN:.4f} "
                           f"(shadow purity={e_clas_sombra:.4f}, pauli={e_pau:.4f})")
    return Q.metricas(ours, q_base, rival_clas), "PASS" if ok else "KILL", razon


def oracle(seed: int, backend: str) -> bool:
    Q.oracle_bell(seed, backend)
    # U-statistic shadow purities: unbiased (mean of 30 repetitions within 5 sigma).
    rng = np.random.default_rng(seed + 8)
    n = 4
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    psi /= np.linalg.norm(psi)
    exacta = Q.exact_window_purities(psi, n, 2)
    reps = np.array([Q.shadow_window_purities(*Q.sample_shadows(psi, n, 400, rng), 2) for _ in range(30)])
    z = np.abs(reps.mean(0) - exacta) / (reps.std(0, ddof=1) / np.sqrt(30) + 1e-12)
    assert z.max() < 5, f"biased shadow purities: z max={z.max():.2f}"
    return True


def parity(seed: int) -> tuple[float, dict]:
    ck = Q.load_checkpoint(IDEA_ID, seed)
    est = ck["X_test"].numpy()
    d = max(np.abs(Q.window_purities_from_bell_probs(Q.bell_probs_pennylane(s, N), N, K_WIN)
                   - Q.window_purities_from_bell_probs(Q.bell_probs_qiskit(s, N), N, K_WIN)).max()
            for s in est)
    clave = "rmse_bell" if "rmse_bell" in ck["params"] else "err_ours"
    return float(ck["params"][clave]), {"max_abs_diff": float(d), "tol": 1e-5, "n_points": int(len(est))}


if __name__ == "__main__":
    Q.CHECKPOINT_SUFIJO = "_campaign" if MODO == "campaign" else ""
    sys.exit(Q.run_cli(IDEA_ID, oracle, experiment, parity))
