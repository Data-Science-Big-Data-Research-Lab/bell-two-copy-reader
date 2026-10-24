"""Shared library for the quantum-data gate line.

Contenido:
- Spin Hamiltonians (TFIM, cluster-Ising, XXZ, ANNNI) and exact ground states.
- Random Pauli classical shadows simulated from the state vector, with optional noise.
- Shadow features: geometrically local Paulis, full-support Paulis, estimated RDM,
  U-statistic purities (via a symbol histogram) and the traces tr(rho^2) and tr(rho^3).
- Two-copy Bell sampling (window purity estimators).
- Differentiable torch state-vector simulator with Rot, RY, CZ, CRZ and CRX gates.
- SPEC JSON contract, guarded execution and checkpoint utilities.

Qubit convention: qubit 0 is the most significant in the index (PennyLane convention).
Qiskit is little-endian: call `to_qiskit_order` before building a `Statevector` or `DensityMatrix`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from typing import Callable, Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import torch

torch.set_default_dtype(torch.float64)

SEMILLAS_OFICIALES: tuple[int, ...] = (0, 1, 7, 42, 123, 456, 789, 1234, 2024, 3407)
RAIZ = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(RAIZ, "checkpoints")

# Active measurement backend: "pennylane" (ruta numpy/torch exacta) o "qiskit" (Statevector /
# DensityMatrix of qiskit.quantum_info with seeded sampling). Set by `run_cli` from --backend.
BACKEND = "pennylane"
# Grouped shadows: B = M / S random bases, S shots per basis (hardware-realistic).
SHOTS_POR_BASE = 10


def set_backend(nombre: str) -> None:
    global BACKEND
    assert nombre in ("pennylane", "qiskit"), nombre
    BACKEND = nombre

# ==========================================
# PAULIS Y HAMILTONIANOS
# ==========================================
I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": PX, "Y": PY, "Z": PZ}


def pauli_op(n: int, ops: dict[int, str]) -> sp.csr_matrix:
    """Producto tensorial disperso de Paulis; `ops` = {qubit: 'X'|'Y'|'Z'}."""
    out = sp.identity(1, dtype=complex, format="csr")
    for q in range(n):
        out = sp.kron(out, sp.csr_matrix(PAULI[ops.get(q, "I")]), format="csr")
    return out


_CACHE_PAULI: dict[tuple, np.ndarray] = {}


def _pauli_dense(n: int, ops: dict[int, str]) -> np.ndarray:
    """Cached dense Pauli (building H is the bottleneck without a cache)."""
    clave = (n, tuple(sorted(ops.items())))
    if clave not in _CACHE_PAULI:
        _CACHE_PAULI[clave] = pauli_op(n, ops).toarray()
    return _CACHE_PAULI[clave]


def hamiltonian(n: int, terms: Iterable[tuple[float, dict[int, str]]]) -> np.ndarray:
    """Dense sum of terms (coef, {qubit: pauli})."""
    h = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for coef, ops in terms:
        if coef != 0.0:
            h += coef * _pauli_dense(n, ops)
    return h


def ground_state(h: np.ndarray) -> np.ndarray:
    """Exact ground state by dense diagonalisation (n <= 10)."""
    vals, vecs = np.linalg.eigh(h)
    psi = vecs[:, 0]
    # Global phase fixed: first component of largest modulus made real and positive (determinism).
    k = int(np.argmax(np.abs(psi)))
    psi = psi * np.exp(-1j * np.angle(psi[k]))
    return psi / np.linalg.norm(psi)


def h_tfim(n: int, h: float) -> sp.csr_matrix:
    """Open TFIM: -sum Z_i Z_{i+1} - h sum X_i. Ordered phase for h < 1."""
    t = [(-1.0, {i: "Z", i + 1: "Z"}) for i in range(n - 1)]
    t += [(-h, {i: "X"}) for i in range(n)]
    return hamiltonian(n, t)


def h_cluster(n: int, h1: float, h2: float = 0.0) -> sp.csr_matrix:
    """Open cluster-Ising: -sum Z X Z - h1 sum X - h2 sum X X. SPT phase for h1 < 1 (h2 = 0)."""
    t = [(-1.0, {i - 1: "Z", i: "X", i + 1: "Z"}) for i in range(1, n - 1)]
    t += [(-h1, {i: "X"}) for i in range(n)]
    t += [(-h2, {i: "X", i + 1: "X"}) for i in range(n - 1)]
    return hamiltonian(n, t)


def h_xxz(n: int, delta: float) -> sp.csr_matrix:
    """Open XXZ: sum (XX + YY + delta ZZ). Neel order for delta > 1, XY for |delta| < 1."""
    t = []
    for i in range(n - 1):
        t += [(1.0, {i: "X", i + 1: "X"}), (1.0, {i: "Y", i + 1: "Y"}),
              (delta, {i: "Z", i + 1: "Z"})]
    return hamiltonian(n, t)


def h_annni(n: int, kappa: float, h: float) -> sp.csr_matrix:
    """ANNNI abierto: -sum Z_i Z_{i+1} + kappa sum Z_i Z_{i+2} - h sum X_i."""
    t = [(-1.0, {i: "Z", i + 1: "Z"}) for i in range(n - 1)]
    t += [(kappa, {i: "Z", i + 2: "Z"}) for i in range(n - 2)]
    t += [(-h, {i: "X"}) for i in range(n)]
    return hamiltonian(n, t)


def annni_ferro(kappa: float, h: float) -> int:
    """1 if (kappa, h) lies in the ferromagnetic phase (Peschel-Emery line, kappa < 0.5)."""
    if kappa >= 0.5:
        return 0
    hc = (1.0 - kappa) / kappa * (1.0 - np.sqrt((1.0 - 3.0 * kappa + 4.0 * kappa ** 2)
                                                / (1.0 - kappa))) if kappa > 1e-9 else 1.0
    return int(h < hc)


# Phase families: (parameter sampler, H builder, label "ordered phase" = 1)
FAMILIAS: dict[str, dict] = {
    "tfim": {"params": lambda rng: (rng.uniform(0.2, 1.8),),
             "H": lambda n, p: h_tfim(n, p[0]), "label": lambda p: int(p[0] < 1.0)},
    "cluster": {"params": lambda rng: (rng.uniform(0.2, 1.8),),
                "H": lambda n, p: h_cluster(n, p[0]), "label": lambda p: int(p[0] < 1.0)},
    "xxz": {"params": lambda rng: (rng.uniform(0.0, 2.0),),
            "H": lambda n, p: h_xxz(n, p[0]), "label": lambda p: int(p[0] > 1.0)},
    "xxz_neg": {"params": lambda rng: (rng.uniform(-2.0, 0.0),),
                "H": lambda n, p: h_xxz(n, p[0]), "label": lambda p: int(p[0] < -1.0)},
    "annni": {"params": lambda rng: (rng.uniform(0.0, 0.45), rng.uniform(0.0, 1.5)),
              "H": lambda n, p: h_annni(n, p[0], p[1]), "label": lambda p: annni_ferro(p[0], p[1])},
}


def sample_family(name: str, n: int, n_states: int, rng: np.random.Generator,
                  balanced: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (states (N, 2^n), labels (N,), parameters (N, k)).

    With `balanced=True`, samples are rejected until there are N/2 per class.
    """
    fam = FAMILIAS[name]
    states, labels, params = [], [], []
    cuenta = {0: 0, 1: 0}
    objetivo = n_states // 2
    intentos = 0
    while len(states) < n_states:
        intentos += 1
        assert intentos < 50 * n_states, f"sampling of {name} does not balance the classes"
        p = fam["params"](rng)
        y = fam["label"](p)
        if balanced and cuenta[y] >= objetivo + (n_states % 2) * y:
            continue
        cuenta[y] += 1
        states.append(ground_state(fam["H"](n, p)))
        labels.append(y)
        params.append(p)
    return np.array(states), np.array(labels), np.array(params, dtype=float)


def split(n_total: int, n_train: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n_total)
    return perm[:n_train], perm[n_train:]


# ==========================================
# OPERATIONS ON STATE VECTORS (NUMPY)
# ==========================================
def apply_1q(psi: np.ndarray, u: np.ndarray, q: int, n: int) -> np.ndarray:
    t = psi.reshape((2,) * n)
    t = np.tensordot(u, t, axes=([1], [q]))
    return np.moveaxis(t, 0, q).reshape(-1)


def apply_2q(psi: np.ndarray, u4: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
    t = psi.reshape((2,) * n)
    u = u4.reshape(2, 2, 2, 2)
    t = np.tensordot(u, t, axes=([2, 3], [q1, q2]))
    return np.moveaxis(t, [0, 1], [q1, q2]).reshape(-1)


def haar_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def rdm(psi: np.ndarray, keep: Sequence[int], n: int) -> np.ndarray:
    """Reduced density matrix over the qubits in `keep` (in that order)."""
    t = psi.reshape((2,) * n)
    rest = [q for q in range(n) if q not in keep]
    t = np.transpose(t, list(keep) + rest).reshape(2 ** len(keep), -1)
    return t @ t.conj().T


def expval_pauli(psi: np.ndarray, n: int, ops: dict[int, str]) -> float:
    """<psi| P |psi>; with BACKEND = "qiskit" it is computed by Statevector.expectation_value."""
    if BACKEND == "qiskit":
        from qiskit.quantum_info import SparsePauliOp, Statevector
        lab = ["I"] * n
        for q, p in ops.items():
            lab[n - 1 - q] = p
        return float(np.real(Statevector(to_qiskit_order(psi, n)).expectation_value(
            SparsePauliOp("".join(lab)))))
    return float(np.real(np.vdot(psi, pauli_op(n, ops) @ psi)))


def to_qiskit_order(vec_or_rho: np.ndarray, n: int) -> np.ndarray:
    """Reorder qubits (PennyLane convention -> Qiskit little-endian)."""
    if vec_or_rho.ndim == 1:
        return vec_or_rho.reshape((2,) * n).transpose(list(range(n))[::-1]).reshape(-1)
    perm = list(range(n))[::-1]
    t = vec_or_rho.reshape((2,) * (2 * n))
    t = t.transpose(perm + [p + n for p in perm])
    return t.reshape(2 ** n, 2 ** n)


# ==========================================
# RANDOM PAULI CLASSICAL SHADOWS
# ==========================================
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
SDG = np.diag([1, -1j])
# Rotation applied before measuring in Z, for bases X (0), Y (1), Z (2).
BASIS_ROT = np.array([HADAMARD, HADAMARD @ SDG, I2])


def _grupos(m: int) -> tuple[int, int]:
    """(number of bases B, shots per basis S) with M = B * S."""
    s = min(SHOTS_POR_BASE, m)
    assert m % s == 0, f"M={m} is not a multiple of SHOTS_POR_BASE={s}"
    return m // s, s


def _bits_de_indices(idx: np.ndarray, n: int) -> np.ndarray:
    return (idx[:, None] >> (n - 1 - np.arange(n))) & 1


def _muestrear_bases_numpy(psi: np.ndarray, n: int, bases_b: np.ndarray, s: int,
                           rng: np.random.Generator) -> np.ndarray:
    b = bases_b.shape[0]
    t = np.broadcast_to(psi.reshape((1,) + (2,) * n), (b,) + (2,) * n).copy()
    for q in range(n):
        u = BASIS_ROT[bases_b[:, q]]  # (B, 2, 2)
        t = np.moveaxis(t, q + 1, -1)
        forma = t.shape
        t = np.matmul(t.reshape(b, -1, 2), np.transpose(u, (0, 2, 1))).reshape(forma)
        t = np.moveaxis(t, -1, q + 1)
    probs = np.abs(t.reshape(b, -1)) ** 2
    probs /= probs.sum(axis=1, keepdims=True)
    acum = np.repeat(probs.cumsum(axis=1), s, axis=0)
    idx = (acum > rng.random((b * s, 1))).argmax(axis=1)
    return _bits_de_indices(idx, n)


_OPS_QISKIT: dict = {}


def _ops_qiskit() -> dict:
    """Cached Qiskit operators (evolving with Operator is ~10x faster than with circuits)."""
    if not _OPS_QISKIT:
        from qiskit.circuit.library import CXGate, HGate
        from qiskit.quantum_info import Operator
        _OPS_QISKIT.update({0: Operator(HADAMARD), 1: Operator(HADAMARD @ SDG),
                            "h": Operator(HGate()), "cx": Operator(CXGate())})
    return _OPS_QISKIT


def _muestrear_bases_qiskit(psi: np.ndarray, n: int, bases_b: np.ndarray, s: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Qiskit evolves the state in each basis and returns the probabilities; S shots are drawn
    with the seeded generator (sampling with `sample_memory` is 5x slower and equivalent)."""
    from qiskit.quantum_info import Statevector
    ops = _ops_qiskit()
    sv0 = Statevector(to_qiskit_order(psi, n))
    rows = []
    for base in bases_b:
        sv = sv0
        for q, p in enumerate(base):
            if p < 2:
                sv = sv.evolve(ops[int(p)], qargs=[q])
        pr = sv.probabilities()
        idx = rng.choice(pr.size, size=s, p=pr / pr.sum())
        # Qiskit order: bit q of the index is qubit q (= PennyLane wire q after reordering).
        rows.append((idx[:, None] >> np.arange(n)) & 1)
    return np.concatenate(rows)


def sample_shadows(psi: np.ndarray, n: int, m: int, rng: np.random.Generator,
                   depol: float = 0.0, readout_flip: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """M grouped random Pauli snapshots: B = M/S bases, S shots per basis.

    Returns (bases (M,n) in {0,1,2}, bits (M,n) in {0,1}); consecutive rows in groups of S
    share a basis. The active backend (`BACKEND`) decides who simulates and samples.
    `depol`: probability that the snapshot is drawn from the maximally mixed state
    (global depolarising). `readout_flip`: probability of inverting each bit.
    """
    b, s = _grupos(m)
    bases_b = rng.integers(0, 3, size=(b, n))
    muestrear = _muestrear_bases_qiskit if BACKEND == "qiskit" else _muestrear_bases_numpy
    bits = muestrear(psi, n, bases_b, s, rng)
    bases = np.repeat(bases_b, s, axis=0)
    if depol > 0.0:
        mix = rng.random(m) < depol
        bits[mix] = rng.integers(0, 2, size=(int(mix.sum()), n))
    if readout_flip > 0.0:
        bits = bits ^ (rng.random((m, n)) < readout_flip)
    return bases, bits.astype(np.int64)


def _coef_por_qubit(bases: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """c[m, q, p] = 3 * [basis == p] * (+-1): estimator of <P_q> for each snapshot."""
    signos = 1 - 2 * bits
    return 3.0 * (bases[:, :, None] == np.arange(3)[None, None, :]) * signos[:, :, None]


def local_pauli_features(bases: np.ndarray, bits: np.ndarray, k_max: int) -> np.ndarray:
    """Estimates of <P> for every non-trivial Pauli string on contiguous windows of
    size exactly k (k = 1..k_max), full support inside the window (Huang style)."""
    m, n = bases.shape
    c = _coef_por_qubit(bases, bits)
    feats = []
    for k in range(1, k_max + 1):
        for s in range(n - k + 1):
            f = c[:, s, :]
            for j in range(1, k):
                f = (f[:, :, None] * c[:, s + j, None, :]).reshape(m, -1)
            feats.append(f.mean(axis=0))
    return np.concatenate(feats)


def support_pauli_features(bases: np.ndarray, bits: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """Estimates of the 3^|S| full-support Pauli strings on `qubits`."""
    m = bases.shape[0]
    c = _coef_por_qubit(bases, bits)
    f = c[:, qubits[0], :]
    for q in qubits[1:]:
        f = (f[:, :, None] * c[:, q, None, :]).reshape(m, -1)
    return f.mean(axis=0)


def _snapshot_mats() -> np.ndarray:
    """mats[b, o] = 3 U_b^dagger |o><o| U_b - I (inverted single-qubit snapshot)."""
    mats = np.zeros((3, 2, 2, 2), dtype=complex)
    for b in range(3):
        for o in range(2):
            ket = np.zeros(2, dtype=complex)
            ket[o] = 1.0
            v = BASIS_ROT[b].conj().T @ ket
            mats[b, o] = 3.0 * np.outer(v, v.conj()) - I2
    return mats


SNAP = _snapshot_mats()


def shadow_snapshots(bases: np.ndarray, bits: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """Inverted snapshots on `qubits`: (M, 2^w, 2^w)."""
    r = SNAP[bases[:, qubits[0]], bits[:, qubits[0]]]
    for q in qubits[1:]:
        s = SNAP[bases[:, q], bits[:, q]]
        mm, a, _ = r.shape
        r = np.einsum("mab,mcd->macbd", r, s).reshape(mm, 2 * a, 2 * a)
    return r


def shadow_rdm(bases: np.ndarray, bits: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
    """Unbiased estimator of the RDM on `qubits` (not necessarily positive semidefinite)."""
    return shadow_snapshots(bases, bits, qubits).mean(axis=0)


def shadow_traces(bases: np.ndarray, bits: np.ndarray, qubits: Sequence[int]) -> tuple[float, float]:
    """Unbiased U-statistics of tr(rho_A^2) and Re tr(rho_A^3) with grouped shadows.

    Only pairs/triples of snapshots from different groups (bases) are counted, since they are
    independent. With G_g = sum of group g: sum_{g!=h} tr(G_g G_h) = tr(S^2) - tr(sum G_g^2), and
    sum_{g,h,l distintos} tr(G_g G_h G_l) = tr(S^3) - 3 tr(S T2) + 2 tr(T3).
    """
    sn = shadow_snapshots(bases, bits, qubits)
    m = sn.shape[0]
    nb, s = _grupos(m)
    g = sn.reshape(nb, s, *sn.shape[1:]).sum(axis=1)
    tot = g.sum(axis=0)
    t2 = np.einsum("gab,gbc->ac", g, g)
    t3 = np.einsum("gab,gbc,gcd->ad", g, g, g)
    p2 = (np.trace(tot @ tot) - np.trace(t2)) / (nb * (nb - 1) * s ** 2)
    p3 = (np.trace(tot @ tot @ tot) - 3 * np.trace(t2 @ tot) + 2 * np.trace(t3)) \
        / (nb * (nb - 1) * (nb - 2) * s ** 3)
    return float(np.real(p2)), float(np.real(p3))


# tr(sigma_i sigma_j) per qubit by symbol (basis*2 + bit): 5 same basis and bit, -4 same basis
# distinto bit, 1/2 base distinta.
_T6 = np.full((6, 6), 0.5)
for _b in range(3):
    _T6[2 * _b, 2 * _b] = _T6[2 * _b + 1, 2 * _b + 1] = 5.0
    _T6[2 * _b, 2 * _b + 1] = _T6[2 * _b + 1, 2 * _b] = -4.0


def _suma_pares_hist(codigo: np.ndarray, k: int) -> float:
    """sum_{i,j} prod_q T(s_iq, s_jq) (diagonal included) via a symbol histogram."""
    hist = np.bincount(codigo, minlength=6 ** k).astype(float).reshape((6,) * k)
    t = hist
    for ax in range(k):
        t = np.tensordot(_T6, t, axes=([1], [ax]))
        t = np.moveaxis(t, 0, ax)
    return float(np.sum(t * hist))


def shadow_window_purities(bases: np.ndarray, bits: np.ndarray, k_max: int) -> np.ndarray:
    """Unbiased tr(rho_W^2) for every contiguous window of size 1..k_max.

    With grouped shadows only pairs from different groups count:
    sum_{g(i) != g(j)} = total sum - sum within each group (both via histogram).
    """
    m, n = bases.shape
    nb, s = _grupos(m)
    sym = bases * 2 + bits
    sym_g = sym.reshape(nb, s, n)
    out = []
    for k in range(1, k_max + 1):
        for st in range(n - k + 1):
            codigo = np.zeros(m, dtype=np.int64)
            for j in range(k):
                codigo = codigo * 6 + sym[:, st + j]
            total = _suma_pares_hist(codigo, k)
            # pairs within each group (diagonal included): product of T per qubit, (B, S, S)
            prod = np.ones((nb, s, s))
            for j in range(k):
                col = sym_g[:, :, st + j]
                prod *= _T6[col[:, :, None], col[:, None, :]]
            out.append((total - float(prod.sum())) / (nb * (nb - 1) * s ** 2))
    return np.array(out)


# ==========================================
# BELL SAMPLING ON TWO COPIES
# ==========================================
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def bell_probs_two_copies(psi: np.ndarray, n: int) -> np.ndarray:
    """Exact distribution of the pairwise Bell measurement (q, n+q) on psi (x) psi.

    Circuit: CNOT(q -> n+q), H(q), measure. Returns probabilities over 2n bits
    (bits 0..n-1 = copia A, n..2n-1 = copia B).
    """
    big = np.kron(psi, psi)
    for q in range(n):
        big = apply_2q(big, CNOT, q, n + q, 2 * n)
        big = apply_1q(big, HADAMARD, q, 2 * n)
    p = np.abs(big) ** 2
    return p / p.sum()


def sample_bell(psi: np.ndarray, n: int, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Bell-measurement samples: returns swap signs (shots, n) in {+1, -1}.

    The SWAP eigenvalue on pair q is (-1)^(a_q b_q) (singlet = a=b=1).
    Con BACKEND = "qiskit" simula y muestrea Qiskit (Statevector de 2n qubits).
    """
    if BACKEND == "qiskit":
        from qiskit.quantum_info import Statevector
        ops = _ops_qiskit()
        sv = Statevector(to_qiskit_order(np.kron(psi, psi), 2 * n))
        for q in range(n):
            sv = sv.evolve(ops["cx"], qargs=[q, n + q])
            sv = sv.evolve(ops["h"], qargs=[q])
        pr = sv.probabilities()
        idx = rng.choice(pr.size, size=shots, p=pr / pr.sum())
        bits = (idx[:, None] >> np.arange(2 * n)) & 1
    else:
        p = bell_probs_two_copies(psi, n)
        idx = rng.choice(p.size, size=shots, p=p)
        bits = (idx[:, None] >> (2 * n - 1 - np.arange(2 * n))) & 1
    a, b = bits[:, :n], bits[:, n:]
    return 1 - 2 * (a & b)


def bell_window_purities(swap_signs: np.ndarray, k_max: int) -> np.ndarray:
    """tr(rho_W^2) = E[prod_{q in W} swap_q] for contiguous windows 1..k_max."""
    _, n = swap_signs.shape
    out = []
    for k in range(1, k_max + 1):
        for s in range(n - k + 1):
            out.append(float(np.prod(swap_signs[:, s:s + k], axis=1).mean()))
    return np.array(out)


def exact_window_purities(psi: np.ndarray, n: int, k_max: int) -> np.ndarray:
    out = []
    for k in range(1, k_max + 1):
        for s in range(n - k + 1):
            r = rdm(psi, list(range(s, s + k)), n)
            out.append(float(np.real(np.trace(r @ r))))
    return np.array(out)


# ==========================================
# STATE-VECTOR SIMULATOR IN TORCH (DIFFERENTIABLE)
# ==========================================
def t_rz(t: torch.Tensor) -> torch.Tensor:
    e = torch.exp(-0.5j * t.to(torch.complex128))
    return torch.diag(torch.stack([e, e.conj()]))


def t_ry(t: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex128)


def t_rx(t: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(t / 2).to(torch.complex128), torch.sin(t / 2).to(torch.complex128)
    return torch.stack([torch.stack([c, -1j * s]), torch.stack([-1j * s, c])])


def t_rot(phi: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """qml.Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)."""
    return t_rz(omega) @ t_ry(theta) @ t_rz(phi)


def t_controlled(u: torch.Tensor) -> torch.Tensor:
    """Puerta controlada 4x4 (control = primer qubit)."""
    out = torch.zeros((4, 4), dtype=torch.complex128)
    out[0, 0] = 1.0
    out[1, 1] = 1.0
    out[2:, 2:] = u
    return out


T_CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex128))


def t_apply_1q(state: torch.Tensor, u: torch.Tensor, q: int, n: int) -> torch.Tensor:
    """state (B, 2^n) complejo."""
    b = state.shape[0]
    t = state.reshape((b,) + (2,) * n)
    t = torch.tensordot(t, u, dims=([q + 1], [1]))  # new axis at the end
    t = torch.movedim(t, -1, q + 1)
    return t.reshape(b, -1)


def t_apply_2q(state: torch.Tensor, u4: torch.Tensor, q1: int, q2: int, n: int) -> torch.Tensor:
    b = state.shape[0]
    t = state.reshape((b,) + (2,) * n)
    u = u4.reshape(2, 2, 2, 2)
    t = torch.tensordot(t, u, dims=([q1 + 1, q2 + 1], [2, 3]))
    t = torch.movedim(t, (-2, -1), (q1 + 1, q2 + 1))
    return t.reshape(b, -1)


def t_unitary(circuit: Callable[[torch.Tensor], torch.Tensor], n: int) -> torch.Tensor:
    """Unitary matrix of the circuit, by applying it to the computational basis."""
    eye = torch.eye(2 ** n, dtype=torch.complex128)
    return circuit(eye).T  # rows = columns of U


def t_expval_z(state: torch.Tensor, qubits: Sequence[int], n: int) -> torch.Tensor:
    """<Z_{q1} ... Z_{qk}> for each state in the batch."""
    idx = torch.arange(2 ** n)
    signo = torch.ones(2 ** n)
    for q in qubits:
        signo = signo * (1 - 2 * ((idx >> (n - 1 - q)) & 1))
    return (state.abs() ** 2 * signo).sum(dim=1)


# ==========================================
# WINDOW QNN (ACTS ON THE RDM OF w QUBITS)
# ==========================================
class WindowQNN:
    """w-qubit QNN: L layers of per-qubit Rot plus a CZ chain; Z readout of the central qubit.

    Salida f(theta, rho) = tr(rho U^dagger Z_c U) y logit = s * f + b (s, b entrenables).
    The torch implementation is checked against PennyLane (default.mixed) and Qiskit in the oracles.
    """

    def __init__(self, w: int, layers: int):
        self.w, self.layers = w, layers
        self.centro = w // 2
        self.n_params = layers * w * 3

    def init(self, gen: torch.Generator) -> dict[str, torch.Tensor]:
        return {"theta": (0.1 * torch.randn(self.layers, self.w, 3, generator=gen)).requires_grad_(),
                "s": torch.tensor(3.0).requires_grad_(), "b": torch.tensor(0.0).requires_grad_()}

    def circuit(self, theta: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        def run(state: torch.Tensor) -> torch.Tensor:
            for l in range(self.layers):
                for q in range(self.w):
                    state = t_apply_1q(state, t_rot(theta[l, q, 0], theta[l, q, 1], theta[l, q, 2]),
                                       q, self.w)
                for q in range(self.w - 1):
                    state = t_apply_2q(state, T_CZ, q, q + 1, self.w)
            return state
        return run

    def observable(self, theta: torch.Tensor) -> torch.Tensor:
        u = t_unitary(self.circuit(theta), self.w)
        idx = torch.arange(2 ** self.w)
        z = (1 - 2 * ((idx >> (self.w - 1 - self.centro)) & 1)).to(torch.complex128)
        return u.conj().T @ torch.diag(z) @ u

    def f(self, params: dict, rhos: torch.Tensor) -> torch.Tensor:
        o = self.observable(params["theta"])
        return torch.einsum("bij,ji->b", rhos, o).real

    def logits(self, params: dict, rhos: torch.Tensor) -> torch.Tensor:
        return params["s"] * self.f(params, rhos) + params["b"]


def bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y.to(logits.dtype))


def error_rate(logits: torch.Tensor | np.ndarray, y: np.ndarray) -> float:
    pred = (np.asarray(logits.detach() if torch.is_tensor(logits) else logits) > 0).astype(int)
    return float(np.mean(pred != np.asarray(y)))


def train_params(params: dict, loss_fn: Callable[[dict], torch.Tensor], steps: int,
                 lr: float) -> dict:
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(params)
        loss.backward()
        opt.step()
    return params


def clone_params(params: dict) -> dict:
    return {k: v.detach().clone().requires_grad_() for k, v in params.items()}


# ==========================================
# CLASSICAL CLASSIFIERS (FIXED IN ADVANCE)
# ==========================================
def logreg_error(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray,
                 c: float = 1.0) -> float:
    """Standardised logistic regression, C fixed in advance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(ytr)) < 2:
        return float(np.mean(yte != ytr[0]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
    clf.fit(xtr, ytr)
    return float(np.mean(clf.predict(xte) != yte))


def svm_rbf_error(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray) -> float:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    clf = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale"))
    clf.fit(xtr, ytr)
    return float(np.mean(clf.predict(xte) != yte))


def binomial_expval(f: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Estimate with `shots` shots of a +-1 observable whose exact mean is f."""
    p = np.clip((1.0 + np.asarray(f)) / 2.0, 0.0, 1.0)
    return 2.0 * rng.binomial(shots, p) / shots - 1.0


# ==========================================
# CONTRATO JSON, CHECKPOINTS Y CLI
# ==========================================
def detail(obj: dict) -> None:
    sys.stderr.write("DETAIL " + json.dumps(obj, default=float) + "\n")
    sys.stderr.flush()


def contract(idea_id: str, seed: int, mode: str, backend: str, oracle_ok: bool, t0: float,
             metrics: dict, status: str, kill_reason: str) -> dict:
    return {"idea_id": idea_id, "backend": backend, "mode": mode, "seed": int(seed),
            "oracle_passed": bool(oracle_ok), "wall_time_sec": round(float(time.time() - t0), 2),
            "metrics": metrics, "status": status, "kill_reason": kill_reason}


def metricas(ours: float, q_base: float, rival: float) -> dict:
    return {"ours_test_loss": round(float(ours), 9), "q_baseline_test_loss": round(float(q_base), 9),
            "classical_rival_test_loss": round(float(rival), 9)}


METRICAS_VACIAS = {"ours_test_loss": 0.0, "q_baseline_test_loss": 0.0,
                   "classical_rival_test_loss": 0.0}


CHECKPOINT_SUFIJO = ""


def checkpoint_path(idea_id: str, seed: int) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{idea_id}_seed{int(seed)}{CHECKPOINT_SUFIJO}.pt")


def save_checkpoint(idea_id: str, seed: int, params: dict, x_test, y_test, config: dict) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({"params": params, "X_test": x_test, "y_test": y_test, "config": dict(config)},
               checkpoint_path(idea_id, seed))


def load_checkpoint(idea_id: str, seed: int) -> dict:
    return torch.load(checkpoint_path(idea_id, seed), weights_only=False)


def run_cli(idea_id: str, oracle: Callable[[int, str], bool],
            experiment: Callable[[int], tuple[dict, str, str]],
            parity: Callable[[int], tuple[float, dict]], argv=None) -> int:
    """Shared CLI: --mode {oracle,gate,campaign,parity} --backend --seed.

    `experiment(seed)` returns (metrics, status, kill_reason) and saves the checkpoint.
    `parity(seed)` devuelve (ours_test_loss evaluada en Qiskit, dict parity).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "gate", "campaign", "parity"], required=True)
    ap.add_argument("--backend", choices=["pennylane", "qiskit"], default="pennylane")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    global CHECKPOINT_SUFIJO
    set_backend(a.backend)
    t0 = time.time()
    try:
        if a.mode == "parity":
            set_backend("pennylane")  # parity explicitly compares both backends
            loss, par = parity(a.seed)
            ok = par["max_abs_diff"] < par["tol"]
            out = contract(idea_id, a.seed, a.mode, a.backend, True, t0,
                           {**METRICAS_VACIAS, "ours_test_loss": round(float(loss), 9)},
                           "PASS" if ok else "KILL",
                           "" if ok else f"max_abs_diff={par['max_abs_diff']:.3e} >= tol={par['tol']:.1e}")
            out["parity"] = par
        else:
            ok = oracle(a.seed, a.backend)
            if a.mode == "oracle":
                out = contract(idea_id, a.seed, a.mode, a.backend, ok, t0, {}, "PASS", "")
            else:
                # With --backend qiskit every measurement and the test evaluation use Qiskit; the
                # checkpoint carries the backend suffix so it does not overwrite the default one.
                if a.backend == "qiskit":
                    CHECKPOINT_SUFIJO = "_qiskit"
                m, status, reason = experiment(a.seed)
                out = contract(idea_id, a.seed, a.mode, a.backend, ok, t0, m, status, reason)
    except AssertionError as exc:
        out = contract(idea_id, a.seed, a.mode, a.backend, False, t0,
                       {} if a.mode == "oracle" else METRICAS_VACIAS, "KILL", f"oraculo: {exc}")
    print(json.dumps(out))
    return 0


# ==========================================
# PENNYLANE / QISKIT MIRRORS OF THE WINDOW QNN (ORACLE AND PARITY)
# ==========================================
def window_qnn_pennylane(qnn: WindowQNN, theta: np.ndarray, rho: np.ndarray) -> float:
    import pennylane as qml
    dev = qml.device("default.mixed", wires=qnn.w)

    @qml.qnode(dev)
    def circ():
        qml.QubitDensityMatrix(rho, wires=range(qnn.w))
        for l in range(qnn.layers):
            for q in range(qnn.w):
                qml.Rot(*theta[l, q], wires=q)
            for q in range(qnn.w - 1):
                qml.CZ(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(qnn.centro))
    return float(circ())


def window_qnn_qiskit(qnn: WindowQNN, theta: np.ndarray, rho: np.ndarray) -> float:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, SparsePauliOp
    w = qnn.w
    qc = QuantumCircuit(w)
    for l in range(qnn.layers):
        for q in range(w):
            phi, th, om = (float(v) for v in theta[l, q])
            qc.rz(phi, q)
            qc.ry(th, q)
            qc.rz(om, q)
        for q in range(w - 1):
            qc.cz(q, q + 1)
    # Qiskit little-endian: PennyLane qubit c is index c in the circuit, but the
    # matriz densidad debe reordenarse.
    dm = DensityMatrix(to_qiskit_order(rho, w)).evolve(qc)
    lab = ["I"] * w
    lab[w - 1 - qnn.centro] = "Z"
    return float(np.real(dm.expectation_value(SparsePauliOp("".join(lab)))))


def random_density(d: int, rng: np.random.Generator, rank: int = 3) -> np.ndarray:
    a = rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank))
    r = a @ a.conj().T
    return r / np.trace(r)


def oracle_window_qnn(qnn: WindowQNN, seed: int, backend: str, n_probe: int = 3) -> bool:
    """Shared oracle: the torch QNN matches the requested backend to 1e-10."""
    rng = np.random.default_rng(seed + 10_000)
    for _ in range(n_probe):
        theta = rng.uniform(-np.pi, np.pi, size=(qnn.layers, qnn.w, 3))
        rho = random_density(2 ** qnn.w, rng)
        f_t = float(qnn.f({"theta": torch.tensor(theta)}, torch.tensor(rho[None]))[0])
        f_b = (window_qnn_pennylane if backend == "pennylane" else window_qnn_qiskit)(qnn, theta, rho)
        assert abs(f_t - f_b) < 1e-10, f"[{backend}] QNN torch vs backend: |d|={abs(f_t - f_b):.2e}"
    return True


def window_qnn_test_logits(qnn: WindowQNN, params: dict, rhos) -> np.ndarray:
    """Evaluation logits: torch (backend pennylane) or a Qiskit circuit over each RDM."""
    rhos_np = rhos.detach().numpy() if torch.is_tensor(rhos) else np.asarray(rhos)
    s, b = float(params["s"].detach()), float(params["b"].detach())
    if BACKEND == "qiskit":
        th = params["theta"].detach().numpy()
        return s * np.array([window_qnn_qiskit(qnn, th, r) for r in rhos_np]) + b
    with torch.no_grad():
        return qnn.logits(params, torch.as_tensor(rhos_np)).numpy()


def parity_window_qnn(qnn: WindowQNN, params: dict, rhos: np.ndarray, y: np.ndarray,
                      n_points: int, tol: float = 1e-5) -> tuple[float, dict]:
    """Evaluate the saved QNN in PennyLane and Qiskit on the first `n_points` test RDMs."""
    theta = params["theta"].detach().numpy()
    s, b = float(params["s"]), float(params["b"])
    k = min(n_points, len(rhos))
    f_pl = np.array([window_qnn_pennylane(qnn, theta, r) for r in rhos[:k]])
    f_qk = np.array([window_qnn_qiskit(qnn, theta, r) for r in rhos[:k]])
    err = error_rate(s * f_qk + b, y[:k])
    return err, {"max_abs_diff": float(np.max(np.abs(f_pl - f_qk))), "tol": tol, "n_points": int(k)}


def windows_rdms(states: np.ndarray, n: int, window: Sequence[int]) -> np.ndarray:
    return np.array([rdm(s, list(window), n) for s in states])


def shadow_rdms(states: np.ndarray, n: int, window: Sequence[int], m: int,
                rng: np.random.Generator) -> np.ndarray:
    out = []
    for s in states:
        b, o = sample_shadows(s, n, m, rng)
        out.append(shadow_rdm(b, o, list(window)))
    return np.array(out)


def local_shadow_dataset(states: np.ndarray, n: int, m: int, k_max: int, rng: np.random.Generator,
                         noise: Callable[[np.ndarray], np.ndarray] | None = None,
                         depol: float = 0.0, readout_flip: float = 0.0) -> np.ndarray:
    feats = []
    for s in states:
        psi = noise(s) if noise is not None else s
        b, o = sample_shadows(psi, n, m, rng, depol=depol, readout_flip=readout_flip)
        feats.append(local_pauli_features(b, o, k_max))
    return np.array(feats)


def all_pauli_strings(k: int) -> list[str]:
    return ["".join(p) for p in product("XYZ", repeat=k)]


# ==========================================
# BELL MEASUREMENT IN PENNYLANE / QISKIT (ORACLE AND PARITY, IDEAS 4 AND 10)
# ==========================================
def bell_probs_pennylane(psi: np.ndarray, n: int) -> np.ndarray:
    import pennylane as qml
    dev = qml.device("default.qubit", wires=2 * n)

    @qml.qnode(dev)
    def circ():
        qml.StatePrep(np.kron(psi, psi), wires=range(2 * n))
        for q in range(n):
            qml.CNOT(wires=[q, n + q])
            qml.Hadamard(wires=q)
        return qml.probs(wires=range(2 * n))
    return np.asarray(circ())


def bell_probs_qiskit(psi: np.ndarray, n: int) -> np.ndarray:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    qc = QuantumCircuit(2 * n)
    for q in range(n):
        qc.cx(q, n + q)
        qc.h(q)
    sv = Statevector(to_qiskit_order(np.kron(psi, psi), 2 * n)).evolve(qc)
    p = np.abs(sv.data) ** 2
    # Back to the PennyLane convention (qubit 0 most significant).
    return to_qiskit_order(p.astype(complex), 2 * n).real


def window_purities_from_bell_probs(p: np.ndarray, n: int, k_max: int) -> np.ndarray:
    idx = np.arange(p.size)
    bits = (idx[:, None] >> (2 * n - 1 - np.arange(2 * n))) & 1
    sg = 1 - 2 * (bits[:, :n] & bits[:, n:])
    return np.array([(p * np.prod(sg[:, s:s + k], axis=1)).sum()
                     for k in range(1, k_max + 1) for s in range(n - k + 1)])


def oracle_bell(seed: int, backend: str, n: int = 3) -> bool:
    """Window purities inferred from the Bell measurement match those of the RDM."""
    rng = np.random.default_rng(seed + 20_000)
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    psi /= np.linalg.norm(psi)
    p = (bell_probs_pennylane if backend == "pennylane" else bell_probs_qiskit)(psi, n)
    d_np = np.abs(p - bell_probs_two_copies(psi, n)).max()
    assert d_np < 1e-10, f"[{backend}] probs Bell backend vs numpy |d|={d_np:.2e}"
    d_pur = np.abs(window_purities_from_bell_probs(p, n, n) - exact_window_purities(psi, n, n)).max()
    assert d_pur < 1e-10, f"[{backend}] purezas Bell vs RDM |d|={d_pur:.2e}"
    return True
