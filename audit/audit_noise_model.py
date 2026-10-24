"""Audit check: is the noise model of dev_noise_sweep.py exact for the Bell route?

Compares, for small n, three numbers for E[prod_{q in W} s_q] (the Bell purity estimator):
  exact    density-matrix: per-qubit depolarising / readout applied channel-by-channel to rho (x) rho,
           then the Bell measurement probabilities, then the exact expectation of the sign product.
  mask     the model used in dev_noise_sweep.py (pair killed w.p. 1-(1-p)^2 -> uniform sign;
           readout -> sign flipped w.p. 2 eps (1-eps)).
  pauli    exact stochastic model on raw bits: depolarising = random Pauli w.p. 3p/4 per qubit, which
           maps to deterministic bit flips of (a, b) after the Bell circuit; readout flips each bit.
"""
from __future__ import annotations

import numpy as np

import qdlib as Q

I2 = np.eye(2, dtype=complex)


def depol_qubit(rho: np.ndarray, q: int, n_tot: int, p: float) -> np.ndarray:
    """Per-qubit depolarising on qubit q of an n_tot-qubit density matrix."""
    t = rho.reshape([2] * (2 * n_tot))
    traced = np.trace(t, axis1=q, axis2=q + n_tot)            # remove qubit q
    mixed = np.tensordot(traced, I2 / 2, axes=0)               # append qubit at the end (ket, bra)
    # move appended axes (last two) back to positions q and q + n_tot
    order = list(range(mixed.ndim))
    k_ket, k_bra = mixed.ndim - 2, mixed.ndim - 1
    rest = order[:-2]
    kets, bras = rest[: n_tot - 1], rest[n_tot - 1:]
    kets.insert(q, k_ket)
    bras.insert(q, k_bra)
    mixed = np.transpose(mixed, kets + bras).reshape(rho.shape)
    return (1 - p) * rho + p * mixed


def bell_probs_from_rho(rho: np.ndarray, n: int) -> np.ndarray:
    u = np.eye(1, dtype=complex)
    m = 2 * n
    big = np.eye(2 ** m, dtype=complex)
    for q in range(n):
        big = np.array([Q.apply_2q(col, Q.CNOT, q, n + q, m) for col in big.T]).T
        big = np.array([Q.apply_1q(col, Q.HADAMARD, q, m) for col in big.T]).T
    out = big @ rho @ big.conj().T
    return np.real(np.diag(out))


def sign_expect(probs: np.ndarray, n: int, W: list[int], eps: float = 0.0) -> float:
    m = 2 * n
    idx = np.arange(probs.size)
    bits = (idx[:, None] >> (m - 1 - np.arange(m))) & 1
    if eps > 0:  # exact readout: average over all flip patterns analytically via enumeration
        tot = 0.0
        for flips in range(2 ** m):
            f = (flips >> (m - 1 - np.arange(m))) & 1
            w = np.prod(np.where(f == 1, eps, 1 - eps))
            b2 = bits ^ f
            a, b = b2[:, :n], b2[:, n:]
            s = np.prod((1 - 2 * (a & b))[:, W], axis=1)
            tot += w * float(probs @ s)
        return tot
    a, b = bits[:, :n], bits[:, n:]
    s = np.prod((1 - 2 * (a & b))[:, W], axis=1)
    return float(probs @ s)


def main() -> None:
    rng = np.random.default_rng(9101)
    n, W = 2, [0, 1]
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    r = Q.rdm(psi, W, n)
    p2 = float(np.real(np.trace(r @ r)))
    rho2 = np.outer(np.kron(psi, psi), np.kron(psi, psi).conj())
    print(f"clean purity p2 = {p2:.4f}")
    print("| channel | value | exact | mask model (dev_noise_sweep) | mask error |")
    print("|---|---|---|---|---|")
    for p in (0.05, 0.10, 0.20):
        rho = rho2.copy()
        for q in range(2 * n):
            rho = depol_qubit(rho, q, 2 * n, p)
        exact = sign_expect(bell_probs_from_rho(rho, n), n, W)
        mask = ((1 - p) ** 2) ** len(W) * p2
        print(f"| depol | {p} | {exact:.4f} | {mask:.4f} | {mask - exact:+.4f} |")
    probs0 = bell_probs_from_rho(rho2, n)
    for e in (0.02, 0.05, 0.10):
        exact = sign_expect(probs0, n, W, eps=e)
        mask = (1 - 4 * e * (1 - e)) ** len(W) * p2
        print(f"| readout | {e} | {exact:.4f} | {mask:.4f} | {mask - exact:+.4f} |")


if __name__ == "__main__":
    main()
