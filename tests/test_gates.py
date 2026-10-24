"""Tests of the common library and of the gate contract.

Run from the repository root:  python -m unittest discover -s tests -t tests -v
Development seeds 9001+ (never the official ones). No test writes to checkpoints/.
"""
import hashlib
import json
import os
import subprocess
import sys
import unittest

import numpy as np
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE)
import qdlib as Q  # noqa: E402
IDEAS = ("bell_qnn",)

CLAVES = {"idea_id", "backend", "mode", "seed", "oracle_passed", "wall_time_sec", "metrics",
          "status", "kill_reason"}
SEED = 9001
ENV = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONWARNINGS": "ignore"}


def hash_checkpoints() -> str:
    d = os.path.join(BASE, "checkpoints")
    h = hashlib.sha256()
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            with open(os.path.join(d, f), "rb") as fh:
                h.update(f.encode() + fh.read())
    return h.hexdigest()


class TestLibreria(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        n = 4
        psi = self.rng.normal(size=2 ** n) + 1j * self.rng.normal(size=2 ** n)
        self.psi, self.n = psi / np.linalg.norm(psi), n

    def test_orden_qiskit_involucion(self):
        v = self.psi
        self.assertLess(np.abs(Q.to_qiskit_order(Q.to_qiskit_order(v, 4), 4) - v).max(), 1e-15)
        r = np.outer(v, v.conj())
        self.assertLess(np.abs(Q.to_qiskit_order(Q.to_qiskit_order(r, 4), 4) - r).max(), 1e-15)

    def test_sombras_insesgadas_rdm(self):
        b, o = Q.sample_shadows(self.psi, self.n, 40000, self.rng)
        err = np.abs(Q.shadow_rdm(b, o, [1, 2]) - Q.rdm(self.psi, [1, 2], self.n)).max()
        self.assertLess(err, 0.05)

    def test_pauli_local_vs_exacto(self):
        b, o = Q.sample_shadows(self.psi, self.n, 40000, self.rng)
        f = Q.local_pauli_features(b, o, 1)
        exacto = [Q.expval_pauli(self.psi, self.n, {q: p}) for q in range(self.n) for p in "XYZ"]
        self.assertLess(np.abs(f - exacto).max(), 5 * np.sqrt(3 / 40000))

    def test_pureza_histograma_igual_a_pares(self):
        b, o = Q.sample_shadows(self.psi, self.n, 60, self.rng)
        hist = Q.shadow_window_purities(b, o, 2)
        sn = Q.shadow_snapshots(b, o, [0, 1])
        m, s = len(sn), Q.SHOTS_POR_BASE
        # solo pares de grupos (bases) distintos
        pares = sum(np.real(np.trace(sn[i] @ sn[j])) for i in range(m) for j in range(m)
                    if i // s != j // s)
        self.assertAlmostEqual(hist[self.n], pares / (m * (m - s)), places=9)

    def test_pureza_agrupada_insesgada(self):
        exacta = Q.exact_window_purities(self.psi, self.n, 2)
        reps = np.array([Q.shadow_window_purities(*Q.sample_shadows(self.psi, self.n, 200, self.rng), 2)
                         for _ in range(40)])
        z = np.abs(reps.mean(0) - exacta) / (reps.std(0, ddof=1) / np.sqrt(40))
        self.assertLess(z.max(), 5)

    def test_bell_exacto(self):
        p = Q.bell_probs_two_copies(self.psi, self.n)
        d = np.abs(Q.window_purities_from_bell_probs(p, self.n, self.n)
                   - Q.exact_window_purities(self.psi, self.n, self.n)).max()
        self.assertLess(d, 1e-12)

    def test_qnn_ventana_tres_simuladores(self):
        qnn = Q.WindowQNN(3, 2)
        self.assertTrue(Q.oracle_window_qnn(qnn, SEED, "pennylane"))
        self.assertTrue(Q.oracle_window_qnn(qnn, SEED, "qiskit"))

    def test_familias_equilibradas(self):
        for f in Q.FAMILIAS:
            _, y, _ = Q.sample_family(f, 6, 10, self.rng)
            self.assertEqual(int(y.sum()), 5, f)

    def test_annni_frontera(self):
        self.assertEqual(Q.annni_ferro(0.0, 0.5), 1)
        self.assertEqual(Q.annni_ferro(0.0, 1.5), 0)
        self.assertEqual(Q.annni_ferro(0.6, 0.1), 0)


class TestBackendQiskit(unittest.TestCase):
    """The Qiskit measurement path reproduces the statistics of the numpy path."""

    def tearDown(self):
        Q.set_backend("pennylane")

    def test_sombras_qiskit_insesgadas(self):
        rng = np.random.default_rng(SEED + 1)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        psi /= np.linalg.norm(psi)
        Q.set_backend("qiskit")
        b, o = Q.sample_shadows(psi, 3, 20000, rng)
        err = np.abs(Q.shadow_rdm(b, o, [0, 2]) - Q.rdm(psi, [0, 2], 3)).max()
        self.assertLess(err, 0.06)

    def test_bell_qiskit(self):
        rng = np.random.default_rng(SEED + 2)
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        psi /= np.linalg.norm(psi)
        Q.set_backend("qiskit")
        est = Q.bell_window_purities(Q.sample_bell(psi, 3, 20000, rng), 3)
        self.assertLess(np.abs(est - Q.exact_window_purities(psi, 3, 3)).max(), 5 / np.sqrt(20000))

    def test_expval_qiskit(self):
        rng = np.random.default_rng(SEED + 3)
        psi = rng.normal(size=16) + 1j * rng.normal(size=16)
        psi /= np.linalg.norm(psi)
        ops = {0: "Z", 1: "X", 3: "Y"}
        a = Q.expval_pauli(psi, 4, ops)
        Q.set_backend("qiskit")
        self.assertAlmostEqual(a, Q.expval_pauli(psi, 4, ops), places=12)


class TestContratoOraculos(unittest.TestCase):
    """Oracle of the bell gate in both backends: exact contract and oracle_passed."""

    @classmethod
    def setUpClass(cls):
        cls.hash0 = hash_checkpoints()

    @classmethod
    def tearDownClass(cls):
        assert hash_checkpoints() == cls.hash0, "los tests modificaron checkpoints/"

    def test_oraculos(self):
        for idea in IDEAS:
            for be in ("pennylane", "qiskit"):
                with self.subTest(idea=idea, backend=be):
                    p = subprocess.run([sys.executable, f"exp_{idea}.py", "--mode", "oracle",
                                        "--backend", be, "--seed", str(SEED)],
                                       cwd=BASE, env=ENV, capture_output=True, text=True, timeout=600)
                    lineas = [l for l in p.stdout.splitlines() if l.strip()]
                    self.assertEqual(len(lineas), 1, p.stderr[-800:])
                    out = json.loads(lineas[0])
                    self.assertEqual(set(out), CLAVES)
                    self.assertEqual(out["metrics"], {})
                    self.assertTrue(out["oracle_passed"], out["kill_reason"])


if __name__ == "__main__":
    unittest.main()
