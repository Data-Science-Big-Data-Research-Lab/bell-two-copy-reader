# How many copies is coherence worth?

Code and archived results for the paper *"How many copies is coherence worth? Exact exchange
rates between two-copy and single-copy readout of quantum data"*.

Repository: <https://github.com/Data-Science-Big-Data-Research-Lab/bell-two-copy-reader> — Data
Science & Big Data Research Lab, Universidad Pablo de Olavide (Seville, Spain).

Two readers of the same quantum state are compared at an **equal copy budget**: the number of
times the unknown state is prepared. The coherent reader measures pairs of copies in the Bell
basis; the single-copy readers use randomised Pauli measurements, locally (random local basis,
one shot each) or globally (one Haar-random unitary on the subsystem). The quantity estimated is
the subsystem purity `tr(rho_A^2)`, the building block of Rényi-2 entropies and of concentratable
entanglement.

The headline numbers, all on ten pre-registered seeds:

| Quantity | Value |
|---|---|
| Purity RMSE, `C = 400`, `|A| = 4`, Bell vs best local reader | 0.051 vs 0.194 (3.8x) |
| Copies a local reader needs to match Bell, `|A| = 2..8` | 2.6, 4.8, 7.8, 16.6, 32, 76, 191 |
| Copies a global reader needs to match Bell, `|A| = 2..8` | 2.0, 2.7, 3.8, 4.9, 7.0, 10.6, 19.3 |
| Exact theory vs published variance bound (`C = 400`, `|A| = 8`) | 213 vs 895 copies |

## What is in this repository

```
qdlib.py                 shared library: state prep, shadow sampling, Bell sampling, estimators,
                         JSON contract, checkpoint helpers
exp_bell_qnn.py          gate (classification) and campaign (RMSE endpoint) for the paper's claim
exp_bell_paper.py        scaling in |A|, copies-to-match, exact noise, NTangled
exp_bell_global.py       global single-copy reader and the published-bound comparison
theory_bell.py           exact U-statistic variance (zeta1, zeta2) and the exchange rate
exp_bell_learn.py        learned labels: preparation depth, von Neumann entropy
exp_bell_emulated.py     transpiled emulation on the IBM Heron r2 model (FakeFez)
fig_paper.py             the three manuscript figures
fill_macros.py           writes every swept number of the manuscript from the result files
tests/test_gates.py      unit and contract tests (development seeds only)
results/bell_qnn/        archived per-seed JSON for every experiment
paper/                   manuscript source, bibliography and built PDF
PROTOCOL.md              pre-registration, endpoints, amendments, statistics
REPRODUCE.md             exact commands and expected outputs
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. the shipped tests (development seeds, never the official ones)
python -m unittest discover -s tests -t tests -v

# 2. the gate, which must reproduce 0.0875 / 0.1625 / 0.2000 bit for bit
python exp_bell_qnn.py --mode gate --seed 42 --backend pennylane

# 3. one confirmatory seed of the campaign (about 6 s)
python exp_bell_qnn.py --mode campaign --seed 42 --backend pennylane

# 4. the closed-form prediction for one size
python theory_bell.py --ks 4 --seeds 42 --n_est 50
```

The full reproduction, including the expensive sweeps, is in `REPRODUCE.md`.

## Branches

| Branch | Contents |
|---|---|
| `main` | Everything needed to reproduce the manuscript: code, tests, archived results, paper source. |
| `dev` | `main` plus `DEVELOPMENT.md`: the development-seed probes behind each design choice and the conventions for new work. New experiments land here first. |
| `audit` | `main` plus the independent audit harness (`audit_*.py`), its archived output and `AUDIT.md`, which lists every finding and how it was resolved. |

Start from `audit` if you want to check the claims rather than reproduce them.

## Scope and known limits

- All results are simulated. No physical device was used. The noise study applies exact
  per-qubit depolarising and readout channels to raw outcomes; `exp_bell_emulated.py` adds
  compilation to a calibrated heavy-hex device model, which still omits crosstalk, leakage and
  drift.
- Self-calibration under noise relies on pure input states, where the full-register purity is
  known. For mixed-state data the anchor disappears and only the raw reader remains.
- The single-copy readers are non-adaptive. Adaptive single-copy protocols can reach the optimal
  `Theta(sqrt d)` scaling, so the exchange rate against the best adaptive protocol may be lower
  than the rates reported here.
- The confirmatory campaign covers `|A| = 4`; the size sweep is exploratory and labelled as such
  in the manuscript.

## Data and code availability

Everything needed to rebuild the manuscript from scratch is in this repository: the archived
per-seed results drive `fig_paper.py` and `fill_macros.py`, so no number in the paper is typed by
hand. NTangled generator weights are external and public
(<https://github.com/LSchatzki/NTangled_Datasets>).

## License

MIT for the code, CC BY 4.0 for the text and figures. See `LICENSE`.

## Contact

Daniel Martín Pérez, Data Science & Big Data Research Lab, Universidad Pablo de Olavide. Issues and pull requests are welcome; please open an issue
before running long sweeps so that we can point you at the archived results instead.
