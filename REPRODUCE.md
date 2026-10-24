# Reproducing the results

All commands run from the repository root. Timings are for a 2-core machine with
`OMP_NUM_THREADS=1`.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## 0. Sanity first

```bash
python -m unittest discover -s tests -t tests -v          # 13 tests, about 70 s
python exp_bell_qnn.py --mode gate --seed 42 --backend pennylane
```

Expected, bit for bit: `ours 0.0875`, `q_baseline 0.1625`, `classical_rival 0.2`, `status PASS`.
If this does not match, stop: everything else is built on the same estimators.

## 1. Confirmatory campaign (table 2 of the paper)

```bash
for s in 0 1 7 42 123 456 789 1234 2024 3407; do
  python exp_bell_qnn.py --mode campaign --seed $s --backend pennylane \
    > results/bell_qnn/campaign_seed$s.json 2> results/bell_qnn/campaign_seed$s.stderr
done
```

Ten seeds take about 6 s each. Expected: Bell RMSE 0.049-0.053, strongest single-copy 0.180-0.201,
mean delta -0.143 (s.d. 0.006), 10/10 negative.

Qiskit parity of the same campaign:

```bash
python audit_qiskit.py            # 10/10 PASS, Bell probabilities agree to 2.2e-16
```

## 2. Size sweep and exchange rate (figure 1, panels a and c)

```bash
python exp_bell_paper.py scale  --ks 2 3 4 5 6          # RMSE vs |A| at C = 400
python exp_bell_paper.py copies --ks 2 3 4 5 6 7 8      # copies-to-match grid
python theory_bell.py --ks 2 3 4 5 6 7 8                # closed-form prediction
python exp_bell_global.py --ks 2 3 4 5 6 7 8            # global rival + bound
```

The `copies` sweep is the expensive one: about 4 min per seed at `|A| = 4` and up to 20 min per
seed at `|A| = 8`. It accepts `--tag` to split seeds across processes and writes one JSON per tag;
`fig_paper.py` merges every `paper_copies_*.json` and removes duplicates by (seed, size).

## 3. Exact noise (figure 2, supplementary table 2)

```bash
python exp_bell_paper.py noise
```

Depolarising `p` and readout `epsilon` up to 0.3 and 0.2, ten seeds, exact channels on raw
outcomes.

## 4. NTangled (figure 3)

```bash
git clone https://github.com/LSchatzki/NTangled_Datasets /tmp/ntangled
python exp_bell_paper.py ntangled
```

The weights are external; the script reads them from `/tmp/ntangled` by default and accepts a
different path.

## 5. Learned labels (table 4)

```bash
python exp_bell_learn.py
```

Two pre-registered tasks, ten official seeds, about 6 min in total.

## 6. Device emulation (table 5)

```bash
python exp_bell_emulated.py --k 2 --n_est 100            # |A| = 2, 16 qubits, ~40 s per seed
python exp_bell_emulated.py --k 2 --n_est 100 --ideal     # noiseless control
python exp_bell_emulated.py --k 3 --n_est 50              # |A| = 3, ~4.5 min per seed
```

The noise model is built once per process and the circuits are transpiled in batched calls.

## 7. Rebuild the figures and the manuscript

```bash
python fig_paper.py                     # fig1_scale.png, fig2_noise.png, fig3_ntangled.png
python fill_macros.py                   # macros_copies.tex, campaign_rows.tex
cd paper && latexmk -pdf bell_two_copy_reader_v3.tex
```

`fill_macros.py` writes every swept number of the manuscript, so a stale result file shows up as a
changed macro rather than as an undetected inconsistency.

## What to expect if something differs

- **Gate numbers differ.** The estimators changed under you, or a dependency changed its RNG
  conventions. Check `numpy` and `qiskit` versions against `requirements.txt` first.
- **Copies sweep differs by a few percent.** The interpolation grid is finite; the archived
  multipliers are the ones quoted in the paper.
- **NTangled fails to load.** The external repository is missing or its layout changed; that path
  is the only external dependency in the project.
