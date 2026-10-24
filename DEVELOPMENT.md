# Development branch

This branch is `main` plus the provenance of every design choice: what was tried on development
seeds, what it showed and why the final configuration won. New experiments land here first and are
merged into `main` only after they have been run on the official seeds with the protocol of
`PROTOCOL.md`.

## Seed discipline

- **Development seeds: `>= 9001`.** Everything exploratory runs there. Nothing on an official seed
  is inspected before the confirmatory run.
- **Official seeds: `0, 1, 7, 42, 123, 456, 789, 1234, 2024, 3407`.** Fixed before the campaign.
  Ten seeds give `1/1024` as the smallest attainable exact Wilcoxon p, which is the floor the
  primary test lives on.
- **Seed 42** is the gate. It must keep reproducing `0.0875 / 0.1625 / 0.2000` on every change,
  even a change that improves other numbers.

## What was tried before the design froze

Probes on development seeds. Numbers are from ten development seeds at `C = 400` unless stated.

| Probe | Question | Outcome |
|---|---|---|
| Rival probe | Is any single-copy classical reader competitive on the classification task? | Bell 0.086; U-statistic shadows 0.135; local Pauli 0.494; full-support Pauli 0.519 (logistic) and 0.403 (SVM); shadow RDM 0.256; hybrid 0.404. The best single-copy reader is the U-statistic one. |
| Endpoint check | Can an 80-state test split resolve a 0.05 margin? | No: resolution 0.0125 against a seed-to-seed delta s.d. of about 0.03. The campaign endpoint moved to estimator RMSE before any official seed. |
| Size probe | Does the gap grow with `|A|`? | Yes, monotonically; RMSE ratios 3.2x and 3.5x at `|A| = 2, 3` with `n_est = 100`. |
| Copy probe | Is the ratio constant in `C`? | No. This is what killed the extrapolated multiplier and forced the measured grid. |
| Dephasing probe | Is a dephasing axis needed? | No: purity is invariant under local unitaries and local Paulis map Bell states to Bell states, so the label and the Bell distribution are unchanged. A numerical check confirmed agreement to machine precision while the state itself moved. |
| Noise probe | Which error model is right? | The sign-masking model was off by up to 0.19 against the density matrix. Replaced by the bit-level model now in `exp_bell_paper.py`. |
| NTangled probe | Do the released generators reproduce the target entanglement? | Yes with real-amplitude product inputs (exact means 0.049 and 0.252 for the classes labelled 0.05 and 0.25); complex inputs drift above target. |
| Campaign dry run | What effect size should the confirmatory run expect? | Delta -0.274 on development seeds, which turned into -0.143 once the stronger baseline was in place. |

## Conventions for new work

1. New experiment, new script named `exp_<topic>.py` with an `oracle()` that self-checks before any
   expensive loop, and a JSON output under `results/bell_qnn/paper_<topic>.json`.
2. Every script must be able to run on a single development seed in under a minute.
3. Never edit an archived result file. New numbers go to a new file, and `fill_macros.py` decides
   which file the manuscript reads.
4. If a change alters the gate numbers, it is a protocol amendment: document it in `PROTOCOL.md`
   with the date and the reason, before running any official seed.
5. Long sweeps accept `--seeds` and `--tag` so they can be split across processes; outputs merge by
   `(seed, size)` in `fig_paper.py`.

## Open threads

- Adaptive single-copy readers: the exchange rate against an adaptive protocol is unknown and the
  paper says so.
- Mixed-state data, where the self-calibration anchor disappears. A reference state of known purity
  would restore it; not implemented.
- A trainable two-copy circuit instead of fixed Bell features. This is a different paper: here the
  coherent circuit has no parameters, which the manuscript states explicitly.
