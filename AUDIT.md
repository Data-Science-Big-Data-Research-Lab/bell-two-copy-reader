# Independent audit: findings and resolutions

This branch carries the audit harness and its archived output. It is the place to start if you
want to check the paper's claims instead of reproducing them. Everything here was run on the ten
official seeds; the audit scripts are read-only with respect to the deposited results, except for
the JSON files they write themselves.

## How to run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

cd audit
python audit_bell.py             # campaign schema, Wilcoxon, Holm m = 2, verdict
python audit_noise_model.py      # bit-level noise model vs density matrices, n = 2
python audit_deep.py --part A     # bias correction, gate/campaign state overlap
python audit_deep.py --part B     # per-window vs scalar purity, grouped vs one-shot
python audit_deep.py --part C     # size sweep and copies-to-match consistency
python audit_qiskit.py           # full campaign on Qiskit, parity of Bell probabilities
python audit_strong_baseline.py  # stronger single-copy rivals on the official seeds
```

Expected verdict from `audit_bell.py`:

```
- Mean delta -0.1426 (sd 0.0062), 10/10 negative, effect size -23.1 sd
- Exact one-sided Wilcoxon: p = 0.00098 (smallest attainable 1/1024 = 0.00098)
- Holm over the family ('qcnn_tolerant', 'bell_qnn'): adjusted p = 0.00195
- Verdict: PASS
```

## Findings

Each finding was raised against an earlier version of the work, fixed, and re-run before the
manuscript was circulated. The two that changed numbers are marked.

### H1. The noise model was not exact (changed numbers)

An earlier study applied depolarising noise as a random sign on the swap eigenvalue. That is wrong:
a depolarised pair does not give a uniform sign, and readout error does not flip the sign only when
exactly one bit flips. The error reached -0.11 at `p = 0.1`, -0.19 at `p = 0.2` and -0.18 at
`epsilon = 0.05` in purity, and it reversed the paper's noise conclusion.

*Fix.* Noise is now applied to the measured bits: `X` on either copy flips `b_q`, `Z` flips `a_q`,
`Y` flips both, each with probability `p/4`, and readout flips each bit with `epsilon`. Validated
against density-matrix evolution on `n = 2` to `7e-4` (`audit_noise_model.py`). The obsolete script
is not shipped.

*Consequence.* Under exact noise both readers share the same bias, so the earlier claim "the more
precise reader is the more fragile one" is false. What is specific to the coherent reader is free
self-calibration from the full-register purity.

### H2. The single-copy baseline was too weak (changed numbers)

The pre-registered baseline grouped `C` snapshots into `C/10` bases with ten shots each. Excluding
same-basis pairs to stay unbiased throws information away. One shot per basis, clipped to
`[2^-|A|, 1]`, is strictly stronger: at `|A| = 4` its RMSE is 0.194 against 0.340.

*Fix.* The campaign scores against the stronger reader. It is sampled with a separate generator, so
no archived Bell number moves, and the campaign therefore becomes harder, not easier, to pass.
Qiskit reproduces it 10/10.

### H3. The copies multiplier had been extrapolated (changed numbers)

An earlier draft converted the RMSE ratio at one budget into copies by assuming a `C^-1/2` law,
which gave 51x at `|A| = 4`. The curves are not parallel over the measured range, so that conversion
is invalid.

*Fix.* The multiplier is now measured on a grid of budgets and interpolated in log-log, never
extrapolated: 7.8x at `|A| = 4` and 191x at `|A| = 8` for the local reader. `theory_bell.py` then
shows why the two curves diverge: the clipped single-copy error falls faster than `C^-1/2` until
estimates move back inside the physical range.

### H4. Bias correction was the identity in the campaign

The global-lambda correction calibrated on the full-register purity is exactly the identity without
noise (`lambda = 1`), so describing the confirmatory campaign as "bias-corrected" would be
misleading. It is not described that way; the correction matters only in the noise study.

### H5. Gate and campaign share states at seed 42

The first 200 of the 400 states of seed 42 are the gate's states, because the protocol uses seed 42
for both. This is disclosed in the manuscript rather than removed, so that the gate stays bit-exact.

### H6. Text and code disagreed in places

Window-purity counts, the definition of the strongest reader and the NTangled split were described
inconsistently between the draft and the code. All three are now generated from the archived JSON by
`fill_macros.py`, so a mismatch surfaces as a changed macro.

## What the audit did not find

- No seed selection: the ten official seeds were fixed before the confirmatory run.
- No unequal copy budgets in any comparison, including the NTangled re-analysis.
- No test leakage: the campaign threshold is computed on the training split only.
- No dependence on the backend: the campaign passes 10/10 on PennyLane and on Qiskit, with Bell
  probabilities agreeing to `2.2e-16`.
- The suite does not touch `checkpoints/`; its hash is unchanged before and after.

## Residual limits that the audit cannot fix

- No hardware data. The noise levels are design targets and the device emulation omits crosstalk,
  leakage and drift.
- Self-calibration needs pure inputs, where the full-register purity is known.
- The single-copy readers are non-adaptive; an adaptive protocol may close part of the gap.
- Both exploratory analyses added after the campaign favour the coherent reader in every seed,
  which is what one expects if the effect is real, but they are not covered by the pre-registered
  test and are labelled as exploratory.
