# Pre-registration and experimental protocol

This file records what was fixed before the confirmatory run, what changed afterwards and why.
It is the document an auditor should read first. It supersedes the internal `SPEC.md` of the
development repository; the amendments below are dated.

## 1. Question and hypotheses

**Primary question.** At an equal number of state preparations `C` per state, how much more
accurately does a two-copy Bell reader estimate a nonlinear functional of the state than the
strongest single-copy reader, and how many copies does the single-copy reader need to close the
gap?

**H1 (primary).** At `C = 400`, `|A| = 4`, Bell RMSE is lower than the best single-copy RMSE by at
least 0.05 on every official seed.

**H0.** The difference is zero or positive.

**Secondary.** The gap grows with `|A|`; it survives when the label must be learned; it survives a
published dataset, exact noise and compilation to a device model.

## 2. Seeds

Ten official seeds, fixed before any confirmatory run:

```
0, 1, 7, 42, 123, 456, 789, 1234, 2024, 3407
```

Development used seeds `>= 9001` only. The single-seed gate uses seed 42. No test in this
repository touches an official seed.

## 3. Endpoints

**Gate (`--mode gate`, historical).** Classification error on `N = 120/80` states, margin
`ours <= q_baseline - 0.05`. Archived result: 0.0875 (Bell) vs 0.1625 (shadows) vs 0.2000
(classical rival). It is a kill filter, not evidence, and it is kept unchanged so that seed 42
still reproduces bit for bit.

**Campaign (`--mode campaign`, confirmatory).** RMSE of the estimator of `tr(rho_A^2)` at `C = 400`
copies, `|A| = 4`, `N = 400` states with a `240/160` split. `ours` is the Bell reader with the
global-lambda bias correction; `q_baseline` is the minimum over the single-copy readers. Per-seed
PASS if `ours < q_baseline`.

**Amendment of 21 October (before any official seed).** The campaign endpoint moved from
classification error to estimator RMSE. Reason: with 80 test items the resolution is 0.0125 while
the seed-to-seed standard deviation of the paired delta is about 0.03, so the pre-registered
margin of 0.05 cannot be resolved. Classification stays in `DETAIL` as a secondary readout. The
gate result is reported unchanged.

## 4. Equal-copy currency

Every reader receives the same `C` copies per state:

- Bell: `C` copies, `C/2` measurements of copy pairs in the Bell basis; every window purity comes
  from the same sample set.
- Local single-copy: `C` copies, one shot per randomly chosen local Pauli basis, U-statistic over
  pairs from different bases, estimate clipped to `[2^-|A|, 1]`.
- Global single-copy: `C` copies, one Haar-random unitary on `A` per shot, U-statistic over the
  inverted snapshots, clipped.

Grouped shadows (`S` shots per basis) are reported for reference because they are common on
hardware; they are never used as the strongest baseline.

## 5. Statistics

Per-seed paired difference `ours - q_baseline`. Primary test: exact one-sided Wilcoxon signed-rank.
Sensitivity: paired t test. Multiplicity: Holm over the pre-registered family `{idea 3, idea 4}`
(`m = 2`); an idea without a campaign enters with `p = 1`. `alpha = 0.05`. With ten seeds the
smallest attainable p is `1/1024 = 9.8e-4`, so the family clears Holm only with 10/10 negative
deltas.

Analyses added after the campaign (size sweep, learned labels, emulation, global reader) are
**exploratory**: they use the official seeds, they are labelled as exploratory in the manuscript,
and they never enter the Holm family.

## 6. Known deviations and errors found by audit

Both were found after the campaign and are corrected in the released code; the manuscript reports
them.

1. **Baseline too weak.** The pre-registered single-copy baseline was the grouped reader
   (`S = 10`). The audit showed that one shot per basis, clipped, is strictly stronger. The
   campaign was re-scored against the stronger reader, which only makes the test harder. The
   stronger baseline is sampled with a separate random generator so that no archived number of the
   Bell reader changes.
2. **Inexact noise model.** An earlier noise study acted on the swap sign instead of the measured
   bits, which is not exact for depolarising and readout channels and was biased against the Bell
   reader (up to -0.19 in purity). It was replaced by the bit-level model, validated against
   density-matrix calculations on `n = 2` to `7e-4`, and the old script is marked obsolete.

A third item, reported in the manuscript for completeness: the first 200 of the 400 states of seed
42 coincide with the gate's states, because the protocol uses seed 42 for both.

## 7. Reproduction contract

`--mode gate --seed 42 --backend pennylane` must print exactly:

```json
{"ours_test_loss": 0.0875, "q_baseline_test_loss": 0.1625, "classical_rival_test_loss": 0.2, "status": "PASS"}
```

Any change that breaks this is a regression, even if it improves other numbers.
