"""Paper figures from results/bell_qnn/paper_*.json (post-audit, official seeds).

  fig1_scale.png     (a) RMSE vs |A| at C=400  (b) RMSE vs C at |A|=4  (c) measured copies-to-match vs |A|
  fig2_noise.png     exact per-qubit depolarising and readout noise, |A|=4, C=400
  fig3_ntangled.png  NTangled 4q: RMSE of CE and classification error per seed

    PY=~/proyectos/active/q-taskfisher/.venv/bin/python
    $PY fig_paper.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import exp_bell_paper as P

R = "results/bell_qnn/"
BELL, SH1, SH10, GREY, GLOB = "#1f3b73", "#c0392b", "#e59866", "#7f8c8d", "#8e44ad"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})


def _ms(rows, key):
    v = np.array([r[key] for r in rows])
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def load_copies() -> list[dict]:
    import glob
    rows = []
    for f in sorted(glob.glob(R + "paper_copies_*.json")):
        rows += json.load(open(f))["rows"]
    seen, out = set(), []
    for r in rows:
        if (r["seed"], r["k"]) not in seen:
            seen.add((r["seed"], r["k"]))
            out.append(r)
    return out


def multipliers() -> dict:
    cp = load_copies()
    out = {}
    for k in sorted({r["k"] for r in cp}):
        v = [P.copies_to_match(r) / 400 for r in cp if r["k"] == k]
        v = [x for x in v if np.isfinite(x)]
        if v:
            out[k] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
    return out


def load_global() -> list[dict]:
    import glob
    rows = []
    for f in sorted(glob.glob(R + "paper_global_*.json")):
        if "_dev" not in f:
            rows += json.load(open(f))["rows"]
    return rows


def global_stats() -> dict:
    """Per |A|: mean/sd of the global-reader multiplier, its RMSE at C=400, and the bound-implied local rate."""
    g = load_global()
    out = {}
    for k in sorted({r["k"] for r in g}):
        sel = [r for r in g if r["k"] == k]
        mg = np.array([r["mult_global"] for r in sel], float)
        out[k] = {"mult": float(np.nanmean(mg)), "sd": float(np.nanstd(mg, ddof=1)), "n": int(np.isfinite(mg).sum()),
                  "rmse400": float(np.mean([r["global_C400"] for r in sel])),
                  "rmse400_sd": float(np.std([r["global_C400"] for r in sel], ddof=1)),
                  "bound": float(np.mean([r["mult_bound_local"] for r in sel])),
                  "bound_inf": float(np.mean([r["mult_inf_bound_local"] for r in sel]))}
    return out


def _theory() -> dict:
    """Mean over seeds of the closed-form predictions (paper_theory.json), by |A|."""
    if not os.path.exists(R + "paper_theory.json"):
        return {}
    rows = json.load(open(R + "paper_theory.json"))["rows"]
    out = {}
    for k in sorted({r["k"] for r in rows}):
        sel = [r for r in rows if r["k"] == k]
        out[k] = {key: float(np.mean([r[key] for r in sel])) for key in ("rmse_bell_pred", "rmse_s1_pred",
                                                                       "mult_pred", "mult_inf")}
    return out


def fig1() -> dict:
    sc = json.load(open(R + "paper_scale.json"))["rows"]
    cp = load_copies()
    ks = sorted({r["k"] for r in sc})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for key, col, mk, lab in (("shadow_S10", SH10, "v", "shadows, 10 shots/basis"),
                              ("shadow_S1_clip", SH1, "o", "shadows, 1 shot/basis, clipped"),
                              ("bell", BELL, "s", "two-copy Bell")):
        m = [_ms([r for r in sc if r["k"] == k], key) for k in ks]
        ax[0].errorbar(ks, [x[0] for x in m], yerr=[x[1] for x in m], fmt=mk + "-", color=col, ms=3.5,
                       lw=1, capsize=2, label=lab)
    ax[0].plot(ks, [np.mean([r["sd_purity"] for r in sc if r["k"] == k]) for k in ks], ":", color=GREY,
               lw=1, label="constant predictor")
    # |A| = 7, 8: same estimators on the same states, from the copies sweep (C = 400 point)
    kx = sorted({r["k"] for r in cp} - set(ks))
    if kx:
        for key, col, mk in (("shadow", SH1, "o"), ("bell", BELL, "s")):
            vals = [[(r["bell_C400"] if key == "bell" else r["shadow"]["400"]) for r in cp if r["k"] == k] for k in kx]
            ax[0].errorbar(kx, [np.mean(v) for v in vals], yerr=[np.std(v, ddof=1) for v in vals], fmt=mk,
                           mfc="white", color=col, ms=3.5, lw=1, capsize=2)
    ks = ks + kx
    gs = global_stats()
    if gs:
        gk = sorted(gs)
        ax[0].errorbar(gk, [gs[k]["rmse400"] for k in gk], yerr=[gs[k]["rmse400_sd"] for k in gk], fmt="D-",
                       color=GLOB, ms=3, lw=1, capsize=2, label="global Haar, 1 shot/unitary, clipped")
    th = _theory()
    if th:
        tk = sorted(th)
        ax[0].plot(tk, [th[k]["rmse_bell_pred"] for k in tk], "-", color=BELL, lw=0.6, alpha=0.5)
    ax[0].set_yscale("log")
    ax[0].set_xticks(ks)
    ax[0].set_xlabel("subsystem size $|A|$")
    ax[0].set_ylabel(r"RMSE of $\mathrm{tr}(\rho_A^2)$")
    ax[0].set_title("(a) $C=400$ copies", fontsize=8)
    ax[0].set_ylim(0.03, 25)
    ax[0].legend(fontsize=5.5, loc="upper left")

    k4 = [r for r in cp if r["k"] == 4]
    cs_b = sorted({int(c) for r in k4 for c in r["bell"]})
    cs_s = sorted({int(c) for r in k4 for c in r["shadow"]})
    mb = [np.mean([r["bell"][str(c)] for r in k4 if str(c) in r["bell"]]) for c in cs_b]
    ms = [np.mean([r["shadow"][str(c)] for r in k4 if str(c) in r["shadow"]]) for c in cs_s]
    ax[1].loglog(cs_s, ms, "o-", color=SH1, ms=3.5, lw=1, label="shadows, 1 shot/basis, clipped")
    ax[1].loglog(cs_b, mb, "s-", color=BELL, ms=3.5, lw=1, label="two-copy Bell")
    ax[1].axhline(np.mean([r["bell_C400"] for r in k4]), color=GREY, ls=":", lw=0.8)
    ax[1].set_xlabel("copies $C$ per state")
    ax[1].set_ylabel(r"RMSE of $\mathrm{tr}(\rho_A^2)$")
    ax[1].set_title("(b) $|A|=4$", fontsize=8)

    mult = multipliers()
    kk = sorted(mult)
    ax[2].errorbar(kk, [mult[k][0] for k in kk], yerr=[mult[k][1] for k in kk], fmt="^", color=SH1,
                   ms=3.5, lw=1, capsize=2, label="local Pauli, measured")
    if gs:
        ax[2].errorbar(gk, [gs[k]["mult"] for k in gk], yerr=[gs[k]["sd"] for k in gk], fmt="D", color=GLOB,
                       ms=3, lw=1, capsize=2, label="global Haar, measured")
        ax[2].plot(gk, [gs[k]["bound"] for k in gk], ":", color=GREY, lw=1, label="local, published bound")
    if th:
        tk = sorted(th)
        ax[2].plot(tk, [th[k]["mult_pred"] for k in tk], "-", color=SH1, lw=1,
                   label="local, exact theory")
        ax[2].plot(tk, [th[k]["mult_inf"] for k in tk], "--", color=SH1, lw=0.8, label=r"local, exact, $C\to\infty$")
    ax[2].set_yscale("log")
    ax[2].set_xticks(kk)
    ax[2].set_xlabel("subsystem size $|A|$")
    ax[2].set_ylabel("copies multiplier")
    ax[2].set_title("(c) copies to match Bell", fontsize=8)
    ax[2].legend(fontsize=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(R + "fig1_scale.png", dpi=300)
    return mult


def fig2() -> None:
    n = json.load(open(R + "paper_noise.json"))["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for a, kind, xl in ((ax[0], "depol", "depolarising $p$ per qubit and copy"),
                        (ax[1], "readout", r"readout error $\varepsilon$ per bit")):
        vs = [0.0] + sorted({r["value"] for r in n if r["kind"] == kind and r["value"] > 0})
        for key, col, mk, lab in (("shadow", SH1, "o", "shadows, 1 shot/basis, clipped"),
                                  ("bell", BELL, "s", "two-copy Bell, raw"),
                                  ("bell_corr", "#27ae60", "^", "two-copy Bell, self-calibrated")):
            m = [_ms([r for r in n if r["value"] == v and (r["kind"] == kind or v == 0.0)], key) for v in vs]
            a.errorbar(vs, [x[0] for x in m], yerr=[x[1] for x in m], fmt=mk + "-", color=col, ms=3.5,
                       lw=1, capsize=2, label=lab)
        a.set_yscale("log")
        a.set_xlabel(xl)
        a.set_ylabel(r"RMSE of $\mathrm{tr}(\rho_A^2)$")
    ax[0].legend(fontsize=6, loc="lower right")
    fig.tight_layout(pad=0.4)
    fig.savefig(R + "fig2_noise.png", dpi=300)


def fig3() -> None:
    d = json.load(open(R + "paper_ntangled.json"))["rows"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for a, kb, ks, yl in ((ax[0], "rmse_bell", "rmse_shadow", "RMSE of CE"),
                          (ax[1], "err_bell", "err_shadow", "test classification error")):
        a.bar(x - 0.19, [r[kb] for r in d], 0.36, color=BELL, label="two-copy Bell")
        a.bar(x + 0.19, [r[ks] for r in d], 0.36, color=SH1, label="shadows, 1 shot/basis, clipped")
        a.set_xticks(x)
        a.set_xticklabels([str(r["seed"]) for r in d], rotation=45, fontsize=6)
        a.set_xlabel("seed")
        a.set_ylabel(yl)
    ax[1].plot(x, [r["err_exact"] for r in d], "k_", ms=8, label="exact CE")
    for a in ax:
        a.set_ylim(0, a.get_ylim()[1] * 1.3)
        a.legend(fontsize=6, ncol=3 if a is ax[1] else 2, loc="upper center")
    fig.tight_layout(pad=0.4)
    fig.savefig(R + "fig3_ntangled.png", dpi=300)


if __name__ == "__main__":
    print("copies multiplier per |A| (mean, sd, n):", fig1())
    fig2()
    fig3()
    print("figures written")
