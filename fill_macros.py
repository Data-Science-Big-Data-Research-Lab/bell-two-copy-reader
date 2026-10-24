"""Write the LaTeX fragments of the paper from the result JSON files (no number is typed by hand).

  paper/macros_copies.tex   \\MultTwo .. \\MultSix (measured copies multipliers) and \\PanelBsentence
  paper/campaign_rows.tex   per-seed rows of the confirmatory table

    PY=~/proyectos/active/q-taskfisher/.venv/bin/python
    $PY fill_macros.py
"""
from __future__ import annotations

import json

import numpy as np

import fig_paper as F

R = "results/bell_qnn/"
NAMES = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}


def fmt(x: float) -> str:
    return f"{x:.0f}" if x >= 20 else f"{x:.1f}"


def slopes(rows: list[dict]) -> tuple[float, float, float]:
    """Local log-log slope of the single-copy curve at |A|=4 below and above C=800, and of Bell."""
    k4 = [r for r in rows if r["k"] == 4]
    def mean_at(key, c):
        v = [r[key][str(c)] for r in k4 if str(c) in r[key]]
        return float(np.mean(v)) if v else float("nan")
    lo = np.log(mean_at("shadow", 800) / mean_at("shadow", 100)) / np.log(8)
    hi_c = max(int(c) for r in k4 for c in r["shadow"])
    hi = np.log(mean_at("shadow", hi_c) / mean_at("shadow", 1600)) / np.log(hi_c / 1600)
    b = np.log(mean_at("bell", 6400) / mean_at("bell", 100)) / np.log(64)
    return float(lo), float(hi), float(b)


def emulated_macros() -> str:
    """\\EmulatedBlock (table tab:emu + one paragraph) and \\EmulatedAbstract from paper_emulated_*.json."""
    import os
    from scipy import stats
    runs = []
    for k in (2, 3):
        for ideal in (False, True):
            f = f"{R}paper_emulated_k{k}{'_ideal' if ideal else ''}.json"
            if os.path.exists(f):
                runs.append((k, ideal, json.load(open(f))["rows"]))
    if not runs:
        return ("\\newcommand{\\EmulatedBlock}{\\textit{[emulation pending]}\\label{tab:emu}}\n"
                "\\newcommand{\\EmulatedAbstract}{[emulation pending].}\n")
    lines, summary = [], {}
    for k, ideal, rows in runs:
        def ms(key):
            v = np.array([r[key] for r in rows])
            return f"{v.mean():.3f} ({v.std(ddof=1):.3f})" if len(v) > 1 else f"{v.mean():.3f}"
        d = np.array([r["rmse_bell_cal"] - r["rmse_shadow"] for r in rows])
        wins = int((d < 0).sum())
        p = float(stats.wilcoxon(d, alternative="less", method="exact").pvalue) if len(d) > 1 and np.any(d) else float("nan")
        n2b = np.mean([r["n2q_bell_mean"] for r in rows])
        n2s = np.mean([r["n2q_shadow_mean"] for r in rows])
        lines.append(f"{k} & {'ideal' if ideal else 'FakeFez noise'} & {len(rows)} & {ms('rmse_bell')} & "
                     f"{ms('rmse_bell_cal')} & {ms('rmse_shadow')} & {n2b:.0f} / {n2s:.0f} & {wins}/{len(rows)} \\\\")
        if not ideal:
            ratio = np.mean([r["rmse_shadow"] for r in rows]) / np.mean([r["rmse_bell_cal"] for r in rows])
            summary[k] = (wins, len(rows), p, ratio, np.mean([r["lam"] for r in rows]),
                          np.mean([r["rmse_bell"] for r in rows]), np.mean([r["rmse_bell_cal"] for r in rows]),
                          np.mean([r["rmse_shadow"] for r in rows]))
    table = ("\\begin{table}[H]\\centering\\small\n\\caption{Transpiled emulation on the IBM Heron r2 model "
             "\\texttt{FakeFez}, $C=400$ copies per state, $n=2|A|$; RMSE of $\\mathrm{tr}(\\rho_A^2)$, mean (s.d.) "
             "over official seeds. 2q gates: mean two-qubit gate count after transpilation (Bell / one single-copy "
             "circuit). Wins: seeds in which the self-calibrated Bell reader beats the single-copy reader.}\n"
             "\\label{tab:emu}\n\\begin{tabular}{rlrrrrrr}\n\\toprule\n$|A|$ & Noise & Seeds & Bell raw & "
             "Bell calibrated & Single-copy & 2q gates & Wins \\\\\n\\midrule\n" + "\n".join(lines) +
             "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    txt = []
    for k, (w, n, p, ratio, lam, rb, rc, rs) in sorted(summary.items()):
        txt.append(f"At $|A|={k}$ the self-calibrated coherent reader reaches an RMSE of ${rc:.3f}$ against "
                   f"${rs:.3f}$ for the single-copy reader (factor ${ratio:.1f}$, lower in {w} of {n} seeds"
                   + (f", exact Wilcoxon $p={p:.2g}$" if np.isfinite(p) else "") +
                   f"); uncorrected it reaches ${rb:.3f}$, and the calibration constant is "
                   f"$\\lambda={lam:.3f}$ per qubit. ")
    all_win = bool(summary) and all(v[0] == v[1] for v in summary.values())
    txt.append("The Bell circuit is several times deeper in two-qubit gates than a single-copy circuit, because "
               "it prepares two copies and routes the pairwise CNOTs on heavy-hex; "
               + ("its estimate nevertheless stays ahead in every seed once the shared contraction is calibrated out."
                  if all_win else "on this device model that overhead is not always compensated, and the "
                  "comparison above should be read seed by seed."))
    k_last = max(summary) if summary else None
    best = summary.get(k_last) if summary else None
    if best and all_win:
        abstract = (f"survives transpilation and calibrated noise on an emulated IBM Heron device "
                    f"(RMSE ${best[6]:.3f}$ against ${best[7]:.3f}$ at $|A|={k_last}$, {best[0]} of {best[1]} seeds).")
    elif best:
        abstract = (f"on an emulated IBM Heron device with calibrated noise wins in {best[0]} of {best[1]} seeds at "
                    f"$|A|={k_last}$ (RMSE ${best[6]:.3f}$ against ${best[7]:.3f}$).")
    else:
        abstract = "[emulation pending]."
    return ("\\newcommand{\\EmulatedBlock}{%\n" + table + "\n" + "".join(txt) + "}\n"
            "\\newcommand{\\EmulatedAbstract}{" + abstract + "}\n")


def main() -> int:
    mult = F.multipliers()
    rows = F.load_copies()
    lo, hi, b = slopes(rows)
    lines = ["% generated by fill_macros.py -- do not edit"]
    for k, name in NAMES.items():
        if k in mult:
            m, s, n = mult[k]
            lines.append(f"\\newcommand{{\\Mult{name}}}{{${fmt(m)}$}}  % sd {s:.2f}, n={n}")
        else:
            lines.append(f"\\newcommand{{\\Mult{name}}}{{n/a}}")
    gs = F.global_stats()
    th = F._theory()
    for k, name in NAMES.items():
        if k in gs:
            lines.append(f"\\newcommand{{\\Glob{name}}}{{${fmt(gs[k]['mult'])}$}}  % sd {gs[k]['sd']:.2f}, n={gs[k]['n']}")
            lines.append(f"\\newcommand{{\\Bound{name}}}{{${fmt(gs[k]['bound'])}$}}")
            lines.append(f"\\newcommand{{\\BoundInf{name}}}{{${fmt(gs[k]['bound_inf'])}$}}")
        if k in th:
            lines.append(f"\\newcommand{{\\Exact{name}}}{{${fmt(th[k]['mult_pred'])}$}}")
            lines.append(f"\\newcommand{{\\ExactInf{name}}}{{${fmt(th[k]['mult_inf'])}$}}")
    lines.append("\\newcommand{\\PanelBsentence}{Over the measured range the single-copy RMSE falls as "
                 f"$C^{{{lo:.2f}}}$ between $C=100$ and $800$ and as $C^{{{hi:.2f}}}$ from $C=1600$ until it "
                 "reaches the coherent reader, faster than $C^{-1/2}$, because clipping to the physical range "
                 "removes a share of the error that shrinks as estimates move back inside it; the coherent "
                 f"reader follows $C^{{{b:.2f}}}$ throughout.}}")
    open("paper/macros_copies.tex", "w").write("\n".join(lines) + "\n")

    c = json.load(open(R + "CAMPAIGN.json"))
    out = []
    for f in c["filas"]:
        det = [json.loads(l[7:]) for l in open(f"{R}campaign_seed{f['seed']}.stderr") if l.startswith("DETAIL")][-1]
        out.append(f"{f['seed']} & {f['ours']:.4f} & {f['q_base']:.4f} & {det['rmse']['sombras']:.4f} & "
                   f"${f['delta']:+.4f}$ \\\\")
    open("paper/campaign_rows.tex", "w").write("\n".join(out) + "\n")
    with open("paper/macros_copies.tex", "a") as fh:
        fh.write("\\newcommand{\\CampaignRows}{%\n" + "\n".join(out) + "}\n")
        fh.write(emulated_macros())
    print("\n".join(lines))
    print({k: v for k, v in mult.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
