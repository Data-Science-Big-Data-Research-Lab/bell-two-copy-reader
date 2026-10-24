"""Audit of the idea-4 campaign (bell_qnn): ten official seeds, Holm m = 2.

Validates the schema, computes the per-seed paired delta (ours - q_baseline, in RMSE), the exact
one-sided Wilcoxon test and the paired t test as a sensitivity check, and applies Holm written by
hand over the pre-registered family {idea 3, idea 4}. An idea with no campaign enters with p = 1.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import stats

RAIZ = "results/bell_qnn"
SEEDS = (0, 1, 7, 42, 123, 456, 789, 1234, 2024, 3407)
FAMILY = ("qcnn_tolerant", "bell_qnn")  # pre-registered primary family (m = 2)
CLAVES = ("idea_id", "backend", "mode", "seed", "oracle_passed", "wall_time_sec",
          "metrics", "status", "kill_reason")


def leer(seed: int) -> tuple[dict, dict]:
    d = json.load(open(f"{RAIZ}/campaign_seed{seed}.json"))
    falta = [k for k in CLAVES if k not in d]
    assert not falta, f"seed {seed}: faltan claves {falta}"
    assert d["mode"] == "campaign" and d["seed"] == seed, f"seed {seed}: cabecera incoherente"
    det = {}
    try:
        lineas = [l for l in open(f"{RAIZ}/campaign_seed{seed}.stderr").read().splitlines()
                  if l.startswith("DETAIL ")]
        if lineas:
            det = json.loads(lineas[-1][7:])
    except FileNotFoundError:
        pass
    return d, det


def holm(pvals: list[float]) -> list[float]:
    """Holm paso a paso con monotonicidad; devuelve los p ajustados en el orden de entrada."""
    m = len(pvals)
    orden = sorted(range(m), key=lambda i: pvals[i])
    ajustados = [0.0] * m
    anterior = 0.0
    for k, i in enumerate(orden):
        f = (m - k) * pvals[i]
        anterior = max(anterior, min(1.0, f))
        ajustados[i] = anterior
    return ajustados


def main() -> int:
    rows = []
    for s in SEEDS:
        d, det = leer(s)
        m = d["metrics"]
        rows.append({
            "seed": s, "status": d["status"], "oracle": bool(d["oracle_passed"]),
            "ours": float(m["ours_test_loss"]), "q_base": float(m["q_baseline_test_loss"]),
            "delta": float(m["ours_test_loss"] - m["q_baseline_test_loss"]),
            "clas_bell": det.get("clasificacion", {}).get("bell"),
            "clas_shadows": det.get("clasificacion", {}).get("sombras"),
            "lam": det.get("lam"),
        })
    assert len(rows) == 10, f"expected 10 seeds, found {len(rows)}"
    assert all(f["oracle"] for f in rows), "some seed has a failed oracle"

    d = np.array([f["delta"] for f in rows])
    w = stats.wilcoxon(d, alternative="less", method="exact")
    t = stats.ttest_rel(d, np.zeros_like(d), alternative="less")
    p_fam = {idea: (float(w.pvalue) if idea == "bell_qnn" else 1.0) for idea in FAMILY}
    p_adj = dict(zip(FAMILY, holm([p_fam[i] for i in FAMILY])))
    verdict = "PASS" if p_adj["bell_qnn"] < 0.05 else "KILL"

    res = {
        "idea": "bell_qnn", "endpoint": "RMSE tr(rho_A^2), bias-corrected, C = 400 equal copies",
        "n_semillas": len(rows), "semillas": list(SEEDS),
        "media": float(d.mean()), "sd": float(d.std(ddof=1)), "negativos": int((d < 0).sum()),
        "p_wilcoxon": float(w.pvalue), "p_t": float(t.pvalue),
        "efecto_sd": float(d.mean() / d.std(ddof=1)),
        "familia": list(FAMILY), "p_familia": p_fam, "p_holm": p_adj, "verdict": verdict,
        "filas": rows,
    }
    json.dump(res, open(f"{RAIZ}/CAMPAIGN.json", "w"), indent=1)

    lineas = ["# Idea-4 campaign `bell_qnn` - ten official seeds", "",
              "Primary endpoint: RMSE of tr(rho_A^2) with bias correction, C = 400 equal copies, "
              "n = 8, |A| = 4, N = 400 states (240/160).", "",
              "| seed | ours (RMSE) | q_baseline | delta | Bell classif. | shadow classif. |",
              "|---|---|---|---|---|---|"]
    for f in rows:
        cb = "—" if f["clas_bell"] is None else f"{f['clas_bell']:.4f}"
        cs = "\u2014" if f["clas_shadows"] is None else f"{f['clas_shadows']:.4f}"
        lineas.append(f"| {f['seed']} | {f['ours']:.4f} | {f['q_base']:.4f} | {f['delta']:+.4f} | "
                      f"{cb} | {cs} |")
    lineas += ["",
               f"- Mean delta **{d.mean():+.4f}** (sd {d.std(ddof=1):.4f}), "
               f"{int((d < 0).sum())}/10 negative, effect size {d.mean() / d.std(ddof=1):.1f} sd",
               f"- Exact one-sided Wilcoxon: p = {w.pvalue:.5f} (smallest attainable 1/1024 = 0.00098)",
               f"- Paired t test (sensitivity): p = {t.pvalue:.5f}",
               f"- Holm over the family {FAMILY}: adjusted p = {p_adj['bell_qnn']:.5f}",
               f"- **Verdict: {verdict}**", ""]
    open(f"{RAIZ}/CAMPAIGN.md", "w").write("\n".join(lineas))
    print("\n".join(lineas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
