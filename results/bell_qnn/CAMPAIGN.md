# Idea-4 campaign `bell_qnn` - ten official seeds

Primary endpoint: RMSE of tr(rho_A^2) with bias correction, C = 400 equal copies, n = 8, |A| = 4, N = 400 states (240/160).

| seed | ours (RMSE) | q_baseline | delta | Bell classif. | shadow classif. |
|---|---|---|---|---|---|
| 0 | 0.0510 | 0.1946 | -0.1437 | 0.0688 | 0.1688 |
| 1 | 0.0501 | 0.2011 | -0.1509 | 0.0500 | 0.1313 |
| 7 | 0.0521 | 0.1931 | -0.1411 | 0.1125 | 0.1688 |
| 42 | 0.0505 | 0.1942 | -0.1437 | 0.0813 | 0.1250 |
| 123 | 0.0530 | 0.1924 | -0.1394 | 0.0563 | 0.1688 |
| 456 | 0.0526 | 0.1966 | -0.1440 | 0.0813 | 0.1812 |
| 789 | 0.0491 | 0.2007 | -0.1516 | 0.0500 | 0.1187 |
| 1234 | 0.0498 | 0.1797 | -0.1299 | 0.1000 | 0.1812 |
| 2024 | 0.0509 | 0.1943 | -0.1434 | 0.0750 | 0.1437 |
| 3407 | 0.0518 | 0.1907 | -0.1389 | 0.0688 | 0.0938 |

- Mean delta **-0.1426** (sd 0.0062), 10/10 negative, effect size -23.1 sd
- Exact one-sided Wilcoxon: p = 0.00098 (smallest attainable 1/1024 = 0.00098)
- Paired t test (sensitivity): p = 0.00000
- Holm over the family ('qcnn_tolerant', 'bell_qnn'): adjusted p = 0.00195
- **Verdict: PASS**
