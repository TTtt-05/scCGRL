# Software versions

## Main scCGRL environment

The verified main environment uses Python 3.10.19, Scanpy 1.11.4, AnnData
0.10.9, NumPy 1.26.4, SciPy 1.15.3, pandas 2.3.3, scikit-learn 1.7.2,
psutil 7.1.3, NetworkX 3.4.2, matplotlib 3.10.7, seaborn 0.13.2,
python-igraph 1.0.0, leidenalg 0.11.0, and adjustText 1.3.0. These versions
are pinned in `environment.yml` and `requirements.txt`.

## Comparison-method environments

| Method | Audited runtime/package |
|---|---|
| PAGA | Scanpy 1.11.5 |
| DPT | Scanpy 1.11.5 |
| Palantir | Palantir 1.4.3 |
| Slingshot | R 4.6.1; Slingshot 2.20.0; source `b0935a945665dff147cf2d5e40090f4110c4e9e0` |
| TSCAN | R 4.6.1; TSCAN 1.50.0; source `f7afcb41e89c8011585a75114446c0da75e2f45b` |
| SLICER | R 4.6.1; SLICER 0.2.0; source `cb1be8ac788f7976bd073553ca8633cc4898c1a2` |
| SCORPIUS | R 4.6.1; SCORPIUS 1.0.10; source `cbb886463e2b82bafbdfe210013bc27d6e61e79f` |
| Monocle 1 | R 3.1.3; Bioconductor 3.0; monocle 1.0.0; HSMMSingleCell 1.0.0; Biobase 2.26.0; VGAM 1.0-1; igraph 1.0.1; irlba 1.0.3 |
| Monocle 2 | R 4.4.3; monocle 2.40.0; igraph 2.0.3; dplyr 0.8.5; DDRTree 0.1.6; VGAM 1.1-14 |
| Monocle 3 | R 4.6.1; monocle3 1.4.27 |

Environment definitions and the exact modern R package installer are under
`benchmark/environments/`. Per-run manifests are authoritative for the
versions actually loaded during a benchmark run.
