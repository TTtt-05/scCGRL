# Software versions

The successful release-validation run reports Windows 10, Python 3.10.19,
Scanpy 1.11.4, AnnData 0.10.9, NumPy 1.26.4, SciPy 1.15.3, pandas 2.3.3,
scikit-learn 1.7.2, psutil 7.1.3, NetworkX 3.4.2, matplotlib 3.10.7,
seaborn 0.13.2, python-igraph 1.0.0, leidenalg 0.11.0, and adjustText 1.3.0.

| Method environment | Key versions |
|---|---|
| Main scCGRL | Python 3.10; exact validated versions pinned in `environment.yml` and `requirements.txt` |
| Python benchmark | Separate recommended environment in `benchmark/environments/benchmark_python.yml`; per-run manifests record the versions actually used |
| Monocle1 | R 3.1.3, Bioconductor 3.0, monocle 1.0.0, HSMMSingleCell 1.0.0, Biobase 2.26.0, VGAM 1.0-1, igraph 1.0.1, irlba 1.0.3 |
| Monocle2 | R 4.4.3, monocle 2.40.0, igraph 2.0.3, dplyr 0.8.5, DDRTree 0.1.6, VGAM 1.1-14 |
| Monocle3 | R 4.6.1, monocle3 1.4.27 |
| Slingshot/TSCAN | R 4.4 family; Slingshot 2.20.0, TSCAN 1.50.0 |
| SLICER/SCORPIUS | SLICER 0.2.0, SCORPIUS 1.0.10 |

The main environment reflects the versions used for the validated seed-42 run;
the benchmark environments intentionally remain isolated because the baseline
packages have different compatibility requirements. Per-run manifests are the
reporting standard. The listed legacy R versions do not imply that those
packages install in an arbitrary modern R environment.
