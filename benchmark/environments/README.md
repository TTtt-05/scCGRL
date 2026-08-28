# Benchmark environments

The comparison methods are isolated from the main scCGRL environment because
their R and Bioconductor requirements are incompatible.

| Environment | Methods | Creation |
|---|---|---|
| `benchmark_python.yml` | PAGA, DPT, Palantir | `conda env create -f benchmark/environments/benchmark_python.yml` |
| `modern_r_baselines.yml` | Slingshot, TSCAN, SLICER, SCORPIUS | Create the YAML environment, activate it, then run `Rscript benchmark/environments/install_modern_r_baselines.R`. |
| `monocle1_r.yml` | Monocle 1 | Historical R 3.1/Bioconductor 3.0 specification; Linux is required. Follow the comments in the file. |
| `monocle2_r.yml` | Monocle 2 | R 4.4.3 and pinned legacy dependencies. |
| `monocle3_r.yml` | Monocle 3 | R 4.6.1 and monocle3 1.4.27 from the official source. |

The modern R installer uses the exact audited upstream source revisions and
checks the resulting package versions. Every benchmark run also writes the
actual runtime package versions, parameters, root-use mode, status, runtime,
and memory to its audit outputs.
