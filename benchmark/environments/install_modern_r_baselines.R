#!/usr/bin/env Rscript

# Install the exact source revisions audited for the modern R baselines.
# This script is deliberately separate from the legacy Monocle environments.

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

BiocManager::install(
  c("SingleCellExperiment", "SummarizedExperiment", "TrajectoryUtils"),
  ask = FALSE,
  update = FALSE
)

packages <- c(
  "bioc/slingshot@b0935a945665dff147cf2d5e40090f4110c4e9e0",
  "bioc/TSCAN@f7afcb41e89c8011585a75114446c0da75e2f45b",
  "jw156605/SLICER@cb1be8ac788f7976bd073553ca8633cc4898c1a2",
  "rcannood/SCORPIUS@cbb886463e2b82bafbdfe210013bc27d6e61e79f"
)

for (package in packages) {
  remotes::install_github(package, upgrade = "never", dependencies = TRUE)
}

expected <- c(
  slingshot = "2.20.0",
  TSCAN = "1.50.0",
  SLICER = "0.2.0",
  SCORPIUS = "1.0.10"
)

observed <- vapply(names(expected), function(package) {
  as.character(utils::packageVersion(package))
}, character(1))

if (!identical(unname(observed), unname(expected))) {
  stop(
    "Installed package versions do not match the audited versions: ",
    paste(names(observed), observed, sep = "=", collapse = ", ")
  )
}

writeLines(capture.output(sessionInfo()), "modern_r_baselines_sessionInfo.txt")
