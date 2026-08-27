# Changelog

## Release 27 - 2026-08-27

- Created a focused portable package containing the core scCGRL model, five
  dataset configurations, verified inputs, ten baseline methods, repeated main
  runs, publication figures, tables, and reproducibility utilities.
- Packaged the verified raw and processed data without changing matrices or
  checksums, and documented each source-specific redistribution boundary.
- Documented the bundled raw counts, processed labels, PCA50, neighbor graphs,
  UMAP3, and the human-bone-marrow ETP/BcellPre exclusion.
- Removed non-core experimental modules and their generated outputs from this
  release scope.
- Removed the dedicated quick-test program and its generated output.
- Updated the README, data-availability statement, result map, and
  reproducibility map to describe only files present in release 27.
- Pinned the main environment to the versions used by the successful seed-42
  validation run and added an upload-gate verification program.
- Added complete software/manuscript citation metadata and an MIT license for
  original scCGRL code; third-party data retain their own terms.
