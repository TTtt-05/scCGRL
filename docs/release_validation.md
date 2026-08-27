# Release validation

## Validation identity

- Dataset: `human_myeloid`
- Seed: 42
- Q-learning episodes: 10,000
- Input stage: integer raw counts followed by the configured common preprocessing
- Exact source commit used: `faa228939abe3e36342242fded8a66db10af52d7`
- Output: `results/human_myeloid/seed42/`

The run manifest records the same source commit. The release-validation run was
compared with seed 42 in
`results/repeat_50_runs/human_myeloid/human_myeloid_50_runs.csv`.

## Numerical comparison

The following values matched the formal repeated-run record with absolute
tolerance `1e-12`:

| Quantity | Seed-42 value |
|---|---:|
| K | 20 |
| Number of paths | 2 |
| Trajectory cells | 67 |
| Global Pearson | 0.920967782267866 |
| Global Spearman | 0.924437054224995 |
| Global Kendall | 0.801631906969445 |
| RF Pearson | 0.912416912259440 |
| RF Spearman | 0.878980960006617 |
| RF Kendall | 0.746150024116406 |
| cor_dist | 0.771392488494538 |
| F1 branches | 0.759175436247590 |
| HIM similarity | 0.806099183738997 |
| wcor features | 0.857175418944892 |
| Overall geometric mean | 0.797570231035418 |

## Resource-measurement scope

This validation run recorded preprocessing 98.06 s, inference 110.69 s,
pipeline 208.75 s, trajectory metrics 44.19 s, peak RSS 1,258.36 MB, and RSS
increase 647.35 MB. Other benchmark processes were active on the workstation,
so these timings verify resource-field capture but are not used as the formal
runtime estimate. Formal runtime and memory summaries remain the repeated-run
records under `results/repeat_50_runs/`.
