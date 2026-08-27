#!/usr/bin/env python
"""Fail-fast verification gate for the public scCGRL release.

The program is read-only. It checks required reproducibility resources, exact
data SHA256 values, processed-data structure, relative-path portability, Git
identity, and Git LFS configuration. It does not run trajectory inference.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "DATA_LICENSES.md",
    "CITATION.cff",
    "environment.yml",
    "requirements.txt",
    "run_sccgrl.py",
    "src/sccgrl/trajectory.py",
    "benchmark/run_benchmark.py",
    "benchmark/common/evaluation.py",
    "figures/human_myeloid/make_figures_human_myeloid.py",
    "figures/mouse_pancreas/make_figures_mouse_pancreas.py",
    "figures/human_bone_marrow/make_figures_human_bone_marrow.py",
    "figures/simulation_2/make_figures_simulation_2.py",
    "figures/simulation_3/make_figures_simulation_3.py",
    "docs/software_versions.md",
    "docs/data_availability.md",
    "reproducibility/data/verify_raw_inputs.py",
]

BASELINE_FILES = [
    "benchmark/paga.py",
    "benchmark/dpt.py",
    "benchmark/palantir.py",
    "benchmark/slingshot.R",
    "benchmark/monocle1.R",
    "benchmark/monocle2.R",
    "benchmark/monocle3.R",
    "benchmark/scorpius.R",
    "benchmark/tscan.R",
    "benchmark/slicer.R",
]

EXPECTED_PROCESSED = {
    "human_myeloid.h5ad": ((3264, 2000), "cluster"),
    "mouse_pancreas.h5ad": ((2780, 2000), "clusters_fig6_broad_final"),
    "human_bone_marrow.h5ad": ((7225, 2000), "celltype"),
    "simulation_2.h5ad": ((2000, 1000), "branch"),
    "simulation_3.h5ad": ((3000, 1000), "branch"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checksum_file(path: Path, base: Path) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = re.split(r"\s+", line.strip(), maxsplit=1)
        records.append((base / relative, expected.lower()))
    return records


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def main() -> int:
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    missing = [name for name in REQUIRED_FILES + BASELINE_FILES if not (ROOT / name).is_file()]
    record("required_reproducibility_files", not missing, "missing=" + repr(missing))

    checksum_specs = [
        (ROOT / "data/checksums.sha256", ROOT / "data"),
        (
            ROOT / "data/processed/processed_inputs_checksums.sha256",
            ROOT / "data/processed",
        ),
    ]
    checksum_failures: list[str] = []
    for checksum_file, base in checksum_specs:
        for file_path, expected in parse_checksum_file(checksum_file, base):
            if not file_path.is_file():
                checksum_failures.append(f"missing:{file_path.relative_to(ROOT)}")
            else:
                observed = sha256(file_path)
                if observed != expected:
                    checksum_failures.append(
                        f"mismatch:{file_path.relative_to(ROOT)}:{observed}"
                    )
    record("data_sha256", not checksum_failures, repr(checksum_failures))

    try:
        import anndata as ad

        structure_failures: list[str] = []
        for filename, (shape, label_column) in EXPECTED_PROCESSED.items():
            adata = ad.read_h5ad(ROOT / "data/processed" / filename, backed="r")
            if tuple(adata.shape) != tuple(shape):
                structure_failures.append(f"{filename}:shape={tuple(adata.shape)}")
            if label_column not in adata.obs.columns:
                structure_failures.append(f"{filename}:missing_obs={label_column}")
            if "X_pca" not in adata.obsm or adata.obsm["X_pca"].shape[1] != 50:
                structure_failures.append(f"{filename}:invalid_X_pca")
            if "X_umap" not in adata.obsm or adata.obsm["X_umap"].shape[1] != 3:
                structure_failures.append(f"{filename}:invalid_X_umap")
            if "connectivities" not in adata.obsp or "distances" not in adata.obsp:
                structure_failures.append(f"{filename}:missing_neighbor_graph")
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        record("processed_h5ad_structure", not structure_failures, repr(structure_failures))
    except Exception as exc:  # verification should fail, not silently skip
        record("processed_h5ad_structure", False, f"{type(exc).__name__}: {exc}")

    text_roots = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "docs", ROOT / "data"]
    absolute_hits: list[str] = []
    for item in text_roots:
        paths = [item] if item.is_file() else list(item.rglob("*.md"))
        for path in paths:
            content = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"(?i)[A-Z]:[\\/]", content):
                absolute_hits.append(str(path.relative_to(ROOT)))
    record("portable_documentation_paths", not absolute_hits, repr(absolute_hits))

    try:
        commit = git_output("rev-parse", "HEAD")
        record("git_commit", bool(re.fullmatch(r"[0-9a-f]{40}", commit)), commit)
    except Exception as exc:
        record("git_commit", False, f"{type(exc).__name__}: {exc}")

    try:
        lfs = subprocess.check_output(
            ["git", "lfs", "version"], text=True, stderr=subprocess.STDOUT
        ).strip()
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        configured = "*.h5ad filter=lfs" in attributes and "*.pkl filter=lfs" in attributes
        record("git_lfs", configured, lfs)
    except Exception as exc:
        record("git_lfs", False, f"{type(exc).__name__}: {exc}")

    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "repository": str(ROOT),
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
