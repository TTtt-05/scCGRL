# Exact audited Python-to-R bridge.
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import psutil
from scipy.io import mmwrite

from .._config import (
    CONDA_EXECUTABLE,
    CURRENT_R_LIBRARY,
    MONOCLE1_RSCRIPT,
    MONOCLE1_R_LIBRARY,
    MONOCLE1_RUNNER,
    MONOCLE2_ENV,
    MONOCLE2_R_LIBRARY,
    MONOCLE2_LEGACY_DPLYR_LIBRARY,
    MONOCLE3_R_LIBRARY,
    R_RUNNER,
    SYSTEM_RSCRIPT,
)


def _write_exchange(baseline_input, directory):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cell_id": baseline_input["cell_ids"]}).to_csv(
        directory / "cell_metadata.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame({"gene_id": baseline_input["gene_ids"]}).to_csv(
        directory / "feature_metadata.csv", index=False, encoding="utf-8"
    )
    matrix = baseline_input["expression"]
    matrix = matrix if sp.issparse(matrix) else sp.csr_matrix(np.asarray(matrix))
    mmwrite(directory / "expression.mtx", matrix)
    (directory / "expression_is_counts.txt").write_text(
        "true\n" if baseline_input["expression_is_counts"] else "false\n", encoding="utf-8"
    )


def run_r_method(method, baseline_input, seed, root_cell_id, method_dir, timeout=None):
    method_dir.mkdir(parents=True, exist_ok=True)
    if method == "Monocle1":
        if not MONOCLE1_RSCRIPT.exists() or not MONOCLE1_R_LIBRARY.exists():
            raise FileNotFoundError(
                "The isolated R 3.1.3 / Bioconductor 3.0 Monocle1 environment is missing"
            )
    elif not SYSTEM_RSCRIPT.exists():
        raise FileNotFoundError(SYSTEM_RSCRIPT)
    bridge_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"{method}_{seed}_", dir=method_dir) as temporary:
        temporary = Path(temporary)
        exchange = temporary / "input"
        output_csv = temporary / "output.csv"
        lineage_csv = temporary / "lineage.csv"
        _write_exchange(baseline_input, exchange)
        exchange_seconds = time.perf_counter() - bridge_started
        if method == "Monocle1":
            command = [
                str(MONOCLE1_RSCRIPT), str(MONOCLE1_RUNNER), str(exchange),
                str(output_csv), str(lineage_csv), str(int(seed)),
                str(root_cell_id), str(MONOCLE1_R_LIBRARY),
            ]
            r_library = MONOCLE1_R_LIBRARY
            r_executable = str(MONOCLE1_RSCRIPT)
        elif method == "Monocle2":
            if not CONDA_EXECUTABLE.exists() or not MONOCLE2_R_LIBRARY.exists():
                raise FileNotFoundError("The isolated Monocle2 environment is missing")
            command = [
                str(CONDA_EXECUTABLE), "run", "-p", str(MONOCLE2_ENV), "Rscript",
                str(R_RUNNER), method, str(exchange), str(output_csv), str(lineage_csv),
                str(int(seed)), str(root_cell_id), str(MONOCLE2_R_LIBRARY),
            ]
            r_library = MONOCLE2_R_LIBRARY
            r_executable = f"conda run -p {MONOCLE2_ENV} Rscript"
        else:
            r_library = MONOCLE3_R_LIBRARY if method == "Monocle3" else CURRENT_R_LIBRARY
            command = [
                str(SYSTEM_RSCRIPT), str(R_RUNNER), method, str(exchange), str(output_csv),
                str(lineage_csv), str(int(seed)), str(root_cell_id), str(r_library),
            ]
            r_executable = str(SYSTEM_RSCRIPT)
        environment = os.environ.copy()
        environment["SCCGRL_MONOCLE2_DPLYR_LIBRARY"] = str(
            MONOCLE2_LEGACY_DPLYR_LIBRARY
        )
        environment["SCCGRL_SHARED_R_LIBRARY"] = str(CURRENT_R_LIBRARY)
        # Monocle3 itself is isolated while its version-matched dependencies
        # remain in the established R 4.6 benchmark library. Monocle2 is fully
        # contained in the activated R 4.4 conda environment.
        environment["R_LIBS_USER"] = (
            str(CURRENT_R_LIBRARY) if method == "Monocle3" else str(r_library)
        )
        started = time.perf_counter()
        process = subprocess.Popen(
            command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            try:
                parent = psutil.Process(process.pid)
                descendants = parent.children(recursive=True)
                for child in descendants:
                    child.kill()
                parent.kill()
                psutil.wait_procs(descendants + [parent], timeout=10)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process.kill()
            stdout, stderr = process.communicate()
            log_path = method_dir / f"error_seed_{seed}.log"
            log_path.write_text(
                f"command={json.dumps(command)}\nreturncode=TIMEOUT\n"
                f"timeout_seconds={timeout}\n\nSTDOUT\n{stdout}\n\nSTDERR\n{stderr}\n",
                encoding="utf-8",
            )
            raise TimeoutError(
                f"{method} exceeded the requested per-run limit of {timeout} seconds"
            ) from error
        completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        elapsed = time.perf_counter() - started
        log_path = method_dir / f"error_seed_{seed}.log"
        log_path.write_text(
            f"command={json.dumps(command)}\nreturncode={completed.returncode}\n\n"
            f"STDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}\n",
            encoding="utf-8",
        )
        if completed.returncode != 0 or not output_csv.exists():
            message = completed.stderr or completed.stdout or "R method produced no output"
            raise RuntimeError(message[-6000:])
        output = pd.read_csv(output_csv)
        order = pd.Index(output["cell_id"].astype(str)).get_indexer(baseline_input["cell_ids"])
        if np.any(order < 0):
            raise ValueError("R output is missing input cells")
        output = output.iloc[order].reset_index(drop=True)
        lineage = {}
        if lineage_csv.exists():
            lineage_frame = pd.read_csv(lineage_csv)
            lineage_frame = lineage_frame.set_index("cell_id").reindex(baseline_input["cell_ids"])
            lineage = {column: lineage_frame[column].to_numpy(float) for column in lineage_frame}
        r_preprocessing = pd.to_numeric(
            output.get("method_preprocessing_seconds", pd.Series([np.nan])), errors="coerce"
        ).dropna()
        r_inference = pd.to_numeric(
            output.get("method_inference_seconds", pd.Series([np.nan])), errors="coerce"
        ).dropna()
        preprocessing_seconds = exchange_seconds + (
            float(r_preprocessing.iloc[0]) if len(r_preprocessing) else 0.0
        )
        inference_seconds = (
            float(r_inference.iloc[0]) if len(r_inference) else max(0.0, elapsed - preprocessing_seconds)
        )
        return {
            "pseudotime": output["pseudotime"].to_numpy(float),
            "branches": output["branch"].astype(str).to_numpy(),
            "lineage_pseudotime": lineage,
            "preprocessing_seconds": preprocessing_seconds,
            "inference_seconds": inference_seconds,
            "parameters": {
                "method_details": str(output["method_details"].dropna().iloc[0])
                if output["method_details"].notna().any() else "",
                "R_executable": r_executable,
                "R_library": str(r_library),
                "terminal_information_supplied": False,
            },
            "has_native_branches": method not in {"SCORPIUS"},
            "has_native_topology": method not in {"SCORPIUS"},
        }
