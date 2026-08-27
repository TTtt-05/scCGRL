"""Shared, transparent long-format summaries for repository result tables."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

IDENTIFIER_COLUMNS = {
    "dataset", "method", "variant", "condition", "status", "error", "error_message",
    "seed", "random_seed", "run", "run_index", "cell_id", "start_index",
    "endpoint_indices", "input_path", "metric", "reference_metric", "value_column",
}


def summarize_numeric(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Return n/mean/median/SD/quartiles/range and Student-t 95% CI."""
    rows = []
    groups = frame.groupby(group_columns, dropna=False, observed=True) if group_columns else [((), frame)]
    for keys, block in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        if "status" in block:
            block = block[block["status"].astype(str).str.lower().eq("success")]
        for column in block.columns:
            if column in set(group_columns) | IDENTIFIER_COLUMNS:
                continue
            values = pd.to_numeric(block[column], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            n = int(len(values)); mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
            margin = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else 0.0
            rows.append({
                **identity, "metric": column, "n": n, "mean": mean,
                "median": float(np.median(values)), "standard_deviation": sd,
                "q1": float(np.percentile(values, 25)), "q3": float(np.percentile(values, 75)),
                "minimum": float(np.min(values)), "maximum": float(np.max(values)),
                "confidence_interval_95_lower": mean - margin,
                "confidence_interval_95_upper": mean + margin,
            })
    return pd.DataFrame(rows)


def read_csvs(paths) -> pd.DataFrame:
    frames = [pd.read_csv(Path(path), encoding="utf-8-sig") for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
