"""Audited shared-root policy for the official baseline benchmark.

The existing benchmark reads the same-seed scCGRL start cell from the stored
50-run CSV.  It never provides terminal identities, terminal counts, branch
labels, or true pseudotime to a baseline method.
"""

import numpy as np

from .._config import ROOT_USAGE
from .preprocessing import build_root_anchor


def locate_root(cell_ids, root_cell_id):
    """Return the exact position of the shared root in aligned baseline cells."""
    matches = np.flatnonzero(np.asarray(cell_ids).astype(str) == str(root_cell_id))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one aligned root {root_cell_id!r}; found {len(matches)}"
        )
    return int(matches[0])


__all__ = ["ROOT_USAGE", "build_root_anchor", "locate_root"]
