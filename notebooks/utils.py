from __future__ import annotations

import math
import re
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot


TREE_NAME = "Events"
NUMERIC_WORDS = ("bool", "float", "double", "int", "short", "long")


def sample_label(path: Path | str) -> str:
    return Path(path).stem


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")


def is_numeric(typename: str) -> bool:
    typename = typename.lower()
    return "char*" not in typename and "string" not in typename and any(word in typename for word in NUMERIC_WORDS)


def branch_values(tree, branch: str, typenames: dict[str, str], entry_stop: int | None = None) -> tuple[np.ndarray, str]:
    array = tree[branch].array(library="ak", entry_stop=entry_stop)
    typename = typenames[branch]

    if ak.fields(array):
        return np.array([], dtype=float), "record"

    if is_numeric(typename):
        values = np.asarray(ak.to_numpy(ak.drop_none(ak.flatten(array, axis=None))), dtype=float).reshape(-1)
        return values[np.isfinite(values)], "values"

    if typename.startswith("std::vector"):
        values = np.asarray(ak.to_numpy(ak.num(array, axis=1)), dtype=float)
        return values[np.isfinite(values)], "multiplicity"

    return np.array([], dtype=float), "non_numeric"


def hist_bins(values: np.ndarray) -> tuple[int, float, float]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    unique = np.unique(values)

    if np.allclose(values, np.round(values)) and unique.size <= 80:
        return max(1, unique.size), math.floor(lo) - 0.5, math.ceil(hi) + 0.5
    if lo == hi:
        pad = max(abs(lo) * 0.1, 0.5)
        return 40, lo - pad, hi + pad

    mean = float(np.mean(values))
    std = float(np.std(values))
    if std > 0:
        plot_lo = max(lo, mean - 3.0 * std)
        plot_hi = min(hi, mean + 3.0 * std)
        if plot_lo < plot_hi:
            return 80, plot_lo, plot_hi
    return 80, lo, hi


def branch_summary(paths: list[Path], branches: list[str] | None = None, tree_name: str = TREE_NAME) -> pd.DataFrame:
    samples = []
    for path in paths:
        tree = uproot.open(path)[tree_name]
        samples.append((sample_label(path), tree, set(tree.keys()), tree.typenames()))

    if branches is None:
        branches = sorted(set().union(*(keys for _, _, keys, _ in samples)))

    rows = []
    for branch in branches:
        missing = []
        empty = []
        types = set()
        chunks = []

        for label, tree, keys, typenames in samples:
            if branch not in keys:
                missing.append(label)
                continue

            values, status = branch_values(tree, branch, typenames)
            types.add(typenames[branch])
            if values.size:
                chunks.append(values)
            else:
                empty.append(label)

        values = np.concatenate(chunks) if chunks else np.array([])
        rows.append(
            {
                "branch": branch,
                "typename": " | ".join(sorted(types)),
                "missing_files": ", ".join(missing),
                "empty_files": ", ".join(empty),
                "is_numeric": bool(values.size),
                "finite_count": int(values.size),
                "nan_count": 0,
                "min": float(np.min(values)) if values.size else math.nan,
                "max": float(np.max(values)) if values.size else math.nan,
                "mean": float(np.mean(values)) if values.size else math.nan,
            }
        )

    return pd.DataFrame(rows)
