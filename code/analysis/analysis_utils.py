"""Small helpers shared by experiment-specific abstraction analysis scripts."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from core import abstraction as AB


Array = np.ndarray


def fit_soft_abstraction(
    metric: Array,
    mu: Array,
    beta: float,
    num_abstract: int,
    *,
    max_outer: int = 30,
    max_inner: int | None = None,
    inner_tol: float | None = None,
):
    """Fit one BA soft abstraction for a fixed beta."""
    kwargs: dict[str, object] = {
        "beta": beta,
        "num_abstract": num_abstract,
        "max_outer": max_outer,
    }
    if max_inner is not None:
        kwargs["max_inner"] = max_inner
    if inner_tol is not None:
        kwargs["inner_tol"] = inner_tol
    return AB.fit_soft_abstraction(metric, mu, **kwargs)


def hard_cluster_members(encoder: Array) -> tuple[Array, dict[int, list[int]]]:
    """Convert a soft encoder into argmax assignments and cluster member lists."""
    hard = np.argmax(np.asarray(encoder, dtype=float), axis=1)
    members: dict[int, list[int]] = defaultdict(list)
    for item_index, cluster_id in enumerate(hard.tolist()):
        members[int(cluster_id)].append(item_index)
    return hard, dict(members)


def write_csv_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write a non-empty list of dictionaries to a CSV file."""
    if not rows:
        raise ValueError(f"Cannot write empty CSV rows to {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_section(title: str) -> None:
    """Print a visible section heading for CLI analysis summaries."""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def purity(values: Sequence[object]) -> tuple[object, float, int]:
    """Return dominant label, its purity, and the number of distinct labels."""
    counts = Counter(values)
    dominant, size = counts.most_common(1)[0]
    return dominant, float(size) / float(len(values)), len(counts)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Compute a weighted average from Python sequences."""
    return float(
        np.average(
            np.asarray(values, dtype=float),
            weights=np.asarray(weights, dtype=float),
        )
    )
