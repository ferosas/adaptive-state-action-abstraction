"""Experiment-side result writing and runner helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence


def ensure_output_dir(path: Path) -> None:
    """Create an output directory and any missing parents."""
    path.mkdir(parents=True, exist_ok=True)


def save_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write a heterogeneous list of dictionaries to CSV."""
    ensure_output_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Write a JSON payload with stable pretty-print formatting."""
    ensure_output_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_beta_schedule(beta_schedule: str) -> list[float]:
    """Parse a comma-separated beta schedule string into floats."""
    values = [item.strip() for item in beta_schedule.split(",")]
    parsed = [float(item) for item in values if item]
    if not parsed:
        raise ValueError("beta_schedule must contain at least one numeric value.")
    return parsed




def build_dense_beta_schedule(
    beta_schedule: Sequence[float],
    adaptive_beta_step: float,
) -> list[float] | None:
    """Build an optional dense adaptive beta ladder from the fixed schedule."""
    if adaptive_beta_step <= 0.0:
        return None
    min_beta, max_beta = min(beta_schedule), max(beta_schedule)
    values = []
    b = min_beta
    while b < max_beta - 1e-9:
        values.append(round(b, 12))
        b += adaptive_beta_step
    values.append(round(max_beta, 12))
    return sorted(set(values))


def resolve_output_dir(output_dir: Path | None, default_output_dir: Path) -> Path:
    """Return the requested output directory or a scenario-specific default."""
    return default_output_dir if output_dir is None else output_dir
