"""Legacy compatibility wrapper for Four-Rooms figure generation.

This script now delegates to ``make_figures.py`` and writes the current report
figures from the CSVs stored in ``final_results``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from exp1_four_rooms import make_figures as MF


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "results" / "four_rooms" / "final_results"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "results" / "four_rooms" / "figures"
)


def parse_args() -> argparse.Namespace:
    """Define the CLI for the legacy Four-Rooms plotting wrapper."""
    parser = argparse.ArgumentParser(
        description="Legacy wrapper around exp1_four_rooms.make_figures."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Rebuild the current Four-Rooms figures from final_results CSVs."""
    args = parse_args()
    MF.make_all_figures(args.results_dir, args.output_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
