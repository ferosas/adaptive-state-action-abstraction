"""Cluster analysis for the soft BA abstractions driven by the
fixed-point pair metric on Four-Rooms.

This script rebuilds the same fixed-point pair metric used by the main
experiment and then studies the resulting abstract clusters. Mirror partners
related by vertical reflection + goal swap + (UP<->DOWN) have zero pairwise
distortion under that metric and can therefore be identified by the encoder.

For each beta:
- Rebuild the soft abstraction with the fixed-point pair distortion.
- Hard-assign states to their argmax abstract.
- Classify clusters by room, goal, and sigma-orbit (agent-in-top-half vs
  mirror partner).  Report cluster-level symmetry purity: for each
  hard cluster, the fraction of its members whose sigma-partner also
  lies in the same cluster (i.e. the cluster is sigma-closed).

Outputs a CSV and a per-beta summary to reports/experimentA/default_run/.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from core import analysis_utils as AU
from exp1_four_rooms import four_rooms_mdp as FRM


REPORT_DIR = THIS_DIR.parent.parent / "reports" / "experimentA" / "default_run"

BETAS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
ETA = 0.10
GAMMA = 0.99


def main() -> None:
    """Rebuild fixed-point Four-Rooms abstractions and summarize symmetry-aware clusters."""
    mdp = FRM.build_four_rooms_mdp(eta=ETA, gamma=GAMMA)
    walls = FRM._build_wall_mask()
    cells, _ = FRM._build_cell_index(walls)
    room_center_cells = [cells.index(rc) for rc in FRM.ROOM_CENTERS]
    metric = FRM.load_distortion(mdp, metric_kind="fixed_point", verbose=False)
    sigma = FRM.sigma_permutation(mdp, cells)

    mu = np.full(mdp.num_states, 1.0 / mdp.num_states, dtype=float)

    rows: List[dict[str, object]] = []

    for beta in BETAS:
        abstraction = AU.fit_soft_abstraction(
            metric,
            mu,
            beta=beta,
            num_abstract=mdp.num_states,
            max_outer=30,
        )
        hard, cluster_members = AU.hard_cluster_members(abstraction.encoder)
        unique_clusters = sorted(cluster_members)

        # Sigma-closure: fraction of s whose sigma-partner sits in the same cluster.
        sigma_closed = int(np.sum(hard == hard[sigma]))

        for cid in unique_clusters:
            member_states = cluster_members[cid]
            member_tuples = []
            for state in member_states:
                cell_idx, goal_idx = mdp.state_labels[state]
                row, col = cells[cell_idx]
                member_tuples.append((cell_idx, goal_idx, row, col))
            members_set = set(member_states)
            closed_count = sum(1 for s in member_states if int(sigma[s]) in members_set)
            room_counts = Counter(FRM.room_label(row, col) for (_, _, row, col) in member_tuples)
            goal_counts = Counter(goal_idx for (_, goal_idx, _, _) in member_tuples)
            cells_in_cluster = set(cell_idx for (cell_idx, _, _, _) in member_tuples)
            contains_center = [
                g for g, cc in enumerate(room_center_cells) if cc in cells_in_cluster
            ]
            contains_hallway = any(
                FRM.room_label(row, col) == "hallway" for (_, _, row, col) in member_tuples
            )
            rows.append({
                "beta": beta,
                "cluster_id": cid,
                "size": len(member_tuples),
                "dominant_room": room_counts.most_common(1)[0][0],
                "dominant_goal": goal_counts.most_common(1)[0][0],
                "rooms_present": "+".join(sorted(room_counts.keys())),
                "goals_present": "+".join(str(g) for g in sorted(goal_counts.keys())),
                "room_purity": max(room_counts.values()) / len(member_tuples),
                "goal_purity": max(goal_counts.values()) / len(member_tuples),
                "n_rooms": len(room_counts),
                "n_goals": len(goal_counts),
                "contains_goal_centers": "+".join(str(g) for g in contains_center),
                "contains_hallway": contains_hallway,
                "sigma_closed_fraction": closed_count / len(member_tuples),
            })

        print(
            f"beta={beta}: Z_act={len(unique_clusters)}, hard_K={len(unique_clusters)}, "
            f"sigma-closed states={sigma_closed}/{mdp.num_states}"
        )

    out_csv = REPORT_DIR / "pair_abstraction_cluster_analysis.csv"
    AU.write_csv_rows(out_csv, rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)")

    AU.print_section("Per-beta summary (fixed-point pair-metric abstractions)")
    for beta in BETAS:
        b_rows = [r for r in rows if float(r["beta"]) == beta]
        z_hard = len(b_rows)
        avg_room = float(np.mean([r["room_purity"] for r in b_rows]))
        avg_goal = float(np.mean([r["goal_purity"] for r in b_rows]))
        avg_sigma = float(np.mean([r["sigma_closed_fraction"] for r in b_rows]))
        max_size = int(max(r["size"] for r in b_rows))
        print(
            f"beta={beta:<5}  hard_K={z_hard:<4} max|C|={max_size:<4} "
            f"room_purity={avg_room:.2f}  goal_purity={avg_goal:.2f}  "
            f"sigma_closure={avg_sigma:.2f}"
        )


if __name__ == "__main__":
    main()
