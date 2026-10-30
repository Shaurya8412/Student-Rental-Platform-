"""What-if recalculation: change budget or distance, re-run the minimum.

Only the agent whose inputs changed is re-run (budget ceiling -> budget
agent, max distance -> location agent); the safety agent's scores never
depend on the profile and are reused, as are unaffected agents' scores
when the caller passes ``cached_scores``. TOPSIS always re-runs (it is
pure numpy and effectively free). Every recalculation is timed and the
elapsed seconds are appended to the what-if log.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .agents import Profile, budget_agent, location_agent, safety_agent
from .decision_agent import build_decision_matrix
from .topsis import ahp_from_importance, topsis


@dataclass
class WhatIfResult:
    """Outcome of one what-if recalculation."""

    ranking: pd.DataFrame  # listing_id, topsis_score, rank
    elapsed_seconds: float
    reran_agents: list
    weights: dict
    consistency_ratio: float
    profile: Profile = field(default=None)


def _log(elapsed: float, changes: dict, reran: list,
         log_path: Path | None) -> None:
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(["timestamp_utc", "changes", "reran_agents",
                                 "elapsed_seconds"])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                json.dumps(changes, default=str),
                ";".join(reran), f"{elapsed:.6f}"])
    except OSError:
        pass  # logging must never break the recalculation


def recalculate(listings: pd.DataFrame, profile: Profile, changes: dict,
                cached_scores: dict | None = None,
                stay_months: int = config.STAY_MONTHS,
                log_path: Path | None = config.WHATIF_LOG_PATH
                ) -> WhatIfResult:
    """Re-rank after applying ``changes`` to ``profile``.

    ``changes`` may contain ``budget_ceiling``, ``max_distance_km``
    and/or ``weights``. ``cached_scores`` maps agent name -> score dict
    from a previous full run; agents absent from it (or affected by the
    change) are recomputed.
    """
    start = time.perf_counter()
    cached_scores = cached_scores or {}
    new_profile = Profile(
        budget_ceiling=float(changes.get("budget_ceiling",
                                         profile.budget_ceiling)),
        max_distance_km=float(changes.get("max_distance_km",
                                          profile.max_distance_km)),
        weights=dict(changes.get("weights", profile.weights)))

    reran: list[str] = []

    def _scores(agent_name: str, compute, affected: bool) -> dict:
        if not affected and agent_name in cached_scores:
            return cached_scores[agent_name]
        reran.append(agent_name)
        return compute().scores

    class _View:  # minimal AgentResult stand-in for build_decision_matrix
        def __init__(self, scores):
            self.scores = scores

    budget_scores = _scores(
        "budget", lambda: budget_agent(listings, new_profile, stay_months),
        affected="budget_ceiling" in changes)
    location_scores = _scores(
        "location", lambda: location_agent(listings, new_profile),
        affected="max_distance_km" in changes)
    safety_scores = _scores(
        "safety", lambda: safety_agent(listings), affected=False)

    agent_views = {"budget": _View(budget_scores),
                   "location": _View(location_scores),
                   "safety": _View(safety_scores)}
    weights, cr = ahp_from_importance(new_profile.weights)
    ids, matrix = build_decision_matrix(listings, agent_views)
    closeness = topsis(matrix, [weights[c] for c in config.CRITERIA])

    # Same safety-floor screening as the Decision Agent: high-risk
    # listings sort below every recommendable one.
    high_risk = [safety_scores[lid] < config.SAFETY_FLOOR for lid in ids]
    ranking = pd.DataFrame({"listing_id": ids,
                            "topsis_score": closeness,
                            "high_risk": high_risk}).sort_values(
        ["high_risk", "topsis_score", "listing_id"],
        ascending=[True, False, True]).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))

    elapsed = time.perf_counter() - start
    _log(elapsed, changes, reran, log_path)
    return WhatIfResult(ranking=ranking, elapsed_seconds=elapsed,
                        reran_agents=reran, weights=weights,
                        consistency_ratio=cr, profile=new_profile)
