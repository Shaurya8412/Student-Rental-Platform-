"""Generate SAMPLE evaluation inputs (student rankings + survey).

These are illustrative sample data so the evaluation harness runs
end-to-end; replace with real collected responses for an actual study
(see the evaluation plan in the literature review, Section VII).

- student_rankings.csv: 12 synthetic students each rank their top 10
  listings. Following the paper's method, each student scores the same
  five criteria the system uses (budget, distance, safety, wifi, food)
  with their *own* budget ceiling, distance limit and importance
  weights, plus judgement noise - so agreement with the system is
  realistic rather than perfect.
- survey.csv: 25 synthetic respondents rating explanation clarity,
  trust and usefulness (1-5) and whether the stated reason matched
  what mattered to them (yes/no).

Run:  python evaluation/generate_sample_eval_data.py [--dataset PATH]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.agents import Profile, budget_agent, location_agent, safety_agent  # noqa: E402
from src.topsis import ahp_from_importance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(config.DATA_PATH))
    args = parser.parse_args()

    rng = random.Random(7)
    df = pd.read_csv(args.dataset)
    safety = safety_agent(df).scores
    by_id = df.set_index("id")

    ranking_rows = []
    for s in range(1, 13):
        profile = Profile(
            budget_ceiling=rng.uniform(8000, 14000),
            max_distance_km=rng.uniform(4.0, 8.0),
            weights={c: 3 for c in config.CRITERIA})
        budget = budget_agent(df, profile).scores
        location = location_agent(df, profile).scores
        # Students state importance 1-5 exactly like the app profile does;
        # per the preference studies, budget/distance/safety dominate.
        importance = {
            "budget": rng.choice([4, 5, 5]),
            "distance": rng.choice([3, 4, 4, 5]),
            "safety": rng.choice([4, 5, 5]),
            "wifi": rng.choice([2, 3, 3, 4]),
            "food": rng.choice([1, 2, 2, 3]),
        }
        w, _ = ahp_from_importance(importance)
        utility = {}
        for lid in df["id"]:
            utility[lid] = (
                w["budget"] * budget[lid]
                + w["distance"] * location[lid]
                + w["safety"] * safety[lid]
                + w["wifi"] * float(by_id.loc[lid, "wifi"])
                + w["food"] * float(by_id.loc[lid, "food_included"])
                + rng.gauss(0, 0.04))
        top10 = sorted(utility, key=utility.get, reverse=True)[:10]
        for rank, lid in enumerate(top10, start=1):
            ranking_rows.append({"student_id": f"S{s:02d}",
                                 "listing_id": int(lid), "rank": rank})
    out_dir = Path(__file__).resolve().parent
    pd.DataFrame(ranking_rows).to_csv(out_dir / "student_rankings.csv",
                                      index=False)

    survey_rows = []
    for r in range(1, 26):
        survey_rows.append({
            "respondent_id": f"R{r:02d}",
            "clarity": rng.choice([3, 4, 4, 5, 5]),
            "trust": rng.choice([3, 3, 4, 4, 5]),
            "usefulness": rng.choice([3, 4, 4, 4, 5]),
            "reason_matched": "yes" if rng.random() < 0.76 else "no",
        })
    pd.DataFrame(survey_rows).to_csv(out_dir / "survey.csv", index=False)
    print(f"Wrote {len(ranking_rows)} ranking rows (12 students) and "
          f"{len(survey_rows)} survey rows for dataset {args.dataset}")


if __name__ == "__main__":
    main()
