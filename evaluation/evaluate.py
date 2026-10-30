"""Evaluation harness (literature review, Section VII).

Computes, for a chosen dataset:
  (a) Spearman correlation between student rankings and the system's
      ranking - per student and averaged;
  (b) explanation-quality survey summary (1-5 ratings + yes/no match);
  (c) red-flag checklist precision / recall / F1 against red_flag_label;
  (d) what-if recalculation latency over 50 runs (mean / median / p95).

Outputs: results/results_table.csv, results/evaluation_summary.md and
matplotlib charts in results/.

Run:  python evaluation/evaluate.py [--dataset data/listings_meerut.csv]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.agents import Profile, budget_agent, location_agent, safety_agent  # noqa: E402
from src.decision_agent import build_decision_matrix  # noqa: E402
from src.fee_estimator import enrich_listings  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.topsis import ahp_from_importance, topsis  # noqa: E402
from src.validator import validate_listings  # noqa: E402
from src.whatif import recalculate  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(config.DATA_PATH))
    parser.add_argument("--outdir", default=str(config.RESULTS_DIR))
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    report = validate_listings(df)
    if not report["ok"]:
        raise SystemExit(f"Dataset failed validation: {report['errors']}")

    profile = Profile.default()
    # Use the same starting budget ceiling the app uses for this dataset.
    for name, path in config.DATASETS.items():
        if Path(args.dataset).resolve() == path.resolve():
            profile.budget_ceiling = config.DATASET_DEFAULT_BUDGET[name]
    client = LLMClient(demo_mode=True)  # offline + deterministic
    enriched = enrich_listings(df, client)
    agents = {"budget": budget_agent(enriched, profile),
              "location": location_agent(enriched, profile),
              "safety": safety_agent(enriched)}
    weights, cr = ahp_from_importance(profile.weights)
    ids, matrix = build_decision_matrix(enriched, agents)
    closeness = topsis(matrix, [weights[c] for c in config.CRITERIA])
    closeness_by_id = {lid: float(closeness[k]) for k, lid in enumerate(ids)}
    # System ranking applies the same safety floor as the app: feasible
    # listings first (by closeness), high-risk ones after.
    ordered_ids = sorted(
        ids, key=lambda i: (agents["safety"].scores[i] < config.SAFETY_FLOOR,
                            -closeness_by_id[i]))
    system_rank = {lid: rank + 1 for rank, lid in enumerate(ordered_ids)}

    metrics: list[tuple[str, float]] = []
    metrics.append(("n_listings", float(len(df))))
    metrics.append(("ahp_consistency_ratio_default_profile", round(cr, 4)))

    # (a) Spearman vs student rankings -------------------------------------
    rankings = pd.read_csv(EVAL_DIR / "student_rankings.csv")
    rhos = []
    for student, group in rankings.groupby("student_id"):
        common = group[group["listing_id"].isin(system_rank)]
        if len(common) >= 3:
            rho, _ = spearmanr(common["rank"],
                               [system_rank[i] for i in common["listing_id"]])
            rhos.append(rho)
    metrics.append(("spearman_n_students", float(len(rhos))))
    metrics.append(("spearman_mean", round(float(np.mean(rhos)), 4)))
    metrics.append(("spearman_median", round(float(np.median(rhos)), 4)))

    # (b) Survey ------------------------------------------------------------
    survey = pd.read_csv(EVAL_DIR / "survey.csv")
    for col in ("clarity", "trust", "usefulness"):
        metrics.append((f"survey_{col}_mean",
                        round(float(survey[col].mean()), 3)))
    match_pct = float((survey["reason_matched"] == "yes").mean() * 100)
    metrics.append(("survey_reason_matched_pct", round(match_pct, 1)))

    # (c) Red-flag checklist quality ---------------------------------------
    safety = agents["safety"]
    y_true = [int(df.set_index("id").loc[i, "red_flag_label"]) for i in ids]
    y_pred = [1 if safety.extras["flags"][i] else 0 for i in ids]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    metrics.append(("redflag_precision", round(float(precision), 4)))
    metrics.append(("redflag_recall", round(float(recall), 4)))
    metrics.append(("redflag_f1", round(float(f1), 4)))

    # (d) What-if latency over 50 runs --------------------------------------
    rng = random.Random(11)
    cached = {name: res.scores for name, res in agents.items()}
    latencies = []
    for _ in range(50):
        changes = {}
        if rng.random() < 0.7:
            changes["budget_ceiling"] = profile.budget_ceiling \
                + rng.choice([-3000, -1500, 1500, 3000])
        if rng.random() < 0.7:
            changes["max_distance_km"] = max(
                0.5, profile.max_distance_km + rng.choice([-2.0, -1.0, 1.0, 2.0]))
        result = recalculate(enriched, profile, changes,
                             cached_scores=cached, log_path=None)
        latencies.append(result.elapsed_seconds)
    lat = np.array(latencies)
    metrics.append(("whatif_latency_mean_s", round(float(lat.mean()), 4)))
    metrics.append(("whatif_latency_median_s", round(float(np.median(lat)), 4)))
    metrics.append(("whatif_latency_p95_s",
                    round(float(np.percentile(lat, 95)), 4)))

    # Outputs ---------------------------------------------------------------
    table = pd.DataFrame(metrics, columns=["metric", "value"])
    table.to_csv(outdir / "results_table.csv", index=False)
    print(table.to_string(index=False))

    plt.figure(figsize=(6, 4))
    plt.hist(rhos, bins=8, color="#4C78A8", edgecolor="white")
    plt.xlabel("Spearman rho per student")
    plt.ylabel("Students")
    plt.title("System vs student rankings")
    plt.tight_layout()
    plt.savefig(outdir / "spearman_hist.png", dpi=120)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(["Precision", "Recall", "F1"],
            [precision, recall, f1], color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylim(0, 1)
    plt.title("Red-flag checklist vs hand labels")
    plt.tight_layout()
    plt.savefig(outdir / "redflag_metrics.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(lat * 1000, bins=10, color="#F58518", edgecolor="white")
    plt.xlabel("Recalculation latency (ms)")
    plt.ylabel("Runs (of 50)")
    plt.title("What-if latency")
    plt.tight_layout()
    plt.savefig(outdir / "latency_hist.png", dpi=120)
    plt.close()

    lines = ["# Evaluation summary", "",
             f"Dataset: `{args.dataset}` ({len(df)} listings, "
             f"default profile, demo mode)", "",
             "| Metric | Value |", "|---|---|"]
    lines += [f"| {m} | {v} |" for m, v in metrics]
    lines += ["",
              "Notes: student rankings and survey responses are the "
              "sample files in evaluation/ (replace with real collected "
              "data for a study). The rule-based checklist is expected "
              "to trail trained classifiers on recall - see README "
              "limitations."]
    (outdir / "evaluation_summary.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
    print(f"\nWrote results_table.csv, evaluation_summary.md and 3 "
          f"charts to {outdir}")


if __name__ == "__main__":
    main()
