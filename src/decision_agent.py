"""Decision Agent: one bounded debate round + AHP-TOPSIS final ranking.

Flow (deliberately shallow, per spec):
1. Run the three specialist agents and note each one's top pick.
2. If the picks disagree, identify the conflicting criteria and ask
   each disagreeing agent exactly ONE targeted follow-up question
   (LLM-answered, with a deterministic numbers-based fallback). No
   second round is run.
3. Compute the final ranking with AHP-TOPSIS regardless of the debate
   outcome - the debate is recorded for transparency, not used as a
   vote. Before presentation, a non-compensatory safety floor
   (config.SAFETY_FLOOR) screens out listings whose safety score is
   below 0.50: bait pricing must not buy its way past fraud signals.
   Their closeness scores are still computed and shown.
4. Explanations are computed from the numbers first (which criterion
   contributed most, true cost, flags). The LLM may only *phrase* those
   facts; the prompt forbids adding new ones, and a template fallback
   phrases the same facts when offline.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .agents import (AgentResult, Profile, budget_agent, location_agent,
                     safety_agent)
from .llm_client import LLMClient
from .topsis import ahp_from_importance, criterion_contributions, topsis

AGENT_ORDER = ["budget", "location", "safety"]
AGENT_CRITERION = {"budget": "budget", "location": "distance",
                   "safety": "safety"}
AGENT_LABEL = {"budget": "Budget agent", "location": "Location agent",
               "safety": "Safety agent"}

DEBATE_SYSTEM = (
    "You are {label} on a student-housing decision panel. Answer the "
    "question using ONLY the facts provided - never invent prices, "
    "distances or features. Weigh ALL the factors in the facts before "
    "answering: the full cost breakdown (rent, deposit, brokerage, "
    "maintenance), distance, every safety flag and its justification, "
    "wifi, food, amenities and fee confidence - not only your own "
    "criterion. Reply with JSON only: "
    "{{\"answer\": string, \"would_change_pick\": boolean}}. "
    "The answer is one or two sentences and cites the numbers.")

ARENA_EXPLAIN_SYSTEM = (
    "You announce the verdict of a rental debate to a student. Use "
    "ONLY the facts given; do not invent features, prices or "
    "amenities. Reply with JSON only: {{\"explanation\": string}} - "
    "two or three plain sentences naming the winner, its strongest "
    "factor, its true cost, and its main caveat if any.")

EXPLAIN_SYSTEM = (
    "You explain a rental ranking to a student. Use ONLY the facts "
    "given; do not invent features, prices, amenities or neighbourhoods. "
    "Reply with JSON only: {{\"explanation\": string}} - one or two "
    "plain sentences.")


@dataclass
class DecisionResult:
    """Everything the UI needs from one decision run."""

    ranking: pd.DataFrame
    agent_results: dict
    weights: dict
    consistency_ratio: float
    picks: dict  # agent name -> top listing id
    conflicts: list
    debate: list  # transcript entries
    listings: pd.DataFrame = field(default=None)


# ---------------------------------------------------------------------------
# Decision matrix
# ---------------------------------------------------------------------------
def build_decision_matrix(listings: pd.DataFrame,
                          agent_results: dict) -> tuple[list, np.ndarray]:
    """Rows = listings (in file order), columns = config.CRITERIA order."""
    ids = [int(i) for i in listings["id"]]
    by_id = listings.set_index("id")
    matrix = np.array([
        [agent_results["budget"].scores[i],
         agent_results["location"].scores[i],
         agent_results["safety"].scores[i],
         float(by_id.loc[i, "wifi"]),
         float(by_id.loc[i, "food_included"])]
        for i in ids], dtype=float)
    return ids, matrix


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------
def _used_amount(row, est_col: str, stated_col: str | None) -> float:
    """Fee amount actually used: verified estimate first, stated second."""
    if est_col in row.index and pd.notna(row[est_col]):
        return float(row[est_col])
    return float(row[stated_col]) if stated_col else 0.0


def _full_facts(listing_id: int, agent_results: dict,
                listings_by_id: pd.DataFrame,
                closeness_by_id: dict) -> dict:
    """ALL factors for one listing - what agents must weigh in a debate."""
    row = listings_by_id.loc[listing_id]
    safety = agent_results["safety"]
    return {
        "listing_id": int(listing_id),
        "title": str(row["title"]) if "title" in row.index
        else f"Listing #{listing_id}",
        "rent": float(row["rent"]),
        "deposit_used": _used_amount(row, "est_deposit", "deposit"),
        "brokerage_used": _used_amount(row, "est_brokerage",
                                       "brokerage_stated"),
        "maintenance_used": _used_amount(row, "est_maintenance", None),
        "true_monthly_cost": round(
            agent_results["budget"].extras["true_costs"][listing_id]),
        "fee_confidence": str(row["fee_confidence"])
        if "fee_confidence" in row.index else "low",
        "needs_manual_review": bool(row["needs_manual_review"])
        if "needs_manual_review" in row.index else False,
        "distance_km": float(row["distance_km"]),
        "agent_scores": {name: round(agent_results[name].scores[listing_id], 3)
                         for name in AGENT_ORDER},
        "safety_score": round(safety.scores[listing_id], 2),
        "safety_flags": safety.extras["flags"][listing_id],
        "flag_justifications": safety.extras["flag_details"][listing_id],
        "wifi": bool(row["wifi"]),
        "food_included": bool(row["food_included"]),
        "amenities": str(row["amenities"]),
        "gender_policy": str(row["gender_policy"]),
        "topsis_closeness": round(float(closeness_by_id[listing_id]), 3),
    }


def _stub_answer(agent_name: str, own_facts: dict,
                 rival_facts: dict) -> dict:
    """Deterministic debate answer weighing ALL factors, not one score.

    The agent concedes only if the rival is at least as good on its own
    criterion and overall, or clearly better overall without being
    worse on flags or much worse on the agent's own criterion.
    """
    criterion = AGENT_CRITERION[agent_name]
    own_c = own_facts["agent_scores"][agent_name]
    rival_c = rival_facts["agent_scores"][agent_name]
    own_close = own_facts["topsis_closeness"]
    rival_close = rival_facts["topsis_closeness"]
    own_flags = len(own_facts["safety_flags"])
    rival_flags = len(rival_facts["safety_flags"])
    would_change = (
        (rival_c >= own_c - 0.05 and rival_close >= own_close - 0.02)
        or (rival_close >= own_close + 0.15 and rival_c >= own_c - 0.20
            and rival_flags <= own_flags))
    shared = (
        f"true cost {config.CURRENCY}{rival_facts['true_monthly_cost']:,}"
        f"/month vs {config.CURRENCY}{own_facts['true_monthly_cost']:,}, "
        f"{rival_flags} red flag(s) vs {own_flags}, overall closeness "
        f"{rival_close:.2f} vs {own_close:.2f}")
    if would_change:
        answer = (
            f"Weighing all factors: on {criterion} my pick "
            f"#{own_facts['listing_id']} scores {own_c:.2f} against "
            f"#{rival_facts['listing_id']} at {rival_c:.2f}, and the "
            f"rival holds up overall ({shared}). I would change my pick.")
    else:
        answer = (
            f"Weighing all factors: my pick #{own_facts['listing_id']} "
            f"leads on {criterion} ({own_c:.2f} vs {rival_c:.2f}), and "
            f"the rival does not dominate overall ({shared}). "
            f"I would not change my pick.")
    return {"answer": answer, "would_change_pick": bool(would_change)}


def run_debate(agent_results: dict, listings: pd.DataFrame,
               closeness_by_id: dict, client: LLMClient) -> tuple[dict, list, list]:
    """One bounded debate round. Returns (picks, conflicts, transcript)."""
    picks = {name: agent_results[name].top_id for name in AGENT_ORDER}
    transcript: list = []
    conflicts: list = []

    if len(set(picks.values())) == 1:
        transcript.append({
            "type": "consensus",
            "text": (f"All three agents picked listing "
                     f"#{picks['budget']} - no debate needed.")})
        return picks, conflicts, transcript

    counts = Counter(picks.values())
    consensus_id = counts.most_common(1)[0][0] if \
        counts.most_common(1)[0][1] >= 2 else None
    listings_by_id = listings.set_index("id")

    for name in AGENT_ORDER:
        own_id = picks[name]
        if consensus_id is not None and own_id == consensus_id:
            continue  # agent is with the consensus; only dissenters are asked
        if consensus_id is not None:
            rival_id = consensus_id
        else:  # three-way split: rival = next agent's differing pick
            rival_id = next(picks[o] for o in AGENT_ORDER
                            if o != name and picks[o] != own_id)
        rival_agents = [o for o in AGENT_ORDER if picks[o] == rival_id]
        rival_criteria = [AGENT_CRITERION[o] for o in rival_agents]
        conflicts.append({
            "agent": name, "own_pick": own_id, "rival_pick": rival_id,
            "criteria": [AGENT_CRITERION[name]] + rival_criteria})

        own_facts = _full_facts(own_id, agent_results, listings_by_id,
                                closeness_by_id)
        rival_facts = _full_facts(rival_id, agent_results, listings_by_id,
                                  closeness_by_id)
        question = (
            f"Your top pick is listing #{own_id}. The "
            f"{' and '.join(AGENT_LABEL[o] for o in rival_agents)} picked "
            f"listing #{rival_id} instead, so the conflict is "
            f"{AGENT_CRITERION[name]} versus "
            f"{' and '.join(rival_criteria)}. Given the facts for both "
            f"listings, would you change your pick to listing #{rival_id}?")
        user = question + "\nFacts: " + json.dumps(
            {"your_pick": own_facts, "rival_pick": rival_facts})
        response = client.call_json(
            DEBATE_SYSTEM.format(label=AGENT_LABEL[name]), user,
            fallback=lambda: _stub_answer(name, own_facts, rival_facts))
        transcript.append({
            "type": "qa", "agent": name, "own_pick": own_id,
            "rival_pick": rival_id, "question": question,
            "answer": str(response.get("answer", "")),
            "would_change_pick": bool(response.get("would_change_pick",
                                                   False))})

    changed = [AGENT_LABEL[e["agent"]] for e in transcript
               if e["type"] == "qa" and e["would_change_pick"]]
    transcript.append({
        "type": "summary",
        "text": ("Debate closed after exactly one round. "
                 + (f"{' and '.join(changed)} would switch to the rival "
                    f"pick. " if changed else
                    "No agent changed its pick. ")
                 + "The final ranking is computed by AHP-TOPSIS over all "
                   "five criteria; the debate is shown for transparency "
                   "and does not override the numbers.")})
    return picks, conflicts, transcript


# ---------------------------------------------------------------------------
# Explanations (facts computed from numbers; LLM only phrases them)
# ---------------------------------------------------------------------------
def explanation_facts(row: dict, rank: int, profile: Profile) -> dict:
    """The complete, code-computed fact set an explanation may use."""
    return {
        "listing_id": int(row["listing_id"]),
        "rank": rank,
        "topsis_score": round(float(row["topsis_score"]), 3),
        "top_criterion": row["top_criterion"],
        "top_criterion_share_pct": round(
            float(row["top_criterion_share"]) * 100),
        "true_monthly_cost": round(float(row["true_monthly_cost"])),
        "budget_ceiling": round(float(profile.budget_ceiling)),
        "cost_pct_of_ceiling": round(
            100 * float(row["true_monthly_cost"]) / profile.budget_ceiling)
        if profile.budget_ceiling else None,
        "distance_km": float(row["distance_km"]),
        "safety_score": round(float(row["safety_score"]), 2),
        "red_flags": list(row["flags"]),
        "wifi": bool(row["wifi"]),
        "food_included": bool(row["food_included"]),
        "fee_confidence": row["fee_confidence"],
        "needs_manual_review": bool(row["needs_manual_review"]),
    }


def _template_explanation(facts: dict) -> str:
    flags = facts["red_flags"]
    flag_text = (f" Note {len(flags)} red flag(s) to check before paying "
                 f"anything." if flags else " No red flags were found.")
    return (
        f"Ranked #{facts['rank']} mainly on its {facts['top_criterion']} "
        f"(about {facts['top_criterion_share_pct']}% of its weighted "
        f"score). True cost is {config.CURRENCY}"
        f"{facts['true_monthly_cost']:,}/month "
        f"({facts['cost_pct_of_ceiling']}% of your ceiling), "
        f"{facts['distance_km']:.1f} km away, safety score "
        f"{facts['safety_score']:.2f}.{flag_text}")


def phrase_explanation(facts: dict, client: LLMClient) -> str:
    """Phrase pre-computed facts; never lets the LLM add new ones."""
    response = client.call_json(
        EXPLAIN_SYSTEM, json.dumps(facts),
        fallback=lambda: {"explanation": _template_explanation(facts)})
    text = response.get("explanation")
    return text if isinstance(text, str) and text.strip() \
        else _template_explanation(facts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_decision(listings: pd.DataFrame, profile: Profile,
                 client: LLMClient,
                 stay_months: int = config.STAY_MONTHS) -> DecisionResult:
    """Run agents -> debate (one round) -> TOPSIS -> explained ranking.

    ``listings`` must already be fee-enriched (see
    fee_estimator.enrich_listings).
    """
    agent_results = {
        "budget": budget_agent(listings, profile, stay_months),
        "location": location_agent(listings, profile),
        "safety": safety_agent(listings),
    }
    weights, cr = ahp_from_importance(profile.weights)
    ids, matrix = build_decision_matrix(listings, agent_results)
    weight_vec = [weights[c] for c in config.CRITERIA]
    closeness = topsis(matrix, weight_vec)
    contributions = criterion_contributions(matrix, weight_vec)
    closeness_by_id = {lid: float(closeness[k]) for k, lid in enumerate(ids)}

    picks, conflicts, debate = run_debate(
        agent_results, listings, closeness_by_id, client)

    by_id = listings.set_index("id")
    safety = agent_results["safety"]

    def _text_col(lid, col):
        if col in by_id.columns:
            value = by_id.loc[lid, col]
            return value if isinstance(value, str) else ""
        return ""

    rows = []
    for k, lid in enumerate(ids):
        top_idx = int(np.argmax(contributions[k]))
        rows.append({
            "listing_id": lid,
            "source": by_id.loc[lid, "source"],
            "title": _text_col(lid, "title") or f"Listing #{lid}",
            "image_url": _text_col(lid, "image_url"),
            "listing_url": _text_col(lid, "listing_url"),
            "rent": float(by_id.loc[lid, "rent"]),
            "distance_km": float(by_id.loc[lid, "distance_km"]),
            "topsis_score": float(closeness[k]),
            "true_monthly_cost":
                agent_results["budget"].extras["true_costs"][lid],
            "budget_score": agent_results["budget"].scores[lid],
            "location_score": agent_results["location"].scores[lid],
            "safety_score": safety.scores[lid],
            "wifi": int(by_id.loc[lid, "wifi"]),
            "food_included": int(by_id.loc[lid, "food_included"]),
            "top_criterion": config.CRITERIA[top_idx],
            "top_criterion_share": float(contributions[k][top_idx]),
            "fee_confidence": by_id.loc[lid, "fee_confidence"]
            if "fee_confidence" in by_id.columns else "low",
            "needs_manual_review": bool(by_id.loc[lid, "needs_manual_review"])
            if "needs_manual_review" in by_id.columns else False,
            "flags": safety.extras["flags"][lid],
            "high_risk": safety.scores[lid] < config.SAFETY_FLOOR,
            "explanation": "",
        })
    # Safety floor first (non-compensatory screen), then TOPSIS closeness.
    ranking = pd.DataFrame(rows).sort_values(
        ["high_risk", "topsis_score", "listing_id"],
        ascending=[True, False, True]).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))

    for pos in range(min(5, len(ranking))):
        row = ranking.iloc[pos].to_dict()
        facts = explanation_facts(row, int(row["rank"]), profile)
        ranking.at[pos, "explanation"] = phrase_explanation(facts, client)

    return DecisionResult(ranking=ranking, agent_results=agent_results,
                          weights=weights, consistency_ratio=cr,
                          picks=picks, conflicts=conflicts, debate=debate,
                          listings=listings)


# ---------------------------------------------------------------------------
# Debate Arena: the user picks the listings, the agents debate those
# ---------------------------------------------------------------------------
@dataclass
class CustomDebateResult:
    """Outcome of a user-picked debate (Debate Arena in the UI)."""

    selected_ids: list
    picks: dict  # agent name -> its pick within the shortlist
    conflicts: list
    debate: list  # transcript entries (same shape as the main debate)
    subset_ranking: pd.DataFrame
    winner_id: int
    winner_explanation: str
    facts: dict  # listing_id -> full fact dict used by the agents
    weights: dict = field(default_factory=dict)
    consistency_ratio: float = 0.0


def _arena_verdict_text(winner: dict, runner_up: dict | None,
                        facts: dict) -> str:
    """Template verdict for the arena winner (facts only, no invention)."""
    w = facts[winner["listing_id"]]
    text = (
        f"Verdict: {w['title']} (#{w['listing_id']}) wins your shortlist "
        f"with a subset TOPSIS score of {winner['topsis_score']:.2f}, "
        f"driven mainly by its {winner['top_criterion']}. True cost is "
        f"{config.CURRENCY}{w['true_monthly_cost']:,}/month, "
        f"{w['distance_km']:.1f} km away, safety {w['safety_score']:.2f} "
        f"with {len(w['safety_flags'])} red flag(s).")
    if runner_up is not None:
        r = facts[runner_up["listing_id"]]
        gaps = {name: w["agent_scores"][name] - r["agent_scores"][name]
                for name in AGENT_ORDER}
        gaps["wifi"] = float(w["wifi"]) - float(r["wifi"])
        gaps["food"] = float(w["food_included"]) - float(r["food_included"])
        best = max(gaps, key=gaps.get)
        text += (
            f" Against the runner-up {r['title']} (#{r['listing_id']}, "
            f"{runner_up['topsis_score']:.2f}), its biggest edge is on "
            f"{best} ({gaps[best]:+.2f}).")
    if w["needs_manual_review"]:
        text += (" Note: some of its fees could not be verified against "
                 "the listing text - confirm them before paying.")
    return text


def run_custom_debate(listings: pd.DataFrame, profile: Profile,
                      client: LLMClient, selected_ids,
                      agent_results: dict | None = None,
                      weights: dict | None = None,
                      consistency_ratio: float = 0.0,
                      stay_months: int = config.STAY_MONTHS
                      ) -> CustomDebateResult:
    """Debate the user's own shortlist, considering all factors.

    Each agent picks its favourite *within the shortlist*; agents whose
    pick is not the subset's AHP-TOPSIS leader are asked exactly one
    follow-up question with the full facts of both listings. The
    verdict is TOPSIS recomputed on the shortlist only (with the
    safety floor applied), so the comparison frame matches the
    student's actual choice set.
    """
    selected = list(dict.fromkeys(int(i) for i in selected_ids))
    if len(selected) < 2:
        raise ValueError("Pick at least two listings for a debate")

    if agent_results is None:
        agent_results = {
            "budget": budget_agent(listings, profile, stay_months),
            "location": location_agent(listings, profile),
            "safety": safety_agent(listings),
        }
    if weights is None:
        weights, consistency_ratio = ahp_from_importance(profile.weights)

    ids, matrix = build_decision_matrix(listings, agent_results)
    position = {lid: k for k, lid in enumerate(ids)}
    sub_matrix = np.array([matrix[position[i]] for i in selected])
    weight_vec = [weights[c] for c in config.CRITERIA]
    sub_close = topsis(sub_matrix, weight_vec)
    sub_contrib = criterion_contributions(sub_matrix, weight_vec)
    closeness_by_id = {lid: float(sub_close[k])
                       for k, lid in enumerate(selected)}
    listings_by_id = listings.set_index("id")
    facts = {lid: _full_facts(lid, agent_results, listings_by_id,
                              closeness_by_id) for lid in selected}

    picks = {name: max(selected, key=lambda lid: (
        agent_results[name].scores[lid], -lid)) for name in AGENT_ORDER}
    transcript: list = []
    conflicts: list = []

    if len(set(picks.values())) == 1:
        only = picks["budget"]
        transcript.append({
            "type": "consensus",
            "text": (f"All three agents picked {facts[only]['title']} "
                     f"(#{only}) from your shortlist - no debate needed.")})
    else:
        leader = max(selected, key=lambda lid: (closeness_by_id[lid], -lid))
        for name in AGENT_ORDER:
            own_id = picks[name]
            if own_id == leader:
                continue  # this agent already backs the subset leader
            conflicts.append({
                "agent": name, "own_pick": own_id, "rival_pick": leader,
                "criteria": [AGENT_CRITERION[name], "overall"]})
            own_facts, rival_facts = facts[own_id], facts[leader]
            question = (
                f"From the student's shortlist, your pick is "
                f"{own_facts['title']} (#{own_id}). The combined "
                f"AHP-TOPSIS comparison of the shortlist currently leads "
                f"with {rival_facts['title']} (#{leader}). Considering "
                f"ALL factors for both listings - cost breakdown, "
                f"distance, safety flags, wifi, food, amenities and fee "
                f"confidence - would you change your pick to #{leader}?")
            user = question + "\nFacts: " + json.dumps(
                {"your_pick": own_facts, "rival_pick": rival_facts})
            response = client.call_json(
                DEBATE_SYSTEM.format(label=AGENT_LABEL[name]), user,
                fallback=lambda: _stub_answer(name, own_facts, rival_facts))
            transcript.append({
                "type": "qa", "agent": name, "own_pick": own_id,
                "rival_pick": leader, "question": question,
                "answer": str(response.get("answer", "")),
                "would_change_pick": bool(response.get(
                    "would_change_pick", False))})
        changed = [AGENT_LABEL[e["agent"]] for e in transcript
                   if e["type"] == "qa" and e["would_change_pick"]]
        transcript.append({
            "type": "summary",
            "text": ("Shortlist debate closed after one round. "
                     + (f"{' and '.join(changed)} would switch to the "
                        f"subset leader. " if changed else
                        "No agent changed its pick. ")
                     + "The verdict below is AHP-TOPSIS recomputed on "
                       "your shortlist only, with the safety floor "
                       "applied.")})

    safety = agent_results["safety"]
    rows = []
    for k, lid in enumerate(selected):
        top_idx = int(np.argmax(sub_contrib[k]))
        rows.append({
            "listing_id": lid, "title": facts[lid]["title"],
            "topsis_score": float(sub_close[k]),
            "true_monthly_cost": facts[lid]["true_monthly_cost"],
            "distance_km": facts[lid]["distance_km"],
            "safety_score": safety.scores[lid],
            "high_risk": safety.scores[lid] < config.SAFETY_FLOOR,
            "flags": safety.extras["flags"][lid],
            "top_criterion": config.CRITERIA[top_idx]})
    subset_ranking = pd.DataFrame(rows).sort_values(
        ["high_risk", "topsis_score", "listing_id"],
        ascending=[True, False, True]).reset_index(drop=True)
    subset_ranking.insert(0, "rank", range(1, len(subset_ranking) + 1))

    winner = subset_ranking.iloc[0].to_dict()
    runner_up = (subset_ranking.iloc[1].to_dict()
                 if len(subset_ranking) > 1 else None)
    template = _arena_verdict_text(winner, runner_up, facts)
    response = client.call_json(
        ARENA_EXPLAIN_SYSTEM,
        json.dumps({"winner": facts[winner["listing_id"]],
                    "runner_up": facts[runner_up["listing_id"]]
                    if runner_up else None,
                    "winner_subset_score": round(
                        float(winner["topsis_score"]), 3)}),
        fallback=lambda: {"explanation": template})
    verdict = response.get("explanation")
    if not isinstance(verdict, str) or not verdict.strip():
        verdict = template

    return CustomDebateResult(
        selected_ids=selected, picks=picks, conflicts=conflicts,
        debate=transcript, subset_ranking=subset_ranking,
        winner_id=int(winner["listing_id"]), winner_explanation=verdict,
        facts=facts, weights=weights,
        consistency_ratio=consistency_ratio)
