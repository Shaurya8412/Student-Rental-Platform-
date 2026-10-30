"""The three specialist agents: budget, location, safety.

Each agent is a plain function that scores every listing 0-1 (higher is
better), gives a one-line reason per listing, and returns its ranking.
Agents never talk to each other; the Decision Agent reconciles them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import config


@dataclass
class Profile:
    """What the student cares about (from the sidebar)."""

    budget_ceiling: float
    max_distance_km: float
    weights: dict  # criterion -> importance 1..5, keys = config.CRITERIA

    @classmethod
    def default(cls) -> "Profile":
        return cls(
            budget_ceiling=float(config.DEFAULT_PROFILE["budget_ceiling"]),
            max_distance_km=float(config.DEFAULT_PROFILE["max_distance_km"]),
            weights=dict(config.DEFAULT_PROFILE["weights"]),
        )


@dataclass
class AgentResult:
    """One agent's scores, reasons and ranking over all listings."""

    name: str
    scores: dict  # listing_id -> score in [0, 1]
    reasons: dict  # listing_id -> one-line reason
    ranked_ids: list  # listing ids, best first (ties broken by id)
    extras: dict = field(default_factory=dict)

    @property
    def top_id(self):
        return self.ranked_ids[0] if self.ranked_ids else None

    @property
    def top_reason(self) -> str:
        return self.reasons.get(self.top_id, "")


def _rank(scores: dict) -> list:
    return sorted(scores, key=lambda lid: (-scores[lid], lid))


# ---------------------------------------------------------------------------
# Budget agent
# ---------------------------------------------------------------------------
def budget_agent(listings: pd.DataFrame, profile: Profile,
                 stay_months: int = config.STAY_MONTHS) -> AgentResult:
    """Score by *true* monthly cost, not the headline rent.

    True monthly cost = rent
        + (deposit + brokerage) / stay_months   (one-time fees amortised)
        + maintenance.
    Deposit/brokerage come from the fee estimator when available and
    fall back to the stated CSV columns otherwise. Score is
    1 - cost/ceiling inside budget and exactly 0 over the ceiling.
    """
    scores: dict = {}
    reasons: dict = {}
    true_costs: dict = {}
    for row in listings.itertuples():
        deposit = row.est_deposit if hasattr(row, "est_deposit") and \
            pd.notna(row.est_deposit) else float(row.deposit)
        brokerage = row.est_brokerage if hasattr(row, "est_brokerage") and \
            pd.notna(row.est_brokerage) else float(row.brokerage_stated)
        maintenance = row.est_maintenance if hasattr(
            row, "est_maintenance") and pd.notna(row.est_maintenance) else 0.0
        amortised = (float(deposit) + float(brokerage)) / stay_months
        true_cost = float(row.rent) + amortised + float(maintenance)
        true_costs[row.id] = true_cost

        ceiling = profile.budget_ceiling
        if ceiling <= 0 or true_cost > ceiling:
            score = 0.0
            reasons[row.id] = (
                f"True cost {config.CURRENCY}{true_cost:,.0f}/month is over "
                f"the {config.CURRENCY}{ceiling:,.0f} ceiling.")
        else:
            score = 1.0 - true_cost / ceiling
            reasons[row.id] = (
                f"True cost {config.CURRENCY}{true_cost:,.0f}/month "
                f"(rent {config.CURRENCY}{float(row.rent):,.0f} + amortised "
                f"fees {config.CURRENCY}{amortised:,.0f} + maintenance "
                f"{config.CURRENCY}{float(maintenance):,.0f}) = "
                f"{true_cost / ceiling:.0%} of the ceiling.")
        scores[row.id] = float(min(1.0, max(0.0, score)))

    return AgentResult(name="budget", scores=scores, reasons=reasons,
                       ranked_ids=_rank(scores),
                       extras={"true_costs": true_costs})


# ---------------------------------------------------------------------------
# Location agent
# ---------------------------------------------------------------------------
def location_agent(listings: pd.DataFrame, profile: Profile) -> AgentResult:
    """Score linearly by distance: 1.0 at 0 km, 0.0 at/past the max."""
    scores: dict = {}
    reasons: dict = {}
    max_d = profile.max_distance_km
    for row in listings.itertuples():
        distance = float(row.distance_km)
        if max_d <= 0:
            score = 1.0 if distance == 0 else 0.0
        else:
            score = min(1.0, max(0.0, 1.0 - distance / max_d))
        scores[row.id] = float(score)
        if distance > max_d:
            reasons[row.id] = (
                f"{distance:.1f} km away, beyond the {max_d:.1f} km limit.")
        else:
            reasons[row.id] = (
                f"{distance:.1f} km away, within the {max_d:.1f} km limit "
                f"({distance / max_d:.0%} of the way to the limit)." if max_d > 0
                else f"{distance:.1f} km away.")
    return AgentResult(name="location", scores=scores, reasons=reasons,
                       ranked_ids=_rank(scores))


# ---------------------------------------------------------------------------
# Safety agent (rule-based red-flag checklist; rules live in config.py)
# ---------------------------------------------------------------------------
def _rule_advance(row, ctx) -> bool:
    return int(row.advance_before_viewing) == 1


def _rule_no_agreement(row, ctx) -> bool:
    return int(row.agreement_mentioned) == 0


def _rule_unverifiable_verified(row, ctx) -> bool:
    if int(row.verified_claim) != 1:
        return False
    text = str(row.free_text_description).lower()
    return not any(keyword in text for keyword in config.PROOF_KEYWORDS)


def _rule_below_market(row, ctx) -> bool:
    threshold = ctx.get("threshold", 0.30)
    return float(row.rent) < (1.0 - threshold) * ctx["median_rent"]


def _rule_no_contact(row, ctx) -> bool:
    text = str(row.free_text_description)
    has_phone = bool(re.search(r"\b\d{10}\b", text)
                     or re.search(r"\b\d{5}\s\d{5}\b", text))
    has_address = any(keyword in text.lower()
                      for keyword in config.ADDRESS_KEYWORDS)
    return not (has_phone or has_address)


def _rule_pressure(row, ctx) -> bool:
    text = str(row.free_text_description).lower()
    return any(phrase in text for phrase in config.PRESSURE_PHRASES)


_RULE_CHECKS = {
    "advance_before_viewing": _rule_advance,
    "no_written_agreement": _rule_no_agreement,
    "unverifiable_verified_claim": _rule_unverifiable_verified,
    "below_market_rent": _rule_below_market,
    "no_contact_or_address": _rule_no_contact,
    "pressure_language": _rule_pressure,
}


def _flag_justification(rule_id: str, row, ctx) -> str:
    """One-line, listing-specific justification for a fired rule."""
    if rule_id == "advance_before_viewing":
        return "The listing asks for money before a viewing is possible."
    if rule_id == "no_written_agreement":
        return "No written rental agreement is mentioned anywhere."
    if rule_id == "unverifiable_verified_claim":
        return ("The listing claims to be 'verified' but shows no ID proof, "
                "registration or licence detail to back the claim.")
    if rule_id == "below_market_rent":
        median = ctx["median_rent"]
        pct = (1.0 - float(row.rent) / median) * 100 if median else 0.0
        return (f"Rent {config.CURRENCY}{float(row.rent):,.0f} is "
                f"{pct:.0f}% below the dataset median of "
                f"{config.CURRENCY}{median:,.0f} - a classic bait pattern.")
    if rule_id == "no_contact_or_address":
        return "No phone number or address detail appears in the text."
    if rule_id == "pressure_language":
        text = str(row.free_text_description).lower()
        found = [p for p in config.PRESSURE_PHRASES if p in text]
        return f"Pressure phrase(s) in the text: {', '.join(found)}."
    return ""


def safety_agent(listings: pd.DataFrame,
                 rules: list | None = None) -> AgentResult:
    """Score by a weighted red-flag checklist (see config.SAFETY_RULES).

    Safety score = 1 - (sum of severities of fired rules), clipped to
    [0, 1]. Returns flags and a justification per flag in ``extras``.
    """
    rules = rules if rules is not None else config.SAFETY_RULES
    median_rent = float(listings["rent"].median()) if len(listings) else 0.0
    scores: dict = {}
    reasons: dict = {}
    flags_by_id: dict = {}
    details_by_id: dict = {}

    for row in listings.itertuples():
        penalty = 0.0
        flags: list[str] = []
        details: dict = {}
        for rule in rules:
            check = _RULE_CHECKS[rule["id"]]
            ctx = dict(rule.get("params", {}))
            ctx["median_rent"] = median_rent
            if check(row, ctx):
                penalty += float(rule["severity"])
                flags.append(rule["label"])
                details[rule["id"]] = _flag_justification(rule["id"], row, ctx)
        score = float(min(1.0, max(0.0, 1.0 - penalty)))
        scores[row.id] = score
        flags_by_id[row.id] = flags
        details_by_id[row.id] = details
        reasons[row.id] = (
            "No red flags found by the checklist." if not flags else
            f"{len(flags)} red flag(s): {'; '.join(flags)}.")

    return AgentResult(
        name="safety", scores=scores, reasons=reasons,
        ranked_ids=_rank(scores),
        extras={"flags": flags_by_id, "flag_details": details_by_id,
                "median_rent": median_rent})
