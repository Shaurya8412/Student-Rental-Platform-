"""Tests for the Decision Agent: debate bound + the safety floor."""

from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.agents import Profile
from src.decision_agent import run_custom_debate, run_decision
from src.fee_estimator import enrich_listings
from src.llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent


def _run_meerut(tmp_path) -> object:
    client = LLMClient(demo_mode=True, cache_path=tmp_path / "cache.json")
    df = enrich_listings(pd.read_csv(ROOT / "data" / "listings_meerut.csv"),
                         client)
    profile = Profile(budget_ceiling=9000.0, max_distance_km=5.0,
                      weights={"budget": 5, "distance": 4, "safety": 5,
                               "wifi": 3, "food": 2})
    return run_decision(df, profile, client)


def test_debate_is_exactly_one_bounded_round(tmp_path):
    result = _run_meerut(tmp_path)
    qa = [e for e in result.debate if e["type"] == "qa"]
    # At most one question per disagreeing agent, never a second round.
    assert len(qa) <= 3
    agents_asked = [e["agent"] for e in qa]
    assert len(agents_asked) == len(set(agents_asked))
    for entry in qa:
        assert entry["question"] and entry["answer"]
        assert isinstance(entry["would_change_pick"], bool)
    # The transcript always closes with a summary (or a consensus note).
    assert result.debate[-1]["type"] in ("summary", "consensus")


def test_safety_floor_keeps_scams_out_of_top5(tmp_path):
    result = _run_meerut(tmp_path)
    top5 = result.ranking.head(5)
    assert not top5["high_risk"].any()
    assert (top5["safety_score"] >= config.SAFETY_FLOOR).all()
    # Every high-risk listing sorts below every recommendable one.
    ranks = result.ranking
    last_safe = ranks[~ranks["high_risk"]]["rank"].max()
    first_risky = ranks[ranks["high_risk"]]["rank"].min()
    assert first_risky > last_safe
    # The planted UPI deposit scam (listing 4 in the Meerut file) is
    # the budget agent's bait - it must not be recommended.
    assert 4 not in set(top5["listing_id"])
    assert bool(ranks.set_index("listing_id").loc[4, "high_risk"])


def test_ranking_is_complete_and_explained(tmp_path):
    result = _run_meerut(tmp_path)
    assert len(result.ranking) == 44
    assert result.ranking["rank"].tolist() == list(range(1, 45))
    for _, row in result.ranking.head(5).iterrows():
        assert row["explanation"]
        assert row["top_criterion"] in config.CRITERIA
        assert 0.0 <= row["topsis_score"] <= 1.0
    assert 0.0 <= result.consistency_ratio < config.CR_THRESHOLD


def _meerut_profile() -> Profile:
    return Profile(budget_ceiling=9000.0, max_distance_km=5.0,
                   weights={"budget": 5, "distance": 4, "safety": 5,
                            "wifi": 3, "food": 2})


def test_custom_debate_on_user_shortlist(tmp_path):
    client = LLMClient(demo_mode=True, cache_path=tmp_path / "c2.json")
    df = enrich_listings(pd.read_csv(ROOT / "data" / "listings_meerut.csv"),
                         client)
    result = run_custom_debate(df, _meerut_profile(), client, [1, 2, 4])
    # The debate covers exactly the user's picks.
    assert set(result.subset_ranking["listing_id"]) == {1, 2, 4}
    assert set(result.picks.values()) <= {1, 2, 4}
    # The planted deposit scam (#4) cannot win a shortlist verdict.
    assert result.winner_id in (1, 2)
    assert result.winner_explanation
    # Still exactly one question per agent, never a second round.
    qa = [e for e in result.debate if e["type"] == "qa"]
    assert len(qa) <= 3
    assert len({e["agent"] for e in qa}) == len(qa)


def test_custom_debate_agents_see_all_factors(tmp_path):
    client = LLMClient(demo_mode=True, cache_path=tmp_path / "c3.json")
    df = enrich_listings(pd.read_csv(ROOT / "data" / "listings_meerut.csv"),
                         client)
    result = run_custom_debate(df, _meerut_profile(), client, [1, 2, 4])
    for facts in result.facts.values():
        for key in ("rent", "deposit_used", "brokerage_used",
                    "maintenance_used", "true_monthly_cost",
                    "fee_confidence", "needs_manual_review", "distance_km",
                    "agent_scores", "safety_score", "safety_flags",
                    "flag_justifications", "wifi", "food_included",
                    "amenities", "gender_policy"):
            assert key in facts


def test_custom_debate_needs_two_listings(tmp_path):
    client = LLMClient(demo_mode=True, cache_path=tmp_path / "c4.json")
    df = pd.read_csv(ROOT / "data" / "listings_meerut.csv")
    with pytest.raises(ValueError):
        run_custom_debate(df, _meerut_profile(), client, [1])
