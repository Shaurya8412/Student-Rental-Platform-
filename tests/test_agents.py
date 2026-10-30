"""Tests for budget/location agents and global scoring bounds."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents import Profile, budget_agent, location_agent, safety_agent

ROOT = Path(__file__).resolve().parent.parent


def _profile(**overrides) -> Profile:
    base = {"budget_ceiling": 10000.0, "max_distance_km": 5.0,
            "weights": {"budget": 5, "distance": 4, "safety": 5,
                        "wifi": 3, "food": 2}}
    base.update(overrides)
    return Profile(**base)


def _listing(**overrides) -> dict:
    row = {"id": 1, "source": "Test", "rent": 5000, "deposit": 10000,
           "brokerage_stated": 0, "distance_km": 2.0,
           "amenities": "WiFi", "food_included": 1, "wifi": 1,
           "gender_policy": "Co-ed",
           "free_text_description": "Room on Main Road. Call 9812045673.",
           "agreement_mentioned": 1, "advance_before_viewing": 0,
           "verified_claim": 0, "red_flag_label": 0, "is_synthetic": 0}
    row.update(overrides)
    return row


def test_budget_true_cost_hand_computed():
    # rent 5000 + (10000 deposit + 0 brokerage)/11 + 500 maintenance
    df = pd.DataFrame([_listing(est_deposit=10000, est_brokerage=0,
                                est_maintenance=500)])
    result = budget_agent(df, _profile(), stay_months=11)
    expected_cost = 5000 + 10000 / 11 + 500
    assert result.extras["true_costs"][1] == pytest.approx(expected_cost)
    assert result.scores[1] == pytest.approx(1 - expected_cost / 10000)


def test_budget_over_ceiling_scores_zero():
    df = pd.DataFrame([_listing(rent=9500, est_deposit=np.nan,
                                est_brokerage=np.nan, est_maintenance=900)])
    # Falls back to stated columns: 9500 + 10000/11 + 900 > 10000 ceiling.
    result = budget_agent(df, _profile())
    assert result.scores[1] == 0.0
    assert "over the" in result.reasons[1]


def test_budget_falls_back_to_stated_columns():
    df = pd.DataFrame([_listing(deposit=11000, brokerage_stated=5500)])
    result = budget_agent(df, _profile())
    expected = 5000 + (11000 + 5500) / 11
    assert result.extras["true_costs"][1] == pytest.approx(expected)


def test_location_scores():
    df = pd.DataFrame([
        _listing(id=1, distance_km=0.0), _listing(id=2, distance_km=2.5),
        _listing(id=3, distance_km=5.0), _listing(id=4, distance_km=9.0)])
    result = location_agent(df, _profile())
    assert result.scores[1] == 1.0
    assert result.scores[2] == pytest.approx(0.5)
    assert result.scores[3] == 0.0
    assert result.scores[4] == 0.0
    assert result.top_id == 1


def test_all_agents_bounded_on_real_datasets():
    profile = Profile.default()
    for name in ("listings.csv", "listings_meerut.csv"):
        df = pd.read_csv(ROOT / "data" / name)
        for result in (budget_agent(df, profile),
                       location_agent(df, profile),
                       safety_agent(df)):
            assert set(result.scores) == set(df["id"])
            assert all(0.0 <= s <= 1.0
                       for s in result.scores.values())
            assert result.top_id in set(df["id"])
            assert result.top_reason  # non-empty justification
            assert sorted(result.ranked_ids) == sorted(df["id"])
