"""Tests for the rule-based safety agent."""

from pathlib import Path

import pandas as pd
import pytest

from src.agents import safety_agent

ROOT = Path(__file__).resolve().parent.parent

_BASE = {
    "source": "Test", "rent": 10000, "deposit": 20000,
    "brokerage_stated": 0, "distance_km": 2.0, "amenities": "WiFi",
    "food_included": 1, "wifi": 1, "gender_policy": "Co-ed",
    "free_text_description":
        "Clean room on Main Road. Call 9812045673 for a visit.",
    "agreement_mentioned": 1, "advance_before_viewing": 0,
    "verified_claim": 0, "red_flag_label": 0, "is_synthetic": 0,
}


def _df(**overrides) -> pd.DataFrame:
    """Five listings at rent 10000 (median 10000); row 1 gets overrides."""
    rows = []
    for lid in range(1, 6):
        row = dict(_BASE, id=lid)
        if lid == 1:
            row.update(overrides)
        rows.append(row)
    return pd.DataFrame(rows)


def _flags(result, lid=1):
    return result.extras["flags"][lid]


def test_clean_listing_scores_one():
    result = safety_agent(_df())
    assert result.scores[1] == 1.0
    assert _flags(result) == []


def test_advance_before_viewing_fires():
    result = safety_agent(_df(advance_before_viewing=1))
    assert result.scores[1] == pytest.approx(0.70)
    assert any("Advance payment" in f for f in _flags(result))


def test_no_agreement_fires():
    result = safety_agent(_df(agreement_mentioned=0))
    assert result.scores[1] == pytest.approx(0.80)
    assert any("agreement" in f for f in _flags(result))


def test_unverifiable_verified_claim_fires_only_without_proof():
    result = safety_agent(_df(verified_claim=1))
    assert result.scores[1] == pytest.approx(0.85)
    assert any("Verified" in f for f in _flags(result))

    with_proof = _df(
        verified_claim=1,
        free_text_description="Verified owner. ID proof available. "
                              "Call 9812045673, Main Road.")
    result2 = safety_agent(with_proof)
    assert result2.scores[1] == 1.0


def test_below_market_rent_fires():
    # Median of the frame is 10000; 6000 is 40% below -> rule fires.
    result = safety_agent(_df(rent=6000))
    assert result.scores[1] == pytest.approx(0.85)
    assert any("below the dataset median" in f for f in _flags(result))
    # 7500 is only 25% below -> does not fire.
    result2 = safety_agent(_df(rent=7500))
    assert result2.scores[1] == 1.0


def test_no_contact_or_address_fires():
    result = safety_agent(_df(
        free_text_description="Nice room available for students."))
    assert result.scores[1] == pytest.approx(0.90)
    assert any("contact" in f.lower() for f in _flags(result))


def test_pressure_language_fires():
    result = safety_agent(_df(
        free_text_description="Limited rooms, pay today! "
                              "Call 9812045673, Main Road."))
    assert result.scores[1] == pytest.approx(0.90)
    assert any("Pressure" in f for f in _flags(result))


def test_all_rules_together_score_zero():
    result = safety_agent(_df(
        rent=5000, agreement_mentioned=0, advance_before_viewing=1,
        verified_claim=1,
        free_text_description="Limited rooms, pay today! Best room."))
    assert result.scores[1] == 0.0
    assert len(_flags(result)) == 6


def test_scores_bounded_on_real_datasets():
    for name in ("listings.csv", "listings_meerut.csv"):
        df = pd.read_csv(ROOT / "data" / name)
        result = safety_agent(df)
        assert set(result.scores) == set(df["id"])
        assert all(0.0 <= s <= 1.0 for s in result.scores.values())
        # Every flag carries a justification.
        for lid, details in result.extras["flag_details"].items():
            assert len(details) == len(result.extras["flags"][lid])
            assert all(details.values())
