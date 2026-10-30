"""Tests for fee extraction: quote verification is the anti-hallucination gate."""

from src.fee_estimator import estimate_fees, stub_extract, verify_and_clean

DESCRIPTION = (
    "2 BHK flat in Ganga Nagar, Meerut. Security deposit: 2 months rent. "
    "No brokerage - owner posted directly. The monthly maintenance is "
    "₹1,650/-. Water charges ₹300 per month. Call 9812045673.")


class FakeClient:
    """Stands in for LLMClient, returning one canned raw response."""

    def __init__(self, raw):
        self.raw = raw

    def call_json(self, system, user, fallback=None):
        return self.raw


def _raw(**overrides):
    raw = {"deposit": 20000, "brokerage": 0, "maintenance": 1650,
           "other_charges": [{"name": "Water charges", "amount": 300,
                              "evidence_quote": "Water charges ₹300"}],
           "evidence": {
               "deposit": "Security deposit: 2 months rent",
               "brokerage": "No brokerage",
               "maintenance": "monthly maintenance is ₹1,650"},
           "confidence": "high", "needs_manual_review": False}
    raw.update(overrides)
    return raw


def test_verify_keeps_charges_with_real_quotes():
    result = verify_and_clean(_raw(), DESCRIPTION)
    assert result["deposit"] == 20000
    assert result["brokerage"] == 0
    assert result["maintenance"] == 1650
    assert result["other_charges"][0]["amount"] == 300
    assert result["needs_manual_review"] is False
    assert result["confidence"] == "high"


def test_fabricated_quote_is_discarded():
    raw = _raw()
    raw["evidence"]["deposit"] = "Deposit of ₹99,999 to be paid in cash"
    result = verify_and_clean(raw, DESCRIPTION)
    assert result["deposit"] is None
    assert result["needs_manual_review"] is True
    # Untouched fields survive.
    assert result["maintenance"] == 1650


def test_value_without_any_quote_is_discarded():
    raw = _raw()
    raw["evidence"]["maintenance"] = None
    result = verify_and_clean(raw, DESCRIPTION)
    assert result["maintenance"] is None
    assert result["needs_manual_review"] is True


def test_other_charge_with_bad_quote_is_dropped():
    raw = _raw(other_charges=[{"name": "Club fee", "amount": 999,
                               "evidence_quote": "Club fee ₹999 yearly"}])
    result = verify_and_clean(raw, DESCRIPTION)
    assert result["other_charges"] == []
    assert result["needs_manual_review"] is True


def test_estimate_fees_end_to_end_with_fake_client():
    raw = _raw()
    raw["evidence"]["brokerage"] = "Brokerage waived for students"  # fake
    result = estimate_fees(DESCRIPTION, 10000, FakeClient(raw))
    assert result["deposit"] == 20000
    assert result["brokerage"] is None
    assert result["needs_manual_review"] is True


def test_stub_extract_matches_llm_schema_and_verifies():
    raw = stub_extract(DESCRIPTION, 10000)
    assert raw["deposit"] == 20000       # 2 months x ₹10,000 rent
    assert raw["brokerage"] == 0         # "No brokerage"
    assert raw["maintenance"] == 1650
    assert raw["other_charges"][0]["amount"] == 300
    cleaned = verify_and_clean(raw, DESCRIPTION)
    assert cleaned["deposit"] == 20000
    assert cleaned["needs_manual_review"] is False


def test_stub_extract_finds_nothing_in_plain_text():
    raw = stub_extract("A quiet room for students near the campus.", 8000)
    assert raw["deposit"] is None and raw["brokerage"] is None
    assert raw["maintenance"] is None and raw["other_charges"] == []
    assert raw["confidence"] == "low"
