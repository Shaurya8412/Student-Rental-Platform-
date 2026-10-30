"""Fee estimation from free-text listing descriptions.

An LLM reads ``free_text_description`` and extracts the charges it can
find. Hard anti-hallucination rules, enforced in code after the call:

- Every extracted charge must carry a verbatim ``evidence_quote`` copied
  from the description. The prompt therefore asks the model for an
  ``evidence`` quote for the three scalar fields as well as for every
  item in ``other_charges``.
- After the call, :func:`verify_and_clean` checks that each quote is an
  exact substring of the source text. A charge whose quote is missing
  or not found is discarded (value set to null) and
  ``needs_manual_review`` is set to True.
- Values are never invented: if the model (or the offline fallback)
  finds nothing, the field stays null.

In demo mode (no API key) a deterministic regex extractor,
:func:`stub_extract`, plays the LLM's role. Because it only ever copies
spans out of the text, its quotes pass verification by construction,
and results are identical on every run.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from .llm_client import LLMClient

SYSTEM_PROMPT = """You extract rental charges from a student housing listing.
Return ONLY a JSON object with this shape:
{"deposit": number|null, "brokerage": number|null,
 "maintenance": number|null,
 "other_charges": [{"name": string, "amount": number,
                    "evidence_quote": string}],
 "evidence": {"deposit": string|null, "brokerage": string|null,
              "maintenance": string|null},
 "confidence": "high"|"medium"|"low", "needs_manual_review": boolean}

Rules:
- Amounts are monthly rupees unless the charge is one-time (deposit,
  brokerage). "2 months rent" means 2 x the monthly rent given to you.
- Every value you report MUST be backed by an evidence_quote copied
  VERBATIM from the listing text. If you cannot quote the text for a
  charge, report null for it. Never guess or infer an amount.
- "No brokerage" means brokerage is 0, quoted from the text.
- Set confidence by how complete and explicit the text is."""


def build_user_prompt(description: str, rent: float) -> str:
    """Build the user prompt for one listing description."""
    return (f"Monthly rent: {rent}\n"
            f"Listing text:\n\"\"\"\n{description}\n\"\"\"")


# ---------------------------------------------------------------------------
# Deterministic offline extractor (demo-mode stand-in for the LLM)
# ---------------------------------------------------------------------------
def _to_int(text: str) -> int:
    return int(text.replace(",", ""))


def _first_match(patterns: list[str], text: str) -> Optional[re.Match]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match
    return None


def stub_extract(description: str, rent: float) -> dict:
    """Extract charges with regexes; quotes are the matched substrings.

    Returns the same shape the LLM is asked for (including ``evidence``),
    so downstream verification treats both paths identically.
    """
    raw: dict = {"deposit": None, "brokerage": None, "maintenance": None,
                 "other_charges": [],
                 "evidence": {"deposit": None, "brokerage": None,
                              "maintenance": None},
                 "confidence": "low", "needs_manual_review": False}

    match = _first_match([
        r"deposit[^.]{0,30}?₹\s*([\d,]+)",
        r"security deposit[^.]{0,30}?₹\s*([\d,]+)",
        r"deposit[^.]{0,25}?(\d+)\s*months?\s*(?:of\s*)?rent",
        r"(\d+)\s*months?\s*(?:rent\s*)?(?:as\s*)?(?:security\s*)?deposit",
    ], description)
    if match:
        value = _to_int(match.group(1))
        if "month" in match.group(0).lower() and value < 100:
            value = int(value * rent)
        raw["deposit"] = value
        raw["evidence"]["deposit"] = match.group(0)

    if re.search(r"no brokerage", description, flags=re.IGNORECASE):
        match = re.search(r"no brokerage", description, flags=re.IGNORECASE)
        raw["brokerage"] = 0
        raw["evidence"]["brokerage"] = match.group(0)
    else:
        match = _first_match([
            r"brokerage[^.]{0,30}?₹\s*([\d,]+)",
            r"brokerage[^.]{0,25}?(\d+)\s*months?",
            r"(\d+)\s*months?\s*brokerage",
        ], description)
        if match:
            value = _to_int(match.group(1))
            if "month" in match.group(0).lower() and value < 100:
                value = int(value * rent)
            raw["brokerage"] = value
            raw["evidence"]["brokerage"] = match.group(0)

    match = _first_match([
        r"maintenance[^.]{0,30}?₹\s*([\d,]+)",
        r"maintenance\s+(?:included|free)",
    ], description)
    if match:
        raw["maintenance"] = (_to_int(match.group(1))
                              if match.lastindex else 0)
        raw["evidence"]["maintenance"] = match.group(0)

    other_patterns = [
        ("Water charges", r"water charges[^.]{0,20}?₹\s*([\d,]+)"),
        ("Electricity charges", r"electricity charges[^.]{0,20}?₹\s*([\d,]+)"),
        ("Cleaning charges", r"cleaning charges[^.]{0,20}?₹\s*([\d,]+)"),
        ("Parking charges", r"parking charges[^.]{0,20}?₹\s*([\d,]+)"),
        ("Mess charges", r"mess charges[^.]{0,20}?₹\s*([\d,]+)"),
    ]
    for name, pattern in other_patterns:
        match = re.search(pattern, description, flags=re.IGNORECASE)
        if match:
            raw["other_charges"].append({
                "name": name, "amount": _to_int(match.group(1)),
                "evidence_quote": match.group(0)})

    found = sum(raw[f] is not None
                for f in ("deposit", "brokerage", "maintenance"))
    raw["confidence"] = "high" if found >= 2 else (
        "medium" if found == 1 else "low")
    return raw


# ---------------------------------------------------------------------------
# Post-call verification (the anti-hallucination gate)
# ---------------------------------------------------------------------------
def verify_and_clean(raw: dict, description: str) -> dict:
    """Verify every evidence quote against the source text.

    Returns exactly the public schema:
    {deposit, brokerage, maintenance, other_charges, confidence,
     needs_manual_review}. Charges without a verbatim quote in the text
    are discarded and flag the listing for manual review.
    """
    result: dict = {"deposit": None, "brokerage": None, "maintenance": None,
                    "other_charges": [], "confidence": "low",
                    "needs_manual_review": False}
    evidence = raw.get("evidence") or {}

    for field in ("deposit", "brokerage", "maintenance"):
        value = raw.get(field)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            result["needs_manual_review"] = True
            continue
        if value < 0:
            result["needs_manual_review"] = True
            continue
        quote = evidence.get(field)
        if not quote or quote not in description:
            # No verbatim evidence -> the value cannot be trusted.
            result["needs_manual_review"] = True
            continue
        result[field] = int(value) if value == int(value) else value

    for charge in raw.get("other_charges") or []:
        quote = charge.get("evidence_quote")
        amount = charge.get("amount")
        if (quote and quote in description
                and isinstance(amount, (int, float)) and amount >= 0):
            result["other_charges"].append({
                "name": str(charge.get("name", "Charge")),
                "amount": amount, "evidence_quote": quote})
        else:
            result["needs_manual_review"] = True

    confidence = raw.get("confidence")
    result["confidence"] = confidence if confidence in (
        "high", "medium", "low") else "low"
    if raw.get("needs_manual_review"):
        result["needs_manual_review"] = True
    return result


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def estimate_fees(description: str, rent: float, client: LLMClient) -> dict:
    """Estimate the fees for one listing (LLM call + verification)."""
    raw = client.call_json(
        SYSTEM_PROMPT,
        build_user_prompt(description, rent),
        fallback=lambda: stub_extract(description, rent),
    )
    return verify_and_clean(raw, description)


def enrich_listings(df: pd.DataFrame, client: LLMClient) -> pd.DataFrame:
    """Add fee-estimate columns to a listings DataFrame.

    Adds: est_deposit, est_brokerage, est_maintenance (float or NaN),
    other_charges (list), fee_confidence, needs_manual_review.
    """
    out = df.copy()
    records = [estimate_fees(str(row.free_text_description), float(row.rent),
                             client)
               for row in out.itertuples()]
    out["est_deposit"] = [r["deposit"] for r in records]
    out["est_brokerage"] = [r["brokerage"] for r in records]
    out["est_maintenance"] = [r["maintenance"] for r in records]
    out["other_charges"] = [r["other_charges"] for r in records]
    out["fee_confidence"] = [r["confidence"] for r in records]
    out["needs_manual_review"] = [r["needs_manual_review"] for r in records]
    return out
