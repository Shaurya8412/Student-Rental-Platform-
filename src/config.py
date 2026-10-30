"""Central configuration for the Student Rental Housing DSS.

All constants live here so behaviour can be tuned in one place:
paths, the user-profile defaults, AHP/TOPSIS settings, the safety
rule checklist, and LLM provider settings.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "listings_user_pg.csv"  # default dataset
DATASETS = {
    "Your PG Listings - Meerut (uploaded)":
        BASE_DIR / "data" / "listings_user_pg.csv",
    "Meerut (MagicBricks)": BASE_DIR / "data" / "listings_meerut.csv",
    "Bengaluru (sample)": BASE_DIR / "data" / "listings.csv",
}
DEFAULT_DATASET = "Your PG Listings - Meerut (uploaded)"
# Starting budget ceiling per dataset (local rent levels differ a lot).
DATASET_DEFAULT_BUDGET = {"Your PG Listings - Meerut (uploaded)": 8000.0,
                          "Meerut (MagicBricks)": 9000.0,
                          "Bengaluru (sample)": 15000.0}
CACHE_PATH = BASE_DIR / "cache" / "llm_cache.json"
RESULTS_DIR = BASE_DIR / "results"
WHATIF_LOG_PATH = RESULTS_DIR / "whatif_log.csv"

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
CITY = "Meerut"
CURRENCY = "₹"
STAY_MONTHS = 11  # assumed tenancy length used to amortise one-time fees

# Criteria used by AHP-TOPSIS, in a fixed order everywhere.
CRITERIA = ["budget", "distance", "safety", "wifi", "food"]

# Default sidebar profile for the demo.
DEFAULT_PROFILE = {
    "budget_ceiling": 15000.0,
    "max_distance_km": 5.0,
    "weights": {"budget": 5, "distance": 4, "safety": 5, "wifi": 3, "food": 2},
}

# ---------------------------------------------------------------------------
# AHP settings
# ---------------------------------------------------------------------------
# Saaty-style mapping from the difference of two 1-5 importance ratings to a
# pairwise-comparison intensity. diff 0 -> equal (1), diff 4 -> extreme (9).
SAATY_DIFF_MAP = {0: 1.0, 1: 3.0, 2: 5.0, 3: 7.0, 4: 9.0}

# Saaty random-consistency index by matrix size n.
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
            8: 1.41, 9: 1.45, 10: 1.49}
CR_THRESHOLD = 0.10  # consistency ratio below this is considered acceptable

# Non-compensatory safety floor applied by the Decision Agent BEFORE the
# compensatory TOPSIS ranking is presented: a listing whose safety score
# falls below this is "high risk" and cannot be recommended, however
# cheap it is. (Standard MCDM screening practice - bait pricing must
# not be able to buy its way past fraud signals.) TOPSIS closeness is
# still computed and shown for every listing for transparency.
SAFETY_FLOOR = 0.50

# Debate Arena: how many listings a user may hand to the agents.
ARENA_MIN_PICKS = 2
ARENA_MAX_PICKS = 5

# ---------------------------------------------------------------------------
# Safety checklist (rule-based). Edit this list to change the safety agent.
# "severity" is the score penalty when the rule fires; severities sum to 1.0
# so a listing that trips every rule scores 0.0.
# ---------------------------------------------------------------------------
SAFETY_RULES = [
    {"id": "advance_before_viewing",
     "label": "Advance payment demanded before viewing",
     "severity": 0.30},
    {"id": "no_written_agreement",
     "label": "No written agreement mentioned",
     "severity": 0.20},
    {"id": "unverifiable_verified_claim",
     "label": "'Verified' claim with no supporting proof in the text",
     "severity": 0.15},
    {"id": "below_market_rent",
     "label": "Rent more than 30% below the dataset median",
     "severity": 0.15,
     "params": {"threshold": 0.30}},
    {"id": "no_contact_or_address",
     "label": "No contact number or address detail in the listing",
     "severity": 0.10},
    {"id": "pressure_language",
     "label": "Pressure language (pay today / limited rooms)",
     "severity": 0.10},
]

# Phrases that count as pressure language (matched case-insensitively).
PRESSURE_PHRASES = [
    "limited rooms", "pay today", "hurry", "last room", "book now",
    "offer ends", "only 1 room left", "only one room left", "act fast",
    "immediately", "first come first served",
]

# Words that count as *some* supporting proof next to a "verified" claim.
PROOF_KEYWORDS = [
    "verified id", "id proof", "aadhaar", "registration", "registered",
    "licence", "license", "police verification", "rera",
]

# Words that count as address detail when no phone number is present.
ADDRESS_KEYWORDS = [
    "road", "nagar", "layout", "sector", "street", "circle", "block",
    "phase", "main", "cross", "near",
]

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
DEFAULT_PROVIDER = "groq"  # override with env LLM_PROVIDER
LLM_PROVIDERS = {
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "model": "gemini-2.0-flash",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "model": "claude-3-5-haiku-latest",
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
    },
}
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 3
