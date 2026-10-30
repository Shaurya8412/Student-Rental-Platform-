# Multi-Agent Decision Support System for Student Rental Housing

A demo-ready decision-support tool that ranks student PG / hostel /
shared-flat listings by **true cost, distance, safety and fraud risk**,
and explains every recommendation. Three specialist agents (Budget,
Location, Safety) score independently; a Decision Agent reconciles them
with **AHP-TOPSIS** after **one bounded debate round**. The design is
deliberately shallow — 3 specialists + 1 decision agent — as a hedge
against the multi-agent coordination failures documented in the
literature.

The implementation follows the five-module architecture proposed in
the project's literature review (*A Multi-Agent Decision Support
System for Rental Housing Selection: An Extended Literature Review
and Research Gap Analysis*):

| Module (paper §V–VI) | Implementation |
|---|---|
| Fee / True-Cost Estimator | `src/fee_estimator.py` — LLM extraction with verbatim-quote verification |
| Red-Flag Checklist | Safety agent in `src/agents.py`, rules in `src/config.py` |
| Decision Engine (AHP-TOPSIS) | `src/topsis.py` + `src/decision_agent.py` |
| Agent Debate Mechanism | One bounded round in `src/decision_agent.py` |
| What-If Recalculation | `src/whatif.py` — re-runs only the affected agent |
| Evaluation Harness | `evaluation/evaluate.py` (paper §VII) |

How the design answers the review's research gaps: explainable scoring
instead of a black-box rank (gap in [4], [14]–[16]); fraud signals
folded *into* the score rather than a standalone classifier ([5],
[21]–[23]); debate applied to a real personal decision ([6], [24],
[25]); AHP-TOPSIS reconciliation instead of a weighted average ([28],
[29]); true cost (deposit + brokerage + maintenance), not headline
rent ([10], [11], [17]); shallow agent graph against coordination
failures ([20]).

## Datasets

Three datasets ship with the repo; the app has a switcher and **the
user's own PG dataset is the default**.

- **`data/listings_user_pg.csv` — the user's collected PG dataset
  (20 real listings, default).** Converted by
  `python data/convert_user_pg_csv.py` from `data/user_pg_raw.csv`
  (PGs collected around CCS University, Meerut, from MagicBricks,
  99acres, Justdial and HexaHome, each row with its source URL and
  area coordinates). Photos come from each row's own source page —
  the MagicBricks rows are matched to their exact listing photos,
  the HexaHome rows use their own photos. Deposit/brokerage are not
  stated in the source data (0, so true cost = rent); the 8 rows
  without a stated distance get a haversine estimate from their
  coordinates to CCS University. These are real rows
  (`is_synthetic = 0`, `red_flag_label = 0`).
- **`data/listings_meerut.csv` — Meerut, sourced from MagicBricks
  patterns (44 listings).** Modelled on live MagicBricks Meerut
  listings checked 2026-10-29: real localities (Ganga Nagar, Shastri
  Nagar, Pallav Puram, Modipuram, Kanker Khera, Vedvyash Puri, Rohta
  Road, Meerut Cantt, Mangal Pandey Nagar, Garh Road…), real rent
  bands (Meerut rentals run ≈ ₹3,000–₹47,000/month; student stock
  clusters at ₹4,000–₹15,000), MagicBricks wording ("posted directly
  by the owner, so no brokerage is payable", "The monthly maintenance
  is ₹…/-"), and 1–2 months' security deposit. `distance_km` is the
  approximate road distance to the CCS University main campus (Garh
  Road), the pilot's reference campus. All rows are a curated sample
  (`is_synthetic = 1`); 8 rows are planted MagicBricks-style scam
  patterns (UPI deposit-before-viewing, fake "verified" owner, bait
  pricing) with `red_flag_label = 1` — including deposit scams of the
  kind the review links to MagicBricks [11]. Regenerate or inspect
  with `python data/generate_meerut_listings.py`.
- **`data/listings.csv` — Bengaluru sample (40 listings)** generated
  by `python data/generate_listings.py`; 8 planted scam rows are the
  ones marked `is_synthetic = 1`, `red_flag_label = 1`.

Both files share one schema:
`id, source, rent, deposit, brokerage_stated, distance_km, amenities,
food_included, wifi, gender_policy, free_text_description,
agreement_mentioned, advance_before_viewing, verified_claim,
red_flag_label, is_synthetic`, plus three presentation columns used
by the UI: `title` (e.g. "Girls PG in Mangal Pandey Nagar"),
`image_url` (real MagicBricks property photos from img.staticmb.com,
harvested from the live Meerut pages on 2026-10-29 and assigned
illustratively by property type), `listing_url` (the real
MagicBricks search page for that locality, where the actual listings
and owner/agent contacts live), and `latitude`/`longitude`
(geocoded locality centres plus a small deterministic jitter),
which power the Near-you feature.

### Replacing the sample with real collected data

1. Collect 30–50 listings for one city (manually, per the paper's
   pilot scope). Record the headline fields exactly as posted.
2. Set `agreement_mentioned`, `advance_before_viewing` and
   `verified_claim` from what the listing actually says/does.
3. Hand-label `red_flag_label` (1 = you verified it is a scam/bait)
   and set `is_synthetic = 0` for genuinely collected rows.
4. Save over the dataset file (or add it to `config.DATASETS`) and
   validate: `python -m src.validator data/<file>.csv`.
5. Clear `cache/llm_cache.json` so fee estimates are recomputed.

## Setup

```bash
cd student-rental-dss
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### LLM configuration (required — the app runs online only)

The app has **no offline/demo mode**: fee extraction, the agent
debate and the explanations call a live LLM API, and the app stops
with setup instructions if no key is configured. Easiest setup:
save `.env.example` as `.env` in the project folder, fill in one
key, and launch — the app loads `.env` automatically at startup
(shell-exported variables work too and take precedence):

| Variable | Values |
|---|---|
| `LLM_PROVIDER` | `gemini` (default) · `openai` · `anthropic` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | your key |
| `LLM_MODEL` | optional model override |

Temperature is fixed at 0 and JSON output is forced. All responses
are cached to `cache/llm_cache.json` (SHA-256 of provider + model +
prompts) to save cost on repeat runs; if an individual API call
fails after retries, that one step uses a deterministic fallback so
a single flaky call never breaks the ranking. (The deterministic
demo mode still exists inside `src/llm_client.py` for the test
suite and the evaluation harness, which run offline by design.)

## Run

```bash
# 1. Validate the datasets
python -m src.validator data/listings_meerut.csv
python -m src.validator data/listings.csv

# 2. Run the test suite (41 tests)
python -m pytest tests/ -q

# 3. Run the evaluation harness -> results/
python evaluation/generate_sample_eval_data.py   # sample students + survey
python evaluation/evaluate.py                    # Meerut by default
python evaluation/evaluate.py --dataset data/listings.csv

# 4. Launch the app
streamlit run app.py
```

## How it works

**Fee estimator.** The LLM reads each free-text description and
returns deposit / brokerage / maintenance / other charges, each
backed by a verbatim `evidence_quote`. Code then verifies every quote
is an exact substring of the listing text; unverifiable charges are
discarded and the listing is flagged `needs_manual_review`. Values are
never invented — no quote, no charge.

**Agents.** Budget: true monthly cost = rent + (deposit +
brokerage)/11 months + maintenance, scored against your ceiling.
Location: linear in distance vs your maximum. Safety: weighted
red-flag checklist (advance before viewing 0.30, no written agreement
0.20, unverifiable "verified" claim 0.15, rent >30% below the dataset
median 0.15, no contact/address 0.10, pressure language 0.10) — edit
`SAFETY_RULES` in `src/config.py`.

**Decision Agent.** If the agents' top picks disagree, each
disagreeing agent is asked exactly one targeted follow-up question
(one round, never two). The final ranking is AHP-TOPSIS: your 1–5
importance ratings become a Saaty pairwise matrix (weights via the
principal eigenvector; consistency ratio reported, 0.03 for the
default profile). Explanations are computed from the numbers — the
criterion that contributed most, true cost, flags — and the LLM only
phrases those facts.

**Safety floor.** Pure TOPSIS is compensatory: a bait-priced scam can
outscore honest listings on budget. Screening is standard MCDM
practice, so listings with safety score < 0.50 (`config.SAFETY_FLOOR`)
are excluded from recommendations; their TOPSIS scores are still
computed and disclosed in the app.

**UI & Debate Arena.** The app presents photo cards (MagicBricks
property images, true cost, safety and fee-confidence chips, red-flag
badges) with a *View original listing* button that opens the source
locality page on MagicBricks for owner/agent contact. In the
**Debate Arena** you hand the agents any 2–5 listings yourself: each
agent states its pick within your shortlist, agents who disagree with
the shortlist's AHP-TOPSIS leader get exactly one follow-up question,
and the debate brief carries **all** factors — cost breakdown
(rent/deposit/brokerage/maintenance), distance, every safety flag
with its justification, WiFi, food, amenities and fee confidence —
not just the agent's own criterion. The verdict is TOPSIS recomputed
on your shortlist only, with the safety floor applied.

**Near you (live location).** The sidebar can request the browser's
live location (via the `streamlit-geolocation` component; the user
must grant permission, and the coordinates stay in the session — they
are never stored or sent to the LLM). Listings carry locality-level
coordinates, so the app computes haversine distances locally
(`src/geo.py`) and shows a **Near you** section: the six closest
listings (safety floor still applied), each annotated with its
distance from the user, plus a **Leaflet map** (via
`streamlit-folium`: user in red, nearest PGs as green markers with
popups, all other listings as grey dots). If the user is far from
the pilot city, the section says so and shows the closest listings
anyway.

**Budget behaviour.** Criterion importance weights are fixed
defaults — there are no 1–5 importance inputs. The budget slider
re-ranks instantly, and the sidebar toggle *"Show only PGs within
my budget"* (on by default) additionally hides every PG whose true
monthly cost exceeds the ceiling from the recommendation and
Near-you lists, so changing the budget visibly changes the PGs
shown; with the toggle off, over-budget PGs stay visible with an
"Over budget" badge and a budget score of 0.

**What-If.** The panel tries a different budget ceiling: only the
Budget agent re-runs, TOPSIS re-runs each time (the recalculation
engine also supports distance changes, but the UI no longer exposes
a max-distance field — the Location agent uses a fixed 5 km
reference). Measured latency ≈ 6 ms mean / 14 ms p95 over 50 runs
(target < 2 s); every app recalculation is logged to
`results/whatif_log.csv`.

## Evaluation results (Meerut modelled sample, offline harness)

From `results/results_table.csv` (sample student/survey data —
replace via `evaluation/` for a real study):

| Metric | Value |
|---|---|
| Spearman ρ vs student rankings (mean, n=12) | 0.48 |
| Survey means: clarity / trust / usefulness (1–5) | 4.24 / 3.72 / 4.04 |
| Survey "reason matched what mattered" | 80% |
| Red-flag checklist precision / recall / F1 | 0.78 / 0.88 / 0.82 |
| What-if latency mean / p95 (50 runs) | 0.007 s / 0.008 s |
| AHP consistency ratio (default profile) | 0.03 |

## Project structure

```
app.py                      Streamlit UI
data/
  listings_meerut.csv       Meerut pilot (MagicBricks), default
  listings.csv              Bengaluru sample
  generate_meerut_listings.py / generate_listings.py
src/
  config.py                 all constants: rules, thresholds, providers
  llm_client.py             cached, retrying, JSON-only LLM access
  validator.py              schema checks (library + CLI)
  fee_estimator.py          quote-verified fee extraction
  agents.py                 Budget / Location / Safety agents
  topsis.py                 AHP weights + TOPSIS (pure numpy)
  decision_agent.py         one debate round + final ranking
  whatif.py                 minimal re-run recalculation + timing log
  geo.py                    haversine distance + nearest listings
tests/                      pytest: TOPSIS, safety, fees, agents, decision, geo
evaluation/                 Spearman, survey, red-flag P/R/F1, latency
results/                    evaluation outputs + what-if log
cache/                      LLM response cache (demo reproducibility)
```

## Limitations

- **Hallucination risk.** The fee estimator reads free text with an
  LLM; quote verification and the manual-review flag mitigate but do
  not eliminate invented or misread charges. In demo mode a regex
  extractor replaces the LLM and cannot invent values, but it also
  misses charges phrased unusually.
- **Small static datasets.** 40–44 listings for one city each, curated
  samples (the Meerut file is modelled on live MagicBricks patterns,
  not a scrape). Results do not generalize to other cities or to the
  live market; there is no live data feed.
- **Rule-based checklist.** Precision/recall (0.70 / 0.88 on the
  Meerut sample) will trail the trained multimodal classifiers in the
  literature (F1 ≈ 0.96 in [5]); one planted scam in each dataset is
  deliberately designed to evade the rules, and real scams will too.
- **Sample evaluation data.** The student rankings and survey
  responses in `evaluation/` are generated samples so the harness
  runs end-to-end; the reported Spearman/survey numbers are
  illustrative, not study results.
- **Neutrality is a design claim, not an audit.** The system earns
  nothing from listings or bookings, but no external party has
  certified this, and dataset curation choices can still bias ranks.
- **Debate is bounded by design.** One round with templated questions
  limits coordination-failure exposure [20] but also limits how much
  disagreement can be resolved; the debate never overrides TOPSIS.
- **Single reference campus per dataset** (CCSU for Meerut) — students
  at other institutions should read distances accordingly.

## Assumptions made

- LLM provider defaults to Gemini; OpenAI/Anthropic selectable by env
  var, all through one client with identical JSON contracts.
- One-time fees are amortized over an 11-month stay (configurable).
- `brokerage_stated` / `deposit` columns are the fallback when the
  text yields no verified quote for that charge.
- All criteria enter TOPSIS as benefit scores (agents convert cost
  and distance), per the paper's five-criterion design.
