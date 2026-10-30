"""Streamlit app: Multi-Agent Decision Support for Student Rental Housing.

Semi-production UI: photo cards with links to the original listings, a
safety-floor banner, the automatic agent debate, a Debate Arena where
the user hands specific listings to the agents, a Near-you section
driven by the browser's live location, and a what-if panel.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load the API key (and provider settings) from a `.env` file in the
# project folder, if one exists. Real shell environment variables win.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()
except ImportError:
    pass

import pandas as pd
import streamlit as st

from src import config
from src.agents import Profile
from src.decision_agent import AGENT_LABEL, run_custom_debate, run_decision
from src.fee_estimator import enrich_listings
from src.geo import add_user_distance
from src.llm_client import LLMClient
from src.validator import validate_listings
from src.whatif import recalculate

try:  # Optional component: the user's live location from their browser.
    from streamlit_geolocation import streamlit_geolocation
except ImportError:  # pragma: no cover - component not installed
    streamlit_geolocation = None

# Streamlit Cloud: keys pasted in App settings -> Secrets are bridged
# into the environment, so the rest of the app sees them like a .env.
try:
    for _name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                  "LLM_PROVIDER", "LLM_MODEL"):
        if _name in st.secrets and not os.getenv(_name):
            os.environ[_name] = str(st.secrets[_name])
except Exception:
    pass

st.set_page_config(page_title="Student Rental DSS", page_icon="🏠",
                   layout="wide")

_CONFIDENCE = {"high": ("🟢", "Fee confidence: high"),
               "medium": ("🟡", "Fee confidence: medium"),
               "low": ("🔴", "Fee confidence: low")}
_CRITERION_LABEL = {"budget": "Budget", "distance": "Distance",
                    "safety": "Safety", "wifi": "WiFi", "food": "Food"}

_CSS = """
<style>
.hero {background: linear-gradient(120deg, #0b2a4a 0%, #0e5e6f 100%);
  color: #fff; padding: 26px 30px; border-radius: 16px; margin-bottom: 18px;}
.hero-title {font-size: 1.9rem; font-weight: 800; letter-spacing: .2px;}
.hero-sub {opacity: .92; margin-top: 6px; font-size: 1.02rem;}
.hero-pills {margin-top: 14px;}
.hero-pills span {background: rgba(255,255,255,.16); border: 1px solid
  rgba(255,255,255,.35); padding: 4px 12px; border-radius: 999px;
  margin-right: 8px; font-size: .85rem; white-space: nowrap;}
.lcard {background: #ffffff; border: 1px solid #e3e8ee; border-radius: 16px;
  overflow: hidden; box-shadow: 0 4px 18px rgba(11,42,74,.08);
  margin-top: 10px;}
.lcard-imgwrap {position: relative; height: 200px;
  background: linear-gradient(135deg, #dbe7f3, #b8d4ea);}
.lcard-imgwrap .ph {position: absolute; inset: 0; display: flex;
  align-items: center; justify-content: center; font-size: 3.2rem;
  opacity: .45;}
.lcard-img {position: absolute; inset: 0; width: 100%; height: 200px;
  object-fit: cover;}
.rank-badge {position: absolute; top: 12px; left: 12px; z-index: 2;
  background: #0b2a4a; color: #fff; font-weight: 700; padding: 5px 12px;
  border-radius: 999px; font-size: .9rem;}
.score-badge {position: absolute; top: 12px; right: 12px; z-index: 2;
  background: rgba(255,255,255,.94); color: #0b2a4a; font-weight: 700;
  padding: 5px 12px; border-radius: 999px; font-size: .9rem;}
.risk-badge {position: absolute; bottom: 12px; left: 12px; z-index: 2;
  background: #b3261e; color: #fff; font-weight: 700; padding: 4px 11px;
  border-radius: 999px; font-size: .8rem;}
.lcard-body {padding: 16px 20px 18px;}
.lcard-title {font-size: 1.22rem; font-weight: 750; color: #10243e;}
.lcard-sub {color: #5b6b7f; font-size: .9rem; margin-top: 2px;}
.cost-row {display: flex; align-items: baseline; gap: 10px;
  margin-top: 10px; flex-wrap: wrap;}
.cost {font-size: 1.55rem; font-weight: 800; color: #0b2a4a;}
.cost-label {color: #5b6b7f; font-size: .88rem;}
.scorebar {height: 8px; background: #e8edf3; border-radius: 99px;
  margin: 12px 0 4px;}
.scorebar > div {height: 8px; border-radius: 99px;
  background: linear-gradient(90deg, #0e7c66, #19b394);}
.chips {margin-top: 10px; line-height: 2;}
.chip {display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: .82rem; margin-right: 6px; background: #eef3f8;
  color: #27435f; border: 1px solid #d8e2ec;}
.chip.green {background: #e6f6ef; color: #0c6b4f; border-color: #bfe6d4;}
.chip.amber {background: #fdf3e0; color: #8a5a00; border-color: #f0dcb2;}
.chip.red {background: #fdecea; color: #9c1c14; border-color: #f5c6c2;}
.expl {color: #33475e; margin-top: 10px; line-height: 1.5;}
.breakdown {color: #7a8aa0; font-size: .83rem; margin-top: 8px;}
.review-banner {background: #fff7e6; border: 1px solid #f0dcb2;
  color: #8a5a00; border-radius: 10px; padding: 8px 12px;
  margin-top: 10px; font-size: .9rem;}
.verdict {background: linear-gradient(120deg, #e9f7f2, #eef6fb);
  border: 1px solid #bfe6d4; border-radius: 14px; padding: 18px 22px;}
.verdict h4 {margin: 0 0 6px; color: #0b2a4a;}
.agent-pick {font-size: 1.02rem; font-weight: 700; color: #10243e;}
.footer {color: #7a8aa0; font-size: .85rem; margin-top: 26px;
  border-top: 1px solid #e3e8ee; padding-top: 12px;}
@media (prefers-color-scheme: dark) {
  .lcard {background: #17263a; border-color: #2c4258;}
  .lcard-title, .cost, .agent-pick {color: #e8f0f8;}
  .expl {color: #c3d2e2;} .lcard-sub, .cost-label {color: #9db1c6;}
  .scorebar {background: #2c4258;}
  .verdict {background: #14332c; border-color: #1f5c4a;}
  .verdict h4 {color: #e8f0f8;}
}
</style>
"""


# ---------------------------------------------------------------------------
# Cached data access (file stamp in the key so regenerated CSVs are picked up)
# ---------------------------------------------------------------------------
def _stamp(path_str: str) -> float:
    try:
        return Path(path_str).stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_listings(path_str: str, stamp: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner="Estimating fees from listing text...")
def get_enriched(provider: str, model: str,
                 path_str: str, stamp: float) -> pd.DataFrame:
    client = LLMClient(provider=provider, model=model, demo_mode=False)
    return enrich_listings(load_listings(path_str, stamp), client)


def _has_api_key() -> bool:
    provider = os.getenv("LLM_PROVIDER", config.DEFAULT_PROVIDER)
    return bool(os.getenv(config.LLM_PROVIDERS[provider]["env_key"]))


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------
def _esc(text) -> str:
    return html.escape(str(text))


def _card_html(row, budget_ceiling: float | None = None) -> str:
    conf_icon, conf_text = _CONFIDENCE.get(row["fee_confidence"],
                                           _CONFIDENCE["low"])
    chips = [
        f'<span class="chip">📍 {_esc(row["distance_km"])} km</span>',
        f'<span class="chip">🛡️ Safety {row["safety_score"]:.2f}</span>',
        f'<span class="chip">{"📶 WiFi" if row["wifi"] else "🚫 No WiFi"}</span>',
        f'<span class="chip">{"🍽️ Food included" if row["food_included"] else "🍳 No food"}</span>',
        f'<span class="chip">{conf_icon} {_esc(conf_text)}</span>',
        f'<span class="chip">Rent ₹{row["rent"]:,.0f}</span>',
    ]
    if budget_ceiling is not None \
            and row["true_monthly_cost"] > budget_ceiling:
        chips.append('<span class="chip amber">💸 Over budget</span>')
    if row["flags"]:
        chips += [f'<span class="chip red">🚩 {_esc(f)}</span>'
                  for f in row["flags"]]
    else:
        chips.append('<span class="chip green">✅ No red flags</span>')
    img = (f'<img class="lcard-img" src="{_esc(row["image_url"])}" '
           f'onerror="this.remove()"/>') if row["image_url"] else ""
    risk = ('<span class="risk-badge">⛔ High risk</span>'
            if row["high_risk"] else "")
    review = ('<div class="review-banner">⚠️ <b>Needs manual review:</b> '
              'some charges in the listing text could not be verified '
              'against a quoted source — confirm fees before paying.</div>'
              if row["needs_manual_review"] else "")
    return f"""
<div class="lcard">
  <div class="lcard-imgwrap"><div class="ph">🏠</div>{img}
    <span class="rank-badge">#{int(row['rank'])}</span>
    <span class="score-badge">TOPSIS {row['topsis_score']:.3f}</span>{risk}
  </div>
  <div class="lcard-body">
    <div class="lcard-title">{_esc(row['title'])}</div>
    <div class="lcard-sub">Listing #{int(row['listing_id'])} ·
      {_esc(row['source'])} · Top contributor:
      {_esc(row['top_criterion'])} ({row['top_criterion_share']:.0%})</div>
    <div class="cost-row"><span class="cost">
      ₹{row['true_monthly_cost']:,.0f}</span>
      <span class="cost-label">true monthly cost (rent + amortised
      deposit/brokerage + maintenance)</span></div>
    <div class="scorebar"><div style="width:{row['topsis_score'] * 100:.0f}%"></div></div>
    <div class="chips">{''.join(chips)}</div>
    <p class="expl">{_esc(row['explanation'])}</p>
    {review}
    <div class="breakdown">Criterion scores — budget
      {row['budget_score']:.2f} · distance {row['location_score']:.2f} ·
      safety {row['safety_score']:.2f} · wifi {row['wifi']:.0f} ·
      food {row['food_included']:.0f}</div>
  </div>
</div>"""


def _render_listing(row, key_prefix: str, note: str | None = None,
                    budget_ceiling: float | None = None) -> None:
    """One listing card plus its action row (original link + arena)."""
    st.markdown(_card_html(row, budget_ceiling), unsafe_allow_html=True)
    if note:
        st.caption(note)
    lid = int(row["listing_id"])
    cols = st.columns([1, 1, 2])
    if row["listing_url"]:
        cols[0].link_button("🔗 View original listing",
                            row["listing_url"],
                            help="Open the source listing page to "
                                 "contact the owner/agent")
    in_arena = lid in st.session_state.get("arena_select", [])
    cols[1].button(
        "✓ In Debate Arena" if in_arena else "⚖️ Debate this",
        key=f"arena_btn_{key_prefix}_{lid}", on_click=_toggle_arena,
        args=(lid,), help="Add or remove this listing from the "
                          "Debate Arena below")
    st.write("")


def _render_map(near: pd.DataFrame, enriched: pd.DataFrame,
                user_lat: float, user_lon: float,
                nearest_km: float) -> None:
    """Leaflet map (via folium): the user in red, the nearest PGs as
    green home markers with popups, every other listing as a small
    grey dot for context."""
    coord = enriched.set_index("id")
    try:
        import folium
        from streamlit_folium import st_folium
    except ImportError:
        pts = pd.DataFrame([{
            "lat": float(coord.loc[i, "latitude"]),
            "lon": float(coord.loc[i, "longitude"])}
            for i in near["listing_id"]])
        st.map(pts)
        return
    zoom = 14 if nearest_km <= 5 else (12 if nearest_km <= 25 else
                                       (9 if nearest_km <= 100 else 5))
    fmap = folium.Map(location=[user_lat, user_lon], zoom_start=zoom,
                      control_scale=True)
    folium.Marker(
        [user_lat, user_lon], tooltip="You are here",
        popup="<b>You are here</b>",
        icon=folium.Icon(color="red", icon="user", prefix="fa"),
    ).add_to(fmap)
    near_ids = {int(i) for i in near["listing_id"]}
    for lid, crow in coord.iterrows():
        if int(lid) in near_ids or pd.isna(crow["latitude"]):
            continue
        folium.CircleMarker(
            [float(crow["latitude"]), float(crow["longitude"])],
            radius=4, color="#9aa7b4", fill=True, fill_opacity=0.6,
            tooltip=_esc(crow["title"])).add_to(fmap)
    dist_by_id = {}
    if "distance_from_user_km" in near.columns:
        dist_by_id = {int(r["listing_id"]): float(
            r["distance_from_user_km"]) for _, r in near.iterrows()}
    for _, row in near.iterrows():
        lid = int(row["listing_id"])
        crow = coord.loc[lid]
        popup = (f"<b>{_esc(crow['title'])}</b><br>"
                 f"True cost ₹{row['true_monthly_cost']:,.0f}/month<br>"
                 f"{dist_by_id.get(lid, 0.0):.1f} km from you · "
                 f"safety {row['safety_score']:.2f}")
        folium.Marker(
            [float(crow["latitude"]), float(crow["longitude"])],
            tooltip=_esc(crow["title"]), popup=popup,
            icon=folium.Icon(color="green", icon="home", prefix="fa"),
        ).add_to(fmap)
    st_folium(fmap, height=430, use_container_width=True,
              returned_objects=[])


def _render_transcript(debate: list) -> None:
    for entry in debate:
        if entry["type"] == "consensus":
            st.success(entry["text"])
        elif entry["type"] == "qa":
            st.markdown(f"**{AGENT_LABEL[entry['agent']]}** "
                        f"(pick #{entry['own_pick']} vs "
                        f"#{entry['rival_pick']})")
            st.markdown(f"*Question:* {entry['question']}")
            st.markdown(f"*Answer:* {entry['answer']}")
            st.markdown("Would change pick: "
                        + ("**yes**" if entry["would_change_pick"]
                           else "no"))
            st.divider()
        else:
            st.info(entry["text"])


def _toggle_arena(listing_id: int) -> None:
    selected = list(st.session_state.get("arena_select", []))
    if listing_id in selected:
        selected.remove(listing_id)
    elif len(selected) < config.ARENA_MAX_PICKS:
        selected.append(listing_id)
    else:
        st.toast(f"Debate Arena holds at most {config.ARENA_MAX_PICKS} "
                 f"listings — remove one first.")
        return
    st.session_state["arena_select"] = selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    # ------------------------------------------------------------ sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        st.markdown("**Dataset**")
        dataset_name = st.selectbox(
            "Pilot city / source", list(config.DATASETS),
            index=list(config.DATASETS).index(config.DEFAULT_DATASET),
            label_visibility="collapsed")
        dataset_path = str(config.DATASETS[dataset_name])
        dkey = dataset_name.split(" ")[0].lower()
        st.markdown("**Your profile**")
        budget = st.slider("Monthly budget ceiling (₹)", 3000, 40000,
                           int(config.DATASET_DEFAULT_BUDGET[dataset_name]),
                           step=250, key=f"budget_{dkey}")
        # No max-distance input (removed at the user's request): the
        # Location agent scores distance against a fixed reference.
        max_distance = float(config.DEFAULT_PROFILE["max_distance_km"])
        within_budget_only = st.toggle(
            "Show only PGs within my budget", value=True,
            help="On: the recommendation and Near-you lists hide every "
                 "PG whose true monthly cost exceeds your ceiling, so "
                 "moving the budget slider visibly changes the list. "
                 "Off: over-budget PGs stay visible with an "
                 "'Over budget' badge and a budget score of 0.")
        # Importance weights are fixed defaults (the 1-5 inputs were
        # removed at the user's request); AHP still uses them.
        weights = dict(config.DEFAULT_PROFILE["weights"])
        st.caption("Criterion importance uses fixed default weights "
                   "(budget 5, distance 4, safety 5, WiFi 3, food 2).")
        st.divider()
        provider = os.getenv("LLM_PROVIDER", config.DEFAULT_PROVIDER)
        key_env = config.LLM_PROVIDERS[provider]["env_key"]
        if _has_api_key():
            st.success(f"🟢 Online — {provider} API key found "
                       f"({key_env}).")
        else:
            st.error(f"🔴 No API key — this app runs online only. "
                     f"Set {key_env} and restart (see the main panel).")
        st.caption("Fee extraction, the agent debate and explanations "
                   "call the LLM API live; responses are disk-cached "
                   "to save cost on repeat runs.")
        st.divider()
        st.markdown("**📍 Your live location**")
        if streamlit_geolocation is not None:
            st.caption("Use your device location to see the listings "
                       "nearest to you. It is used only in this session "
                       "to compute distances — never stored, never sent "
                       "to the LLM.")
            loc = streamlit_geolocation()
            if loc and loc.get("latitude") is not None \
                    and loc.get("longitude") is not None:
                st.session_state["user_loc"] = (
                    float(loc["latitude"]), float(loc["longitude"]))
        else:
            st.caption("Install streamlit-geolocation to enable the "
                       "Near-you section.")
        if "user_loc" in st.session_state:
            _ulat, _ulon = st.session_state["user_loc"]
            st.success(f"Location active ({_ulat:.4f}, {_ulon:.4f}) — "
                       f"see the Near-you section below.")

    profile = Profile(budget_ceiling=float(budget),
                      max_distance_km=float(max_distance), weights=weights)

    # ------------------------------------------------- online-only gate
    if not _has_api_key():
        st.error(
            f"**This app runs online only — no API key found.** "
            f"The LLM features (fee extraction, agent debate, "
            f"explanations) need a live {provider} API key in the "
            f"`{key_env}` environment variable.")
        st.markdown(
            "**Easiest:** create a file named exactly `.env` in the "
            "project folder (next to `app.py`) containing one line — "
            f"`{key_env}=your-key-here` — then restart the app. "
            "The app loads it automatically.\n\n"
            "Or set it in the terminal **before** launching:\n\n"
            f"```powershell\n# Windows (PowerShell)\n"
            f"$env:{key_env}=\"your-key-here\"\n"
            f"streamlit run app.py\n```\n"
            f"```bash\n# macOS / Linux\n"
            f"export {key_env}=\"your-key-here\"\n"
            f"streamlit run app.py\n```\n"
            "To use a different provider, also set `LLM_PROVIDER` to "
            "`gemini`, `openai` or `anthropic` (and that provider's "
            "key variable). See `.env.example` in the project folder.")
        return

    # ------------------------------------------------------------- data
    try:
        raw = load_listings(dataset_path, _stamp(dataset_path))
    except FileNotFoundError:
        st.error(f"Dataset not found at {dataset_path}. "
                 f"Generate it with the scripts in data/.")
        return
    report = validate_listings(raw)
    if not report["ok"]:
        st.error("Dataset failed schema validation: "
                 + "; ".join(report["errors"]))
        return
    if report["warnings"]:
        st.warning("Data warnings: " + "; ".join(report["warnings"]))

    probe = LLMClient(provider=provider, demo_mode=False)
    enriched = get_enriched(probe.provider, probe.model,
                            dataset_path, _stamp(dataset_path))
    client = LLMClient(provider=probe.provider, demo_mode=False)

    # --------------------------------------------------------- decision
    try:
        with st.spinner("Agents are scoring and debating..."):
            decision = run_decision(enriched, profile, client)
    except Exception as exc:  # the UI must never crash on a bad run
        st.error(f"Decision run failed: {exc}")
        return
    ranking = decision.ranking

    # -------------------------------------------------------------- hero
    mode = f"Online · live {provider}"
    st.markdown(f"""
<div class="hero">
  <div class="hero-title">🏠 Student Rental Decision Support</div>
  <div class="hero-sub">Budget, Location and Safety agents score every
    listing, debate their disagreements once, and AHP-TOPSIS decides —
    with quote-verified fees and fraud flags built into every score.</div>
  <div class="hero-pills"><span>📍 {_esc(dataset_name)}</span>
    <span>{len(enriched)} listings</span><span>{mode}</span>
    <span>AHP consistency {decision.consistency_ratio:.2f}</span></div>
</div>""", unsafe_allow_html=True)

    # ---------------------------------------------------- safety floor
    n_high_risk = int(ranking["high_risk"].sum())
    if n_high_risk:
        excluded = ranking[ranking["high_risk"]]
        st.info(
            f"🛡️ **Safety floor applied:** {n_high_risk} listing(s) scored "
            f"below {config.SAFETY_FLOOR} on safety and are excluded from "
            f"the recommendations however cheap they are: "
            + ", ".join(f"#{int(r['listing_id'])} (safety "
                        f"{r['safety_score']:.2f})"
                        for _, r in excluded.iterrows())
            + ". Their TOPSIS scores are still computed and shown in the "
              "full ranking below.")

    # Arena state: reset when the dataset changes; default to the top 3.
    if st.session_state.get("arena_dataset") != dataset_name:
        st.session_state["arena_dataset"] = dataset_name
        st.session_state["arena_select"] = [
            int(i) for i in ranking.head(3)["listing_id"]]
        st.session_state.pop("arena_result", None)
    valid_ids = [int(i) for i in enriched["id"]]
    st.session_state["arena_select"] = [
        i for i in st.session_state.get("arena_select", [])
        if i in valid_ids][:config.ARENA_MAX_PICKS]

    # --------------------------------------------------------- near you
    user_loc = st.session_state.get("user_loc")
    if user_loc and {"latitude", "longitude"} <= set(enriched.columns):
        ulat, ulon = user_loc
        dist = add_user_distance(
            enriched[["id", "latitude", "longitude"]], ulat, ulon
        ).set_index("id")["distance_from_user_km"]
        ranked = ranking.copy()
        ranked["distance_from_user_km"] = ranked["listing_id"].map(dist)
        near_all = ranked.sort_values("distance_from_user_km")
        nearest_km = float(near_all.iloc[0]["distance_from_user_km"])
        st.header("📍 Near you")
        if nearest_km > 50:
            st.caption(
                f"Your location is about {nearest_km:,.0f} km from the "
                f"nearest listing in this dataset — showing the closest "
                f"ones anyway. (Distances use locality-level listing "
                f"coordinates.)")
        else:
            st.caption("The listings closest to your live location. "
                       "High-risk listings are excluded here too, as in "
                       "the recommendations.")
        n_skipped = int(near_all.head(6)["high_risk"].sum())
        near = near_all[~near_all["high_risk"]]
        if within_budget_only:
            near = near[near["true_monthly_cost"] <= budget]
        near = near.head(6)
        for _, row in near.iterrows():
            _render_listing(
                row, key_prefix="near",
                note=f"📍 {row['distance_from_user_km']:.1f} km from "
                     f"your location",
                budget_ceiling=budget)
        if n_skipped:
            st.warning(
                f"⚠️ {n_skipped} of the 6 listings closest to you were "
                f"excluded by the safety floor (high risk) — they are "
                f"in the full ranking below.")
        _render_map(near, enriched, ulat, ulon, nearest_km)

    # ---------------------------------------------------------- top five
    st.header("🏆 Top recommendations")
    feasible = ranking[~ranking["high_risk"]]
    within = feasible[feasible["true_monthly_cost"] <= budget]
    st.caption(
        f"Budget ceiling ₹{budget:,} — {len(within)} of {len(ranking)} "
        f"listings fit within it. This list re-ranks instantly when you "
        f"move the budget slider.")
    display = (within if within_budget_only else feasible).head(5)
    if within_budget_only and display.empty:
        st.warning(
            f"No PG fits within a ₹{budget:,} true monthly cost on this "
            f"dataset. Raise the budget slider, or turn off "
            f"'Show only PGs within my budget' in the sidebar.")
    for _, row in display.iterrows():
        _render_listing(row, key_prefix="top", budget_ceiling=budget)

    # ----------------------------------------------------- debate arena
    st.header("⚖️ Debate Arena — your picks, agents debate")
    st.caption("Pick any 2–5 listings (from the cards above or the list) "
               "and the three agents will debate exactly those, weighing "
               "every factor — cost breakdown, distance, safety flags, "
               "WiFi, food, amenities and fee confidence — before the "
               "Decision Agent gives a verdict.")
    by_id = enriched.set_index("id")
    labels = {int(i): (f"#{int(i)} · "
                       f"{by_id.loc[i, 'title'] if 'title' in by_id.columns else 'Listing'}"
                       f" — ₹{float(by_id.loc[i, 'rent']):,.0f}/mo · "
                       f"{float(by_id.loc[i, 'distance_km'])} km")
              for i in valid_ids}
    selected = st.multiselect(
        "Listings for the agents to debate", options=valid_ids,
        format_func=lambda i: labels[i], key="arena_select",
        max_selections=config.ARENA_MAX_PICKS)
    profile_sig = (budget, max_distance,
                   tuple(sorted(weights.items())), dataset_name)

    if st.button("▶ Run agent debate on my picks", type="primary"):
        if len(selected) < config.ARENA_MIN_PICKS:
            st.warning(f"Pick at least {config.ARENA_MIN_PICKS} listings "
                       f"for a debate.")
            st.session_state.pop("arena_result", None)
        else:
            try:
                with st.spinner("Agents are debating your shortlist..."):
                    result = run_custom_debate(
                        enriched, profile, client, selected,
                        agent_results=decision.agent_results,
                        weights=decision.weights,
                        consistency_ratio=decision.consistency_ratio)
                st.session_state["arena_result"] = result
                st.session_state["arena_key"] = (tuple(sorted(selected)),
                                                 profile_sig)
            except Exception as exc:
                st.error(f"Debate failed: {exc}")

    arena = st.session_state.get("arena_result")
    if arena is not None:
        if st.session_state.get("arena_key") != (tuple(sorted(selected)),
                                                 profile_sig):
            st.caption("Selection or profile changed since this debate — "
                       "run it again for an up-to-date verdict.")
        # Agent positions within the shortlist.
        st.subheader("Agent positions on your shortlist")
        pos_cols = st.columns(3)
        crit_of = {"budget": "budget", "location": "distance",
                   "safety": "safety"}
        for col, name in zip(pos_cols, ("budget", "location", "safety")):
            pick = arena.picks[name]
            facts = arena.facts[pick]
            with col:
                with st.container(border=True):
                    st.markdown(f"**{AGENT_LABEL[name]}**")
                    st.markdown(
                        f"<div class='agent-pick'>{_esc(facts['title'])}"
                        f" (#{pick})</div>", unsafe_allow_html=True)
                    st.caption(
                        f"Scores {facts['agent_scores'][name]:.2f} on "
                        f"{crit_of[name]} within your shortlist · true "
                        f"cost ₹{facts['true_monthly_cost']:,} · safety "
                        f"{facts['safety_score']:.2f}")
        # All-factor comparison table.
        st.subheader("All factors, side by side")
        comp = pd.DataFrame([{
            "Listing": f"{f['title']} (#{lid})",
            "True cost ₹/mo": f["true_monthly_cost"],
            "Rent ₹": f["rent"], "Deposit ₹": f["deposit_used"],
            "Brokerage ₹": f["brokerage_used"],
            "Maintenance ₹": f["maintenance_used"],
            "Distance km": f["distance_km"],
            "Safety": f["safety_score"],
            "Red flags": len(f["safety_flags"]),
            "WiFi": "✓" if f["wifi"] else "—",
            "Food": "✓" if f["food_included"] else "—",
            "Fee confidence": f["fee_confidence"],
            "Shortlist TOPSIS": round(f["topsis_closeness"], 3),
        } for lid, f in arena.facts.items()])
        st.dataframe(comp, hide_index=True)
        # Transcript.
        st.subheader("Debate transcript")
        _render_transcript(arena.debate)
        # Verdict.
        winner_row = arena.subset_ranking.iloc[0]
        st.markdown(
            f"<div class='verdict'><h4>🏆 Verdict — "
            f"{_esc(winner_row['title'])} "
            f"(#{int(winner_row['listing_id'])})</h4>"
            f"<div>{_esc(arena.winner_explanation)}</div></div>",
            unsafe_allow_html=True)
        st.dataframe(
            arena.subset_ranking[["rank", "title", "topsis_score",
                                  "true_monthly_cost", "safety_score",
                                  "high_risk"]],
            hide_index=True)

    # -------------------------------------------- how the ranking works
    with st.expander("📐 How the ranking works (AHP weights + agent picks)"):
        weight_df = pd.DataFrame(
            {"criterion": [_CRITERION_LABEL[c] for c in config.CRITERIA],
             "default importance (fixed)": [profile.weights[c]
                                            for c in config.CRITERIA],
             "AHP weight": [round(decision.weights[c], 3)
                            for c in config.CRITERIA]})
        st.dataframe(weight_df, hide_index=True)
        st.write(f"AHP consistency ratio: "
                 f"**{decision.consistency_ratio:.3f}** "
                 f"(below {config.CR_THRESHOLD} is considered consistent).")
        for name, result in decision.agent_results.items():
            st.write(f"**{AGENT_LABEL[name]}** top pick: listing "
                     f"#{result.top_id} — {result.top_reason}")

    # ------------------------------------------------- debate transcript
    with st.expander("🗣️ Automatic debate transcript (one bounded round)"):
        _render_transcript(decision.debate)
        if decision.conflicts:
            st.caption("Conflicting criteria identified: "
                       + "; ".join(
                           f"{AGENT_LABEL[c['agent']]} "
                           f"({c['criteria'][0]} vs "
                           f"{'/'.join(c['criteria'][1:])})"
                           for c in decision.conflicts))

    # ------------------------------------------------------ full ranking
    with st.expander("📋 Full ranking (all listings)"):
        full = ranking[["rank", "title", "topsis_score",
                        "true_monthly_cost", "rent", "distance_km",
                        "safety_score", "fee_confidence", "high_risk",
                        "listing_url"]].copy()
        st.dataframe(
            full, hide_index=True,
            column_config={
                "listing_url": st.column_config.LinkColumn(
                    "Original listing", display_text="Open ↗"),
                "topsis_score": st.column_config.NumberColumn(
                    format="%.3f"),
                "true_monthly_cost": st.column_config.NumberColumn(
                    format="₹%d")})

    # ------------------------------------------------------------ what-if
    st.header("🔮 What-if: try a different budget")
    new_budget = st.slider("New budget ceiling (₹)", 3000, 40000,
                           int(budget), step=250, key=f"wi_budget_{dkey}")
    if st.button("Recalculate"):
        changes = {}
        if new_budget != budget:
            changes["budget_ceiling"] = float(new_budget)
        if not changes:
            st.info("Nothing changed — move the slider first.")
        else:
            cached_scores = {name: res.scores for name, res
                             in decision.agent_results.items()}
            result = recalculate(enriched, profile, changes,
                                 cached_scores=cached_scores)
            st.success(
                f"Recalculated in {result.elapsed_seconds * 1000:.0f} ms "
                f"({result.elapsed_seconds:.3f} s). Agents re-run: "
                f"{', '.join(result.reran_agents) or 'none (TOPSIS only)'}"
                f". Target: under 2 s with cache.")
            merged = result.ranking.head(5).merge(
                ranking[["listing_id", "title", "true_monthly_cost"]],
                on="listing_id", how="left")
            st.dataframe(merged, hide_index=True)

    st.markdown(
        "<div class='footer'>Demo build: static pilot datasets, "
        "rule-based safety checklist, LLM used only for fee extraction, "
        "one bounded debate round and explanation phrasing. Photos are "
        "MagicBricks property images; “View original listing” opens the "
        "source search page for that locality, where owners/agents can "
        "be contacted. Your live location, if shared, is used only "
        "in-session to compute distances and is never stored. "
        "See README for limitations.</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
