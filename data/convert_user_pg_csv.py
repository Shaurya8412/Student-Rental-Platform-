"""Convert the user's collected PG CSV to the project schema.

Input : data/user_pg_raw.csv - the user's own dataset (20 real PG
        listings collected around CCS University, Meerut, with
        latitude/longitude per area and a source URL per row).
Output: data/listings_user_pg.csv - the same listings in the project
        schema (see README), ready for the app and the agents.

Mapping choices (kept explicit on purpose):
- deposit / brokerage: the source data states neither, so both are 0
  and the composed description says so - the fee estimator therefore
  finds nothing to extract and the true cost equals the rent.
- agreement_mentioned / advance_before_viewing / verified_claim: not
  stated in the source data, so 0; the safety checklist treats
  "no agreement mentioned" uniformly across these real listings.
- food_included: 1 only when the source says food is included.
- distance_km: 8 rows (the non-MagicBricks sources) state no distance;
  for those it is filled with the haversine distance from the row's
  coordinates to CCS University (28.96925, 77.74049, geocoded) and
  rounded to 0.1 km - an estimate, marked in the converter output.
- image_url: photos taken from each row's own source page. The 12
  MagicBricks rows were matched to their exact listing photos on the
  "PG near Chaudhary Charan Singh University" page (matched by name,
  area and rent, 2026-10-30); the two HexaHome rows use their own
  listing photos; rows whose source page exposed no photo (99acres
  blocks scraping, Justdial) use other photos from the same
  MagicBricks source page.
- free_text_description is composed from the row's own fields
  (name, area, rent basis, amenities, food, notes, safety note).

Run:  python data/convert_user_pg_csv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.geo import haversine_km  # noqa: E402
from src.validator import validate_listings  # noqa: E402

CCSU = (28.96925, 77.74049)  # Chaudhary Charan Singh University

DATA_DIR = Path(__file__).resolve().parent
MB = "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images"

# Photos from the rows' source pages, by listing id (see docstring).
IMAGE_BY_ID = {
    1: f"{MB}/2025/Apr/09/Photo_h400_w540/GR2-489515-2443949_400_540.jpg",
    2: f"{MB}/2025/Feb/09/Photo_h400_w540/GR2-482039-2390297_400_540.jpg",
    3: f"{MB}/2025/Apr/09/Photo_h400_w540/GR2-489515-2443953_400_540.jpg",
    4: f"{MB}/2025/Apr/09/Photo_h400_w540/GR2-489515-2443957_400_540.jpg",
    5: f"{MB}/2025/Jun/03/Photo_h400_w540/GR2-496037-2489647_400_540.jpg",
    6: f"{MB}/2024/Aug/21/Photo_h400_w540/GR2-459557-2226891_400_540.jpg",
    7: f"{MB}/2025/Jan/03/Photo_h400_w540/GR2-477549-2354595_400_540.jpg",
    8: f"{MB}/2023/Nov/24/Photo_h400_w540/GR2-408091-1951359_400_540.jpeg",
    9: f"{MB}/2025/Mar/03/Photo_h400_w540/GR2-484539-2409429_400_540.jpg",
    10: f"{MB}/2025/Jun/08/Photo_h400_w540/GR2-496569-2493381_400_540.jpg",
    11: f"{MB}/2025/Mar/18/Photo_h400_w540/GR2-486783-2423979_400_540.jpg",
    12: f"{MB}/2025/Mar/03/Photo_h400_w540/GR2-484539-2409427_400_540.jpg",
    13: f"{MB}/2025/Jan/03/Photo_h400_w540/GR2-477549-2354603_400_540.jpg",
    14: f"{MB}/2025/Mar/18/Photo_h400_w540/GR2-486783-2423981_400_540.jpg",
    15: f"{MB}/2023/Nov/24/Photo_h400_w540/GR2-408091-1951357_400_540.jpeg",
    16: f"{MB}/2025/Jun/08/Photo_h400_w540/GR2-496569-2493389_400_540.jpg",
    17: f"{MB}/2025/Apr/09/Photo_h400_w540/GR2-489515-2443947_400_540.jpg",
    18: f"{MB}/2024/Aug/21/Photo_h400_w540/GR2-459557-2226899_400_540.jpg",
    19: "https://img.hexahome.in/media/pg/15-jan-2025/"
        "pg-photo-gpaty40q.webp?&rs=fill:430:380",
    20: "https://img.hexahome.in/media/pg/15-jan-2025/"
        "pg-photo-qkdm1wac.webp?&rs=fill:430:380",
}

SOURCE_BY_DOMAIN = {"magicbricks.com": "MagicBricks",
                    "99acres.com": "99acres",
                    "justdial.com": "Justdial",
                    "hexahome.in": "HexaHome"}


def _domain(url: str) -> str:
    return url.split("/")[2].replace("www.", "")


def convert(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    rows = []
    filled_distance: list[int] = []
    for _, r in raw.iterrows():
        lid = int(str(r["listing_id"]).replace("PG", ""))
        distance = r["distance_km"]
        if pd.isna(distance):
            distance = round(haversine_km(float(r["latitude"]),
                                          float(r["longitude"]), *CCSU), 1)
            filled_distance.append(lid)
        amenities = str(r["amenities"]).replace(";", ",")
        food_included = 1 if str(r["food"]).startswith("Included") else 0
        rating = ""
        if pd.notna(r["rating_out_of_5"]):
            rating = (f"Rated {r['rating_out_of_5']}/5 by "
                      f"{int(r['rating_count'])} tenant(s). ")
        other = ""
        if pd.notna(r["other_room_options"]):
            other = f"Other options: {r['other_room_options']}. "
        description = (
            f"{r['listing_name']} in {r['area']}, Meerut, "
            f"{distance} km from Chaudhary Charan Singh "
            f"University. Rent basis: {r['rent_basis']}. {other}"
            f"Amenities: {amenities}. Food: {r['food']}. "
            f"Wi-Fi: {r['wifi']}. {rating}"
            f"{r['safety_info_present']}. {r['description_notes']} "
            f"Security deposit and brokerage are not stated in the "
            f"source listing.")
        rows.append({
            "id": lid,
            "source": SOURCE_BY_DOMAIN.get(_domain(r["source_url"]),
                                           "MagicBricks"),
            "rent": int(r["rent_monthly"]),
            "deposit": 0,
            "brokerage_stated": 0,
            "distance_km": float(distance),
            "amenities": amenities,
            "food_included": food_included,
            "wifi": 1 if r["wifi"] == "Yes" else 0,
            "gender_policy": str(r["gender_policy"]),
            "free_text_description": description,
            "agreement_mentioned": 0,
            "advance_before_viewing": 0,
            "verified_claim": 0,
            "red_flag_label": 0,
            "is_synthetic": 0,
            "title": f"{r['listing_name']} in {r['area']}",
            "image_url": IMAGE_BY_ID[lid],
            "listing_url": str(r["source_url"]),
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
        })
    return pd.DataFrame(rows), filled_distance


def main() -> None:
    raw = pd.read_csv(DATA_DIR / "user_pg_raw.csv")
    out, filled = convert(raw)
    report = validate_listings(out)
    if not report["ok"]:
        raise SystemExit(f"Converted data failed validation: "
                         f"{report['errors']}")
    path = DATA_DIR / "listings_user_pg.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} rows to {path}")
    print(f"distance_km estimated from coordinates for ids: {filled}")
    print(f"Validator warnings: {report['warnings']}")
    from src.agents import safety_agent
    safety = safety_agent(out)
    flagged = {lid: f for lid, f in safety.extras["flags"].items() if f}
    print(f"Safety checklist flags: {flagged}")


if __name__ == "__main__":
    main()
