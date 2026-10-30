"""Generate the 40-row SAMPLE dataset data/listings.csv.

All rows are generated for the demo. 32 rows follow realistic legitimate
patterns; 8 rows are deliberately planted scam patterns (advance before
viewing, no agreement, unverifiable "verified" claim, far-below-market
rent, pressure language) and are the rows marked ``is_synthetic=1``
together with ``red_flag_label=1``. One planted scam (id pattern: token
payment buried in otherwise clean text) is designed to evade the
rule-based checklist, so evaluation recall is realistically < 1.

Run:  python data/generate_listings.py
"""

from __future__ import annotations

import random
import sys
import zlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLUMNS = ["id", "source", "rent", "deposit", "brokerage_stated",
           "distance_km", "amenities", "food_included", "wifi",
           "gender_policy", "free_text_description", "agreement_mentioned",
           "advance_before_viewing", "verified_claim", "red_flag_label",
           "is_synthetic", "title", "image_url", "listing_url",
           "latitude", "longitude"]

# Area centre coordinates (geocoded 2026-10-29); listings get the area
# centre plus a small deterministic jitter for map display.
AREA_COORDS = {
    "Koramangala": (12.93309, 77.62327),
    "HSR Layout": (12.90052, 77.65744),
    "Indiranagar": (12.97394, 77.64390),
    "Whitefield": (12.97935, 77.73369),
    "Electronic City": (12.84541, 77.66434),
    "JP Nagar": (12.90831, 77.59024),
    "Bellandur": (12.92734, 77.67169),
    "Marathahalli": (12.95466, 77.70752),
    "Yelahanka": (13.09931, 77.59259),
    "BTM Layout": (12.91350, 77.61003),
    "Hebbal": (13.04981, 77.58903),
    "Peenya": (13.03188, 77.52654),
}


def _jitter(key: str, scale: float = 0.004) -> tuple[float, float]:
    """Deterministic +/-scale-degree offset, so listings in one area
    do not stack on a single map pin."""
    h = zlib.crc32(key.encode())
    return (((h % 1000) / 1000 - 0.5) * 2 * scale,
            (((h >> 10) % 1000) / 1000 - 0.5) * 2 * scale)

# MagicBricks property photos (img.staticmb.com), used illustratively
# for the modelled rows; same pools as the Meerut dataset.
FLAT_IMAGES = [
    "https://img.staticmb.com/mbphoto/property/cropped_images/ver2/XIwvQlc61t8ZIpanz4mAnjVB5tA9dMsNP5_VxHoRtvggCIXv1DoI/Photo_h300_w450/78681415_5_PropertyImage437-51807497089044_300_450.jpg",
    "https://img.staticmb.com/mbphoto/property/cropped_images/ver2/XIwvQlc61t8ZIpanz4mAmDVP9dY9ds4q28eVseah3ZS28VC_Iywb/Photo_h300_w450/69171901_2_Snapchat-812160502_300_450.jpg",
    "https://img.staticmb.com/mbphoto/property/cropped_images/ver2/XIwvQlc61t8ZIpanz4mAnTVN99s9dcpQS3hGddjQOjeuU18hAqMl/Photo_h300_w450/84601367_1_PropertyImage776-6427211876235_300_450.jpg",
    "https://img.staticmb.com/mbphoto/property/cropped_images/ver2/XIwvQlc61t8ZIpanz4mAmDVG88A9dsow8uZ41ycPSc5VF89y15od/Photo_h300_w450/65711441_8_PropertyImage543-9954209285723_300_450.jpg",
    "https://img.staticmb.com/mbphoto/property/cropped_images/ver2/XIwvQlc61t8ZIpanz4mAnTVP9dY9dM1cZq8k19fAeVRkIzbeR4Ru/Photo_h300_w450/1873_1791348697374.2030_300_450.jpg",
]
PG_IMAGES = [
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2023/Jan/15/Photo_h400_w540/GR2-343345-1625657_400_540.jpg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2022/Dec/12/Photo_h400_w540/GR2-335757-1583523_400_540.jpeg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2024/Feb/20/Photo_h400_w540/GR2-414989-2049305_400_540.jpeg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2026/May/04/Photo_h400_w540/GR2-537493-2771481_400_540.jpg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2026/Feb/09/Photo_h400_w540/GR2-526803-2704855_400_540.jpeg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2022/Sep/20/Photo_h400_w540/GR2-318445-1497763_400_540.jpeg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2022/Jan/22/Photo_h400_w540/GR2-266273-1246377_400_540.jpg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2026/Sep/16/Photo_h400_w540/GR2-557659-2883777_400_540.jpeg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2024/Aug/30/Photo_h400_w540/GR2-461039-2237569_400_540.jpeg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2025/Jun/03/Photo_h400_w540/GR2-495965-2489215_400_540.jpeg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2024/May/12/Photo_h400_w540/GR2-440533-2130515_400_540.jpg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2021/Mar/03/Photo_h400_w540/GR2-156483-772775_400_540.jpg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2021/Sep/28/Photo_h400_w540/GR2-238427-1123507_400_540.jpg",
    "https://img.staticmb.com/mbphoto/pg/grd2/cropped_images/2021/Aug/22/Photo_h400_w540/GR2-220441-1052063_400_540.jpeg",
]
KORAMANGALA_PG_URL = "https://www.magicbricks.com/pg-under-7000-in-koramangala-block-4-bangalore-pppfr"
KORAMANGALA_FLAT_URL = "https://www.magicbricks.com/flats-for-rent-in-koramangala-bangalore-pppfr"

AREAS = ["Koramangala", "HSR Layout", "Indiranagar", "Whitefield",
         "Electronic City", "JP Nagar", "Bellandur", "Marathahalli",
         "Yelahanka", "BTM Layout", "Hebbal", "Peenya"]
LANDMARKS = ["Christ University", "the metro station",
             "City Engineering College", "the tech park", "the bus depot"]
AMENITY_POOL = ["WiFi", "AC", "Power Backup", "Laundry", "Parking",
                "Geyser", "RO Water", "CCTV", "Attached Bathroom",
                "Study Table", "Wardrobe", "Housekeeping"]
SOURCES = ["NoBroker", "MagicBricks", "99acres", "Housing.com", "OLX",
           "Facebook Group"]


def _phone(rng: random.Random) -> str:
    return str(rng.choice([98, 97, 96, 95, 94, 93, 91, 90, 89, 88, 87, 86])) \
        + f"{rng.randint(10000000, 99999999)}"


def _legit_row(rng: random.Random, flaw: str | None = None) -> dict:
    kind = rng.choice(["PG", "PG", "Hostel", "Shared flat"])
    area = rng.choice(AREAS)
    addr = f"{rng.randint(2, 140)}, {rng.randint(1, 12)}th Cross, {area}"
    phone = _phone(rng)
    gender = rng.choice(["Boys", "Girls", "Co-ed"])
    if kind == "PG":
        rent = rng.randrange(9500, 15001, 100)
    elif kind == "Hostel":
        rent = rng.randrange(9500, 12501, 100)
    else:
        rent = rng.randrange(11000, 21001, 100)
    amenities = ";".join(rng.sample(AMENITY_POOL, rng.randint(4, 7)))
    wifi = 1 if kind != "Shared flat" or rng.random() < 0.8 else 0
    food = 1 if kind in ("PG", "Hostel") or rng.random() < 0.25 else 0

    # Deposit: months-of-rent, fixed amount, or unstated in the text.
    dep_mode = rng.choice(["months", "months", "fixed", "silent"])
    if dep_mode == "months":
        months = rng.choice([1, 2, 2, 3])
        deposit = months * rent
        deposit_text = f"Deposit: {months} months rent (refundable)."
    elif dep_mode == "fixed":
        deposit = rng.choice([10000, 15000, 20000, 25000])
        deposit_text = f"Deposit ₹{deposit:,} (refundable)."
    else:
        deposit = 2 * rent
        deposit_text = ""

    # Brokerage: none (owner), one month, fixed fee, or unstated.
    bro_mode = rng.choice(["none", "none", "month", "fixed", "silent"])
    if bro_mode == "none":
        brokerage = 0
        brokerage_text = "No brokerage - direct owner listing."
    elif bro_mode == "month":
        brokerage = rent
        brokerage_text = "Brokerage: 1 month rent."
    elif bro_mode == "fixed":
        brokerage = rng.choice([5000, 6000, 7500, 9000])
        brokerage_text = f"Brokerage ₹{brokerage:,}."
    else:
        brokerage = 0
        brokerage_text = ""

    maint_mode = rng.choice(["amount", "amount", "included", "silent"])
    if maint_mode == "amount":
        maint_text = (f"Maintenance ₹{rng.choice([500, 800, 1000, 1200, 1500, 2000]):,}"
                      " per month.")
    elif maint_mode == "included":
        maint_text = "Maintenance included in rent."
    else:
        maint_text = ""

    food_text = "Food included (3 meals a day)." if food else \
        "Food not included; kitchen access available."
    extra_text = rng.choice(["", "", " Water charges ₹300 per month.",
                             " Parking charges ₹500 per month."])

    verified = 0
    proof_text = ""
    if flaw == "verified_no_proof":
        verified = 1  # claims verified, text offers no proof -> FP by design
    elif rng.random() < 0.08:
        verified = 1
        proof_text = (" Owner ID proof available; police verification "
                      "done for all tenants.")

    agreement = 0 if flaw == "no_agreement" else 1
    pressure_text = ""
    if flaw == "pressure":
        pressure_text = (" Limited rooms for the new semester - "
                         "book now for the August intake.")

    sharing = rng.choice(["Single", "Double", "Triple"])
    if kind == "PG":
        desc = (f"Spacious {sharing.lower()}-sharing PG room in {area}, "
                f"{addr}. Amenities: {amenities}. Rent ₹{rent:,}/month. "
                f"{deposit_text} {brokerage_text} {maint_text} {food_text}"
                f"{extra_text}{proof_text}{pressure_text} "
                f"Call {phone} to schedule a visit.")
    elif kind == "Hostel":
        desc = (f"{gender} hostel at {addr}, near {rng.choice(LANDMARKS)}. "
                f"Clean {sharing.lower()}-sharing rooms with {amenities}. "
                f"Rent ₹{rent:,}/month. {deposit_text} {maint_text} "
                f"{food_text}{extra_text}{proof_text}{pressure_text} "
                f"Warden on site. Contact {phone}.")
    else:
        desc = (f"One room in a shared flat at {addr}. {sharing} occupancy "
                f"with {amenities}. Rent ₹{rent:,}/month. {deposit_text} "
                f"{brokerage_text} {maint_text} {food_text}{extra_text}"
                f"{proof_text}{pressure_text} "
                f"Call the owner at {phone}.")

    if kind == "PG":
        title = f"{sharing}-Sharing PG in {area}"
    elif kind == "Hostel":
        title = f"{gender} Hostel in {area}"
    else:
        title = f"Room in a Shared Flat in {area}"
    image_url = rng.choice(PG_IMAGES if kind in ("PG", "Hostel")
                           else FLAT_IMAGES)
    if area == "Koramangala":
        listing_url = KORAMANGALA_PG_URL if kind in ("PG", "Hostel") \
            else KORAMANGALA_FLAT_URL
    else:
        listing_url = ""
    base_lat, base_lon = AREA_COORDS[area]
    dlat, dlon = _jitter(f"{area}|{rent}|{desc[:24]}")
    latitude = round(base_lat + dlat, 5)
    longitude = round(base_lon + dlon, 5)

    return {
        "source": rng.choice(SOURCES), "rent": rent, "deposit": deposit,
        "brokerage_stated": brokerage,
        "distance_km": round(rng.uniform(0.4, 9.5), 1),
        "amenities": amenities, "food_included": food, "wifi": wifi,
        "gender_policy": gender, "free_text_description": desc,
        "agreement_mentioned": agreement, "advance_before_viewing": 0,
        "verified_claim": verified, "red_flag_label": 0, "is_synthetic": 0,
        "title": title, "image_url": image_url, "listing_url": listing_url,
        "latitude": latitude, "longitude": longitude,
    }


def _scam_rows() -> list[dict]:
    """Eight hand-built scam patterns (red_flag_label=1, is_synthetic=1)."""
    base = {"amenities": "WiFi;AC;Power Backup", "food_included": 0,
            "wifi": 1, "gender_policy": "Co-ed", "red_flag_label": 1,
            "is_synthetic": 1}
    rows = [
        # 1. Everything at once: advance, no agreement, fake verified,
        #    far below market, pressure, no contact detail.
        {**base, "source": "Facebook Group", "rent": 4200, "deposit": 4200,
         "brokerage_stated": 0, "distance_km": 2.1,
         "free_text_description":
         "Verified premium PG in Koramangala at an unbelievable price! "
         "Only ₹4,200 per month. Limited rooms, pay today to block yours. "
         "Send the advance by UPI to confirm. No time wasters please.",
         "agreement_mentioned": 0, "advance_before_viewing": 1,
         "verified_claim": 1},
        # 2. Advance before viewing + pressure, with a phone number.
        {**base, "source": "OLX", "rent": 5000, "deposit": 5000,
         "brokerage_stated": 0, "distance_km": 3.4,
         "free_text_description":
         "Cheap single room, HSR Layout side. Owner is out of station so "
         "pay the advance before viewing. Hurry, last room left. "
         "Call 9812045673.",
         "agreement_mentioned": 0, "advance_before_viewing": 1,
         "verified_claim": 0},
        # 3. Fake verified + no agreement + below market, no contact.
        {**base, "source": "Broker WhatsApp", "rent": 4800, "deposit": 9600,
         "brokerage_stated": 4800, "distance_km": 5.2,
         "free_text_description":
         "100% verified flat, best deal in Whitefield. No agreement "
         "needed, just move in with your bags. Water charges ₹300 extra.",
         "agreement_mentioned": 0, "advance_before_viewing": 0,
         "verified_claim": 1},
        # 4. Token advance + fake verified + below market (has address).
        {**base, "source": "Facebook Group", "rent": 6500, "deposit": 6500,
         "brokerage_stated": 3250, "distance_km": 1.8,
         "free_text_description":
         "Verified owner listing. Beautiful 1RK on 100 Feet Road, "
         "Indiranagar. Pay a token advance to book before viewing, fully "
         "refundable. Brokerage: half month.",
         "agreement_mentioned": 1, "advance_before_viewing": 1,
         "verified_claim": 1},
        # 5. No agreement + below market + pressure, no contact.
        {**base, "source": "OLX", "rent": 3900, "deposit": 0,
         "brokerage_stated": 0, "distance_km": 6.8,
         "free_text_description":
         "Room available in Electronic City. No agreement, no deposit "
         "headache. Book now, offer ends soon. Move in this week itself.",
         "agreement_mentioned": 0, "advance_before_viewing": 0,
         "verified_claim": 0},
        # 6. Advance before viewing + no agreement (has phone).
        {**base, "source": "Broker WhatsApp", "rent": 5500, "deposit": 5500,
         "brokerage_stated": 0, "distance_km": 4.4,
         "free_text_description":
         "Owner posted. Room in BTM Layout 2nd Stage. Send one month "
         "advance before visiting, owner travels frequently. "
         "Call 9742018856.",
         "agreement_mentioned": 0, "advance_before_viewing": 1,
         "verified_claim": 0},
        # 7. Fake verified + pressure + below market (agreement exists).
        {**base, "source": "Facebook Group", "rent": 4600, "deposit": 9200,
         "brokerage_stated": 0, "distance_km": 2.9,
         "free_text_description":
         "Fully verified PG near the campus, JP Nagar. Pay today and "
         "get the first month free. Limited rooms only, genuine "
         "students preferred.",
         "agreement_mentioned": 1, "advance_before_viewing": 0,
         "verified_claim": 1},
        # 8. The subtle one: columns look clean; only the text mentions a
        #    pre-viewing token, phrased softly. Designed to evade the
        #    checklist (advance_before_viewing column says 0).
        {**base, "source": "OLX", "rent": 9800, "deposit": 19600,
         "brokerage_stated": 0, "distance_km": 3.1,
         "free_text_description":
         "Spacious room in Hebbal, near the lake. Deposit: 2 months "
         "rent. To block the room before your visit, transfer a small "
         "token of ₹2,000 and we will hold it for you. "
         "Call 9686112407 to discuss.",
         "agreement_mentioned": 1, "advance_before_viewing": 0,
         "verified_claim": 0},
    ]
    meta = [
        ("PG in Koramangala", PG_IMAGES[5], KORAMANGALA_PG_URL,
         "Koramangala"),
        ("Single Room in HSR Layout", PG_IMAGES[6], "", "HSR Layout"),
        ("Flat in Whitefield", FLAT_IMAGES[3], "", "Whitefield"),
        ("1RK in Indiranagar", FLAT_IMAGES[4], "", "Indiranagar"),
        ("Room in Electronic City", PG_IMAGES[7], "", "Electronic City"),
        ("Room in BTM Layout", PG_IMAGES[8], "", "BTM Layout"),
        ("PG in JP Nagar", PG_IMAGES[9], "", "JP Nagar"),
        ("Room in Hebbal", FLAT_IMAGES[0], "", "Hebbal"),
    ]
    for row, (title, image_url, listing_url, area) in zip(rows, meta):
        base_lat, base_lon = AREA_COORDS[area]
        dlat, dlon = _jitter(f"scam|{title}")
        row.update({"title": title, "image_url": image_url,
                    "listing_url": listing_url,
                    "latitude": round(base_lat + dlat, 5),
                    "longitude": round(base_lon + dlon, 5)})
    return rows


def main() -> None:
    rng = random.Random(42)
    flaws = ["no_agreement", "pressure", "verified_no_proof"]
    legit = []
    for i in range(32):
        flaw = flaws[i] if i < len(flaws) else None
        legit.append(_legit_row(rng, flaw=flaw))

    scams = _scam_rows()
    insert_at = [4, 10, 16, 22, 27, 32, 35, 38]  # positions in final list
    rows = list(legit)
    for pos, scam in zip(insert_at, scams):
        rows.insert(pos, scam)
    for idx, row in enumerate(rows, start=1):
        row["id"] = idx

    df = pd.DataFrame(rows)[COLUMNS]
    out_path = Path(__file__).resolve().parent / "listings.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Median rent: ₹{df['rent'].median():,.0f} | "
          f"scam rows: {int(df['red_flag_label'].sum())} | "
          f"synthetic rows: {int(df['is_synthetic'].sum())}")

    # Sanity check: schema + what the safety checklist would say.
    from src.agents import safety_agent
    from src.validator import validate_listings
    report = validate_listings(df)
    print("Validator:", "OK" if report["ok"] else report["errors"])
    result = safety_agent(df)
    tp = fp = fn = 0
    for _, row in df.iterrows():
        predicted = len(result.extras["flags"][row["id"]]) > 0
        if predicted and row["red_flag_label"] == 1:
            tp += 1
        elif predicted:
            fp += 1
        elif row["red_flag_label"] == 1:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"Safety checklist on sample data: TP={tp} FP={fp} FN={fn} "
          f"precision={precision:.2f} recall={recall:.2f}")


if __name__ == "__main__":
    main()
