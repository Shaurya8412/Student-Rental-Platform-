"""Generate the Meerut pilot dataset data/listings_meerut.csv.

The sample is modelled on live MagicBricks Meerut rental listings
(checked 2026-10-29): the same localities (Ganga Nagar, Shastri Nagar,
Pallav Puram, Modipuram, Kanker Khera, Vedvyash Puri, Rohta Road,
Meerut Cantt, Mangal Pandey Nagar...), the same rent bands (Meerut
rentals run roughly ₹3,000-₹47,000/month; student stock clusters at
₹4,000-₹15,000), MagicBricks-style wording ("posted directly by the
owner, no brokerage", "The monthly maintenance is ₹.../-"), and
per-listing security deposits of 1-2 months' rent. All rows carry
source = "MagicBricks".

``distance_km`` is the approximate road distance to the Chaudhary
Charan Singh University (CCSU) main campus on Garh Road, the pilot's
reference campus.

Every row in this file is a curated sample row (is_synthetic = 1):
36 follow realistic legitimate patterns and 8 are planted scam
patterns of the kind reported around MagicBricks-style listings
(deposit-before-viewing UPI scams, fake "verified" owners, bait
pricing) with red_flag_label = 1. One planted scam hides its token
payment in otherwise clean text and is designed to evade the
rule-based checklist, so evaluation recall is realistically < 1.

Run:  python data/generate_meerut_listings.py
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

# Locality centre coordinates (geocoded 2026-10-29). Each listing gets
# its locality centre plus a small deterministic jitter, so the app
# can compute live distance from the user's location.
LOCALITY_COORDS = {
    "Mangal Pandey Nagar": (28.96853, 77.73576),
    "Garh Road": (28.95554, 77.75474),
    "Jagriti Vihar": (28.95381, 77.74874),
    "Brahmpuri": (28.97315, 77.69756),
    "Delhi Road": (28.92531, 77.65392),
    "Hapur Road": (28.97153, 77.66825),
    "Shastri Nagar": (28.95478, 77.73310),
    "Sadar Bazaar": (28.99176, 77.70057),
    "Lisari Road": (28.97191, 77.71002),
    "Meerut Cantt": (29.02195, 77.69042),
    "Ganga Nagar": (29.00758, 77.75963),
    "Defence Colony": (29.01060, 77.74284),
    "Rohta Road": (28.99144, 77.65525),
    "Vedvyash Puri": (28.94910, 77.65056),
    "Kanker Khera": (29.02870, 77.66914),
    "Shatabdi Nagar": (28.94072, 77.67723),
    "Pallav Puram": (29.05391, 77.71390),
    "Modipuram": (29.04684, 77.70562),
}


def _jitter(key: str, scale: float = 0.004) -> tuple[float, float]:
    """Deterministic +/-scale-degree offset, so listings in one
    locality do not stack on a single map pin."""
    h = zlib.crc32(key.encode())
    return (((h % 1000) / 1000 - 0.5) * 2 * scale,
            (((h >> 10) % 1000) / 1000 - 0.5) * 2 * scale)

# Real MagicBricks property photos (img.staticmb.com) harvested from the
# live Meerut flats/PG pages on 2026-10-29; assigned illustratively to
# the modelled rows by property type.
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

# Real MagicBricks Meerut search pages (verbatim URLs) - where a user
# finds and contacts the actual listings for each locality.
DEFAULT_LISTING_URL = "https://www.magicbricks.com/flats-for-rent-in-meerut-pppfr"
LOCALITY_URLS = {
    "Ganga Nagar": "https://www.magicbricks.com/flats-for-rent-in-ganga-nagar-meerut-pppfr",
    "Pallav Puram": "https://www.magicbricks.com/independent-house-for-rent-in-pallav-puram-meerut-price-5000-to-10000-pppfr",
    "Shastri Nagar": "https://www.magicbricks.com/1-bhk-independent-house-for-rent-in-shastri-nagar-meerut-price-15000-to-20000-pppfr",
    "Rohta Road": "https://www.magicbricks.com/paying-guest-for-girls-in-rohta-road-meerut-pppfr",
    "Vedvyash Puri": "https://www.magicbricks.com/flats-for-rent-in-vedvyash-puri-meerut-price-10000-to-15000-pppfr",
    "Ansal Colony": "https://www.magicbricks.com/property-for-rent-in-ansal-colony-meerut-pppfr",
    "Ganga Sagar": "https://www.magicbricks.com/independent-house-for-rent-in-ganga-sagar-meerut-pppfr",
}

# locality -> (kinds, rent range, distance-to-CCSU range, nearby institute)
LOCALITIES = {
    "Mangal Pandey Nagar": (["PG", "Room"], (4800, 6500), (0.8, 2.0),
                            "CCS University"),
    "Garh Road": (["PG", "1BHK"], (4800, 8500), (1.0, 3.0),
                  "CCS University"),
    "Jagriti Vihar": (["Room", "1BHK"], (4800, 7500), (2.5, 3.5),
                      "CCS University"),
    "Brahmpuri": (["Room", "1BHK"], (4800, 6500), (3.0, 4.5),
                  "the city campus"),
    "Delhi Road": (["PG", "Hostel"], (4800, 7000), (3.0, 5.0),
                   "Dewan Institute"),
    "Hapur Road": (["Room", "PG"], (4800, 6500), (3.5, 5.5),
                   "the polytechnic"),
    "Shastri Nagar": (["1BHK", "2BHK"], (5500, 12000), (4.0, 5.5),
                      "the university"),
    "Sadar Bazaar": (["Room"], (4800, 5500), (4.0, 5.0), "the cantonment"),
    "Lisari Road": (["Room"], (4800, 5200), (4.5, 6.0), "the city campus"),
    "Meerut Cantt": (["PG", "Room"], (5000, 9500), (4.5, 6.0),
                     "St Mary's Academy"),
    "Ganga Nagar": (["1BHK", "2BHK", "PG"], (5500, 15000), (5.5, 7.0),
                    "IIMT College"),
    "Defence Colony": (["2BHK"], (9000, 12000), (5.5, 6.5), "the cantonment"),
    "Rohta Road": (["PG"], (4800, 6500), (5.0, 7.5), "Translam Institute"),
    "Vedvyash Puri": (["PG"], (4800, 7000), (6.0, 7.5),
                      "Kishan Institute of Technology"),
    "Kanker Khera": (["1BHK", "2BHK"], (5000, 9000), (6.5, 8.0),
                     "the cantonment"),
    "Shatabdi Nagar": (["2BHK"], (8000, 13000), (7.5, 9.0), "MIET"),
    "Pallav Puram": (["1BHK", "2BHK"], (6000, 10000), (9.0, 11.5),
                     "Subharti University"),
    "Modipuram": (["PG"], (4800, 7000), (9.5, 11.0), "Shobhit University"),
}
AMENITY_POOL = ["WiFi", "AC", "Power Backup", "Laundry", "Parking",
                "Geyser", "RO Water", "CCTV", "Attached Bathroom",
                "Study Table", "Wardrobe", "Housekeeping", "Balcony",
                "Lift", "Security Guard"]
OWNER_NAMES = ["Rohit Sharma", "Sandeep Kumar", "Rahul Gupta",
               "Priya Singh", "Amit Tyagi", "Neha Agarwal",
               "Vishal Chauhan", "Kavita Rani", "Manoj Verma",
               "Sunita Devi", "Arjun Singh", "Pooja Yadav",
               "Deepak Pal", "Anjali Gupta", "Sanjay Rathi"]


def _phone(rng: random.Random) -> str:
    return str(rng.choice([98, 97, 96, 95, 94, 93, 91, 90, 89, 88, 87, 86])) \
        + f"{rng.randint(10000000, 99999999)}"


def _fees(rng: random.Random, rent: int, kind: str) -> tuple[int, int, str, str, str]:
    """Return (deposit, brokerage, deposit_text, brokerage_text, maint_text)."""
    dep_mode = rng.choice(["months", "months", "fixed", "silent"])
    if dep_mode == "months":
        months = rng.choice([1, 2, 2])
        deposit = months * rent
        deposit_text = f"Security deposit: {months} months rent."
    elif dep_mode == "fixed":
        deposit = rng.choice([5000, 8000, 10000, 15000])
        deposit_text = f"Security Deposit ₹{deposit:,}."
    else:
        deposit = 2 * rent
        deposit_text = ""

    bro_mode = rng.choice(["owner", "owner", "month", "fixed", "silent"])
    if bro_mode == "owner":
        brokerage = 0
        brokerage_text = ("This property is posted directly by the owner, "
                          "so no brokerage is payable.")
    elif bro_mode == "month":
        brokerage = rent
        brokerage_text = "Brokerage: 1 month rent."
    elif bro_mode == "fixed":
        brokerage = rng.choice([3000, 4000, 5000])
        brokerage_text = f"Brokerage ₹{brokerage:,}."
    else:
        brokerage = 0
        brokerage_text = ""

    if kind in ("PG", "Hostel"):
        maint_text = rng.choice(["Maintenance included in rent.", "", ""])
    else:
        maint_text = rng.choice([
            f"The monthly maintenance is ₹{rng.choice([500, 750, 1000, 1250, 1650]):,}/-.",
            "Maintenance included in rent.", ""])
    return deposit, brokerage, deposit_text, brokerage_text, maint_text


def _legit_row(rng: random.Random, locality: str,
               flaw: str | None = None) -> dict:
    kinds, (rent_lo, rent_hi), (dist_lo, dist_hi), institute = \
        LOCALITIES[locality]
    kind = rng.choice(kinds)
    if flaw == "cheap_pg":
        kind, rent = "PG", 4200
    else:
        rent = rng.randrange(rent_lo, rent_hi + 1, 100)
    distance = round(rng.uniform(dist_lo, dist_hi), 1)
    owner = rng.choice(OWNER_NAMES)
    phone = _phone(rng)
    amenities = ";".join(rng.sample(AMENITY_POOL, rng.randint(4, 7)))
    gender = {"PG": rng.choice(["Boys", "Girls", "Co-ed"]),
              "Hostel": rng.choice(["Boys", "Girls"])}.get(
                  kind, rng.choice(["Boys", "Girls", "Co-ed", "Co-ed"]))
    food = 1 if kind in ("PG", "Hostel") or rng.random() < 0.2 else 0
    wifi = 1 if rng.random() < 0.85 else 0
    deposit, brokerage, dep_t, bro_t, maint_t = _fees(rng, rent, kind)
    food_text = ("Food included (3 meals a day)." if food
                 else "Food not included; kitchen access available.")
    addr = f"{rng.choice(['A', 'B', 'C', 'K', 'L', 'P'])}-{rng.randint(1, 240)}, {locality}"
    furnishing = rng.choice(["Semi-Furnished", "Furnished", "Unfurnished"])

    verified = 0
    proof_text = ""
    if flaw == "verified_no_proof":
        verified = 1  # claims verified; text offers no proof -> FP by design
    elif rng.random() < 0.08:
        verified = 1
        proof_text = (" Owner ID proof available on request; police "
                      "verification is mandatory for all tenants.")
    agreement = 0 if flaw == "no_agreement" else 1

    sharing = rng.choice(["Single", "Twin", "Triple"])
    if kind == "PG":
        desc = (f"Move into a professionally managed {gender.lower()} PG "
                f"in {locality}, Meerut. {sharing}-sharing rooms with "
                f"{amenities}. Rent ₹{rent:,}/month. {dep_t} {maint_t} "
                f"{food_text} Located near {institute}. "
                f"Call {phone} to schedule a visit.")
    elif kind == "Hostel":
        desc = (f"{gender} hostel in {locality}, Meerut, near {institute}. "
                f"{sharing}-sharing rooms with {amenities}. "
                f"Rent ₹{rent:,}/month. {dep_t} {maint_t} {food_text} "
                f"Warden on site. Contact {phone}.")
    elif kind == "Room":
        desc = (f"Single room for rent in {locality}, Meerut. "
                f"{furnishing}, with {amenities}. Monthly rent "
                f"₹{rent:,}. {dep_t} {bro_t} {maint_t} {food_text} "
                f"Address: {addr}, Meerut, Uttar Pradesh. "
                f"Contact owner {owner} at {phone}.")
    else:  # 1BHK / 2BHK
        desc = (f"{kind} flat for rent in {locality}, Meerut. "
                f"{furnishing}, tenant preferred Bachelors/Family. "
                f"The flat is available at a monthly rent of ₹{rent:,}. "
                f"{dep_t} {bro_t} {maint_t} You can visit the flat at "
                f"{addr}, {locality}, Meerut, Uttar Pradesh. "
                f"Contact owner {owner} at {phone}.")
    desc = (desc + proof_text).replace("  ", " ").strip()

    if kind == "PG":
        title = f"{gender} PG in {locality}"
    elif kind == "Hostel":
        title = f"{gender} Hostel in {locality}"
    elif kind == "Room":
        title = f"Single Room in {locality}"
    else:
        title = f"{kind} Flat in {locality}"
    image_url = rng.choice(PG_IMAGES if kind in ("PG", "Hostel")
                           else FLAT_IMAGES)
    listing_url = LOCALITY_URLS.get(locality, DEFAULT_LISTING_URL)
    base_lat, base_lon = LOCALITY_COORDS[locality]
    dlat, dlon = _jitter(f"{locality}|{rent}|{distance}")
    latitude = round(base_lat + dlat, 5)
    longitude = round(base_lon + dlon, 5)

    return {"source": "MagicBricks", "rent": rent, "deposit": deposit,
            "brokerage_stated": brokerage, "distance_km": distance,
            "amenities": amenities, "food_included": food, "wifi": wifi,
            "gender_policy": gender, "free_text_description": desc,
            "agreement_mentioned": agreement, "advance_before_viewing": 0,
            "verified_claim": verified, "red_flag_label": 0,
            "is_synthetic": 1, "title": title, "image_url": image_url,
            "listing_url": listing_url, "latitude": latitude,
            "longitude": longitude}


def _scam_rows() -> list[dict]:
    """Eight planted MagicBricks-style scam patterns (label=1)."""
    base = {"source": "MagicBricks", "amenities": "WiFi;AC;Power Backup",
            "food_included": 1, "wifi": 1, "gender_policy": "Co-ed",
            "red_flag_label": 1, "is_synthetic": 1}
    rows = [
        # 1. Deposit-before-viewing UPI scam near the university.
        {**base, "rent": 2499, "deposit": 2499, "brokerage_stated": 0,
         "distance_km": 1.2,
         "free_text_description":
         "Verified boys PG in Mangal Pandey Nagar, cheapest in Meerut! "
         "Only ₹2,499 per month with food. Only 1 room left - pay today "
         "by UPI and block your bed. Owner will share location after "
         "payment. No calls, chat only.",
         "agreement_mentioned": 0, "advance_before_viewing": 1,
         "verified_claim": 1},
        # 2. Advance before viewing, owner "out of station" (has phone).
        {**base, "rent": 3000, "deposit": 3000, "brokerage_stated": 0,
         "distance_km": 6.1, "food_included": 0,
         "free_text_description":
         "1 BHK house for rent in Ganga Nagar, Meerut at just ₹3,000. "
         "Owner is out of station, transfer the security deposit to "
         "confirm your visit slot. First come first served. "
         "Call 9837012456.",
         "agreement_mentioned": 0, "advance_before_viewing": 1,
         "verified_claim": 0},
        # 3. Fake "verified owner" + no agreement + bait rent, no contact.
        {**base, "rent": 3500, "deposit": 7000, "brokerage_stated": 3500,
         "distance_km": 4.8,
         "free_text_description":
         "MagicBricks verified owner property. Spacious room, Hapur "
         "side, Meerut. No agreement needed, shift with your bags "
         "today. Price fixed, no negotiation.",
         "agreement_mentioned": 0, "advance_before_viewing": 0,
         "verified_claim": 1},
        # 4. Token advance + pressure + bait 2BHK (has address & phone).
        {**base, "rent": 4500, "deposit": 4500, "brokerage_stated": 2250,
         "distance_km": 4.6, "food_included": 0,
         "free_text_description":
         "2 BHK flat for rent in Shastri Nagar, Meerut for only "
         "₹4,500 - below market deal. Pay today a small token to "
         "confirm before someone else takes it. Address: B-42, "
         "Shastri Nagar, Meerut. Call 9412703388.",
         "agreement_mentioned": 1, "advance_before_viewing": 1,
         "verified_claim": 0},
        # 5. No agreement + bait rent + pressure, no contact detail.
        {**base, "rent": 3200, "deposit": 0, "brokerage_stated": 0,
         "distance_km": 5.3, "food_included": 0,
         "free_text_description":
         "Room available, Lisari side, Meerut. No agreement, no "
         "deposit tension. Book now, offer ends this week. "
         "Genuine students only.",
         "agreement_mentioned": 0, "advance_before_viewing": 0,
         "verified_claim": 0},
        # 6. Advance-before-viewing deposit scam at a Cantt PG (has phone).
        {**base, "rent": 4000, "deposit": 4000, "brokerage_stated": 0,
         "distance_km": 5.1,
         "free_text_description":
         "Girls PG in Meerut Cantt with food and WiFi at ₹4,000 only. "
         "Send one month advance before visiting to reserve your bed, "
         "warden will confirm on WhatsApp. Call 9045517823.",
         "agreement_mentioned": 0, "advance_before_viewing": 1,
         "verified_claim": 0},
        # 7. Fake verified + pressure + bait rent near Subharti.
        {**base, "rent": 3800, "deposit": 7600, "brokerage_stated": 0,
         "distance_km": 10.2,
         "free_text_description":
         "Fully verified PG near Subharti University, Modipuram. "
         "Limited rooms at this price - pay today and get 10% off "
         "the first month. All facilities included.",
         "agreement_mentioned": 1, "advance_before_viewing": 0,
         "verified_claim": 1},
        # 8. The subtle one: clean columns; the pre-viewing token is
        #    buried in soft wording (advance_before_viewing says 0).
        {**base, "rent": 6800, "deposit": 13600, "brokerage_stated": 0,
         "distance_km": 1.9, "food_included": 0,
         "free_text_description":
         "Single room for rent on Garh Road, Meerut, walking distance "
         "to the university gate. Security deposit: 2 months rent. "
         "To hold the room before your visit, transfer a small "
         "refundable token of ₹1,500 and we will keep it booked for "
         "you. Contact owner Rakesh Pal at 9720014589.",
         "agreement_mentioned": 1, "advance_before_viewing": 0,
         "verified_claim": 0},
    ]
    meta = [
        ("Boys PG in Mangal Pandey Nagar", PG_IMAGES[0],
         DEFAULT_LISTING_URL, "Mangal Pandey Nagar"),
        ("1 BHK House in Ganga Nagar", FLAT_IMAGES[0],
         LOCALITY_URLS["Ganga Nagar"], "Ganga Nagar"),
        ("Single Room near Hapur Road", PG_IMAGES[1], DEFAULT_LISTING_URL,
         "Hapur Road"),
        ("2 BHK Flat in Shastri Nagar", FLAT_IMAGES[1],
         LOCALITY_URLS["Shastri Nagar"], "Shastri Nagar"),
        ("Single Room in Lisari Road", PG_IMAGES[2], DEFAULT_LISTING_URL,
         "Lisari Road"),
        ("Girls PG in Meerut Cantt", PG_IMAGES[3], DEFAULT_LISTING_URL,
         "Meerut Cantt"),
        ("PG in Modipuram", PG_IMAGES[4], DEFAULT_LISTING_URL,
         "Modipuram"),
        ("Single Room on Garh Road", FLAT_IMAGES[2], DEFAULT_LISTING_URL,
         "Garh Road"),
    ]
    for row, (title, image_url, listing_url, locality) in zip(rows, meta):
        base_lat, base_lon = LOCALITY_COORDS[locality]
        dlat, dlon = _jitter(f"scam|{title}")
        row.update({"title": title, "image_url": image_url,
                    "listing_url": listing_url,
                    "latitude": round(base_lat + dlat, 5),
                    "longitude": round(base_lon + dlon, 5)})
    return rows


def main() -> None:
    rng = random.Random(123)
    localities = list(LOCALITIES)
    legit = []
    cheap_pg_used = False
    for i in range(36):
        locality = localities[i % len(localities)] if i < len(localities) \
            else rng.choice(localities)
        flaw = None
        if locality == "Rohta Road" and not cheap_pg_used:
            flaw = "cheap_pg"
            cheap_pg_used = True
        legit.append(_legit_row(rng, locality, flaw=flaw))
    # Deterministic designed flaws on three specific rows.
    legit[3]["agreement_mentioned"] = 0            # no-agreement FP
    legit[7]["verified_claim"] = 1                 # verified-no-proof FP
    if "ID proof" in legit[7]["free_text_description"]:
        legit[7]["free_text_description"] = legit[7][
            "free_text_description"].split(" Owner ID proof")[0]

    scams = _scam_rows()
    insert_at = [3, 9, 15, 21, 26, 31, 35, 39]
    rows = list(legit)
    for pos, scam in zip(insert_at, scams):
        rows.insert(pos, scam)
    for idx, row in enumerate(rows, start=1):
        row.pop("_flaw", None)
        row["id"] = idx

    df = pd.DataFrame(rows)[COLUMNS]
    out_path = Path(__file__).resolve().parent / "listings_meerut.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Median rent: ₹{df['rent'].median():,.0f} | "
          f"rent range: ₹{df['rent'].min():,}–₹{df['rent'].max():,} | "
          f"scam rows: {int(df['red_flag_label'].sum())}")

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
    print(f"Safety checklist on Meerut data: TP={tp} FP={fp} FN={fn} "
          f"precision={precision:.2f} recall={recall:.2f}")


if __name__ == "__main__":
    main()
