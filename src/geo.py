"""Geographic helpers for the 'Near you' feature.

Listings carry locality-level coordinates (latitude/longitude columns,
geocoded locality centres plus a small deterministic jitter). The
user's live location comes from the browser; distances are computed
locally with the haversine formula - the user's coordinates are never
stored or sent anywhere.
"""

from __future__ import annotations

import math

import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float,
                 lon2: float) -> float:
    """Great-circle distance between two points, in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def add_user_distance(listings: pd.DataFrame, user_lat: float,
                      user_lon: float) -> pd.DataFrame:
    """Copy of *listings* with a `distance_from_user_km` column.

    Rows without coordinates get NaN and sort last in nearest_listings.
    """
    out = listings.copy()
    out["distance_from_user_km"] = [
        haversine_km(user_lat, user_lon, lat, lon)
        if pd.notna(lat) and pd.notna(lon) else float("nan")
        for lat, lon in zip(out["latitude"], out["longitude"])]
    return out


def nearest_listings(listings: pd.DataFrame, user_lat: float,
                     user_lon: float, n: int = 6) -> pd.DataFrame:
    """The *n* listings closest to the user, nearest first."""
    with_dist = add_user_distance(listings, user_lat, user_lon)
    return (with_dist.sort_values("distance_from_user_km",
                                  na_position="last")
            .head(n).reset_index(drop=True))
