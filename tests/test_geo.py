"""Tests for the geographic helpers (Near-you feature)."""

import math
from pathlib import Path

import pandas as pd
import pytest

from src.geo import add_user_distance, haversine_km, nearest_listings

ROOT = Path(__file__).resolve().parent.parent

# CCS University area, Garh Road, Meerut (the pilot's reference campus).
CAMPUS = (28.9555, 77.7547)


def test_haversine_zero_and_known_distance():
    assert haversine_km(28.95, 77.75, 28.95, 77.75) == 0.0
    # Delhi (Connaught Place) to Meerut is roughly 65-75 km straight-line.
    d = haversine_km(28.6315, 77.2167, *CAMPUS)
    assert 60.0 < d < 80.0
    # Symmetry.
    assert math.isclose(d, haversine_km(*CAMPUS, 28.6315, 77.2167),
                        rel_tol=1e-9)


def test_nearest_listings_from_campus():
    df = pd.read_csv(ROOT / "data" / "listings_meerut.csv")
    near = nearest_listings(df, *CAMPUS, n=6)
    assert len(near) == 6
    distances = near["distance_from_user_km"].tolist()
    assert distances == sorted(distances)
    # From the campus, the closest listings are within ~3 km (Garh Road /
    # Jagriti Vihar / Mangal Pandey Nagar stock).
    assert distances[0] < 3.0
    # Every returned listing has real coordinates.
    assert near["latitude"].notna().all()


def test_add_user_distance_handles_missing_coords():
    df = pd.DataFrame({"latitude": [28.95, None],
                       "longitude": [77.75, None]})
    out = add_user_distance(df, *CAMPUS)
    assert out.loc[0, "distance_from_user_km"] < 1.0
    assert math.isnan(out.loc[1, "distance_from_user_km"])
    with pytest.raises(KeyError):
        add_user_distance(pd.DataFrame({"x": [1]}), *CAMPUS)
