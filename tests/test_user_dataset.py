"""Tests for the user's uploaded PG dataset (converted to schema)."""

from pathlib import Path

import pandas as pd

from src.validator import validate_listings

ROOT = Path(__file__).resolve().parent.parent


def test_user_pg_dataset_is_valid_and_complete():
    df = pd.read_csv(ROOT / "data" / "listings_user_pg.csv")
    report = validate_listings(df)
    assert report["ok"], report["errors"]
    assert report["warnings"] == []
    assert len(df) == 20
    # Real collected rows, each with a photo from its source page,
    # a link back to that page, and coordinates near CCS University.
    assert (df["is_synthetic"] == 0).all()
    assert df["image_url"].str.startswith("http").all()
    assert df["listing_url"].str.startswith("http").all()
    assert df["latitude"].between(28.9, 29.1).all()
    assert df["longitude"].between(77.6, 77.8).all()
    assert df["distance_km"].notna().all()
    assert df["rent"].between(2000, 15000).all()
