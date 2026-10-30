"""Schema validator for data/listings.csv.

Checks the required columns, basic types, value ranges and binary flags,
and reports missing values per column. Usable as a library function and
as a CLI:

    python -m src.validator data/listings.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

REQUIRED_COLUMNS = [
    "id", "source", "rent", "deposit", "brokerage_stated", "distance_km",
    "amenities", "food_included", "wifi", "gender_policy",
    "free_text_description", "agreement_mentioned", "advance_before_viewing",
    "verified_claim", "red_flag_label", "is_synthetic",
]
NUMERIC_COLUMNS = ["rent", "deposit", "brokerage_stated", "distance_km"]
BINARY_COLUMNS = [
    "food_included", "wifi", "agreement_mentioned", "advance_before_viewing",
    "verified_claim", "red_flag_label", "is_synthetic",
]


def validate_listings(df: pd.DataFrame) -> dict:
    """Validate a listings DataFrame against the project schema.

    Returns a report dict: {"ok", "errors", "warnings",
    "missing_by_column", "n_rows"}.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return {"ok": False, "errors": errors, "warnings": warnings,
                "missing_by_column": {}, "n_rows": len(df)}

    missing_by_column = {
        c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS
        if int(df[c].isna().sum()) > 0
    }
    if missing_by_column:
        warnings.append(f"Missing values by column: {missing_by_column}")

    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].tolist()
        errors.append(f"Duplicate listing ids: {dupes}")

    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().sum() > int(df[col].isna().sum()):
            errors.append(f"Column {col!r} contains non-numeric values")

    if (pd.to_numeric(df["rent"], errors="coerce") <= 0).any():
        errors.append("Column 'rent' must be > 0 for every row")
    if (pd.to_numeric(df["distance_km"], errors="coerce") < 0).any():
        errors.append("Column 'distance_km' must be >= 0 for every row")

    for col in BINARY_COLUMNS:
        values = set(pd.to_numeric(df[col], errors="coerce").dropna().unique())
        if not values.issubset({0, 1}):
            errors.append(
                f"Column {col!r} must contain only 0/1, found {sorted(values)}")

    if df["free_text_description"].astype(str).str.strip().eq("").any():
        warnings.append("Some rows have an empty free_text_description")

    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "missing_by_column": missing_by_column, "n_rows": len(df)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: validate a CSV file and print the report."""
    parser = argparse.ArgumentParser(description="Validate a listings CSV.")
    parser.add_argument("csv_path", help="Path to the listings CSV")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv_path)
    report = validate_listings(df)
    print(f"Rows: {report['n_rows']}")
    print(f"Missing values by column: {report['missing_by_column'] or 'none'}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    print("Schema OK" if report["ok"] else "Schema INVALID")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
