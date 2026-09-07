#!/usr/bin/env python3
"""Generate the TRIDENT-2 species->lifestage->route allowlist.

The allowlist is built from the preprocessed training data and retains only
species/route/lifestage combinations whose observed count is at least a
configurable minimum threshold.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd


DEFAULT_ZIP_PATH = Path("preprocessed_tox_data_SK_2026-04-17.csv.zip")
DEFAULT_OUTPUT_PATH = Path("data/trident2/permitted_route_lifestage_combinations.json")


def build_permitted_combinations(
    df: pd.DataFrame,
    min_count: int = 5,
) -> dict[str, dict[str, list[str]]]:
    """Return a nested dict keyed as species_taxid -> lifestage -> routes."""
    allowed = (
        df.loc[
            df["NCBI_match"].notna()
            & df["organism_lifestage_categorized"].notna()
            & df["administration_route_categorized"].notna(),
            [
                "NCBI_match",
                "organism_lifestage_categorized",
                "administration_route_categorized",
            ],
        ]
        .dropna()
        .groupby(
            ["NCBI_match", "organism_lifestage_categorized", "administration_route_categorized"]
        )
        .size()
        .reset_index(name="count")
    )

    permitted: dict[str, dict[str, list[str]]] = {}
    for taxid, group in allowed.groupby("NCBI_match"):
        valid = group[group["count"] >= min_count]
        if valid.empty:
            continue

        permitted[str(int(taxid))] = {
            str(lifestage): sorted(routes)
            for lifestage, routes in valid.groupby("organism_lifestage_categorized")[
                "administration_route_categorized"
            ]
        }

    return dict(sorted(permitted.items(), key=lambda item: int(item[0])))


def generate_allowlist(
    zip_path: str | Path = DEFAULT_ZIP_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    min_count: int = 5,
) -> dict[str, dict[str, list[str]]]:
    zip_path = Path(zip_path)
    output_path = Path(output_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(f"No CSV file found inside {zip_path}")

        with archive.open(csv_members[0]) as fh:
            df = pd.read_csv(fh)

    permitted = build_permitted_combinations(df=df, min_count=min_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(permitted, indent=2, sort_keys=True))
    return permitted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the TRIDENT-2 species->lifestage->route allowlist using a minimum "
            "observation threshold from the preprocessed training data."
        )
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default=str(DEFAULT_ZIP_PATH),
        help="Path to the preprocessed_tox_data_SK_2026-04-17.csv.zip file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the generated JSON allowlist.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum number of observations required for a species/lifestage/route combination.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    permitted = generate_allowlist(
        zip_path=args.zip_path,
        output_path=args.output_path,
        min_count=args.min_count,
    )
    print(f"Wrote {len(permitted)} species entries to {args.output_path} with min_count={args.min_count}")
