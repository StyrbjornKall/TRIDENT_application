#!/usr/bin/env python3
"""Generate TRIDENT-2 training-data observation counts.

Produces two JSON files from the preprocessed training data:
  - taxid -> number of observations (across all species groups/units/routes/lifestages)
  - species_group -> conc_unit -> administration_route -> organism_lifestage -> number
    of observations, so the app can show how much experimental evidence backs a
    given prediction's exact combination.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_ZIP_PATH = Path("tmp/preprocessed_tox_data_SK_2026-04-17.csv.zip")
DEFAULT_TAXID_OUTPUT_PATH = Path("data/trident2/taxid2n_obs.json")
DEFAULT_COMBO_OUTPUT_PATH = Path("data/trident2/species_group_unit_route_lifestage_n_obs.json")

NOT_SPECIFIED = "Not specified"

COLUMNS = [
    "NCBI_rank_species",
    "species_group_corrected",
    "conc_unit",
    "administration_route_categorized",
    "organism_lifestage_categorized",
]


def build_taxid_counts(df: pd.DataFrame) -> dict[str, int]:
    taxid = df["NCBI_rank_species"].dropna().astype(float).astype(int).astype(str)
    counts = taxid.value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def build_combo_counts(df: pd.DataFrame) -> dict:
    combo_df = df[
        ["species_group_corrected", "conc_unit", "administration_route_categorized", "organism_lifestage_categorized"]
    ].copy()
    combo_df["administration_route_categorized"] = combo_df["administration_route_categorized"].fillna(
        NOT_SPECIFIED
    )
    combo_df["organism_lifestage_categorized"] = combo_df["organism_lifestage_categorized"].fillna(NOT_SPECIFIED)

    counts = combo_df.groupby(
        ["species_group_corrected", "conc_unit", "administration_route_categorized", "organism_lifestage_categorized"]
    ).size()

    nested: dict = {}
    for (group, unit, route, lifestage), count in counts.items():
        nested.setdefault(group, {}).setdefault(unit, {}).setdefault(route, {})[lifestage] = int(count)
    return nested


def generate_observation_counts(
    zip_path: str | Path = DEFAULT_ZIP_PATH,
    taxid_output_path: str | Path = DEFAULT_TAXID_OUTPUT_PATH,
    combo_output_path: str | Path = DEFAULT_COMBO_OUTPUT_PATH,
) -> tuple[dict[str, int], dict]:
    zip_path = Path(zip_path)
    taxid_output_path = Path(taxid_output_path)
    combo_output_path = Path(combo_output_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(f"No CSV file found inside {zip_path}")

        with archive.open(csv_members[0]) as fh:
            df = pd.read_csv(fh, usecols=COLUMNS, low_memory=False)

    taxid_counts = build_taxid_counts(df)
    combo_counts = build_combo_counts(df)

    taxid_output_path.parent.mkdir(parents=True, exist_ok=True)
    taxid_output_path.write_text(json.dumps(taxid_counts, indent=2, sort_keys=True))

    combo_output_path.parent.mkdir(parents=True, exist_ok=True)
    combo_output_path.write_text(json.dumps(combo_counts, indent=2, sort_keys=True))

    return taxid_counts, combo_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate taxid and species group/unit/route/lifestage observation counts."
    )
    parser.add_argument("--zip-path", type=str, default=str(DEFAULT_ZIP_PATH))
    parser.add_argument("--taxid-output-path", type=str, default=str(DEFAULT_TAXID_OUTPUT_PATH))
    parser.add_argument("--combo-output-path", type=str, default=str(DEFAULT_COMBO_OUTPUT_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    taxid_counts, combo_counts = generate_observation_counts(
        zip_path=args.zip_path,
        taxid_output_path=args.taxid_output_path,
        combo_output_path=args.combo_output_path,
    )
    print(f"Wrote {len(taxid_counts)} taxid entries to {args.taxid_output_path}")
    print(f"Wrote {len(combo_counts)} species group entries to {args.combo_output_path}")
