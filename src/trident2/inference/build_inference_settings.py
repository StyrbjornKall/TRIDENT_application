import pandas as pd
import random
import random
import numpy as np
import argparse
from loguru import logger
from trident2.preprocessing.preprocess_data import explicit_casting


# Assign units based on species group (only for standardized cases) and cases where species group implies a unit
def assign_standard_units(species_group: str):
    if species_group == "fish":
        return "mg/l"
    elif species_group == "crustaceans":
        return "mg/l"
    elif species_group == "algae":
        return "mg/l"
    elif species_group == "other mammals":
        return "mg/kg"
    elif species_group == "rodents":
        return "mg/kg"
    elif species_group == "birds":
        return "mg/kg"
    elif species_group == "reptiles":
        return "mg/kg"
    elif species_group == "amphibians":
        return "mg/l"
    elif species_group == "echinoderms":
        return "mg/l"
    elif species_group == "cnidarians and bryozoans":
        return "mg/l"
    else:
        return None


# Secondary assignment based on habitat
def assign_units_habitat(habitat):
    if not pd.isna(habitat):
        if habitat == "Water":
            return "mg/l"
        elif habitat == "Soil":
            return "mg/kg soil"
        else:
            return None
    else:
        return None


# Backup assignment based on most frequent unit per taxon
def assign_units_taxon(species: str, species_unit_map: dict):
    if species in species_unit_map:
        return species_unit_map[species]
    else:
        return None


def assign_unit_species_group(species_group: str, species_group_map: dict):
    if species_group in species_group_map:
        return species_group_map[species_group][0]
    else:
        return None


# Final assignment (in case of ties in the species unit map) are assigned unit the unit of the species group if included in ties, else assign first
def adjust_ties(ties: list, species_group: str, species_group_map: dict):
    if species_group_map[species_group] in ties:
        return species_group_map[species_group][0]
    else:
        return ties[0]


# Apply the assignment functions
def final_unit_assignment(row, species_unit_map, species_group_unit_map):
    unit = assign_standard_units(row["species_group_corrected"])
    if unit is not None:
        return unit

    unit = assign_units_habitat(row["organism_habitat"])
    if unit is not None:
        return unit

    unit = assign_units_taxon(row["NCBI_rank_family"], species_unit_map)
    if unit is not None:
        if isinstance(unit, list) and len(unit) > 1:
            # Handle ties
            return adjust_ties(
                unit, row["species_group_corrected"], species_group_unit_map
            )
        else:
            return unit[0] if isinstance(unit, list) else unit

    else:
        unit = assign_unit_species_group(
            row["species_group_corrected"], species_group_unit_map
        )

    return None


# Assign effects based on species group (only for standardized cases) and cases where species group implies a effect
def assign_standard_acute_effect(species_group: str):
    if species_group == "fish":
        return "MOR"
    elif species_group == "crustaceans":
        return "MOR"
    elif species_group == "algae":
        return "POP"
    elif species_group == "cyanobacteria":
        return "POP"
    elif species_group == "other mammals":
        return "MOR"
    elif species_group == "rodents":
        return "MOR"
    elif species_group == "birds":
        return "MOR"
    elif species_group == "reptiles":
        return "MOR"
    elif species_group == "amphibians":
        return "MOR"
    else:
        return None


def assign_standard_chronic_effect(species_group: str):
    if species_group == "fish":
        return "REP"
    elif species_group == "crustaceans":
        return "REP"
    elif species_group == "algae":
        return "POP"
    elif species_group == "cyanobacteria":
        return "POP"
    elif species_group == "other mammals":
        return "MOR"
    elif species_group == "rodents":
        return "MOR"
    elif species_group == "birds":
        return "MOR"
    elif species_group == "reptiles":
        return "MOR"
    elif species_group == "amphibians":
        return "MOR"
    else:
        return None


# Backup assignment based on most frequent effect per taxon
def assign_effect_taxon(species: str, species_effect_map: dict):
    if species in species_effect_map:
        return species_effect_map[species]
    else:
        return None


def assign_effect_species_group(species_group: str, species_group_map: dict):
    if species_group in species_group_map:
        return species_group_map[species_group][0]
    else:
        return None


# Apply the assignment functions
def assign_acute_effect(row, species_effect_map, species_group_effect_map):
    effect = assign_standard_acute_effect(row["species_group_corrected"])
    if effect is not None:
        return effect

    effect = assign_effect_taxon(row["NCBI_rank_family"], species_effect_map)
    if effect is not None:
        if isinstance(effect, list) and len(effect) > 1:
            # Handle ties
            return adjust_ties(
                effect, row["species_group_corrected"], species_group_effect_map
            )
        else:
            return effect[0] if isinstance(effect, list) else effect

    if effect is None:
        return assign_effect_species_group(
            row["species_group_corrected"], species_group_effect_map
        )

    return None


def assign_chronic_effect(row, species_effect_map, species_group_effect_map):
    effect = assign_standard_acute_effect(row["species_group_corrected"])
    if effect is not None:
        return effect

    effect = assign_effect_taxon(row["NCBI_rank_family"], species_effect_map)
    if effect is not None:
        if isinstance(effect, list) and len(effect) > 1:
            # Handle ties
            return adjust_ties(
                effect, row["species_group_corrected"], species_group_effect_map
            )
        else:
            return effect[0] if isinstance(effect, list) else effect

    if effect is None:
        return assign_effect_species_group(
            row["species_group_corrected"], species_group_effect_map
        )

    return None


def print_effect_assignment_source_stats(
    df, species_acute_effect_map, species_group_acute_effect_map
):
    sources = {
        "standard_assignment": 0,
        "taxon_assignment": 0,
        "species_group_assignment": 0,
        "unassigned": 0,
    }
    for _, row in df.iterrows():
        if assign_standard_acute_effect(row["species_group_corrected"]) is not None:
            sources["standard_assignment"] += 1
        elif (
            assign_effect_taxon(row["NCBI_rank_family"], species_acute_effect_map)
            is not None
        ):
            sources["taxon_assignment"] += 1
        elif (
            assign_effect_species_group(
                row["species_group_corrected"], species_group_acute_effect_map
            )
            is not None
        ):
            sources["species_group_assignment"] += 1
        else:
            sources["unassigned"] += 1

    total = len(df)
    logger.info("Effect assignment source stats:")
    for key, count in sources.items():
        logger.info(f"  {key}: {count} ({(count / total) * 100:.2f}%)")


def print_unit_assignment_source_stats(df, species_unit_map, species_group_unit_map):
    sources = {
        "standard_assignment": 0,
        "habitat_assignment": 0,
        "taxon_assignment": 0,
        "species_group_assignment": 0,
        "unassigned": 0,
    }
    for _, row in df.iterrows():
        if assign_standard_units(row["species_group_corrected"]) is not None:
            sources["standard_assignment"] += 1
        elif assign_units_habitat(row["organism_habitat"]) is not None:
            sources["habitat_assignment"] += 1
        elif assign_units_taxon(row["NCBI_rank_family"], species_unit_map) is not None:
            sources["taxon_assignment"] += 1
        elif (
            assign_unit_species_group(
                row["species_group_corrected"], species_group_unit_map
            )
            is not None
        ):
            sources["species_group_assignment"] += 1
        else:
            sources["unassigned"] += 1

    total = len(df)
    logger.info("Unit assignment source stats:")
    for key, count in sources.items():
        logger.info(f"  {key}: {count} ({(count / total) * 100:.2f}%)")


# Assign duration based on species group (only for standardized cases) and cases where species group implies a duration
def assign_standard_acute_duration(species_group: str):
    if species_group == "fish":
        return np.log10(96)
    elif species_group == "crustaceans":
        return np.log10(48)
    elif species_group == "algae":
        return np.log10(96)
    elif species_group == "cyanobacteria":
        return np.log10(96)
    else:
        return np.nan


# Backup assignment based on most frequent duration per taxon
def assign_duration_taxon(species: str, species_duration_map: dict):
    if species in species_duration_map:
        return species_duration_map[species]
    else:
        return None


def assign_duration_species_group(species_group: str, species_group_map: dict):
    if species_group in species_group_map:
        return species_group_map[species_group][0]
    else:
        return None


# Apply the assignment functions
def assign_acute_duration(row, species_duration_map, species_group_duration_map):
    duration = assign_standard_acute_duration(row["species_group_corrected"])
    if duration is not None:
        return duration

    duration = assign_duration_taxon(row["NCBI_rank_family"], species_duration_map)
    if duration is not None:
        if isinstance(duration, list) and len(duration) > 1:
            # Handle ties
            return adjust_ties(
                duration, row["species_group_corrected"], species_group_duration_map
            )
        else:
            return duration[0] if isinstance(duration, list) else duration

    if duration is None:
        return assign_duration_species_group(
            row["species_group_corrected"], species_group_duration_map
        )

    return None


def print_duration_assignment_source_stats(
    df, species_acute_duration_map, species_group_acute_duration_map
):
    sources = {
        "standard_assignment": 0,
        "taxon_assignment": 0,
        "species_group_assignment": 0,
        "unassigned": 0,
    }
    for _, row in df.iterrows():
        if assign_standard_acute_duration(row["species_group_corrected"]) is not None:
            sources["standard_assignment"] += 1
        elif (
            assign_duration_taxon(row["NCBI_rank_family"], species_acute_duration_map)
            is not None
        ):
            sources["taxon_assignment"] += 1
        elif (
            assign_duration_species_group(
                row["species_group_corrected"], species_group_acute_duration_map
            )
            is not None
        ):
            sources["species_group_assignment"] += 1
        else:
            sources["unassigned"] += 1

    total = len(df)
    logger.info("Duration assignment source stats:")
    for key, count in sources.items():
        logger.info(f"  {key}: {count} ({(count / total) * 100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Build inference settings for TRIDENT2"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/skall/TRIDENT_2/data/Preprocessed_tox_data_SK_20250716_step3.pkl.zip",
        help="Path to the preprocessed toxicity data",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/skall/TRIDENT_2/data/taxa_assigned_units_effects_durations_20251219.csv",
        help="Path to save the output CSV with assigned units, effects, and durations",
    )
    args = parser.parse_args()

    if args.data_path.endswith(".pkl.zip"):
        df = pd.read_pickle(args.data_path)
    if args.data_path.endswith(".csv.zip"):
        df = pd.read_csv(args.data_path)
    logger.success(f"Loaded data from {args.data_path} with shape {df.shape}")

    # Preprocess data to ensure correct types and taxonomic ranks
    logger.info("Preprocessing data for correct types and taxonomic ranks...")
    df = explicit_casting(df)

    logger.info(
        f"Removing {df['NCBI_rank_species'].isna().sum()} records with missing NCBI_rank_species"
    )
    df = df[df.NCBI_rank_species.notna()]

    logger.info("Preprocessing data for inference settings assignment...")
    df["effect"] = df.effect.str.replace("<", "").str.replace(">", "")

    # Build species to most common unit map (can include ties in the form of lists)
    ranks = [
        "superkingdom",
        "kingdom",
        "phylum",
        "subphylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    logger.info(
        "Building species to most common unit map for taxon-based assignment..."
    )
    species_unit_map = (
        df.groupby("NCBI_rank_family")["conc_unit"]
        .agg(lambda x: x.mode(dropna=False).tolist())
        .to_dict()
    )
    species_group_unit_map = (
        df.groupby("species_group_corrected")["conc_unit"]
        .agg(lambda x: x.mode(dropna=False).tolist())
        .to_dict()
    )
    df["assigned_conc_unit"] = df[
        ["species_group_corrected", "organism_habitat"]
        + [f"NCBI_rank_{rank}" for rank in ranks]
    ].apply(
        final_unit_assignment, axis=1, args=(species_unit_map, species_group_unit_map)
    )

    print_unit_assignment_source_stats(df, species_unit_map, species_group_unit_map)

    # Acute effects only
    df_acute = df[df["effect"].isin(["MOR", "ITX", "POP", "GRO"])]
    df_chronic = df[df.endpoint.isin(["NOEC", "LOEC", "EC10"])]
    # Build species to most common effect map (can include ties in the form of lists)
    ranks = [
        "superkingdom",
        "kingdom",
        "phylum",
        "subphylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    logger.info(
        "Building species to most common effect map for taxon-based assignment..."
    )
    species_acute_effect_map = (
        df_acute.groupby("NCBI_rank_family")["effect"]
        .agg(lambda x: x.mode(dropna=False).tolist())
        .to_dict()
    )
    species_group_acute_effect_map = (
        df_acute.groupby("species_group_corrected")["effect"]
        .agg(lambda x: x.mode(dropna=False).tolist())
        .to_dict()
    )
    species_chronic_effect_map = (
        df_chronic.groupby("NCBI_rank_family")["effect"]
        .agg(lambda x: x.mode(dropna=False).tolist())
        .to_dict()
    )
    species_group_chronic_effect_map = (
        df_chronic.groupby("species_group_corrected")["effect"]
        .agg(lambda x: x.mode(dropna=False).tolist())
        .to_dict()
    )
    df["assigned_acute_effect"] = df[
        ["species_group_corrected", "effect"] + [f"NCBI_rank_{rank}" for rank in ranks]
    ].apply(
        assign_acute_effect,
        axis=1,
        args=(species_acute_effect_map, species_group_acute_effect_map),
    )
    df["assigned_chronic_effect"] = df[
        ["species_group_corrected", "effect"] + [f"NCBI_rank_{rank}" for rank in ranks]
    ].apply(
        assign_chronic_effect,
        axis=1,
        args=(species_chronic_effect_map, species_group_chronic_effect_map),
    )

    print_effect_assignment_source_stats(
        df, species_acute_effect_map, species_group_acute_effect_map
    )

    # Limit to acute duration being above np.log10(24 hours) or 1e-6 (since this indicates NaN)
    df_acute = df[(df["duration"] > 0) & (df["duration"] < np.log10(505))]

    # Build species to most common duration map (can include ties in the form of lists)
    logger.info(
        "Building species to most common duration map for taxon-based assignment..."
    )
    ranks = [
        "superkingdom",
        "kingdom",
        "phylum",
        "subphylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    species_acute_duration_map = (
        df_acute.groupby("NCBI_rank_family")["duration"]
        .agg(lambda x: x.mode(dropna=False).tolist())
        .to_dict()
    )
    species_group_acute_duration_map = (
        df_acute.groupby("species_group_corrected")["duration"]
        .agg(lambda x: x.mode(dropna=False).tolist())
        .to_dict()
    )
    df["assigned_acute_duration"] = df[
        ["species_group_corrected", "duration"]
        + [f"NCBI_rank_{rank}" for rank in ranks]
    ].apply(
        assign_acute_duration,
        axis=1,
        args=(species_acute_duration_map, species_group_acute_duration_map),
    )

    # Transform back from log10 hours to hours
    df["assigned_acute_duration"] = df["assigned_acute_duration"].apply(
        lambda x: 10**x if pd.notna(x) else np.nan
    )

    print_duration_assignment_source_stats(
        df, species_acute_duration_map, species_group_acute_duration_map
    )

    # Save dataframe mapping taxa to assigned units, effects, and durations
    taxa_assignment = df[
        [f"NCBI_rank_{rank}" for rank in ranks]
        + [
            "species_group_corrected",
            "assigned_conc_unit",
            "assigned_acute_effect",
            "assigned_acute_duration",
        ]
    ].drop_duplicates()

    logger.success("Assigned units, effects, and durations for taxa. Saving to CSV...")
    taxa_assignment.to_csv(
        args.output_path,
        index=False,
    )
    logger.success(f"Saved taxa assignment to {args.output_path}")


if __name__ == "__main__":
    main()
