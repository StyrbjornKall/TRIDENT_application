import os
import numpy as np
import pandas as pd
import json
from loguru import logger


# Deterministic one-hot encoding for concentration units
CONC_UNIT_ENCODING = {
    "mg/l":	        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "ppm":	        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "mg/kg":	    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "kg/m2":	    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "%":	        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "mg/kg bw/d":	[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "mg/org":	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "mg/kg diet":	[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "mg/kg bw":	    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "mg/kg org":	[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "mg/kg soil":	[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "mg/l air":	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "ml/kg bw":	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "mg/org/d":	    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
}

DURATION_MISSING_ENCODING: dict[str, list[int]] = {
    "1": [1],
    "0": [0],
}

# Supported --inference_effect codes
VALID_EFFECTS = {"MOR", "POP", "GRO", "BEH", "REP", "ITX", "DVP", "PHY", "MPH", "CAR"}

# Supported --inference_administration_route values
VALID_ADMINISTRATION_ROUTES = {
    "static",
    "oral",
    "renewal",
    "intraperitoneal",
    "environmental",
    "flow_through",
    "spray",
    "intravenous",
    "food",
    "direct_application",
    "subcutaneous",
    "gavage",
    "inhalation",
    "culture_media",
    "granular",
    "topical",
    "intramuscular",
    "soaking",
    "dermal",
    "drinking",
    "in_vitro",
    "injection",
    "parenteral",
}

# Supported --inference_organism_lifestage_categorized values
VALID_ORGANISM_LIFESTAGES = {"early", "mid", "adult"}


def _normalize_values(values) -> list:
    """Normalize a scalar, list, or None into a flat list (empty if None/NaN)."""
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    return [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]


def validate_inference_effect(inference_effect) -> None:
    """Raise ValueError if any --inference_effect value is not supported.

    None/missing values are allowed (they fall back to the per-species
    inference mapping CSV).

    Args:
        inference_effect: A single value, a list of values, or None.
    """
    values = _normalize_values(inference_effect)
    invalid = sorted(set(values) - VALID_EFFECTS)
    if invalid:
        raise ValueError(
            f"Invalid --inference_effect value(s): {invalid}. "
            f"Must be one of: {sorted(VALID_EFFECTS)}"
        )


def validate_administration_route(inference_administration_route) -> None:
    """Raise ValueError if any --inference_administration_route value is not supported.

    None/missing values are allowed (they fall back to the per-species
    inference mapping CSV).

    Args:
        inference_administration_route: A single value, a list of values, or None.
    """
    values = _normalize_values(inference_administration_route)
    invalid = sorted(set(values) - VALID_ADMINISTRATION_ROUTES)
    if invalid:
        raise ValueError(
            f"Invalid --inference_administration_route value(s): {invalid}. "
            f"Must be one of: {sorted(VALID_ADMINISTRATION_ROUTES)}"
        )


def validate_organism_lifestage(inference_organism_lifestage_categorized) -> None:
    """Raise ValueError if any --inference_organism_lifestage_categorized value is not supported.

    None/missing values are allowed (they fall back to the per-species
    inference mapping CSV).

    Args:
        inference_organism_lifestage_categorized: A single value, a list of values, or None.
    """
    values = _normalize_values(inference_organism_lifestage_categorized)
    invalid = sorted(set(values) - VALID_ORGANISM_LIFESTAGES)
    if invalid:
        raise ValueError(
            f"Invalid --inference_organism_lifestage_categorized value(s): {invalid}. "
            f"Must be one of: {sorted(VALID_ORGANISM_LIFESTAGES)}"
        )


def map_family_to_inference_setting(family: str, mapping_dict: dict):
    if family in mapping_dict:
        effect, conc_unit, duration = mapping_dict[family]
    else:
        effect, conc_unit, duration = None, None, None
    return effect, conc_unit, duration


def map_species_group_to_inference_setting(species_group: str, mapping_dict: dict):
    if species_group in mapping_dict:
        effect, conc_unit, duration = mapping_dict[species_group]
    else:
        effect, conc_unit, duration = None, None, None
    return effect, conc_unit, duration


def deterministic_onehot_encoding_conc_unit(conc_unit: str):
    """
    Get deterministic one-hot encoding for a concentration unit.

    Args:
        conc_unit: Concentration unit string (e.g., 'mg/l', 'ppm').

    Returns:
        14-element one-hot encoded list.

    Example:
        >>> deterministic_onehot_encoding('mg/l')
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    """
    return CONC_UNIT_ENCODING.get(conc_unit, [0] * len(CONC_UNIT_ENCODING))


def deterministic_onehot_encoding_duration_missing(duration_missing: str):
    """
    Get deterministic one-hot encoding for duration missing flag.

    Args:
        duration_missing: Duration missing flag string ('1' for missing, '0' for present).

    Returns:
        2-element one-hot encoded list.

    Example:
        >>> deterministic_onehot_encoding_duration_missing('1')
        [1, 0]
    """
    return DURATION_MISSING_ENCODING.get(duration_missing, [0] * len(DURATION_MISSING_ENCODING))


def generate_inference_metadata(
    df: pd.DataFrame,
    inference_mapping_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generate inference metadata columns for the input dataframe.

    This function adds/updates the following columns:
    - inference_effect: Effect type (e.g., 'MOR' for mortality)
    - inference_duration: Log10-transformed exposure duration
    - inference_conc_unit: Concentration unit
    - inference_onehot: One-hot encoded concentration unit

    Args:
        df: Input dataframe with inference_ncbi_taxid column.
        inference_mapping_df: Optional dataframe mapping taxa to metadata.
            Should have columns: NCBI_rank_species, assigned_acute_effect,
            assigned_acute_duration, assigned_conc_unit.

    Returns:
        DataFrame with added metadata columns.

    Raises:
        ValueError: If metadata is None and inference_mapping_df not provided.

    Example:
        >>> df = pd.DataFrame({
        ...     'inference_ncbi_taxid': ['7955', '7227'],
        ...     'inference_effect': [None, None],
        ...     'inference_duration': [None, None],
        ...     'inference_conc_unit': [None, None],
        ... })
        >>> df = generate_inference_metadata(df, mapping_df)
    """
    df = df.copy()

    # Check if we need the mapping dataframe
    has_all_metadata = (
        df["inference_effect"].notna().all()
        and df["inference_duration"].notna().all()
        and df["inference_conc_unit"].notna().all()
    )

    if not has_all_metadata and inference_mapping_df is None:
        raise ValueError(
            "If any of inference_effect, inference_duration, or inference_conc_unit "
            "is None, inference_mapping_df must be provided."
        )

    # Create mapping dictionaries from inference_mapping_df
    if inference_mapping_df is not None:
        # Convert taxids to strings to match the format in the inference dataframe
        taxid_to_effect = dict(
            zip(
                inference_mapping_df["NCBI_rank_species"].astype(int).astype("string"),
                inference_mapping_df["assigned_acute_effect"],
            )
        )
        taxid_to_duration = dict(
            zip(
                inference_mapping_df["NCBI_rank_species"].astype(int).astype("string"),
                inference_mapping_df["assigned_acute_duration"],
            )
        )
        taxid_to_conc_unit = dict(
            zip(
                inference_mapping_df["NCBI_rank_species"].astype(int).astype("string"),
                inference_mapping_df["assigned_conc_unit"],
            )
        )

        # Apply mappings where values are None
        if df["inference_effect"].isna().any():
            logger.info("Assigning effect from mapping.")
            mask = df["inference_effect"].isna()
            df.loc[mask, "inference_effect"] = df.loc[mask, "inference_ncbi_taxid"].map(
                taxid_to_effect
            )

        if df["inference_duration"].isna().any():
            logger.info("Assigning duration from mapping.")
            mask = df["inference_duration"].isna()
            df.loc[mask, "inference_duration"] = df.loc[
                mask, "inference_ncbi_taxid"
            ].map(taxid_to_duration)

        if df["inference_conc_unit"].isna().any():
            logger.info("Assigning concentration unit from mapping.")
            mask = df["inference_conc_unit"].isna()
            df.loc[mask, "inference_conc_unit"] = df.loc[
                mask, "inference_ncbi_taxid"
            ].map(taxid_to_conc_unit)

    # Fill remaining missing values with defaults
    if df["inference_effect"].isna().any():
        missing_count = df["inference_effect"].isna().sum()
        logger.info(f"Filling {missing_count} missing effects with MOR (mortality).")
        df["inference_effect"] = df["inference_effect"].fillna("MOR")

    if df["inference_conc_unit"].isna().any():
        missing_count = df["inference_conc_unit"].isna().sum()
        logger.info(f"Filling {missing_count} missing concentration units with mg/l.")
        df["inference_conc_unit"] = df["inference_conc_unit"].fillna("mg/l")

    # One-hot encode concentration units and duration-missing flag, then combine
    df["inference_onehot_conc_unit"] = df["inference_conc_unit"].map(
        deterministic_onehot_encoding_conc_unit
    )
    df["inference_onehot_duration_missing"] = (
        df["inference_duration"].isna().astype(int).astype(str).map(
            deterministic_onehot_encoding_duration_missing
        )
    )
    # Replace NaN durations with 1e-6 (sentinel used by the model)
    df["inference_duration"] = df["inference_duration"].fillna(1e-6)
    # Combine into single 15-dim vector expected by MultiModalDataset
    df["inference_onehot"] = [
        conc + dur
        for conc, dur in zip(
            df["inference_onehot_conc_unit"], df["inference_onehot_duration_missing"]
        )
    ]
    assert df["inference_onehot"].apply(len).eq(15).all(), "One-hot vectors must be 15 elements long."
    logger.info("Generated inference_onehot column with combined one-hot encodings.")
    
    # Convert duration to numeric type and log10 transform
    logger.info("Filling missing durations with 1e-6 and Log10 transforming.")
    df["inference_duration"] = df["inference_duration"].fillna(1e-6)  # Sentinel for missing
    df["inference_duration"] = pd.to_numeric(df["inference_duration"], errors='coerce')
    df["inference_duration"] = np.log10(df["inference_duration"])

    # Format effect for tokenizer (add angle brackets)
    df["inference_effect"] = [
        f"<{x}>" if not pd.isna(x) else None for x in df["inference_effect"]
    ]

    # Format optional metadata columns
    if "inference_administration_route_categorized" in df.columns:
        df["inference_administration_route_categorized"] = [
            f"<{x}>" if not pd.isna(x) else None
            for x in df["inference_administration_route_categorized"]
        ]

    if "inference_organism_lifestage_categorized" in df.columns:
        df["inference_organism_lifestage_categorized"] = [
            f"<{x}>" if not pd.isna(x) else None
            for x in df["inference_organism_lifestage_categorized"]
        ]

    logger.info(f"Example input after metadata generation:\n{df.iloc[0][['inference_effect', 'inference_administration_route_categorized', 'inference_organism_lifestage_categorized']].to_dict()}")
    logger.success("Inference metadata generation complete.")

    return df