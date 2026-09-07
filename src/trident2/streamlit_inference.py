"""
Streamlit-facing inference helpers for the TRIDENT-2 page.

Thin wrapper around the reusable pieces in ``src/trident2/inference/*`` and
``src/trident2/utils/*`` (dataset/dataloader/model code shared with the
``trident2_quick_inference.py`` CLI), adding Streamlit caching for the model,
tokenizer, and taxonomy/mapping data, plus a single ``predict_toxicity``
entry point for the page's single- and batch-prediction modes.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.trident2.inference.model import build_model_and_tokenizer, predict
from src.trident2.inference.preprocess_input import (
    CONC_UNIT_ENCODING,
    VALID_ADMINISTRATION_ROUTES,
    VALID_EFFECTS,
    VALID_ORGANISM_LIFESTAGES,
    generate_inference_metadata,
)
from src.trident2.inference.torch_dataset import MultiModalCollator, MultiModalDataset
from src.trident2.inference.torch_loaders import build_dataloader
from src.trident2.utils.chem_utils import canonicalize_smiles, is_valid_smiles

MODEL_REPO_ID = "StyrbjornKall/TRIDENT_2"

TAXID_TO_COMMONNAME_PATH = "data/trident2/taxid2commonname.json"
TAXID_TO_SCINAME_PATH = "data/trident2/taxid2sciname.json"
TAXA_MAPPING_CSV_PATH = "data/trident2/taxa_assigned_units_effects_durations_2026-04-17.csv"
PERMITTED_ROUTE_LIFESTAGE_COMBINATIONS_PATH = (
    "data/trident2/permitted_route_lifestage_combinations.json"
)

NOT_SPECIFIED = "Not specified"
ALL_SPECIES_GROUPS = "All species groups"

ENDPOINTS = ["EC50", "EC10", "NOEC", "LOEC"]
EFFECT_OPTIONS = sorted(VALID_EFFECTS)
ADMINISTRATION_ROUTE_OPTIONS = [NOT_SPECIFIED] + sorted(VALID_ADMINISTRATION_ROUTES)
ORGANISM_LIFESTAGE_OPTIONS = [NOT_SPECIFIED] + sorted(VALID_ORGANISM_LIFESTAGES)
CONC_UNIT_OPTIONS = sorted(CONC_UNIT_ENCODING.keys())


@st.cache_resource(show_spinner="Loading TRIDENT-2 model from Hugging Face (this can take a moment)...")
def load_model_and_tokenizer():
    model, tokenizer = build_model_and_tokenizer(model_name=MODEL_REPO_ID, device="cpu")
    model.eval()
    return model, tokenizer


@st.cache_data(show_spinner="Loading permitted route/lifestage combinations...")
def load_permitted_route_lifestage_combinations() -> dict[str, dict[str, list[str]]]:
    """Load the species->lifestage->route allowlist built from >=5 observations.

    The nested dictionary is created once from the preprocessed training data and
    loaded at app startup so dropdowns only expose combinations supported by the
    source dataset.
    """
    with open(PERMITTED_ROUTE_LIFESTAGE_COMBINATIONS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading species list...")
def load_species_options() -> pd.DataFrame:
    """Build the species selection table used by the page.

    Only species present in both the name dictionaries, the taxa mapping CSV, and
    the >=5-observation route/lifestage allowlist are included, so every
    selectable species has a display label, a eukaryotic species group, and
    default effect/conc_unit/duration values to pre-fill the form with.
    """
    permitted_combinations = load_permitted_route_lifestage_combinations()

    with open(TAXID_TO_COMMONNAME_PATH) as f:
        taxid2commonname: dict[str, str] = json.load(f)
    with open(TAXID_TO_SCINAME_PATH) as f:
        taxid2sciname: dict[str, str] = json.load(f)

    mapping_df = pd.read_csv(TAXA_MAPPING_CSV_PATH)
    mapping_df["NCBI_rank_species"] = mapping_df["NCBI_rank_species"].astype(int).astype(str)
    mapping_df = mapping_df.drop_duplicates(subset=["NCBI_rank_species"]).set_index(
        "NCBI_rank_species"
    )

    rows = []
    for taxid, common_name in taxid2commonname.items():
        if taxid not in taxid2sciname or taxid not in mapping_df.index:
            continue
        if taxid not in permitted_combinations:
            continue
        latin_name = taxid2sciname[taxid]
        # A handful of entries have no recorded common name; fall back to the latin name alone.
        label = f"{common_name} ({latin_name})" if common_name else latin_name
        default = mapping_df.loc[taxid]
        rows.append(
            {
                "taxid": taxid,
                "common_name": common_name,
                "latin_name": latin_name,
                "label": label,
                "species_group": default["species_group_corrected"],
                "default_effect": default["assigned_acute_effect"],
                "default_conc_unit": default["assigned_conc_unit"],
                "default_duration": default["assigned_acute_duration"],
            }
        )

    return pd.DataFrame(rows).sort_values("latin_name").reset_index(drop=True)


def get_valid_lifestage_options_for_species(taxid: str) -> list[str]:
    """Return organism lifestages valid for a species under the >=5 rule."""
    allowed = load_permitted_route_lifestage_combinations().get(str(taxid), {})
    return [NOT_SPECIFIED] + sorted(allowed.keys())


def get_valid_route_options_for_species_and_lifestage(taxid: str, lifestage: Optional[str]) -> list[str]:
    """Return administration routes valid for a species and selected lifestage."""
    allowed = load_permitted_route_lifestage_combinations().get(str(taxid), {})
    if not allowed:
        return ADMINISTRATION_ROUTE_OPTIONS

    if lifestage and lifestage != NOT_SPECIFIED:
        routes = allowed.get(lifestage, [])
    else:
        routes = sorted({route for route_list in allowed.values() for route in route_list})
    return [NOT_SPECIFIED] + sorted(routes)


def predict_toxicity(
    smiles_list: list[str],
    taxid: str,
    effect: str,
    conc_unit: str,
    duration_hours: Optional[float],
    administration_route: Optional[str] = None,
    organism_lifestage: Optional[str] = None,
    species_common_name: Optional[str] = None,
    species_latin_name: Optional[str] = None,
    species_group: Optional[str] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Run TRIDENT-2 inference for a list of SMILES against one species/effect/route/lifestage/unit.

    Args:
        smiles_list: SMILES strings to predict.
        taxid: NCBI taxid string of the selected species.
        effect: One of ``VALID_EFFECTS``.
        conc_unit: One of ``CONC_UNIT_OPTIONS``.
        duration_hours: Exposure duration in hours, or ``None`` if unknown/unavailable
            for this species (the model's missing-duration flag is used instead).
        administration_route: One of ``VALID_ADMINISTRATION_ROUTES``, or ``None``.
        organism_lifestage: One of ``VALID_ORGANISM_LIFESTAGES``, or ``None``.
        species_common_name: Common name of the selected species, for display in the output.
        species_latin_name: Latin name of the selected species, for display in the output.
        species_group: Eukaryotic species group of the selected species, for display in the output.
        batch_size: Inference batch size.

    Returns:
        DataFrame with one row per input SMILES (original order/duplicates
        preserved) containing the canonicalized SMILES, a validity flag, the
        species/effect/duration/route/lifestage/unit metadata used, and log10
        + back-transformed predictions for EC50/EC10/NOEC/LOEC in the chosen
        concentration unit (all log10 columns first, followed by all linear
        columns). Invalid SMILES get ``NaN`` predictions.
    """
    model, tokenizer = load_model_and_tokenizer()

    df = pd.DataFrame({"SMILES": smiles_list})
    df["SMILES valid"] = df["SMILES"].apply(is_valid_smiles)
    df["inference_smiles"] = canonicalize_smiles(
        df["SMILES"].tolist(), canonical=True, isomericSmiles=False, drop_erroneous_smiles=True
    )

    df["Species common name"] = species_common_name
    df["Species latin name"] = species_latin_name
    df["NCBI taxid"] = taxid
    df["Species group"] = species_group
    df["Effect"] = effect
    df["Administration route"] = administration_route if administration_route else NOT_SPECIFIED
    df["Organism lifestage"] = organism_lifestage if organism_lifestage else NOT_SPECIFIED
    df["Concentration unit"] = conc_unit
    df["Duration (h)"] = duration_hours if duration_hours is not None else np.nan

    log10_columns = [f"{endpoint} log10({conc_unit})" for endpoint in ENDPOINTS]
    linear_columns = [f"{endpoint} ({conc_unit})" for endpoint in ENDPOINTS]
    for column in log10_columns + linear_columns:
        df[column] = np.nan

    valid_index = df.index[df["inference_smiles"].notna()]
    if len(valid_index) == 0:
        return df

    work_df = df.loc[valid_index].copy()
    work_df["inference_ncbi_taxid"] = taxid
    work_df["inference_effect"] = effect
    work_df["inference_conc_unit"] = conc_unit
    work_df["inference_duration"] = duration_hours
    work_df["inference_administration_route_categorized"] = administration_route
    work_df["inference_organism_lifestage_categorized"] = organism_lifestage

    mapping_df = pd.read_csv(TAXA_MAPPING_CSV_PATH)
    work_df = generate_inference_metadata(df=work_df, inference_mapping_df=mapping_df)

    dataset = MultiModalDataset(
        df=work_df,
        smiles_column="inference_smiles",
        token_metadata_columns=[
            "inference_effect",
            "inference_administration_route_categorized",
            "inference_organism_lifestage_categorized",
        ],
        taxid_column="inference_ncbi_taxid",
        duration_column="inference_duration",
        onehot_column="inference_onehot",
        sep_token=tokenizer.special_tokens_map["sep_token"],
    )
    collate_fn = MultiModalCollator(
        tokenizer=tokenizer,
        max_len=200,
        padding="longest",
        truncation=True,
        padding_idx=tokenizer.pad_token_id,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    preds, _ = predict(model=model, dataloader=dataloader, device="cpu", return_cls_embeddings=False)

    for endpoint, values in preds.items():
        df.loc[valid_index, f"{endpoint} log10({conc_unit})"] = values
        df.loc[valid_index, f"{endpoint} ({conc_unit})"] = 10 ** values

    return df
