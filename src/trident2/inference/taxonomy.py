"""
Taxonomic information utilities.

This module provides functions to load taxonomic data files
used for TRIDENT inference, including
species group mappings, and lineage information.

It also includes functions to map taxonomic lineages to species groups
and to fill in missing taxonomy information in a DataFrame.

And other utilities related to taxonomy processing for TRIDENT.
"""

from __future__ import annotations
import json
import pandas as pd


def load_taxonomic_information(
    taxid2speciesgroup_dict_path: str | None = None,
    taxid2speciesname_dict_path: str | None = None,
    taxid2lineage_csv_path: str | None = None,
) -> tuple[dict | None, dict | None, pd.DataFrame | None]:
    """
    Load taxonomic information from various data files.

    Args:
        taxid2speciesgroup_dict_path: Path to JSON file with taxid to species group mappings.
        taxid2speciesname_dict_path: Path to JSON file with taxid to latin name mappings.
        taxid2lineage_csv_path: Path to CSV file with taxid to lineage mappings.

    Returns:
        Tuple of (speciesgroup_dict, speciesname_dict, lineage_df).
        Any item is None if its path was not provided.

    Example:
        >>> group_dict, name_dict, lineage_df = load_taxonomic_information(
        ...     taxid2speciesname_dict_path="taxid_names.json"
        ... )
    """
    taxid2speciesgroup_dict = None
    taxid2speciesname_dict = None
    taxid2lineage_df = None


    # Load species group mapping
    if taxid2speciesgroup_dict_path is not None:
        with open(taxid2speciesgroup_dict_path) as f:
            taxid2speciesgroup_dict = json.load(f)
        print(f"Loaded species group mapping with {len(taxid2speciesgroup_dict)} entries")
    else:
        print("No species group mapping provided")

    # Load taxid to species name mapping
    if taxid2speciesname_dict_path is not None:
        with open(taxid2speciesname_dict_path) as f:
            taxid2speciesname_dict = json.load(f)
        print(f"Loaded taxid to species name mapping with {len(taxid2speciesname_dict)} entries")
    else:
        print("No taxid to species name mapping provided")

    # Load taxid to lineage mapping
    if taxid2lineage_csv_path is not None:
        taxid2lineage_df = pd.read_csv(taxid2lineage_csv_path)
        print(f"Loaded taxid to lineage mapping with {len(taxid2lineage_df)} entries")
    else:
        print("No taxid to lineage mapping provided")

    return taxid2speciesgroup_dict, taxid2speciesname_dict, taxid2lineage_df


def map_lineage_to_species_group(ranked_lineage: dict) -> str:
    # mammals
    if ranked_lineage["class"] == "40674":
        if ranked_lineage["order"] == "9989":
            return "rodents"
        else:
            return "other mammals"
    # fish
    if (ranked_lineage["class"] == "186623") or ranked_lineage["class"] in [
        "7777",
        "2682552",
        "117569",
    ]:
        return "fish"
    # amphibians
    if ranked_lineage["class"] == "8292":
        return "amphibians"
    # anthropods
    if ranked_lineage["phylum"] == "6656":
        if (
            ranked_lineage["subphylum"] == "6657" or ranked_lineage["class"] == "6844"
        ):  # Pure crustaceans or horsehoe crabs
            return "crustaceans"
        elif ranked_lineage["class"] == "50557":  # These are the pure insects
            return "insects"
        else:
            return "insects"  # put other terrestrial anthropods here, like springtails, millipedes, they are basically anthropods that are neither insects nore crustaceans, they make up roughly 3000 datapoints
    # arachnids
    if ranked_lineage["phylum"] == "6854":
        return "spiders"
    # mollusks
    if ranked_lineage["phylum"] == "6447":  # These are pure mollusca
        return "mollusks"
    if ranked_lineage["phylum"] == "7568":  # These are other shells and snails
        return "mollusks"
    # fungi
    if ranked_lineage["kingdom"] == "4751":  # fungi
        return "fungi"
    # birds
    if ranked_lineage["class"] == "8782":  # aves
        return "birds"
    # plants
    if ranked_lineage["phylum"] in [
        "35493",
    ]:  # green plants
        return "plants"
    if ranked_lineage["class"] in [
        "2201463"
    ]:  # mosses and worts (they also belong to green plants though but for clarity)
        return "plants"
    # algae
    if ranked_lineage["phylum"] in [
        "2836",
        "3041",
        "2830",
        "2763",
    ]:  # diatoms, green algae, haptophytes, red algae,  etc
        return "algae"
    if (
        ranked_lineage["class"]
        in [
            "2870",
            "2864",
            "3035",
            "3027",
            "2825",
            "5747",
            "38410",
            "33859",
            "35675",
            "304573",
        ]
    ):  # brown algae, dinoflagellates, euglenids, cryptomonads, golden algae, Eustigmatophyceae, Raphidophytes, Synurids, Pelagophyceae
        return "algae"
    # cyanobacteria
    if ranked_lineage["phylum"] == "1117":  # apparently this is cyanobacteria
        return "cyanobacteria"
    # rotifers
    if ranked_lineage["phylum"] == "10190":  # rotifera
        return "rotifers"
    # echinoderms
    if ranked_lineage["phylum"] in ["7586"]:  # sea stars, sea urchins, sea cucumbers
        return "echinoderms"
    # worms
    if (
        ranked_lineage["phylum"]
        in ["33310", "6157", "43120", "10229", "6340", "6231", "6217", "6178", "10229"]
    ):  # nematodes, annelids, flatworms, goblet worms, segmnented worms, horse hair worms, arrow worms
        return "worms"
    # cnidarians and bryozoans
    if (
        ranked_lineage["phylum"] == "6073"
    ):  # Pure cnidarians = jellyfish, corals, sea anemones.
        return "cnidarians and bryozoans"
    if (
        ranked_lineage["phylum"] in ["10197", "10205"]
    ):  # ctenophores, bryozoans, (comb jellies, bryozoans here since they are aquatic and look like corals)
        return "cnidarians and bryozoans"
    # reptiles
    if ranked_lineage["class"] in ["8504"]:  # lepidosaurs
        return "reptiles"
    if ranked_lineage["order"] in ["8459", "1294634"]:  # turtles, alligators and others
        return "reptiles"
    # tunicates and sponges
    if ranked_lineage["subphylum"] == "7712":  # tunicates
        return "tunicates and sponges"
    if ranked_lineage["phylum"] == "6040":  # sponges
        return "tunicates and sponges"
    # protozoans
    if any(
        t
        in [
            "2605435",
            "136419",
            "5878",
            "6020",
            "5977",
            "6015",
            "37471",
            "5988",
            "33827",
            "194287",
            "422676",
            "1280412",
            "33829",
            "5653",
            "2779609",
            "6000",
            "1485085",
            "2681632",
            "555280",
            "2497438",
        ]
        for t in ranked_lineage.values()
    ):
        return "protozoans"
    return "unmapped"


# Construct ranked dict for each row in results and apply mapping function
def construct_ranked_lineage(row) -> dict:
    return {
        "kingdom": str(row["NCBI_rank_kingdom"]),
        "phylum": str(row["NCBI_rank_phylum"]),
        "subphylum": str(row["NCBI_rank_subphylum"]),
        "class": str(row["NCBI_rank_class"]),
        "order": str(row["NCBI_rank_order"]),
        "family": str(row["NCBI_rank_family"]),
        "genus": str(row["NCBI_rank_genus"]),
        "species": str(row["NCBI_rank_species"]),
    }


def fill_missing_taxonomy(df: pd.DataFrame, backwards: bool = True) -> pd.DataFrame:
    """
    Fill missing taxonomy information in a DataFrame.
    Parameters:
    - df: The input DataFrame containing taxonomy information.
    - backwards: If True, fill missing values by propagating the next valid observation backward. If False, fill missing values by propagating the last valid observation forward.

    Returns:
    - DataFrame: The DataFrame with missing taxonomy information filled.

    Example:
    >>> data = {
    ...     "NCBI_rank_superkingdom": ["Bacteria"],
    ...     "NCBI_rank_kingdom": [None],
    ...     "NCBI_rank_phylum": ["Proteobacteria"],
    ...     "NCBI_rank_class": [None],
    ...     "NCBI_rank_order": [None],
    ...     "NCBI_rank_family": ["Cyanobacteriota"],
    ...     "NCBI_rank_genus": [None],
    ...     "NCBI_rank_species": [None],
    ... }
    >>> df = pd.DataFrame(data)
    >>> filled_df = fill_missing_taxonomy(df, backwards=True)
    >>> print(filled_df)
    >>> {
    ...     "NCBI_rank_superkingdom": ["Bacteria"],
    ...     "NCBI_rank_kingdom": ["Proteobacteria_p"],
    ...     "NCBI_rank_phylum": ["Proteobacteria"],
    ...     "NCBI_rank_class": ["Cyanobacteriota_f_o"],
    ...     "NCBI_rank_order": ["Cyanobacteriota_f"],
    ...     "NCBI_rank_family": ["Cyanobacteriota"],
    ...     "NCBI_rank_genus": [None],
    ...     "NCBI_rank_species": [None],
    ... }
    >>> df = pd.DataFrame(data)
    >>> filled_df = fill_missing_taxonomy(df, backwards=False)
    >>> print(filled_df)
    >>> {
    ...     "NCBI_rank_superkingdom": ["Bacteria"],
    ...     "NCBI_rank_kingdom": ["Bacteria_sk"],
    ...     "NCBI_rank_phylum": ["Proteobacteria"],
    ...     "NCBI_rank_class": ["Proteobacteria_p"],
    ...     "NCBI_rank_order": ["Proteobacteria_p_c"],
    ...     "NCBI_rank_family": ["Cyanobacteriota"],
    ...     "NCBI_rank_genus": ["Cyanobacteriota_f"],
    ...     "NCBI_rank_species": ["Cyanobacteriota_f_g"],
    ... }
    """
    ranks = [
        "NCBI_rank_superkingdom",
        "NCBI_rank_kingdom",
        "NCBI_rank_phylum",
        "NCBI_rank_subphylum",
        "NCBI_rank_class",
        "NCBI_rank_order",
        "NCBI_rank_family",
        "NCBI_rank_genus",
        "NCBI_rank_species",
    ]
    suffixes = {
        "NCBI_rank_superkingdom": "sk",
        "NCBI_rank_kingdom": "k",
        "NCBI_rank_phylum": "p",
        "NCBI_rank_subphylum": "subp",
        "NCBI_rank_class": "c",
        "NCBI_rank_order": "o",
        "NCBI_rank_family": "f",
        "NCBI_rank_genus": "g",
        "NCBI_rank_species": "sp",
    }
    fill_order = ranks.copy()  # Order in which to fill missing ranks
    if backwards:
        fill_order.reverse()  # Start from species
    # Iterate over rows
    for i, row in df.iterrows():
        last_valid = None
        last_valid_rank = None

        for rank in fill_order:
            val = row[rank]

            if pd.notnull(val):
                # record the most recent non-null value and its rank
                last_valid = str(val)
                last_valid_rank = rank
            else:
                # build synthetic ID based on the most recent valid value's rank (append that rank's suffix)
                if last_valid and last_valid_rank:
                    suf = suffixes.get(last_valid_rank, suffixes.get(rank, ""))
                    new_val = f"{last_valid}_{suf}"
                    df.at[i, rank] = new_val
                    # update last_valid to the newly created synthetic value and mark its rank as the current filled rank
                    last_valid = new_val
                    last_valid_rank = rank
    return df
