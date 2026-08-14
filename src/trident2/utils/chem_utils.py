"""
Chemistry utilities for TRIDENT.

This module provides utility functions for processing chemical
data, particularly SMILES strings.
"""

from __future__ import annotations
from typing import List, Optional, Union
from rdkit import Chem
from rdkit.Chem import inchi, rdMolDescriptors


def canonicalize_smiles(
    smiles: Union[List[str], str],
    canonical: bool = True,
    isomericSmiles: bool = False,
    drop_erroneous_smiles: bool = False,
) -> Union[List[str], List[None]]:
    """
    Canonicalize SMILES strings using RDKit.

    Args:
        smiles: Single SMILES string or list of SMILES.
        canonical: Whether to return canonical SMILES.
        isomericSmiles: Whether to preserve stereochemistry.
        drop_erroneous_smiles: If True, return None for invalid SMILES;
            otherwise return the original string.

    Returns:
        List of canonicalized SMILES (or None for invalid ones).

    Example:
        >>> canonicalize_smiles(['CCO', 'c1ccccc1', 'invalid'])
        ['CCO', 'c1ccccc1', 'invalid']  # or [None] if drop_erroneous_smiles=True
    """

    # Handle single string input
    if isinstance(smiles, str):
        smiles = [smiles]

    # Get unique SMILES to avoid redundant computations
    unique_smiles = list(set(smiles))

    # Create mapping from original to canonical
    smiles_to_canonical = {
        s: _canonicalize_single(s, canonical=canonical, isomericSmiles=isomericSmiles, flag_erroneous_smiles=drop_erroneous_smiles) for s in unique_smiles
    }

    # Map back to original order
    canonical_smiles = [smiles_to_canonical[s] for s in smiles]

    return canonical_smiles


def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is valid.

    Args:
        smiles: SMILES string to validate.

    Returns:
        True if valid, False otherwise.

    Example:
        >>> is_valid_smiles('CCO')
        True
        >>> is_valid_smiles('invalid')
        False
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def enumerate_smiles(smiles: str) -> str:
    """
    Generate a random non-canonical SMILES representation.

    Useful for data augmentation during training.

    Args:
        smiles: Input SMILES string.

    Returns:
        Randomized SMILES representation.

    Example:
        >>> enumerate_smiles('c1ccccc1')  # May return 'C1=CC=CC=C1' or similar
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        return smiles


def _canonicalize_single(smi: str, canonical: bool = True, isomericSmiles: bool = False, flag_erroneous_smiles: bool = False) -> Union[str, None]:
    try:
        mol = _mol_from_smiles(smi)
        if mol is None:
            raise ValueError("Invalid SMILES")
        return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomericSmiles)
    except Exception:
        if flag_erroneous_smiles:
            return None
        return smi
        

def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES; log a warning and return None on failure."""
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol


def calculate_rdkit_descriptors(smiles: Union[str, List[str]]) -> dict:
    """
    Compute descriptors from SMILES using RDKit.

    Returns a flat dictionary.  All keys use snake_case.  Missing or
    uncalculable values are stored as None.
    """
    # Handle single string input
    if isinstance(smiles, str):
        smiles = [smiles]

    # Get unique SMILES to avoid redundant computations
    unique_smiles = list(set(smiles))

    result: dict = {"smiles_input": unique_smiles}
    
    for smi in unique_smiles:
        mol = _mol_from_smiles(smi)
        if mol is None:
            result[smi] = {
                "canonical_smiles": None,
                "inchi": None,
                "inchikey": None,
            }
            continue

        try:
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        except Exception:
            canonical = None

        try:
            inchi_str = inchi.MolToInchi(mol)
        except Exception:
            inchi_str = None

        try:
            inchikey_str = inchi.InchiToInchiKey(inchi_str) if inchi_str else None
        except Exception:
            inchikey_str = None

        result[smi] = {
            "canonical_smiles": canonical,
            "inchi": inchi_str,
            "inchikey": inchikey_str,
        }

    return result
