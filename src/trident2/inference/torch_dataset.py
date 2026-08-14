"""
PyTorch Dataset classes for TRIDENT.

This module provides Dataset implementations for loading and processing
multimodal data for TRIDENT inference.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    """
    Dataset for multimodal chemical toxicity data.

    Handles SMILES strings, taxonomic IDs, metadata tokens, and various
    numerical features (duration, one-hot encodings).

    Args:
        df: DataFrame containing the data.
        smiles_column: Column name for SMILES strings.
        token_metadata_columns: List of columns for tokenizable metadata.
        taxid_column: Column name for taxonomic IDs.
        embedding_columns: List of columns for external embeddings.
        duration_column: Column name for exposure duration.
        onehot_column: Column name for one-hot encoded features.
        sep_token: Token to separate different input modalities.

    Example:
        >>> dataset = MultiModalDataset(
        ...     df=df,
        ...     smiles_column='SMILES',
        ...     taxid_column='ncbi_taxid',
        ...     duration_column='duration',
        ...     onehot_column='conc_unit_onehot'
        ... )
        >>> item = dataset[0]
        >>> print(item['input_text'])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        smiles_column: str | None = None,
        token_metadata_columns: list[str] | None = None,
        taxid_column: str | None = None,
        duration_column: str | None = None,
        onehot_column: str | None = None,
        sep_token: str = "</s>",
    ) -> None:
        self.df = df.copy().reset_index(drop=True)
        self.sep_token = sep_token

        # Column names
        self.smiles_column = smiles_column
        self.token_metadata_columns = token_metadata_columns
        self.taxid_column = taxid_column
        self.duration_column = duration_column
        self.onehot_column = onehot_column

        # Pre-extract columns into arrays for efficiency
        self.smiles = df[smiles_column].values if smiles_column else None
        self.metadata_arrays = (
            [df[c].values for c in self.token_metadata_columns]
            if self.token_metadata_columns
            else None
        )
        self.taxids = df[taxid_column].values if taxid_column else None
        self.durations = df[duration_column].values if duration_column else None
        self.onehots = df[onehot_column].values if onehot_column else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        texts = []

        # Add metadata tokens
        if self.metadata_arrays:
            for arr in self.metadata_arrays:
                metadata = arr[idx]
                if metadata is not None and not pd.isna(metadata):
                    texts.append(metadata)
            # Add separator after metadata
            texts.append(self.sep_token)

        # Add SMILES (optionally augmented)
        if self.smiles is not None:
            smiles = self.smiles[idx]
            texts.append(smiles)

        input_text = "".join([t for t in texts if t])

        # Taxonomic ID (with optional augmentation)
        taxid = None
        if self.taxids is not None:
            taxid = self.taxids[idx]

        return {
            "input_text": input_text,
            "taxid": taxid,
            "duration": [self.durations[idx]] if self.durations is not None else [np.nan],
            "one_hot_encoding": self.onehots[idx] if self.onehots is not None else None,
        }


class MultiModalCollator:
    """
    Collator for batching multimodal data.

    Tokenizes text inputs and stacks numerical features into tensors.

    Args:
        tokenizer: HuggingFace tokenizer for text tokenization.
        max_len: Maximum token sequence length.
        padding: Padding strategy ('longest', 'max_length').
        truncation: Whether to truncate sequences.
        padding_idx: Token ID used for padding.

    Example:
        >>> collator = MultiModalCollator(tokenizer, max_len=512)
        >>> batch = collator([dataset[i] for i in range(32)])
        >>> print(batch['input_ids'].shape)
    """

    def __init__(
        self,
        tokenizer,
        max_len: int = 512,
        padding: str = "longest",
        truncation: bool = True,
        padding_idx: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding = padding
        self.truncation = truncation
        self.padding_idx = padding_idx

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        # One-hot encoding
        one_hot_encoding = [item["one_hot_encoding"] for item in batch]
        one_hot_encoding = (
            torch.tensor(np.stack(one_hot_encoding), dtype=torch.float32)
            if None not in one_hot_encoding
            else None
        )

        # Durations
        durations = [item["duration"] for item in batch]
        durations = torch.tensor(durations, dtype=torch.float32)

        # Taxids
        taxids = [item["taxid"] for item in batch]
        taxids = taxids if any(taxids) else None

        # Tokenize text
        texts = [item["input_text"] for item in batch]
        encodings = self.tokenizer.batch_encode_plus(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_len,
            return_tensors="pt",
        )
        encodings["attention_mask"][encodings["input_ids"] == self.padding_idx] = 0

        # Build output batch
        batch_output = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

        if taxids is not None:
            batch_output["taxids"] = taxids
        if one_hot_encoding is not None:
            batch_output["one_hot_encoding"] = one_hot_encoding
        if durations is not None:
            batch_output["duration"] = durations

        return batch_output
