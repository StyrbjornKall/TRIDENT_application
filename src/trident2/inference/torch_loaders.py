"""
DataLoader utilities for TRIDENT.

This module provides functions for building DataLoaders with various
sampling strategies for training and inference.
"""

from __future__ import annotations

from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler
)

def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    collate_fn=None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader with specified sampling strategy.

    Args:
        dataset: PyTorch Dataset to create loader for.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        collate_fn: Custom collation function.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        Configured DataLoader instance.

    Example:
        >>> loader = build_dataloader(
        ...     dataset,
        ...     sampler='SequentialSampler',
        ...     batch_size=64,
        ...     collate_fn=collator
        ... )
        >>> for batch in loader:
        ...     # Process batch
        ...     pass
    """

    sampler_instance = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        sampler=sampler_instance,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    print(f"Built dataloader with {len(dataset)} samples, {len(dataloader)} batches")
    return dataloader
