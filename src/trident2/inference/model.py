"""
Prediction functions for TRIDENT inference.

This module provides the main prediction functionality for running
inference with TRIDENT models.
"""

from __future__ import annotations
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import os
from loguru import logger
from tqdm import tqdm


def build_model_and_tokenizer(
    model_name: str = "/home/skall/trident2_inference/models/eyr2owlv_final_model_final_epoch_hf_trident2",
    device=None,
):
    trident_2 = AutoModel.from_pretrained(
        model_name, token=os.environ.get("HF_ACCESS_TOKEN"), trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=os.environ.get("HF_ACCESS_TOKEN"), trust_remote_code=True
    )

    logger.success(
        f"Loaded model and tokenizer from {model_name} successfully."
    )

    if device is not None:
        trident_2 = trident_2.to(device)
        logger.info(f"Moved model to device: {device}")

    return trident_2, tokenizer


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda:0",
    mixed_precision: bool = False,
    return_cls_embeddings: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """
    Run inference with a TRIDENT model.

    Args:
        model: The TRIDENT model to use for predictions.
        dataloader: DataLoader containing the inference data.
        device: Device to run inference on ('cuda:0', 'cpu', etc.).
        mixed_precision: Whether to use FP16 mixed precision.
        return_cls_embeddings: If False, CLS embeddings are never accumulated
            or transferred off the GPU, which avoids unnecessary memory
            growth (768-dim float32 per sample) when embeddings are not
            needed by the caller.

    Returns:
        Tuple of (predictions dict, CLS embeddings array or None).
        Predictions dict has keys: 'EC50', 'EC10', 'NOEC', 'LOEC'.
        CLS embeddings is None when return_cls_embeddings is False.

    Example:
        >>> model = load_model(checkpoint_path)
        >>> dataloader = build_dataloader(dataset)
        >>> preds, embeddings = predict(model, dataloader, device='cuda:0')
        >>> print(preds['EC50'].shape)  # [num_samples]
    """
    model.eval()

    if mixed_precision:
        print("Using mixed precision inference")
    elif device != "cpu":
        print(
            "Using full precision inference on GPU. "
            "Consider enabling mixed precision for faster inference."
        )

    cls_embeddings = [] if return_cls_embeddings else None
    ec50_preds = []
    ec10_preds = []
    noec_preds = []
    loec_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            # Extract batch samples (exclude non-tensor items)
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key not in ["labels", "taxids", "endpoint"]
            }
            taxids = batch.get("taxids")

            if not mixed_precision:
                preds, roberta_output = model(**inputs, taxids=taxids)
            else:
                with torch.autocast(device_type=torch.device(device).type):
                    preds, roberta_output = model(**inputs, taxids=taxids)

            if return_cls_embeddings:
                cls_embeddings.append(roberta_output.detach().cpu().numpy())
            ec50_preds.append(preds["EC50"].float().detach().cpu().numpy())
            ec10_preds.append(preds["EC10"].float().detach().cpu().numpy())
            noec_preds.append(preds["NOEC"].float().detach().cpu().numpy())
            loec_preds.append(preds["LOEC"].float().detach().cpu().numpy())

    # Concatenate all predictions
    total_preds = {
        "EC50": np.concatenate(ec50_preds, axis=0),
        "EC10": np.concatenate(ec10_preds, axis=0),
        "NOEC": np.concatenate(noec_preds, axis=0),
        "LOEC": np.concatenate(loec_preds, axis=0),
    }
    if return_cls_embeddings:
        cls_embeddings = np.concatenate(cls_embeddings, axis=0)

    return total_preds, cls_embeddings


def run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda:0",
    mixed_precision: bool = False,
    return_embeddings: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Simplified inference function that returns only predictions.

    This is a convenience wrapper around `predict()` for cases where
    you only need the predictions without CLS embeddings.

    Args:
        model: The TRIDENT model to use for predictions.
        dataloader: DataLoader containing the inference data.
        device: Device to run inference on.
        mixed_precision: Whether to use FP16 mixed precision.
        return_embeddings: If True, also return CLS embeddings.

    Returns:
        If return_embeddings is False: Dictionary of predictions.
        If return_embeddings is True: Tuple of (predictions, embeddings).

    Example:
        >>> predictions = run_inference(model, dataloader)
        >>> print(predictions.keys())
        dict_keys(['EC50', 'EC10', 'NOEC', 'LOEC'])
    """
    preds, embeddings = predict(
        model=model,
        dataloader=dataloader,
        device=device,
        mixed_precision=mixed_precision,
        return_cls_embeddings=return_embeddings,
    )

    if return_embeddings:
        return preds, embeddings
    return preds