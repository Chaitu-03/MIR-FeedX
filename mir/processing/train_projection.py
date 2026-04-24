"""
One-time offline training script: learn CLIP-512 → MiniLM-384 projection.

Prerequisites
-------------
1. Download images via the crawler.
2. Generate caption pairs (requires BLIP-2):

    python -m mir.processing.generate_caption_pairs   # produces data/projection_pairs.npz

Run training
------------
    python -m mir.processing.train_projection

Reads:  data/projection_pairs.npz   (keys: clip_embeddings, caption_embeddings)
Writes: models/clip_to_miniLM_projection.pt

Retrain whenever the text embedding model changes (e.g. MiniLM → BGE):
regenerate caption_embeddings with the new model and re-run this script.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)

_PAIRS_PATH = Path("data/projection_pairs.npz")
_OUT_PATH = Path("models/clip_to_miniLM_projection.pt")
_CLIP_DIM = 512
_TEXT_DIM = 384
_HIDDEN_DIM = 768
_VAL_SIZE = 500
_EPOCHS = 80
_BATCH_SIZE = 256
_LR = 3e-4
_TARGET_COS = 0.75


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return (a_n * b_n).sum(dim=-1).mean().item()


def _combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE + cosine loss. MSE aligns magnitude; cosine aligns direction."""
    mse = F.mse_loss(pred, target)
    cos = (1 - F.cosine_similarity(pred, target)).mean()
    return mse + 0.5 * cos


def _build_projection() -> torch.nn.Sequential:
    """Non-linear projection: CLIP-512 → 768 → MiniLM-384."""
    model = torch.nn.Sequential(
        torch.nn.Linear(_CLIP_DIM, _HIDDEN_DIM),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(_HIDDEN_DIM, _TEXT_DIM),
    )
    # Initialise both linear layers with orthogonal weights
    torch.nn.init.orthogonal_(model[0].weight)
    torch.nn.init.orthogonal_(model[3].weight)
    return model


def train(
    pairs_path: Path = _PAIRS_PATH,
    out_path: Path = _OUT_PATH,
    epochs: int = _EPOCHS,
    batch_size: int = _BATCH_SIZE,
    lr: float = _LR,
    val_size: int = _VAL_SIZE,
) -> float:
    """
    Train the projection and return the final validation cosine similarity.
    Raises FileNotFoundError if projection_pairs.npz is missing.
    """
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Projection pairs not found at {pairs_path}.\n"
            "Generate them first:\n"
            "    python -m mir.processing.generate_caption_pairs"
        )

    data = np.load(pairs_path)
    clip_embs = torch.from_numpy(data["clip_embeddings"]).float()      # (N, 512)
    text_embs = torch.from_numpy(data["caption_embeddings"]).float()   # (N, 384)

    n = len(clip_embs)
    if n < val_size + batch_size:
        raise ValueError(
            f"Need at least {val_size + batch_size} pairs, got {n}. "
            "Generate more caption pairs."
        )

    log.info("Loaded %d pairs from %s", n, pairs_path)

    # Shuffle and split
    perm = torch.randperm(n)
    val_perm = perm[:val_size]
    train_perm = perm[val_size:]

    train_ds = TensorDataset(clip_embs[train_perm], text_embs[train_perm])
    val_clip = clip_embs[val_perm]
    val_text = text_embs[val_perm]

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Non-linear projection: CLIP-512 → 768 → MiniLM-384
    layer = _build_projection()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log.info("Training projection: %d train / %d val pairs, %d epochs", len(train_perm), val_size, epochs)

    for epoch in range(1, epochs + 1):
        layer.train()
        epoch_loss = 0.0
        for clip_b, text_b in loader:
            optimizer.zero_grad()
            pred = layer(clip_b)
            loss = _combined_loss(pred, text_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            layer.eval()
            with torch.no_grad():
                val_cos = _cosine_sim(layer(val_clip), val_text)
            log.info(
                "Epoch %3d/%d  loss=%.5f  val_cos=%.4f  lr=%.2e",
                epoch, epochs, epoch_loss / len(loader), val_cos,
                scheduler.get_last_lr()[0],
            )

    # Final validation
    layer.eval()
    with torch.no_grad():
        final_cos = _cosine_sim(layer(val_clip), val_text)

    log.info("Training complete. Val cosine similarity: %.4f (target ≥ %.2f)", final_cos, _TARGET_COS)
    if final_cos < _TARGET_COS:
        log.warning(
            "Val cosine %.4f < %.2f target. "
            "Consider more pairs (aim for 5 000+) or additional epochs.",
            final_cos, _TARGET_COS,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": layer.state_dict(), "clip_dim": _CLIP_DIM, "hidden_dim": _HIDDEN_DIM, "text_dim": _TEXT_DIM}, str(out_path))
    log.info("Saved projection weights → %s", out_path)

    return final_cos


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    train()
