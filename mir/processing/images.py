from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from mir.config import settings

log = logging.getLogger(__name__)

_CLIP_DIM = 512
_TEXT_DIM = 384
_HIDDEN_DIM = 768


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Projection layer: CLIP 512-d → MiniLM 384-d
# ---------------------------------------------------------------------------

class ProjectionLayer:
    """
    Trainable nn.Linear(512, 384, bias=False) that maps CLIP embeddings into
    the MiniLM sentence-embedding space.

    If trained weights exist on disk they are loaded.  Otherwise an orthogonal
    random matrix is used as a geometry-preserving placeholder (NOT truncation —
    truncation destroys distance relationships).

    Retrain whenever the text embedding model changes (MiniLM → BGE, etc.):
        python -m mir.processing.train_projection
    """

    def __init__(
        self,
        weights_path: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self._device = device

        path = weights_path or settings.clip_miniLM_projection_path
        if path.exists():
            checkpoint = torch.load(str(path), map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # New format: non-linear projection saved with metadata dict
                hidden = checkpoint.get("hidden_dim", _HIDDEN_DIM)
                self._linear = nn.Sequential(
                    nn.Linear(_CLIP_DIM, hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, _TEXT_DIM),
                )
                self._linear.load_state_dict(checkpoint["state_dict"])
            elif isinstance(checkpoint, dict) and any(k.startswith("0.") for k in checkpoint):
                # Sequential state_dict saved bare (e.g. torch.save(layer.state_dict(), path))
                # Infer hidden dim from the first linear weight shape
                hidden = checkpoint["0.weight"].shape[0]
                self._linear = nn.Sequential(
                    nn.Linear(_CLIP_DIM, hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, _TEXT_DIM),
                )
                self._linear.load_state_dict(checkpoint)
            else:
                # True legacy: bare nn.Linear(512, 384, bias=False) state_dict
                log.warning("Loading legacy linear projection weights — retrain recommended")
                self._linear = nn.Linear(_CLIP_DIM, _TEXT_DIM, bias=False)
                self._linear.load_state_dict(checkpoint)
            log.info("Loaded projection weights from %s", path)
        else:
            # Untrained fallback: non-linear with orthogonal init
            self._linear = nn.Sequential(
                nn.Linear(_CLIP_DIM, _HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(_HIDDEN_DIM, _TEXT_DIM),
            )
            nn.init.orthogonal_(self._linear[0].weight)
            nn.init.orthogonal_(self._linear[3].weight)
            log.warning(
                "Using untrained projection layer — run "
                "`python -m mir.processing.train_projection` first"
            )

        self._linear = self._linear.to(device).eval()

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project a 512-d CLIP embedding (or batch N×512) to 384-d, L2-normalised.

        Returns shape (384,) for 1-d input, (N, 384) for 2-d input.
        """
        squeeze = embedding.ndim == 1
        t = torch.from_numpy(np.atleast_2d(embedding)).float().to(self._device)
        with torch.no_grad():
            out = self._linear(t)
            out = F.normalize(out, dim=-1)
        result = out.cpu().numpy()
        return result.squeeze(0) if squeeze else result


# ---------------------------------------------------------------------------
# Image processor
# ---------------------------------------------------------------------------

class ImageProcessor:
    """
    CLIP ViT-B/32 image embedder with Redis-backed phash cache.

    Pass an existing (clip_model, preprocess) from NSFWClassifier to avoid
    loading CLIP weights a second time.
    """

    def __init__(
        self,
        clip_model: Any | None = None,
        preprocess: Any | None = None,
        device: str | None = None,
        redis_client: Any | None = None,          # sync redis.Redis
        weights_path: Path | None = None,
    ) -> None:
        self._device = device or _resolve_device()

        if clip_model is not None and preprocess is not None:
            self._model = clip_model
            self._preprocess = preprocess
        else:
            self._model, self._preprocess = self._load_clip()

        self._model = self._model.to(self._device).eval()
        self._redis = redis_client
        self._projection = ProjectionLayer(
            weights_path=weights_path, device=self._device
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _load_clip() -> tuple[Any, Any]:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        return model, preprocess

    # ------------------------------------------------------------------
    # Redis cache  (key = mir:img_emb:{phash}, value = JSON float list)
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(phash: str) -> str:
        return f"mir:img_emb:{phash}"

    def _cache_get(self, phash: str) -> np.ndarray | None:
        if not self._redis:
            return None
        raw = self._redis.get(self._cache_key(phash))
        if raw is None:
            return None
        return np.array(json.loads(raw), dtype=np.float32)

    def _cache_set(self, phash: str, embedding: np.ndarray) -> None:
        if not self._redis:
            return
        self._redis.set(self._cache_key(phash), json.dumps(embedding.tolist()))

    # ------------------------------------------------------------------
    # Internal embedding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _phash(image: Image.Image) -> str:
        import imagehash
        return str(imagehash.phash(image))

    def _encode_batch(self, tensors: list[torch.Tensor]) -> np.ndarray:
        """Stack, encode, L2-normalise; returns (B, 512) float32."""
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_image(batch)
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Return 512-d L2-normalised CLIP embedding for one image.

        Checks Redis cache by phash first; stores on miss. No TTL — embeddings
        are stable for a given image content.
        """
        img = Image.open(image_path).convert("RGB")
        ph = self._phash(img)

        cached = self._cache_get(ph)
        if cached is not None:
            log.debug("Cache hit phash=%s", ph)
            return cached

        tensor = self._preprocess(img)
        embedding = self._encode_batch([tensor])[0]   # (512,)
        self._cache_set(ph, embedding)
        return embedding

    def embed_images_batch(
        self,
        image_paths: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Embed a list of image paths; returns shape (N, 512), L2-normalised.

        Each image is loaded once (phash + preprocess from same PIL object).
        Cache is checked per image; only misses go through CLIP.
        """
        if not image_paths:
            return np.zeros((0, _CLIP_DIM), dtype=np.float32)

        n = len(image_paths)
        results: list[np.ndarray | None] = [None] * n
        miss_idx: list[int] = []
        miss_tensors: list[torch.Tensor] = []
        miss_phashes: list[str] = []

        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
            except Exception as exc:
                log.warning("Skipping unreadable image %s: %s", path, exc)
                continue
            ph = self._phash(img)
            cached = self._cache_get(ph)
            if cached is not None:
                results[i] = cached
            else:
                miss_idx.append(i)
                miss_tensors.append(self._preprocess(img))
                miss_phashes.append(ph)

        # Batch-encode cache misses
        for b_start in range(0, len(miss_tensors), batch_size):
            b_tensors = miss_tensors[b_start : b_start + batch_size]
            b_idx = miss_idx[b_start : b_start + batch_size]
            b_phashes = miss_phashes[b_start : b_start + batch_size]

            embeddings = self._encode_batch(b_tensors)   # (B, 512)
            for local_j, (idx, ph, emb) in enumerate(
                zip(b_idx, b_phashes, embeddings)
            ):
                results[idx] = emb
                self._cache_set(ph, emb)

        valid = [r for r in results if r is not None]
        if not valid:
            return np.zeros((0, _CLIP_DIM), dtype=np.float32)
        return np.stack(valid)

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Project a 512-d (or N×512) CLIP embedding to 384-d MiniLM space."""
        return self._projection.project(embedding)
