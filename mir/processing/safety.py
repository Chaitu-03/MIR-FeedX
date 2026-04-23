from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from mir.config import settings

log = logging.getLogger(__name__)

NSFW_PROMPTS: list[str] = [
    "an explicit photo",
    "nudity",
    "pornographic content",
    "sexually explicit image",
]
SFW_PROMPTS: list[str] = [
    "a safe for work photo",
    "a normal photograph",
    "a landscape",
    "art",
]

_ALL_PROMPTS = NSFW_PROMPTS + SFW_PROMPTS
_N_NSFW = len(NSFW_PROMPTS)


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class NSFWClassifier:
    """
    CLIP ViT-B/32-based NSFW image classifier + keyword blocklist for text posts.

    Must run at ingestion time — before any post is written to the search indexes.
    NSFW posts are stored in Postgres for audit purposes but are excluded from
    Qdrant and do not count toward TARGET_POST_COUNT.

    Pass an existing `clip_model` + `preprocess` to share the CLIP instance with
    ImageProcessor and avoid loading the model weights a second time.
    """

    def __init__(
        self,
        clip_model: Any | None = None,
        preprocess: Any | None = None,
        device: str | None = None,
        blocklist_path: Path | None = None,
    ) -> None:
        self._device = device or _resolve_device()

        if clip_model is not None and preprocess is not None:
            self._model = clip_model
            self._preprocess = preprocess
        else:
            self._model, self._preprocess = self._load_clip()

        self._model = self._model.to(self._device).eval()

        self._tokenizer = self._build_tokenizer()
        # Precompute and cache prompt features — these never change between calls.
        self._text_features: torch.Tensor = self._encode_prompts()

        _bl_path = blocklist_path or settings.nsfw_blocklist_path
        self.text_blocklist: list[str] = self._load_blocklist(_bl_path)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_clip() -> tuple[Any, Any]:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        return model, preprocess

    @staticmethod
    def _build_tokenizer() -> Any:
        import open_clip
        return open_clip.get_tokenizer("ViT-B-32")

    def _encode_prompts(self) -> torch.Tensor:
        """Encode all prompts once and return L2-normalised feature matrix (8 × D)."""
        tokens = self._tokenizer(_ALL_PROMPTS).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = F.normalize(features, dim=-1)
        return features  # (8, D)

    @staticmethod
    def _load_blocklist(path: Path) -> list[str]:
        if not path.exists():
            log.warning(
                "NSFW text blocklist not found at %s — text classifier is disabled.", path
            )
            return []
        terms = [
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        log.info("Loaded %d NSFW blocklist terms from %s", len(terms), path)
        return terms

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def is_image_nsfw(
        self,
        image: Image.Image,
        threshold: float | None = None,
    ) -> tuple[bool, float]:
        """
        Classify a single PIL image.

        Returns (is_nsfw, nsfw_score) where nsfw_score is the summed softmax
        probability mass over the NSFW prompt set.
        """
        threshold = threshold if threshold is not None else settings.nsfw_threshold

        tensor = self._preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(tensor)
            image_features = F.normalize(image_features, dim=-1)

        # Cosine similarity → softmax across all prompts (both NSFW and SFW)
        probs = (image_features @ self._text_features.T).softmax(dim=-1)
        nsfw_score: float = probs[0, :_N_NSFW].sum().item()

        return nsfw_score > threshold, nsfw_score

    def is_text_nsfw(self, text: str) -> bool:
        """Fast blocklist scan for text-only posts that have no downloadable images."""
        text_lower = text.lower()
        return any(term in text_lower for term in self.text_blocklist)

    def classify_post(
        self,
        post: dict,
        images: list[Image.Image] | None = None,
    ) -> dict[str, bool | float | None]:
        """
        Classify a post for NSFW content.

        - If `images` is provided (pre-loaded by ImageDownloader), run the image
          classifier on each one and flag the post if ANY image exceeds the threshold.
        - If no images are provided (text-only post), fall back to the keyword blocklist.

        Returns:
            {
                "nsfw":       bool         — True if the post should be excluded from indexes,
                "nsfw_score": float | None — highest image NSFW score; None for text-only posts.
            }

        Caller responsibilities:
            - NSFW posts MUST be excluded from all Qdrant collections.
            - NSFW posts MUST NOT be counted toward TARGET_POST_COUNT.
            - NSFW posts MAY be stored in Postgres for audit / reclassification.
        """
        nsfw = False
        nsfw_score: float | None = None

        if images:
            max_score = 0.0
            for img in images:
                is_nsfw, score = self.is_image_nsfw(img)
                if score > max_score:
                    max_score = score
                if is_nsfw:
                    nsfw = True
            nsfw_score = max_score
        else:
            text = post.get("body_clean") or post.get("body_raw") or ""
            if text:
                nsfw = self.is_text_nsfw(text)

        return {"nsfw": nsfw, "nsfw_score": nsfw_score}
