from __future__ import annotations

import html as _html_stdlib
import logging
import re
import unicodedata
from typing import Any

import bleach
import html2text as _html2text
import numpy as np

from mir.config import settings

log = logging.getLogger(__name__)

_WINDOW_TOKENS = 512
_OVERLAP_TOKENS = 64
_EMBED_DIM = 384
_BATCH_SIZE = 64


class TextProcessor:
    """
    HTML → clean text → language detection → sentence embedding.

    Lifecycle
    ---------
    - SentenceTransformer and fasttext are loaded once in __init__.
    - Pass device='mps' for Apple Silicon, 'cuda' for NVIDIA, 'cpu' as fallback.
    - embed_tags() embeds EACH tag individually then mean-pools — do NOT concatenate
      tags into a single string before embedding.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "mps",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._device = device
        self._st_model = SentenceTransformer(model_name, device=device)
        self._tokenizer = self._st_model.tokenizer

        self._ft_model: Any = self._load_fasttext()

        self._h2t = _html2text.HTML2Text()
        self._h2t.ignore_links = True
        self._h2t.ignore_images = True
        self._h2t.body_width = 0  # no line-wrapping

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_fasttext() -> Any | None:
        path = settings.fasttext_model_path
        if not path.exists():
            log.warning(
                "fasttext LID model not found at %s — language detection disabled. "
                "Download from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
                path,
            )
            return None
        try:
            import fasttext

            fasttext.FastText.eprint = lambda *_: None  # suppress noisy stderr
            model = fasttext.load_model(str(path))
            log.info("fasttext LID model loaded from %s", path)
            return model
        except Exception as exc:
            log.warning("Failed to load fasttext model: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    # Matches <style>...</style> and <script>...</script> including their text content.
    _STYLE_SCRIPT_RE = re.compile(
        r"<(style|script)[^>]*>.*?</(style|script)>",
        re.DOTALL | re.IGNORECASE,
    )

    def clean(self, html: str) -> str:
        """
        Strip all HTML tags and return normalised plain text for embedding.

        Steps:
        1. Remove <style> and <script> elements entirely (content + tags).
        2. bleach.clean() strips remaining tags.
        3. html.unescape() decodes HTML entities (e.g. &eacute; → é).
        4. NFKC Unicode normalisation.
        5. Whitespace collapse.
        """
        if not html:
            return ""
        without_blocks = self._STYLE_SCRIPT_RE.sub("", html)
        stripped = bleach.clean(without_blocks, tags=[], strip=True)
        decoded = _html_stdlib.unescape(stripped)
        normalised = unicodedata.normalize("NFKC", decoded)
        return " ".join(normalised.split())

    def to_display(self, html: str) -> str:
        """
        Convert HTML to readable plain text preserving paragraph structure.

        Uses html2text so line breaks and list items survive. Intended for
        display snippets — not for embedding (use clean() for that).
        """
        if not html:
            return ""
        raw = self._h2t.handle(html)
        return unicodedata.normalize("NFKC", raw).strip()

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> str:
        """
        Return ISO 639-1 language code detected by fasttext LID-176.

        Falls back to 'und' (undetermined) when the model is unavailable or
        the text is too short to classify reliably.
        """
        if not self._ft_model or not text.strip():
            return "und"
        try:
            label = self._ft_model.predict(text.replace("\n", " "))[0][0]
            return label.replace("__label__", "")
        except Exception as exc:
            log.debug("Language detection failed: %s", exc)
            return "und"

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Batch-encode a list of strings.

        Returns an array of shape (N, 384). Empty input returns shape (0, 384).
        """
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        return self._st_model.encode(
            texts,
            batch_size=_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def embed_tags(self, tags: list[str]) -> np.ndarray:
        """
        Embed each tag individually and return the mean-pooled 384-d vector.

        Embedding tags individually (then pooling) preserves the semantics of
        each tag. Concatenating all tags into one string before embedding
        produces a noisier vector and loses individual tag boundaries.

        Returns shape (384,). Returns zeros if tags is empty.
        """
        if not tags:
            return np.zeros(_EMBED_DIM, dtype=np.float32)
        per_tag = self.embed(tags)           # (N, 384)
        return per_tag.mean(axis=0)          # (384,)

    # ------------------------------------------------------------------
    # Chunking for long posts
    # ------------------------------------------------------------------

    def _token_count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _chunk_text(
        self,
        text: str,
        window: int = _WINDOW_TOKENS,
        overlap: int = _OVERLAP_TOKENS,
    ) -> list[str]:
        """
        Split text into overlapping token-level windows.

        Tokenises without special tokens, slices into windows of `window` tokens
        with `overlap` token stride, then decodes each window back to a string.
        The sentence-transformers encoder will re-add special tokens when encoding.
        """
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        step = window - overlap
        chunks: list[str] = []
        start = 0
        while start < len(ids):
            end = min(start + window, len(ids))
            window_ids = ids[start:end]
            chunk = self._tokenizer.decode(
                window_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            chunks.append(chunk)
            if end >= len(ids):  # covered the last token — stop
                break
            start += step
        return chunks

    def _embed_with_chunking(self, text: str) -> np.ndarray:
        """
        Encode a (potentially long) text, chunking if needed.

        If the text fits within 512 tokens: encode directly.
        Otherwise: encode all overlapping windows and MEAN-POOL the embeddings.
        Returns shape (384,).
        """
        if self._token_count(text) <= _WINDOW_TOKENS:
            return self.embed([text])[0]
        chunks = self._chunk_text(text)
        chunk_embeddings = self.embed(chunks)   # (K, 384)
        return chunk_embeddings.mean(axis=0)    # (384,)

    # ------------------------------------------------------------------
    # Full post pipeline
    # ------------------------------------------------------------------

    def process_post(self, post: dict) -> dict:
        """
        Run the full text pipeline on a raw post dict.

        Returns:
            body_clean      (str)        bleach-cleaned text, stored in posts.body_clean
            body_display    (str)        html2text version for UI snippets (not stored in DB)
            lang            (str)        ISO 639-1 code, stored in posts.lang
            text_embedding  (np.ndarray) shape (384,) for Qdrant
            tag_embedding   (np.ndarray) shape (384,) for Qdrant
        """
        body_raw: str = post.get("body_raw") or ""

        body_clean = self.clean(body_raw)
        body_display = self.to_display(body_raw)

        lang = self.detect_language(body_clean) if body_clean else "und"

        text_embedding = (
            self._embed_with_chunking(body_clean)
            if body_clean
            else np.zeros(_EMBED_DIM, dtype=np.float32)
        )

        tags: list[str] = post.get("tags") or []
        tag_embedding = self.embed_tags(tags)

        return {
            "body_clean": body_clean,
            "body_display": body_display,
            "lang": lang,
            "text_embedding": text_embedding,
            "tag_embedding": tag_embedding,
        }
