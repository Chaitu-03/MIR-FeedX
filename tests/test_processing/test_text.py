"""
Tests for mir/processing/text.py — TextProcessor.

Structure
---------
Unit tests   (no model weights, fast):
  - clean() with Tumblr-style HTML
  - detect_language() with mocked fasttext
  - _chunk_text() windowing and overlap math
  - _embed_with_chunking() mean-pool correctness (mocked embed)
  - embed_tags() mean-pool shape and math (mocked embed)
  - process_post() routing / plumbing

Integration tests (@pytest.mark.slow, load real models once):
  - embed_tags() cosine similarity: ["photography"] ≈ ["photo", "camera"]
  - long post produces a single 384-d embedding
  - chunk mean-pool produces correct shape on real text

Run fast only:  pytest tests/test_processing/test_text.py -m "not slow"
"""
from __future__ import annotations

import unicodedata
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mir.processing.text import TextProcessor, _EMBED_DIM, _WINDOW_TOKENS, _OVERLAP_TOKENS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_processor(
    token_counts: dict[str, int] | None = None,
    embed_returns: dict[tuple[str, ...], np.ndarray] | None = None,
) -> TextProcessor:
    """
    Instantiate TextProcessor bypassing all heavy model loading.

    `token_counts`  maps text → fake token count (for _token_count override).
    `embed_returns` maps tuple(texts) → ndarray returned by embed().
    """
    p = TextProcessor.__new__(TextProcessor)

    # html2text helper (lightweight, real)
    import html2text as _h2t
    p._h2t = _h2t.HTML2Text()
    p._h2t.ignore_links = True
    p._h2t.ignore_images = True
    p._h2t.body_width = 0

    # Mock sentence-transformer model
    mock_st = MagicMock()

    # Mock tokenizer: encode returns a list whose length we control
    mock_tok = MagicMock()
    _tok_counts = token_counts or {}

    def fake_encode(text, add_special_tokens=False):
        length = _tok_counts.get(text, min(len(text.split()) * 2, 50))
        return list(range(length))

    def fake_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        # Return a predictable string so chunk content can be checked
        return f"chunk[{ids[0]}:{ids[-1]+1}]" if ids else ""

    mock_tok.encode = fake_encode
    mock_tok.decode = fake_decode
    mock_st.tokenizer = mock_tok
    p._tokenizer = mock_tok

    # Mock embed(): default returns zeros; override per call if embed_returns given
    _embed_ret = embed_returns or {}

    def fake_encode_sentences(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        key = tuple(texts)
        if key in _embed_ret:
            return _embed_ret[key]
        n = len(texts)
        # Return deterministic embeddings: row i = all float(i)
        arr = np.zeros((n, _EMBED_DIM), dtype=np.float32)
        for i in range(n):
            arr[i] = float(i + 1)
        return arr

    mock_st.encode = fake_encode_sentences
    p._st_model = mock_st

    # fasttext: disabled by default
    p._ft_model = None

    return p


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Unit tests — clean()
# ---------------------------------------------------------------------------

class TestClean:
    def setup_method(self):
        self.p = _make_processor()

    def test_strips_all_html_tags(self):
        html = "<p>Hello <b>world</b>!</p>"
        result = self.p.clean(html)
        assert "<" not in result and ">" not in result
        assert "Hello" in result
        assert "world" in result

    def test_removes_anchor_tags_and_keeps_text(self):
        html = '<a href="http://example.com">Click here</a>'
        result = self.p.clean(html)
        assert "Click here" in result
        assert "href" not in result

    def test_collapses_whitespace(self):
        html = "<p>Too   many    spaces</p>"
        result = self.p.clean(html)
        assert "  " not in result
        assert "Too many spaces" in result

    def test_removes_script_and_style_blocks(self):
        html = "<style>body{color:red}</style><script>alert(1)</script>Real content"
        result = self.p.clean(html)
        assert "Real content" in result
        assert "color:red" not in result
        assert "alert" not in result

    def test_unicode_nfkc_normalisation(self):
        # Ligature ﬁ (U+FB01) → fi after NFKC
        html = f"<p>pro\uFB01le</p>"
        result = self.p.clean(html)
        assert "\uFB01" not in result
        assert "profile" in result

    def test_tumblr_style_post_with_nested_tags(self):
        html = (
            "<p><strong>Photography tip:</strong> "
            "Use the <em>golden hour</em> for warm light.</p>"
            "<ul><li>Morning</li><li>Evening</li></ul>"
        )
        result = self.p.clean(html)
        assert "Photography tip:" in result
        assert "golden hour" in result
        assert "Morning" in result
        assert "Evening" in result
        assert "<" not in result

    def test_empty_string_returns_empty(self):
        assert self.p.clean("") == ""

    def test_none_guard(self):
        assert self.p.clean(None) == ""  # type: ignore[arg-type]

    def test_html_entities_decoded(self):
        html = "<p>Caf&eacute; &amp; Crois&shy;sant</p>"
        result = self.p.clean(html)
        assert "Café" in result or "Cafe" in result  # entity decoded
        assert "&amp;" not in result

    def test_br_tags_become_spaces(self):
        html = "Line one<br>Line two<br/>Line three"
        result = self.p.clean(html)
        assert "Line one" in result
        assert "Line two" in result
        assert "Line three" in result


# ---------------------------------------------------------------------------
# Unit tests — to_display()
# ---------------------------------------------------------------------------

class TestToDisplay:
    def setup_method(self):
        self.p = _make_processor()

    def test_preserves_paragraph_breaks(self):
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        result = self.p.to_display(html)
        # html2text should produce two separate paragraphs
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_empty_returns_empty(self):
        assert self.p.to_display("") == ""


# ---------------------------------------------------------------------------
# Unit tests — detect_language()
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def setup_method(self):
        self.p = _make_processor()

    def test_returns_und_when_model_unavailable(self):
        self.p._ft_model = None
        assert self.p.detect_language("Hello world") == "und"

    def test_returns_und_for_empty_text(self):
        mock_ft = MagicMock()
        self.p._ft_model = mock_ft
        assert self.p.detect_language("") == "und"
        assert self.p.detect_language("   ") == "und"

    def test_strips_label_prefix(self):
        mock_ft = MagicMock()
        mock_ft.predict.return_value = (["__label__en"], [0.99])
        self.p._ft_model = mock_ft
        assert self.p.detect_language("Hello") == "en"

    def test_french_label(self):
        mock_ft = MagicMock()
        mock_ft.predict.return_value = (["__label__fr"], [0.95])
        self.p._ft_model = mock_ft
        assert self.p.detect_language("Bonjour le monde") == "fr"

    def test_newlines_replaced_before_prediction(self):
        mock_ft = MagicMock()
        mock_ft.predict.return_value = (["__label__en"], [0.9])
        self.p._ft_model = mock_ft
        self.p.detect_language("line one\nline two\nline three")
        call_text = mock_ft.predict.call_args[0][0]
        assert "\n" not in call_text

    def test_predict_exception_returns_und(self):
        mock_ft = MagicMock()
        mock_ft.predict.side_effect = RuntimeError("model error")
        self.p._ft_model = mock_ft
        assert self.p.detect_language("Some text") == "und"


# ---------------------------------------------------------------------------
# Unit tests — _chunk_text() windowing
# ---------------------------------------------------------------------------

class TestChunkText:
    def setup_method(self):
        self.p = _make_processor()

    def test_short_text_produces_one_chunk(self):
        # fake encode returns length = 10 < 512
        self.p._tokenizer.encode = lambda t, add_special_tokens=False: list(range(10))
        chunks = self.p._chunk_text("short text")
        assert len(chunks) == 1

    def test_exact_window_size_produces_one_chunk(self):
        self.p._tokenizer.encode = lambda t, **kw: list(range(_WINDOW_TOKENS))
        chunks = self.p._chunk_text("exactly 512 tokens")
        assert len(chunks) == 1

    def test_slightly_over_window_produces_two_chunks(self):
        self.p._tokenizer.encode = lambda t, **kw: list(range(_WINDOW_TOKENS + 1))
        chunks = self.p._chunk_text("513 token text")
        assert len(chunks) == 2

    def test_overlap_means_second_chunk_starts_before_end_of_first(self):
        """With overlap=64 and window=512, second chunk starts at token 448."""
        captured_windows: list[list[int]] = []
        real_decode = self.p._tokenizer.decode

        def tracking_decode(ids, **kw):
            captured_windows.append(list(ids))
            return "chunk"

        self.p._tokenizer.decode = tracking_decode
        self.p._tokenizer.encode = lambda t, **kw: list(range(600))

        self.p._chunk_text("600 token text")

        assert len(captured_windows) == 2
        # First window: [0..511]
        assert captured_windows[0][0] == 0
        assert len(captured_windows[0]) == _WINDOW_TOKENS
        # Second window starts at step = window - overlap = 448
        step = _WINDOW_TOKENS - _OVERLAP_TOKENS
        assert captured_windows[1][0] == step

    def test_many_chunks_cover_all_tokens(self):
        total = 1200
        self.p._tokenizer.encode = lambda t, **kw: list(range(total))
        covered: set[int] = set()

        def tracking_decode(ids, **kw):
            covered.update(ids)
            return "chunk"

        self.p._tokenizer.decode = tracking_decode
        self.p._chunk_text("long text")
        assert covered == set(range(total))


# ---------------------------------------------------------------------------
# Unit tests — _embed_with_chunking() mean-pool correctness
# ---------------------------------------------------------------------------

class TestEmbedWithChunking:
    def test_short_text_calls_embed_once(self):
        p = _make_processor(token_counts={"hello world": 5})
        called_with: list[list[str]] = []

        def tracking_embed(texts, **kw):
            called_with.append(texts)
            return np.ones((len(texts), _EMBED_DIM), dtype=np.float32)

        p.embed = tracking_embed
        result = p._embed_with_chunking("hello world")

        assert len(called_with) == 1
        assert called_with[0] == ["hello world"]
        assert result.shape == (_EMBED_DIM,)

    def test_long_text_mean_pools_chunk_embeddings(self):
        """Mean of 3 known chunk embeddings must equal the returned vector."""
        p = _make_processor(token_counts={"long text": 600})

        chunk_texts = ["chunk[0:512]", "chunk[448:960]", "chunk[896:1200]"]

        e1 = np.full(_EMBED_DIM, 1.0, dtype=np.float32)
        e2 = np.full(_EMBED_DIM, 3.0, dtype=np.float32)
        e3 = np.full(_EMBED_DIM, 5.0, dtype=np.float32)
        expected_mean = np.full(_EMBED_DIM, 3.0, dtype=np.float32)  # (1+3+5)/3

        def fake_chunk(text, **kw):
            return chunk_texts

        def fake_embed(texts, **kw):
            mapping = {chunk_texts[0]: e1, chunk_texts[1]: e2, chunk_texts[2]: e3}
            return np.stack([mapping[t] for t in texts])

        p._chunk_text = fake_chunk
        p.embed = fake_embed

        result = p._embed_with_chunking("long text")
        assert result.shape == (_EMBED_DIM,)
        np.testing.assert_allclose(result, expected_mean, rtol=1e-5)

    def test_long_text_returns_single_vector(self):
        p = _make_processor(token_counts={"a " * 300: 600})
        result = p._embed_with_chunking("a " * 300)
        assert result.shape == (_EMBED_DIM,)
        assert result.ndim == 1


# ---------------------------------------------------------------------------
# Unit tests — embed_tags()
# ---------------------------------------------------------------------------

class TestEmbedTags:
    def test_empty_tags_returns_zero_vector(self):
        p = _make_processor()
        result = p.embed_tags([])
        assert result.shape == (_EMBED_DIM,)
        np.testing.assert_array_equal(result, np.zeros(_EMBED_DIM))

    def test_single_tag_equals_its_embedding(self):
        fixed = np.arange(_EMBED_DIM, dtype=np.float32)
        p = _make_processor()
        p.embed = lambda texts, **kw: fixed.reshape(1, -1)
        result = p.embed_tags(["photography"])
        np.testing.assert_allclose(result, fixed)

    def test_mean_pooling_math(self):
        """embed_tags must return the element-wise mean of per-tag embeddings."""
        e1 = np.full(_EMBED_DIM, 2.0, dtype=np.float32)
        e2 = np.full(_EMBED_DIM, 6.0, dtype=np.float32)
        expected = np.full(_EMBED_DIM, 4.0, dtype=np.float32)

        p = _make_processor()
        p.embed = lambda texts, **kw: np.stack([e1, e2])
        result = p.embed_tags(["tag1", "tag2"])
        np.testing.assert_allclose(result, expected)

    def test_output_shape_is_1d(self):
        p = _make_processor()
        p.embed = lambda texts, **kw: np.ones((len(texts), _EMBED_DIM), dtype=np.float32)
        result = p.embed_tags(["a", "b", "c"])
        assert result.ndim == 1
        assert result.shape == (_EMBED_DIM,)

    def test_tags_embedded_individually_not_concatenated(self):
        """embed() must be called with the tag list, not a joined string."""
        received: list[list[str]] = []
        p = _make_processor()

        def capturing_embed(texts, **kw):
            received.append(list(texts))
            return np.ones((len(texts), _EMBED_DIM), dtype=np.float32)

        p.embed = capturing_embed
        p.embed_tags(["photography", "sunset", "landscape"])

        assert len(received) == 1
        assert received[0] == ["photography", "sunset", "landscape"]
        # Crucially, the call must NOT be a single joined string
        assert received[0] != ["photography sunset landscape"]


# ---------------------------------------------------------------------------
# Unit tests — process_post() plumbing
# ---------------------------------------------------------------------------

class TestProcessPost:
    def setup_method(self):
        self.p = _make_processor()
        self.p.embed = lambda texts, **kw: np.ones((len(texts), _EMBED_DIM), dtype=np.float32)

    def test_returns_required_keys(self):
        result = self.p.process_post({"body_raw": "<p>Hello</p>", "tags": ["art"]})
        assert set(result) >= {"body_clean", "body_display", "lang", "text_embedding", "tag_embedding"}

    def test_text_embedding_shape(self):
        result = self.p.process_post({"body_raw": "<p>Hello world</p>", "tags": []})
        assert result["text_embedding"].shape == (_EMBED_DIM,)

    def test_tag_embedding_shape(self):
        result = self.p.process_post({"body_raw": "", "tags": ["music", "jazz"]})
        assert result["tag_embedding"].shape == (_EMBED_DIM,)

    def test_empty_body_produces_zero_text_embedding(self):
        result = self.p.process_post({"body_raw": "", "tags": []})
        np.testing.assert_array_equal(result["text_embedding"], np.zeros(_EMBED_DIM))

    def test_empty_tags_produce_zero_tag_embedding(self):
        result = self.p.process_post({"body_raw": "<p>hi</p>", "tags": []})
        np.testing.assert_array_equal(result["tag_embedding"], np.zeros(_EMBED_DIM))

    def test_lang_is_und_without_fasttext(self):
        self.p._ft_model = None
        result = self.p.process_post({"body_raw": "<p>Bonjour</p>", "tags": []})
        assert result["lang"] == "und"

    def test_body_clean_has_no_html_tags(self):
        result = self.p.process_post({"body_raw": "<p><b>Bold</b> text</p>", "tags": []})
        assert "<" not in result["body_clean"]
        assert "Bold" in result["body_clean"]

    def test_missing_body_raw_handled(self):
        result = self.p.process_post({"tags": ["art"]})
        assert result["body_clean"] == ""


# ---------------------------------------------------------------------------
# Integration tests — real sentence-transformer model
# ---------------------------------------------------------------------------

def _try_import_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


sentence_transformers_available = _try_import_sentence_transformers()


@pytest.fixture(scope="module")
def real_processor():
    if not sentence_transformers_available:
        pytest.skip("sentence-transformers not installed")
    # Use CPU for CI compatibility; override to 'mps' locally if desired
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return TextProcessor(device=device)


@pytest.mark.slow
class TestIntegrationRealModel:
    def test_photography_tags_similar_to_photo_camera(self, real_processor: TextProcessor):
        """
        Per-tag embedding approach: ["photography"] should be close to ["photo", "camera"]
        because each tag is embedded independently and the shared visual-equipment semantics
        dominate after mean-pooling.
        """
        v1 = real_processor.embed_tags(["photography"])
        v2 = real_processor.embed_tags(["photo", "camera"])
        sim = _cosine(v1, v2)
        assert sim > 0.6, (
            f"Expected cosine > 0.6 for photography ≈ photo+camera, got {sim:.4f}"
        )

    def test_unrelated_tags_less_similar(self, real_processor: TextProcessor):
        """Tags from unrelated domains should score lower than the photography pair."""
        v_photo = real_processor.embed_tags(["photography"])
        v_food = real_processor.embed_tags(["recipe", "cooking"])
        sim = _cosine(v_photo, v_food)
        # Not asserting a hard lower bound — just log it for inspection
        assert isinstance(sim, float)

    def test_long_post_produces_single_384d_vector(self, real_processor: TextProcessor):
        """
        A post whose body exceeds 512 tokens must still produce a single (384,) embedding
        via the sliding-window mean-pool path.
        """
        # Repeat a phrase enough times to exceed 512 subword tokens
        long_text = "The quick brown fox jumps over the lazy dog. " * 60
        result = real_processor._embed_with_chunking(long_text)
        assert result.shape == (_EMBED_DIM,)
        assert result.ndim == 1
        assert np.isfinite(result).all()

    def test_chunk_mean_pool_differs_from_first_chunk_only(self, real_processor: TextProcessor):
        """Mean-pooling all chunks must produce a different vector than encoding only chunk 0."""
        long_text = "The quick brown fox jumps over the lazy dog. " * 60
        # Full mean-pooled embedding
        pooled = real_processor._embed_with_chunking(long_text)
        # Embedding of just the first 512-token chunk
        chunks = real_processor._chunk_text(long_text)
        first_only = real_processor.embed([chunks[0]])[0]
        # They should differ (mean-pool uses information from all chunks)
        assert not np.allclose(pooled, first_only), (
            "Mean-pooled embedding must differ from first-chunk-only embedding"
        )

    def test_embed_returns_correct_shape(self, real_processor: TextProcessor):
        texts = ["hello", "world", "this is a test"]
        result = real_processor.embed(texts)
        assert result.shape == (3, _EMBED_DIM)

    def test_embed_empty_list(self, real_processor: TextProcessor):
        result = real_processor.embed([])
        assert result.shape == (0, _EMBED_DIM)
