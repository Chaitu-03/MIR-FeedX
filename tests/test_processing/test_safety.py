"""
Tests for mir/processing/safety.py — NSFWClassifier.

Structure:
  - Unit tests  (fast, no model weights): mock CLIP to test scoring math and branching.
  - Integration tests (slow, real model): marked @pytest.mark.slow; load CLIP once per
    session via a module-scoped fixture and test against synthetic SFW images.

Run unit tests only:  pytest tests/test_processing/test_safety.py -m "not slow"
Run all (incl. model): pytest tests/test_processing/test_safety.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from mir.processing.safety import (
    NSFWClassifier,
    NSFW_PROMPTS,
    SFW_PROMPTS,
    _N_NSFW,
    _ALL_PROMPTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_image(color: tuple[int, int, int] = (0, 0, 0), size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), color)


def _mock_classifier(
    nsfw_logits: list[float] | None = None,
    sfw_logits: list[float] | None = None,
    blocklist: list[str] | None = None,
) -> NSFWClassifier:
    """
    Build an NSFWClassifier with a fully mocked CLIP model.

    `nsfw_logits` and `sfw_logits` control the raw logits returned by
    encode_image @ encode_text^T before softmax.
    Defaults: very low NSFW logits, high SFW logits → clearly SFW.
    """
    nsfw_logits = nsfw_logits or [0.1, 0.1, 0.1, 0.1]
    sfw_logits = sfw_logits or [5.0, 5.0, 5.0, 5.0]
    all_logits = nsfw_logits + sfw_logits  # length 8

    n_prompts = len(_ALL_PROMPTS)
    dim = 512

    # Precomputed text features — unit vectors along dim=0..n_prompts
    text_feats = torch.zeros(n_prompts, dim)
    for i in range(n_prompts):
        text_feats[i, i] = 1.0

    # image encode returns a vector whose dot product with text_feats[i] equals all_logits[i]
    img_vec = torch.zeros(1, dim)
    for i, v in enumerate(all_logits):
        img_vec[0, i] = v

    mock_model = MagicMock()
    mock_model.encode_image.return_value = img_vec
    mock_model.encode_text.return_value = text_feats

    # preprocess: identity — return a random 3×64×64 tensor so .unsqueeze(0) works
    mock_preprocess = MagicMock(return_value=torch.zeros(3, 64, 64))

    clf = NSFWClassifier.__new__(NSFWClassifier)
    clf._device = "cpu"
    clf._model = mock_model
    clf._preprocess = mock_preprocess
    clf._tokenizer = MagicMock(return_value=torch.zeros(n_prompts, 77, dtype=torch.long))

    # Compute real text features from mock (skips model encode_text path)
    clf._text_features = torch.nn.functional.normalize(text_feats, dim=-1)
    clf.text_blocklist = blocklist if blocklist is not None else []
    return clf


# ---------------------------------------------------------------------------
# Unit tests — scoring math
# ---------------------------------------------------------------------------

class TestScoringLogic:
    def test_high_sfw_logits_produce_sfw_result(self):
        clf = _mock_classifier(nsfw_logits=[0.0, 0.0, 0.0, 0.0], sfw_logits=[10.0] * 4)
        img = _solid_image()
        is_nsfw, score = clf.is_image_nsfw(img, threshold=0.75)
        assert not is_nsfw
        assert score < 0.75

    def test_high_nsfw_logits_produce_nsfw_result(self):
        # The mock image vector [10,10,10,10,0,...] is L2-normalised inside
        # is_image_nsfw before the dot product, yielding effective logits
        # [0.5,0.5,0.5,0.5,0,...] and nsfw_score ≈ 0.622.
        # Threshold must be below that geometric ceiling (< 0.622).
        clf = _mock_classifier(nsfw_logits=[10.0] * 4, sfw_logits=[0.0] * 4)
        img = _solid_image()
        is_nsfw, score = clf.is_image_nsfw(img, threshold=0.5)
        assert is_nsfw, f"Expected NSFW classification, got score={score:.4f}"
        assert score > 0.5

    def test_score_is_sum_of_nsfw_softmax_probs(self):
        """nsfw_score must equal probs[0:_N_NSFW].sum() after the joint softmax.

        The mock image tensor passes through F.normalize before the dot product,
        so the expected value must be computed on the *normalised* vector.
        """
        import torch.nn.functional as _F

        nsfw_l = [2.0, 1.5, 1.0, 0.5]
        sfw_l = [3.0, 3.0, 3.0, 3.0]
        clf = _mock_classifier(nsfw_logits=nsfw_l, sfw_logits=sfw_l)

        img = _solid_image()
        _, score = clf.is_image_nsfw(img)

        # Reproduce the full pipeline of is_image_nsfw with the mock's geometry:
        # encode_image → raw tensor → F.normalize → dot with unit text vecs → softmax
        raw = torch.zeros(1, 512)
        for i, v in enumerate(nsfw_l + sfw_l):
            raw[0, i] = v
        img_norm = _F.normalize(raw, dim=-1)   # identical to what is_image_nsfw does
        text_f = torch.zeros(8, 512)
        for i in range(8):
            text_f[i, i] = 1.0                 # unit vectors (already normalised)
        logits = img_norm @ text_f.T            # (1, 8)
        probs = logits.softmax(dim=-1)
        expected = probs[0, :_N_NSFW].sum().item()

        assert abs(score - expected) < 1e-5

    def test_custom_threshold_respected(self):
        clf = _mock_classifier(nsfw_logits=[2.0] * 4, sfw_logits=[2.0] * 4)
        img = _solid_image()

        # Equal logits → each prompt gets 1/8 probability → nsfw_score ≈ 0.5
        _, score = clf.is_image_nsfw(img)
        assert abs(score - 0.5) < 1e-4

        is_nsfw_low, _ = clf.is_image_nsfw(img, threshold=0.49)
        assert is_nsfw_low  # 0.5 > 0.49

        is_nsfw_high, _ = clf.is_image_nsfw(img, threshold=0.51)
        assert not is_nsfw_high  # 0.5 < 0.51

    def test_any_image_nsfw_flags_post(self):
        """classify_post must return nsfw=True if ANY image in the post exceeds threshold."""
        clf = _mock_classifier()
        sfw_img = _solid_image((135, 206, 235))

        # Override is_image_nsfw to return nsfw=True for the second image only
        call_count = 0

        def fake_is_image_nsfw(img, threshold=None):
            nonlocal call_count
            call_count += 1
            return (call_count == 2), 0.8 if call_count == 2 else 0.1

        clf.is_image_nsfw = fake_is_image_nsfw

        result = clf.classify_post({}, images=[sfw_img, sfw_img])
        assert result["nsfw"] is True
        assert result["nsfw_score"] == 0.8

    def test_all_sfw_images_produce_sfw_post(self):
        clf = _mock_classifier(nsfw_logits=[0.0] * 4, sfw_logits=[10.0] * 4)
        images = [_solid_image((r, 100, 200)) for r in [50, 100, 150]]
        result = clf.classify_post({}, images=images)
        assert result["nsfw"] is False
        assert result["nsfw_score"] is not None
        assert result["nsfw_score"] < 0.75


# ---------------------------------------------------------------------------
# Unit tests — text blocklist
# ---------------------------------------------------------------------------

class TestTextBlocklist:
    def test_obvious_safe_text_passes(self):
        clf = _mock_classifier(blocklist=["pornography", "explicit content", "gore"])
        assert not clf.is_text_nsfw("I love hiking in the mountains on weekends")
        assert not clf.is_text_nsfw("My cat knocked over my coffee this morning")
        assert not clf.is_text_nsfw("Beautiful sunset over the Pacific Ocean")

    def test_exact_blocked_term_flagged(self):
        clf = _mock_classifier(blocklist=["pornography", "explicit content", "gore"])
        assert clf.is_text_nsfw("This post contains explicit content for adults")

    def test_blocklist_is_case_insensitive(self):
        clf = _mock_classifier(blocklist=["gore"])
        assert clf.is_text_nsfw("Graphic GORE WARNING")
        assert clf.is_text_nsfw("gore")
        assert clf.is_text_nsfw("GORE")

    def test_partial_substring_match(self):
        clf = _mock_classifier(blocklist=["nsfw"])
        assert clf.is_text_nsfw("#nsfw #art #photography")

    def test_empty_blocklist_never_flags(self):
        clf = _mock_classifier(blocklist=[])
        assert not clf.is_text_nsfw("literally anything goes here including explicit content")

    def test_empty_text_never_flags(self):
        clf = _mock_classifier(blocklist=["gore", "pornography"])
        assert not clf.is_text_nsfw("")

    def test_text_only_post_uses_blocklist(self):
        clf = _mock_classifier(blocklist=["pornography"])
        result = clf.classify_post({"body_clean": "this is pornography"}, images=None)
        assert result["nsfw"] is True
        assert result["nsfw_score"] is None  # text path → no score

    def test_safe_text_post_not_flagged(self):
        clf = _mock_classifier(blocklist=["pornography"])
        result = clf.classify_post({"body_clean": "A photo of my dog at the park"}, images=None)
        assert result["nsfw"] is False
        assert result["nsfw_score"] is None

    def test_body_raw_fallback_when_clean_missing(self):
        clf = _mock_classifier(blocklist=["explicit content"])
        result = clf.classify_post(
            {"body_raw": "<p>explicit content warning</p>"}, images=None
        )
        assert result["nsfw"] is True


# ---------------------------------------------------------------------------
# Unit tests — blocklist file loading
# ---------------------------------------------------------------------------

class TestBlocklistLoading:
    def test_loads_from_file(self, tmp_path: Path):
        bl = tmp_path / "bl.txt"
        bl.write_text("gore\npornography\n# comment\nexplicit content\n")

        clf = _mock_classifier()
        clf.text_blocklist = NSFWClassifier._load_blocklist(bl)

        assert "gore" in clf.text_blocklist
        assert "pornography" in clf.text_blocklist
        assert "explicit content" in clf.text_blocklist
        assert not any(t.startswith("#") for t in clf.text_blocklist)

    def test_missing_file_returns_empty_list(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.txt"
        result = NSFWClassifier._load_blocklist(missing)
        assert result == []

    def test_blank_lines_skipped(self, tmp_path: Path):
        bl = tmp_path / "bl.txt"
        bl.write_text("\n\ngore\n\nexplicit content\n\n")
        result = NSFWClassifier._load_blocklist(bl)
        assert result == ["gore", "explicit content"]

    def test_terms_are_lowercased(self, tmp_path: Path):
        bl = tmp_path / "bl.txt"
        bl.write_text("GORE\nExplicit Content\n")
        result = NSFWClassifier._load_blocklist(bl)
        assert all(t == t.lower() for t in result)


# ---------------------------------------------------------------------------
# Unit tests — classify_post routing
# ---------------------------------------------------------------------------

class TestClassifyPostRouting:
    def test_images_take_priority_over_text(self):
        """When images are provided, the text blocklist should NOT be consulted."""
        clf = _mock_classifier(
            nsfw_logits=[0.0] * 4,
            sfw_logits=[10.0] * 4,
            blocklist=["safe photo"],  # would flag the text if consulted
        )
        result = clf.classify_post(
            {"body_clean": "a safe photo of a landscape"},
            images=[_solid_image()],
        )
        # Image path should be taken; text not consulted; result is SFW
        assert result["nsfw"] is False
        assert result["nsfw_score"] is not None

    def test_empty_post_is_sfw(self):
        clf = _mock_classifier(blocklist=["pornography"])
        result = clf.classify_post({}, images=None)
        assert result["nsfw"] is False
        assert result["nsfw_score"] is None


# ---------------------------------------------------------------------------
# Integration tests — real CLIP model, synthetic SFW images
# ---------------------------------------------------------------------------

def _try_import_open_clip():
    try:
        import open_clip  # noqa: F401
        return True
    except ImportError:
        return False


open_clip_available = _try_import_open_clip()


@pytest.fixture(scope="module")
def real_classifier():
    """Load the real CLIP model once per test module (expensive)."""
    if not open_clip_available:
        pytest.skip("open_clip not installed")
    return NSFWClassifier()


@pytest.mark.slow
class TestIntegrationRealModel:
    """
    Integration tests against the real CLIP ViT-B/32 model.
    All test images are synthetic (generated via PIL) — no network required.
    """

    def test_solid_black_image_is_sfw(self, real_classifier: NSFWClassifier):
        """A completely black image has no explicit content and must not be flagged."""
        black = Image.new("RGB", (224, 224), (0, 0, 0))
        is_nsfw, score = real_classifier.is_image_nsfw(black)
        assert not is_nsfw, f"Black image falsely flagged as NSFW (score={score:.4f})"
        assert score < real_classifier._text_features.new_tensor(0.75).item() + 0.01

    def test_sky_blue_image_is_sfw(self, real_classifier: NSFWClassifier):
        """A flat sky-blue image should score well below the NSFW threshold."""
        sky = Image.new("RGB", (224, 224), (135, 206, 235))
        is_nsfw, score = real_classifier.is_image_nsfw(sky)
        assert not is_nsfw, f"Sky-blue image falsely flagged (score={score:.4f})"

    def test_green_landscape_image_is_sfw(self, real_classifier: NSFWClassifier):
        """A flat green (grass/forest) image should not be flagged."""
        green = Image.new("RGB", (224, 224), (34, 139, 34))
        is_nsfw, score = real_classifier.is_image_nsfw(green)
        assert not is_nsfw, f"Green image falsely flagged (score={score:.4f})"

    def test_grey_image_is_sfw(self, real_classifier: NSFWClassifier):
        """Mid-grey image — no content whatsoever."""
        grey = Image.new("RGB", (224, 224), (128, 128, 128))
        is_nsfw, score = real_classifier.is_image_nsfw(grey)
        assert not is_nsfw, f"Grey image falsely flagged (score={score:.4f})"

    def test_score_is_float_in_unit_interval(self, real_classifier: NSFWClassifier):
        """nsfw_score must always be a float in [0, 1]."""
        img = Image.new("RGB", (224, 224), (200, 200, 200))
        _, score = real_classifier.is_image_nsfw(img)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_model_runs_on_expected_device(self, real_classifier: NSFWClassifier):
        """Model should be on MPS (or CPU on non-Apple hardware)."""
        device = real_classifier._device
        assert device in {"mps", "cuda", "cpu"}

    def test_text_features_shape(self, real_classifier: NSFWClassifier):
        """Precomputed text features must have one row per prompt."""
        feats = real_classifier._text_features
        assert feats.shape[0] == len(NSFW_PROMPTS) + len(SFW_PROMPTS)
        assert feats.ndim == 2
