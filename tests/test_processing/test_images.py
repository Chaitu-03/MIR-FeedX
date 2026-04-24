"""
Unit tests for mir/processing/images.py.

Unit tests (fast, no model weights):
  - ProjectionLayer: output shape, L2-normalised, batch input, load fallback
  - Cache: hit/miss with mocked Redis, no-Redis passthrough, key format

Integration tests (@pytest.mark.slow):
  - Real CLIP model: embed_image returns 512-d unit vector
  - Project round-trip produces valid 384-d unit vector
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mir.processing.images import ImageProcessor, ProjectionLayer, _CLIP_DIM, _TEXT_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_clip_emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_CLIP_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _rand_clip_batch(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    batch = rng.standard_normal((n, _CLIP_DIM)).astype(np.float32)
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    return batch / norms


def _solid_image_file(tmp_path: Path, color=(128, 64, 200), size=64) -> Path:
    from PIL import Image
    p = tmp_path / "test.jpg"
    Image.new("RGB", (size, size), color).save(str(p), format="JPEG")
    return p


def _mock_processor(redis_client=None) -> ImageProcessor:
    """ImageProcessor with mocked CLIP — no model weights loaded."""
    mock_model = MagicMock()
    mock_preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))

    def fake_encode_image(tensor):
        b = tensor.shape[0]
        # Return deterministic unit vectors
        out = torch.zeros(b, _CLIP_DIM)
        out[:, 0] = 1.0
        return out

    mock_model.encode_image.side_effect = fake_encode_image

    p = ImageProcessor.__new__(ImageProcessor)
    p._device = "cpu"
    p._model = mock_model
    p._preprocess = mock_preprocess
    p._redis = redis_client
    p._projection = ProjectionLayer(device="cpu")   # real projection, random orthogonal
    return p


# ---------------------------------------------------------------------------
# ProjectionLayer — unit tests
# ---------------------------------------------------------------------------

class TestProjectionLayer:
    def test_output_shape_1d_input(self):
        layer = ProjectionLayer(device="cpu")
        emb = _rand_clip_emb()
        out = layer.project(emb)
        assert out.shape == (_TEXT_DIM,), f"Expected ({_TEXT_DIM},), got {out.shape}"

    def test_output_shape_2d_batch(self):
        layer = ProjectionLayer(device="cpu")
        batch = _rand_clip_batch(5)
        # project each individually (public interface is 1-d per call)
        outs = np.stack([layer.project(v) for v in batch])
        assert outs.shape == (5, _TEXT_DIM)

    def test_output_is_l2_normalised_1d(self):
        layer = ProjectionLayer(device="cpu")
        out = layer.project(_rand_clip_emb())
        norm = float(np.linalg.norm(out))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm:.6f}"

    def test_output_is_l2_normalised_2d(self):
        layer = ProjectionLayer(device="cpu")
        batch = _rand_clip_batch(8)
        for v in batch:
            out = layer.project(v)
            norm = float(np.linalg.norm(out))
            assert abs(norm - 1.0) < 1e-5

    def test_output_dtype_float32(self):
        layer = ProjectionLayer(device="cpu")
        out = layer.project(_rand_clip_emb())
        assert out.dtype == np.float32

    def test_fallback_warning_when_no_weights(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="mir.processing.images"):
            layer = ProjectionLayer(
                weights_path=Path("/nonexistent/weights.pt"),
                device="cpu",
            )
        assert any("untrained" in r.message.lower() for r in caplog.records)
        # Still functional — returns correct shape
        out = layer.project(_rand_clip_emb())
        assert out.shape == (_TEXT_DIM,)

    def test_loads_weights_from_file(self, tmp_path: Path):
        # Save a fresh layer's weights, reload, verify same projection
        original = ProjectionLayer(device="cpu")
        weights_path = tmp_path / "proj.pt"
        torch.save(original._linear.state_dict(), str(weights_path))

        loaded = ProjectionLayer(weights_path=weights_path, device="cpu")
        emb = _rand_clip_emb()
        assert np.allclose(original.project(emb), loaded.project(emb), atol=1e-5)

    def test_different_inputs_produce_different_outputs(self):
        layer = ProjectionLayer(device="cpu")
        a = layer.project(_rand_clip_emb(seed=1))
        b = layer.project(_rand_clip_emb(seed=2))
        assert not np.allclose(a, b)

    def test_orthogonal_fallback_is_not_truncation(self):
        """
        Orthogonal init must not simply copy the first 384 dims of CLIP.
        Check: the first projection layer weight is NOT an identity-like matrix.
        ProjectionLayer now uses Sequential; first linear layer is _linear[0].
        """
        layer = ProjectionLayer(device="cpu")
        W = layer._linear[0].weight.detach().numpy()   # (hidden_dim, 512)
        # If it were truncation: W would equal eye(hidden, 512)
        identity_like = np.eye(*W.shape)
        assert not np.allclose(W, identity_like, atol=0.01)


# ---------------------------------------------------------------------------
# Redis cache — unit tests
# ---------------------------------------------------------------------------

class TestRedisCache:
    def test_cache_key_format(self):
        assert ImageProcessor._cache_key("abc123") == "mir:img_emb:abc123"

    def test_cache_miss_calls_clip_and_stores(self, tmp_path: Path):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None   # cache miss

        p = _mock_processor(redis_client=mock_redis)
        img_path = _solid_image_file(tmp_path)

        result = p.embed_image(str(img_path))

        # CLIP was called
        p._model.encode_image.assert_called_once()
        # Result stored in Redis
        mock_redis.set.assert_called_once()
        stored_key = mock_redis.set.call_args[0][0]
        assert stored_key.startswith("mir:img_emb:")
        # Stored value is valid JSON list of floats
        stored_json = mock_redis.set.call_args[0][1]
        stored = json.loads(stored_json)
        assert len(stored) == _CLIP_DIM

    def test_cache_hit_skips_clip(self, tmp_path: Path):
        cached_emb = _rand_clip_emb(seed=42)
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(cached_emb.tolist()).encode()

        p = _mock_processor(redis_client=mock_redis)
        img_path = _solid_image_file(tmp_path)

        result = p.embed_image(str(img_path))

        # CLIP never called
        p._model.encode_image.assert_not_called()
        # Returned cached value
        np.testing.assert_allclose(result, cached_emb, atol=1e-5)

    def test_no_redis_still_works(self, tmp_path: Path):
        """ImageProcessor works without Redis (cache disabled)."""
        p = _mock_processor(redis_client=None)
        img_path = _solid_image_file(tmp_path)
        result = p.embed_image(str(img_path))
        assert result.shape == (_CLIP_DIM,)

    def test_batch_cache_partial_hit(self, tmp_path: Path):
        """Some images cached, others not — only misses go through CLIP."""
        cached_emb = _rand_clip_emb(seed=7)

        # img_0 → cache hit; img_1 → cache miss
        img0 = _solid_image_file(tmp_path / "img0.jpg" if False else tmp_path, color=(10, 20, 30))
        img1_dir = tmp_path / "sub"
        img1_dir.mkdir()
        from PIL import Image
        Image.new("RGB", (64, 64), (200, 100, 50)).save(str(img1_dir / "img1.jpg"))
        img1 = str(img1_dir / "img1.jpg")

        # The two images have different phashes → different cache keys
        # We'll make get() return cached_emb for the first key and None for the second
        call_count = [0]

        def fake_get(key):
            call_count[0] += 1
            # First call = first image's phash key → hit
            if call_count[0] == 1:
                return json.dumps(cached_emb.tolist()).encode()
            return None   # second image → miss

        mock_redis = MagicMock()
        mock_redis.get.side_effect = fake_get

        p = _mock_processor(redis_client=mock_redis)
        results = p.embed_images_batch([str(img0), img1])

        assert results.shape == (2, _CLIP_DIM)
        # CLIP called exactly once (only for the miss)
        assert p._model.encode_image.call_count == 1

    def test_no_ttl_on_cache_set(self, tmp_path: Path):
        """Embeddings are stable — must be stored with no TTL (no expiry)."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        p = _mock_processor(redis_client=mock_redis)
        p.embed_image(str(_solid_image_file(tmp_path)))

        set_call_kwargs = mock_redis.set.call_args[1] if mock_redis.set.call_args else {}
        set_call_args = mock_redis.set.call_args[0] if mock_redis.set.call_args else ()
        # Neither 'ex' nor 'px' nor 'exat' nor 'pxat' should be in the call
        assert "ex" not in set_call_kwargs
        assert "px" not in set_call_kwargs

    def test_embed_image_shape(self, tmp_path: Path):
        p = _mock_processor()
        result = p.embed_image(str(_solid_image_file(tmp_path)))
        assert result.shape == (_CLIP_DIM,)
        assert result.dtype == np.float32

    def test_embed_images_batch_empty(self):
        p = _mock_processor()
        result = p.embed_images_batch([])
        assert result.shape == (0, _CLIP_DIM)

    def test_embed_images_batch_shape(self, tmp_path: Path):
        paths = []
        from PIL import Image
        for i in range(4):
            fp = tmp_path / f"img{i}.jpg"
            Image.new("RGB", (64, 64), (i * 30, i * 20, i * 10)).save(str(fp))
            paths.append(str(fp))

        p = _mock_processor()
        results = p.embed_images_batch(paths)
        assert results.shape == (4, _CLIP_DIM)


# ---------------------------------------------------------------------------
# Integration tests — real CLIP model
# ---------------------------------------------------------------------------

def _try_open_clip() -> bool:
    try:
        import open_clip  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def real_processor(tmp_path_factory):
    if not _try_open_clip():
        pytest.skip("open_clip not installed")
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return ImageProcessor(device=device)


@pytest.mark.slow
class TestIntegrationRealCLIP:
    def test_embed_image_shape(self, real_processor, tmp_path):
        p = _solid_image_file(tmp_path)
        out = real_processor.embed_image(str(p))
        assert out.shape == (_CLIP_DIM,)

    def test_embed_image_unit_norm(self, real_processor, tmp_path):
        out = real_processor.embed_image(str(_solid_image_file(tmp_path)))
        norm = float(np.linalg.norm(out))
        assert abs(norm - 1.0) < 1e-4

    def test_embed_batch_consistent_with_single(self, real_processor, tmp_path):
        from PIL import Image
        paths = []
        for i in range(3):
            fp = tmp_path / f"b{i}.jpg"
            Image.new("RGB", (128, 128), (i * 40, i * 30, i * 20)).save(str(fp))
            paths.append(str(fp))
        batch = real_processor.embed_images_batch(paths)
        singles = np.stack([real_processor.embed_image(p) for p in paths])
        np.testing.assert_allclose(batch, singles, atol=1e-4)

    def test_project_output_shape(self, real_processor, tmp_path):
        clip_emb = real_processor.embed_image(str(_solid_image_file(tmp_path)))
        proj = real_processor.project(clip_emb)
        assert proj.shape == (_TEXT_DIM,)

    def test_project_output_unit_norm(self, real_processor, tmp_path):
        clip_emb = real_processor.embed_image(str(_solid_image_file(tmp_path)))
        proj = real_processor.project(clip_emb)
        norm = float(np.linalg.norm(proj))
        assert abs(norm - 1.0) < 1e-4
