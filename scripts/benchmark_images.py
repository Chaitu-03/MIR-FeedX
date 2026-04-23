#!/usr/bin/env python3
"""
Benchmark CLIP image embedding throughput on MPS.

Target: ~8 ms/image on Apple M2 Pro.

Usage
-----
    # Default: 100 synthetic images, batch_size=32
    python scripts/benchmark_images.py

    # Custom
    python scripts/benchmark_images.py --n 200 --batch-size 16 --device mps

    # Use real images from the crawl corpus
    python scripts/benchmark_images.py --image-dir data/raw_images --n 100
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_synthetic_images(n: int, tmp_dir: Path, size: int = 224) -> list[str]:
    """Write n random RGB images to tmp_dir; return their paths."""
    rng = np.random.default_rng(42)
    paths: list[str] = []
    for i in range(n):
        arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        p = tmp_dir / f"img_{i:04d}.jpg"
        Image.fromarray(arr).save(str(p), format="JPEG", quality=90)
        paths.append(str(p))
    return paths


def _collect_real_images(image_dir: Path, n: int) -> list[str]:
    paths = list(image_dir.rglob("*.jpg"))[:n]
    if len(paths) < n:
        print(f"[warn] Only {len(paths)} images found in {image_dir} (requested {n})")
    return [str(p) for p in paths]


def run_benchmark(
    image_paths: list[str],
    batch_size: int,
    device: str,
) -> dict:
    from mir.processing.images import ImageProcessor

    processor = ImageProcessor(device=device)
    n = len(image_paths)

    # ------------------------------------------------------------------
    # Warm-up (first batch excluded from timing — MPS JIT compilation)
    # ------------------------------------------------------------------
    warmup = image_paths[:batch_size]
    _ = processor.embed_images_batch(warmup, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Timed run — single batch call covering all images
    # ------------------------------------------------------------------
    # Per-image latency via individual embed_image calls
    per_image_times: list[float] = []
    for p in image_paths:
        t0 = time.perf_counter()
        processor.embed_image(p)
        per_image_times.append((time.perf_counter() - t0) * 1000)   # ms

    # Batch throughput
    t0 = time.perf_counter()
    embeddings = processor.embed_images_batch(image_paths, batch_size=batch_size)
    batch_elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "n": n,
        "batch_size": batch_size,
        "device": device,
        "embedding_shape": embeddings.shape,
        # Per-image (sequential) stats
        "per_image_mean_ms": statistics.mean(per_image_times),
        "per_image_p50_ms": statistics.median(per_image_times),
        "per_image_p95_ms": sorted(per_image_times)[int(0.95 * n)],
        # Batch throughput stats
        "batch_total_ms": batch_elapsed_ms,
        "batch_ms_per_image": batch_elapsed_ms / n,
        "batch_img_per_sec": n / (batch_elapsed_ms / 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CLIP image embedding")
    parser.add_argument("--n", type=int, default=100, help="Number of images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="mps | cuda | cpu (auto-detect)")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Use real images from this directory instead of synthetic",
    )
    args = parser.parse_args()

    import torch
    device = args.device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    with tempfile.TemporaryDirectory() as tmp:
        if args.image_dir:
            image_paths = _collect_real_images(Path(args.image_dir), args.n)
        else:
            print(f"Generating {args.n} synthetic images…")
            image_paths = _make_synthetic_images(args.n, Path(tmp))

        if not image_paths:
            print("No images to benchmark. Exiting.")
            sys.exit(1)

        print(f"Benchmarking {len(image_paths)} images on {device} (batch_size={args.batch_size})…")
        results = run_benchmark(image_paths, args.batch_size, device)

    target_ms = 8.0
    batch_ms = results["batch_ms_per_image"]
    status = "✓" if batch_ms <= target_ms else "✗"

    print("\n── Results ────────────────────────────────────────────")
    print(f"  Device          : {results['device']}")
    print(f"  Images          : {results['n']}")
    print(f"  Batch size      : {results['batch_size']}")
    print(f"  Embedding shape : {results['embedding_shape']}")
    print()
    print("  Per-image (sequential, includes Redis cache hits after 1st run):")
    print(f"    mean   : {results['per_image_mean_ms']:.2f} ms")
    print(f"    p50    : {results['per_image_p50_ms']:.2f} ms")
    print(f"    p95    : {results['per_image_p95_ms']:.2f} ms")
    print()
    print("  Batch throughput (embed_images_batch):")
    print(f"    total  : {results['batch_total_ms']:.1f} ms")
    print(f"    per img: {batch_ms:.2f} ms  {status}  (target ≤ {target_ms} ms)")
    print(f"    img/s  : {results['batch_img_per_sec']:.1f}")
    print("────────────────────────────────────────────────────────")

    sys.exit(0 if batch_ms <= target_ms else 1)


if __name__ == "__main__":
    main()
