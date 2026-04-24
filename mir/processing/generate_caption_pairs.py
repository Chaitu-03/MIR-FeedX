"""
Generate projection training pairs: CLIP-512 ↔ MiniLM-384.

Two modes
---------
1. **Image mode** (default when data/raw_images/ contains images):
   - Scan raw_images for JPEGs.
   - Encode each with CLIP image encoder → 512-d.
   - Generate a caption via a lightweight captioner (BLIP-2 or git-base-coco).
   - Encode caption with MiniLM → 384-d.
   - Saves N pairs where N = number of images found.

2. **Text bootstrap mode** (--bootstrap, or auto-fallback when no images):
   - Use a large curated set of diverse English sentences.
   - Encode each sentence with CLIP text encoder → 512-d.
   - Encode same sentence with MiniLM → 384-d.
   - Valid shortcut: CLIP text ↔ MiniLM text teaches the projection
     the right geometry without needing image-caption pairs.
   - Target ≥ 5 000 pairs; script generates ~6 000 by default.

Usage
-----
    # Bootstrap (no images needed):
    python -m mir.processing.generate_caption_pairs --bootstrap

    # Image mode (run after crawling):
    python -m mir.processing.generate_caption_pairs --image-dir data/raw_images --n 5000

Output: data/projection_pairs.npz  (keys: clip_embeddings, caption_embeddings)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

_OUT_PATH = Path("data/projection_pairs.npz")
_CLIP_DIM = 512
_TEXT_DIM = 384

# ---------------------------------------------------------------------------
# Bootstrap corpus — diverse sentences spanning Tumblr content genres
# ---------------------------------------------------------------------------

_BOOTSTRAP_SENTENCES: list[str] = [
    # Art & illustration
    "digital painting of a woman with flowing hair in soft pastel colors",
    "ink drawing of a dragon perched on a medieval castle",
    "watercolor landscape with mountains reflected in a still lake",
    "abstract geometric art with overlapping triangles in neon colors",
    "oil painting of sunflowers in a vase on a wooden table",
    "pencil sketch of a cat sleeping on a windowsill",
    "pixel art character with a sword standing in a fantasy dungeon",
    "vector illustration of a city skyline at sunset",
    "surrealist painting of melting clocks over a desert landscape",
    "charcoal portrait of an elderly man with deep wrinkles",
    "comic book panel with a superhero flying over a burning city",
    "minimalist poster design with bold typography",
    "concept art for a science fiction space station interior",
    "stained glass window pattern with floral motifs",
    "linocut print of a fox in a snowy forest",
    # Photography
    "black and white photograph of a street in New York City at night",
    "close-up photograph of dew drops on a spider web",
    "portrait photograph of a young woman laughing in natural light",
    "aerial photograph of a coastal city at sunrise",
    "macro photograph of a butterfly wing showing intricate patterns",
    "long exposure photograph of car light trails on a highway",
    "documentary photograph of a market in Southeast Asia",
    "fashion photograph of a model in a minimalist white studio",
    "astrophotography of the Milky Way over a dark forest",
    "underwater photograph of colorful coral reef fish",
    # Nature & animals
    "a wolf howling at the full moon in a snowy pine forest",
    "a hummingbird hovering next to a bright red flower",
    "a tortoiseshell cat curled up asleep in a sunny spot",
    "a group of elephants walking across the African savanna",
    "cherry blossoms falling in a Japanese garden with a stone lantern",
    "autumn leaves in shades of red orange and gold covering a forest path",
    "a golden retriever puppy playing in fallen leaves",
    "a peacock displaying its iridescent tail feathers",
    "a whale breaching the surface of the ocean at sunset",
    "a red fox kit exploring a meadow full of wildflowers",
    # Fashion & aesthetics
    "vintage 1970s fashion with bell-bottom jeans and a floral shirt",
    "dark academia aesthetic with old books candles and antique maps",
    "cottagecore style dress in a sunlit wildflower field",
    "cyberpunk fashion with neon lights and futuristic accessories",
    "cozy winter outfit with an oversized sweater and plaid scarf",
    "elegant ballgown in deep burgundy velvet at a formal event",
    "streetwear style with oversized hoodie and high-top sneakers",
    "bohemian jewelry with turquoise stones and silver rings",
    "pastel kawaii fashion with ruffles and cute accessories",
    "gothic aesthetic with black lace velvet and silver jewelry",
    # Food & lifestyle
    "a stack of fluffy pancakes drizzled with maple syrup and berries",
    "a latte art coffee in a ceramic mug on a rustic wooden table",
    "a colorful acai bowl topped with granola banana and coconut flakes",
    "homemade sourdough bread fresh from the oven on a cutting board",
    "a charcuterie board with cheese grapes nuts and cured meats",
    "a bowl of ramen with soft-boiled egg nori and bamboo shoots",
    "french macarons in pastel colors arranged on a marble surface",
    "a cozy reading nook with fairy lights blankets and a hot drink",
    "a terracotta pot with a thriving monstera plant by a window",
    "flat lay of a minimalist desk setup with plants and stationery",
    # Architecture & interiors
    "a modern minimalist living room with floor-to-ceiling windows",
    "the interior of a gothic cathedral with towering stone arches",
    "a traditional Japanese ryokan room with tatami mats and paper screens",
    "a narrow colorful street in a Mediterranean village",
    "a treehouse nestled in a dense green forest",
    "a mid-century modern home with an open floor plan",
    "the grand staircase of a baroque European palace",
    "a tiny cabin in the mountains surrounded by snow",
    "a bohemian bedroom with macrame hanging plants and fairy lights",
    "the Eiffel Tower illuminated against a purple twilight sky",
    # Space & science
    "the surface of Mars with red dust and rocky terrain",
    "a nebula with swirling clouds of purple and blue gas",
    "the International Space Station orbiting above Earth",
    "a total solar eclipse with the corona visible around the moon",
    "bioluminescent plankton glowing blue in ocean waves at night",
    "a cross-section diagram of a human brain",
    "snowflake crystals photographed under a microscope",
    "the aurora borealis dancing in green and purple over a frozen lake",
    "lightning striking a city skyline during a thunderstorm",
    "a time-lapse of clouds moving over a mountain range",
    # Music & culture
    "a guitarist playing on stage with dramatic backlighting",
    "vinyl records stacked next to a vintage turntable",
    "a street musician playing saxophone in a rainy city",
    "a ballet dancer mid-leap on an empty stage",
    "an orchestra performing in a grand concert hall",
    "graffiti mural covering the side of a building in a city",
    "a film camera and strips of developed film on a light table",
    "a library with floor-to-ceiling bookshelves and rolling ladders",
    "a typewriter on a desk with crumpled papers around it",
    "a collection of vintage postcards and photographs spread out",
    # Fantasy & fiction
    "a dragon made of ice soaring over a frozen tundra",
    "a enchanted forest with glowing mushrooms and fireflies at night",
    "a wizard casting a spell with swirling magical energy",
    "a mermaid sitting on a rock watching a ship pass by",
    "a steampunk city with clockwork towers and airships in the sky",
    "an ancient library filled with floating magical books",
    "a knight in silver armor standing before a ruined castle",
    "a fairy sitting on a mushroom in a miniature forest",
    "a portal to another dimension glowing with energy in a field",
    "a ship sailing through clouds in a sky kingdom",
    # Emotions & abstract
    "solitude and peace by a still lake at dawn with morning mist",
    "the warmth of sunlight streaming through curtains on a lazy afternoon",
    "the feeling of nostalgia looking at old family photographs",
    "quiet joy found in a cup of tea and a good book on a rainy day",
    "the excitement of a city at night with neon signs and crowds",
    "melancholy blue hour in an empty city street after rain",
    "hope represented by a single flower growing through cracked concrete",
    "the energy and chaos of a busy street market",
    "the serenity of a zen garden with raked sand and smooth stones",
    "wonder and awe at the vastness of a starry night sky",
    # Additional variety
    "a polaroid photograph fading at the edges with handwritten caption",
    "handwritten calligraphy letters in black ink on cream paper",
    "a collection of crystals and gemstones arranged on velvet",
    "an antique compass and old map of unknown territories",
    "pressed flowers arranged in a botanical illustration style",
    "a ceramic bowl with an imperfect wabi-sabi glaze",
    "a glass greenhouse filled with rare tropical plants",
    "a patchwork quilt in earthy tones spread on a bed",
    "candles of different heights burning on a mantelpiece",
    "a hammock strung between two palm trees over turquoise water",
    "a typeface poster with expressive lettering in bold primary colors",
    "a diorama in miniature showing a city street scene",
    "a mosaic pattern made from broken tiles in vibrant colors",
    "an old wooden boat reflected perfectly in still water",
    "a collection of vintage cameras on a wooden shelf",
    "shadow puppets casting shapes on a white wall",
    "frosted glass with abstract shapes visible through it",
    "a spiral staircase viewed from above",
    "a field of lavender under a clear blue sky in Provence",
    "a close-up of tree bark with lichen and moss growing on it",
]

# Expand corpus by adding prompt variations
_PREFIXES = [
    "", "a beautiful ", "an artistic rendering of ", "a stunning ",
    "a detailed illustration of ", "a photograph of ", "a painting of ",
    "a sketch of ", "a minimalist representation of ", "a vibrant ",
]


def _expand_corpus(base: list[str], target: int) -> list[str]:
    """Expand base sentences with prefix variations to reach target size.

    Generates all (prefix, sentence) combos deterministically, then cycles
    with index-offset shuffling to avoid infinite loop when combos < target.
    """
    seen: set[str] = set(base)
    sentences: list[str] = list(base)

    # First pass: all unique prefix × sentence combos
    for prefix in _PREFIXES:
        if not prefix:
            continue  # bare sentence already in base
        for sentence in base:
            candidate = (prefix + sentence).strip()
            if candidate not in seen:
                seen.add(candidate)
                sentences.append(candidate)
                if len(sentences) >= target:
                    return sentences[:target]

    # Still short — repeat with minor word-order variants to pad
    extras = [
        f"{s}, highly detailed" for s in base
    ] + [
        f"{s}, award winning" for s in base
    ] + [
        f"{s}, professional photography" for s in base
    ] + [
        f"{s}, 4k resolution" for s in base
    ] + [
        f"{s}, cinematic lighting" for s in base
    ]
    for candidate in extras:
        if candidate not in seen:
            seen.add(candidate)
            sentences.append(candidate)
        if len(sentences) >= target:
            break

    if len(sentences) < target:
        log.warning(
            "Corpus capped at %d (target %d). Add more base sentences for better diversity.",
            len(sentences), target,
        )

    return sentences[:target]


# ---------------------------------------------------------------------------
# Bootstrap mode: CLIP text → MiniLM text pairs
# ---------------------------------------------------------------------------

def generate_bootstrap_pairs(
    n_pairs: int = 6000,
    device: str | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode sentences with both CLIP text encoder and MiniLM.

    Returns (clip_embeddings: (N,512), caption_embeddings: (N,384)).
    """
    import open_clip
    from sentence_transformers import SentenceTransformer

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    log.info("Generating %d bootstrap pairs on %s", n_pairs, device)

    sentences = _expand_corpus(_BOOTSTRAP_SENTENCES, n_pairs)
    log.info("Corpus size: %d unique sentences", len(sentences))

    # Load models
    log.info("Loading CLIP ViT-B/32…")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    log.info("Loading MiniLM…")
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # CLIP text encoding
    clip_embs_list: list[np.ndarray] = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            feats = clip_model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)
        clip_embs_list.append(feats.cpu().numpy().astype(np.float32))
        if (i // batch_size) % 10 == 0:
            log.info("CLIP text encoded %d/%d", min(i + batch_size, len(sentences)), len(sentences))

    clip_embs = np.concatenate(clip_embs_list, axis=0)  # (N, 512)

    # MiniLM encoding
    log.info("Encoding with MiniLM…")
    text_embs = st_model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)  # (N, 384)

    log.info("Generated %d pairs — CLIP: %s  MiniLM: %s", len(sentences), clip_embs.shape, text_embs.shape)
    return clip_embs, text_embs


# ---------------------------------------------------------------------------
# Image mode: CLIP image + lightweight captioner → MiniLM
# ---------------------------------------------------------------------------

def generate_image_pairs(
    image_dir: Path,
    n: int = 5000,
    device: str | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode real images with CLIP and generate captions with BLIP (git-base-coco).

    Falls back to filename-derived descriptions if captioner unavailable.
    """
    import open_clip
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    paths = list(image_dir.rglob("*.jpg"))[:n]
    if not paths:
        raise FileNotFoundError(f"No JPEG images found under {image_dir}")

    log.info("Found %d images in %s", len(paths), image_dir)

    # Load CLIP
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()

    # Try to load captioner
    captioner = None
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        log.info("Loading BLIP captioner…")
        captioner_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        captioner = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        captioner.eval()
        log.info("BLIP loaded")
    except Exception as exc:
        log.warning("BLIP not available (%s) — using filename-derived captions", exc)

    clip_embs_list: list[np.ndarray] = []
    captions: list[str] = []

    for i, path in enumerate(paths):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        # CLIP image embedding
        tensor = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(tensor)
            feat = F.normalize(feat, dim=-1)
        clip_embs_list.append(feat.cpu().numpy().astype(np.float32))

        # Caption
        if captioner is not None:
            inputs = captioner_proc(img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = captioner.generate(**inputs, max_new_tokens=30)
            caption = captioner_proc.decode(out[0], skip_special_tokens=True)
        else:
            # Fallback: use parent directory name as rough topic label
            caption = path.parent.name.replace("_", " ").replace("-", " ")
        captions.append(caption)

        if i % 100 == 0:
            log.info("Processed %d/%d images", i, len(paths))

    clip_embs = np.concatenate(clip_embs_list, axis=0)

    log.info("Encoding %d captions with MiniLM…", len(captions))
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    text_embs = st_model.encode(
        captions,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    log.info("Generated %d image pairs", len(captions))
    return clip_embs, text_embs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description="Generate CLIP↔MiniLM projection training pairs")
    parser.add_argument("--bootstrap", action="store_true", help="Text-only bootstrap (no images needed)")
    parser.add_argument("--image-dir", default="data/raw_images", help="Root dir of crawled images")
    parser.add_argument("--n", type=int, default=6000, help="Number of pairs to generate")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(_OUT_PATH))
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    has_images = image_dir.exists() and any(image_dir.rglob("*.jpg"))

    if args.bootstrap or not has_images:
        if not args.bootstrap:
            log.info("No images found in %s — falling back to text bootstrap mode", image_dir)
        clip_embs, text_embs = generate_bootstrap_pairs(n_pairs=args.n, device=args.device)
    else:
        clip_embs, text_embs = generate_image_pairs(
            image_dir=image_dir, n=args.n, device=args.device
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), clip_embeddings=clip_embs, caption_embeddings=text_embs)
    log.info("Saved %d pairs → %s", len(clip_embs), out)


if __name__ == "__main__":
    main()
