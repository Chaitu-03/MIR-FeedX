"""
Seed blog list for the MIR crawler.

Pass to Crawler.crawl(seed_blogs=SEED_BLOGS) to bootstrap CrawlState.
Discovery from reblog chains will expand the corpus automatically after
the first run.
"""
from __future__ import annotations

SEED_BLOGS_BY_GENRE: dict[str, list[str]] = {
    "art_design": [
        "thepaintsociety",
        "designobserver",
        "artdirection",
        "illustration-daily",
        "graphic-design-inspiration",
    ],
    "photography": [
        "photography-daily",
        "landscape-photography",
        "portrait-photography",
        "street-photography-hub",
        "nature-photography-now",
    ],
    "fashion_style": [
        "fashionably-yours",
        "streetstyle-blog",
        "vintage-fashion-diary",
        "sustainable-fashion",
        "high-fashion-news",
    ],
    "music": [
        "indie-music-blog",
        "electronic-beats",
        "vinyl-obsession",
        "music-production-tips",
        "live-concert-photography",
    ],
    "writing_literature": [
        "writing-prompts",
        "poetry-daily",
        "book-reviews-blog",
        "creative-writing-hub",
        "short-story-archive",
    ],
    "gaming": [
        "gaming-news-daily",
        "indie-games-spotlight",
        "esports-coverage",
        "game-design-theory",
        "retro-gaming-blog",
    ],
    "anime_manga": [
        "anime-screenshots",
        "manga-adaptations",
        "anime-news-network",
        "cosplay-central",
        "anime-music-videos",
    ],
    "science_technology": [
        "science-daily",
        "tech-innovations",
        "space-exploration",
        "artificial-intelligence-news",
        "quantum-physics-explained",
    ],
    "nature_environment": [
        "wildlife-photography-hub",
        "ocean-conservation",
        "botanical-gardens",
        "climate-action-now",
        "national-parks-blog",
    ],
    "travel": [
        "travel-inspiration-blog",
        "urban-exploration",
        "backpacking-adventures",
        "architecture-travel",
        "hidden-gems-worldwide",
    ],
    "food_cooking": [
        "food-photography",
        "recipe-inspiration",
        "vegan-cooking-blog",
        "dessert-goals",
        "street-food-culture",
    ],
    "film_tv": [
        "film-noir-classics",
        "tv-series-reviews",
        "cinematography-blog",
        "movie-poster-art",
        "independent-cinema",
    ],
    "humor_memes": [
        "funny-posts-daily",
        "meme-culture",
        "dark-humor-blog",
        "comedy-sketches",
        "internet-humor",
    ],
    "history": [
        "historical-photographs",
        "ancient-history-blog",
        "military-history",
        "vintage-ephemera",
        "historical-artifacts",
    ],
    "beauty_wellness": [
        "skincare-routine",
        "makeup-tutorials",
        "yoga-meditation",
        "fitness-motivation",
        "mental-health-matters",
    ],
    "pets_animals": [
        "cute-animals-daily",
        "dog-photography",
        "cat-lover-blog",
        "wildlife-facts",
        "exotic-pets",
    ],
}

# Flat list for passing directly to Crawler.crawl()
SEED_BLOGS: list[str] = [
    blog for blogs in SEED_BLOGS_BY_GENRE.values() for blog in blogs
]
