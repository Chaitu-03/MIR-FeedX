"""
Seed blog list for the MIR crawler.

Real, known-active Tumblr blogs organised by genre.
Pass to Crawler.crawl(seed_blogs=SEED_BLOGS) to bootstrap CrawlState.
Discovery from reblog chains will expand the corpus automatically after
the first run.
"""
from __future__ import annotations

SEED_BLOGS_BY_GENRE: dict[str, list[str]] = {
    "art_design": [
        "staff",
        "theartidote",
        "asylum-art",
        "itscolossal",
        "crossconnectmag",
    ],
    "photography": [
        "photojojo",
        "natgeofound",
        "lensblr-network",
        "thephotographerssociety",
        "magnumphotos",
    ],
    "fashion_style": [
        "fashionistable",
        "wmagazine",
        "manrepeller",
        "thesartorialist",
        "fuckyeahfashioncouples",
    ],
    "music": [
        "pitchfork",
        "npr-music",
        "stereogum",
        "songaday",
        "8tracks",
    ],
    "writing_literature": [
        "writingpromptsworld",
        "poetryofresistance",
        "thewritingcafe",
        "wordsnquotes",
        "yeoldenews",
    ],
    "gaming": [
        "gamespot",
        "kotaku",
        "ilovevideogamessomuch",
        "pixelatedcrown",
        "theomeganerd",
    ],
    "anime_manga": [
        "animatedtext",
        "studioghibligifs",
        "shonenjump",
        "fuckyeahanime",
        "mangacap",
    ],
    "science_technology": [
        "nasa",
        "scishow",
        "spaceplasma",
        "neurosciencestuff",
        "thescienceofreality",
    ],
    "nature_environment": [
        "earth-song",
        "theworldofphotography",
        "animalworld",
        "wildlifegifsblog",
        "oceanatdawn",
    ],
    "travel": [
        "travelingcolors",
        "travelthisworld",
        "wanderlusting",
        "departured",
        "theworldlookslikethis",
    ],
    "food_cooking": [
        "foodffs",
        "tastykitchen",
        "veganrecipeclub",
        "fullcravings",
        "yummytastyfood",
    ],
    "film_tv": [
        "cinemastatic",
        "classicfilmheroines",
        "filmforlife",
        "criterion",
        "screenmusings",
    ],
    "humor_memes": [
        "tastefullyoffensive",
        "funnyordie",
        "textsfromlastnight",
        "4gifs",
        "lolsomeone",
    ],
    "history": [
        "historicaltimes",
        "thehistoryoftheworld",
        "retronaut",
        "medievalpoc",
        "oldnewyork",
    ],
    "beauty_wellness": [
        "thelipsticklesbians",
        "reallifeskincare",
        "yogaholics",
        "fitnessgifs",
        "mindful-recovery",
    ],
    "pets_animals": [
        "dailybunny",
        "cuteoverload",
        "dogsofinstaworld",
        "catsof",
        "zfrankenfluffy",
    ],
}

# Flat list for passing directly to Crawler.crawl()
SEED_BLOGS: list[str] = [
    blog for blogs in SEED_BLOGS_BY_GENRE.values() for blog in blogs
]
