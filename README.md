# MIR-FeedX

A multimodal information retrieval engine for Tumblr. Crawls blogs and tag feeds, processes posts through ML pipelines, and serves hybrid keyword + semantic search via a REST API.

> Built for the AFIR course, PES University Semester 6 — Chaitanya Makkar & G Harish

---

## What It Does

1. **Crawls** Tumblr blogs and tag feeds via the Tumblr API v2
2. **Processes** posts through text embedding (MiniLM), image embedding (CLIP + projection layer), NSFW detection (CLIP zero-shot), and language detection (fasttext)
3. **Indexes** fused 384-d vectors in Qdrant for approximate nearest-neighbour search
4. **Serves** a REST API with four search modes: hybrid general search, account name, tag, and community

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn |
| Task queue | Celery + Beat |
| Primary DB | PostgreSQL 16 + pgvector |
| Vector search | Qdrant (HNSW, 384-d cosine) |
| Cache + broker | Redis |
| Text embeddings | `all-MiniLM-L6-v2` (384-d) |
| Image embeddings | OpenCLIP `ViT-B/32` (512-d → projected to 384-d) |
| NSFW filter | CLIP zero-shot classification |
| Language detection | fasttext LID-176 |
| Community clustering | HDBSCAN (accounts), KMeans (tags) |

---

## Prerequisites

- Python 3.11+
- Docker Desktop
- Tumblr API consumer key — register at [tumblr.com/oauth/apps](https://www.tumblr.com/oauth/apps)

---

## Setup

```bash
git clone <repo> && cd MIR-FeedX
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set:

```env
TUMBLR_API_CREDENTIALS=[{"consumer_key": "YOUR_KEY", "secret": "YOUR_SECRET"}]
API_KEYS_ENABLED=false
```

Optionally download the fasttext language model (falls back to `"und"` if absent):

```bash
curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -o data/lid.176.bin
```

Apply database migrations once infrastructure is up:

```bash
.venv/bin/alembic upgrade head
```

---

## Running Locally

Four processes must run simultaneously — open a separate terminal for each.

**Terminal A — Infrastructure (Docker)**
```bash
docker compose up
```
Wait until all three services (`postgres`, `qdrant`, `redis`) show `healthy` before proceeding.

**Terminal B — API Server**
```bash
source .venv/bin/activate
.venv/bin/uvicorn mir.api.app:app --reload --port 8000
```
Swagger docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

**Terminal C — Celery Worker + Beat Scheduler**
```bash
source .venv/bin/activate
.venv/bin/celery -A mir.workers.celery_app worker \
  --beat --queues default,gpu --loglevel INFO
```
This handles crawling, ML processing, community rebuilding, and scheduled tasks.

**Terminal D — Monitor (optional)**
```bash
source .venv/bin/activate
python3 scripts/monitor.py --watch
```

---

## Verify It's Working

```bash
# Check all dependencies are healthy
curl http://localhost:8000/api/v1/health

# Run a search (API_KEYS_ENABLED=false means no key needed)
curl -X POST http://localhost:8000/api/v1/search/general \
  -H "Content-Type: application/json" \
  -d '{"query": "space photography", "limit_posts": 5}'
```

---

## Search Algorithm

General search runs two retrievals in parallel and fuses them with Reciprocal Rank Fusion (RRF):

- **Keyword**: PostgreSQL `websearch_to_tsquery` + GIN index — supports `"quoted phrases"`, `OR`, `-negation`
- **Semantic**: Qdrant HNSW cosine search on fused post vectors

The per-post embedding is a weighted average of three modalities:
```
post_vec = 0.5 × text_vec + 0.35 × image_vec + 0.15 × tag_vec
```

RRF merges the two ranked lists using only rank positions (scale-invariant, no score normalisation needed):
```
rrf_score(post) = Σ 1 / (60 + rank_i)   for each list i containing this post
```

A post appearing in both lists scores higher than one appearing in just one — RRF rewards consensus. Final scores are then multiplied by two re-ranking boosts:
```
engagement_boost = 1 + 0.1 × log₂(1 + note_count)
recency_boost    = 0.5 ^ (age_days / 90)           ← halves every 90 days
final_score      = rrf_score × engagement_boost × recency_boost
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/v1/search/general` | Hybrid keyword + semantic search |
| `GET` | `/api/v1/search/accounts?q=` | Fuzzy blog name search (pg_trgm) |
| `GET` | `/api/v1/search/tags?q=` | Exact + prefix + semantic tag search |
| `GET` | `/api/v1/search/communities?q=` | HDBSCAN topic cluster search |
| `GET` | `/api/v1/health` | Dependency health + latency |
| `GET` | `/api/v1/stats` | Post counts, queue depth, crawl status |
| `POST` | `/api/v1/admin/reindex` | Re-embed all posts *(admin key)* |
| `POST` | `/api/v1/admin/cache/clear` | Flush Redis cache *(admin key)* |
| `POST` | `/api/v1/admin/rebuild_communities` | Re-run clustering *(admin key)* |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TUMBLR_API_CREDENTIALS` | — | JSON array of `{consumer_key, secret}`. Multiple keys = round-robin with rate-limit rotation. |
| `DATABASE_URL` | `postgresql+asyncpg://mir:mir@localhost:5432/mir` | PostgreSQL DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis DSN |
| `QDRANT_HOST` | `localhost` | Qdrant hostname |
| `API_KEYS_ENABLED` | `true` | Set `false` for open dev mode |
| `NSFW_THRESHOLD` | `0.75` | CLIP NSFW score cutoff (0–1) |
| `TARGET_POST_COUNT` | `50000` | Crawl stops at this many SFW posts |
| `CRAWL_INTERVAL_MINUTES` | `15` | Beat crawl frequency (use `360` in production) |
| `LOG_JSON` | `false` | Set `true` for structured JSON logs |
| `POST_WEIGHT_TEXT` | `0.5` | Text weight in embedding fusion |
| `POST_WEIGHT_IMAGE` | `0.35` | Image weight in embedding fusion |
| `POST_WEIGHT_TAG` | `0.15` | Tag weight in embedding fusion |

---

## Authentication & API Keys

Authentication is controlled by `API_KEYS_ENABLED` in `.env`. When enabled, every request must include an `X-API-Key: <key>` header. Keys are stored as bcrypt hashes (work factor 12) — the plaintext is never written to the database.

There are two roles: regular keys can use search endpoints, admin keys can additionally call `/admin/*` routes.

**Creating a key** — insert directly into PostgreSQL:

```sql
-- Connect: psql postgresql://mir:mir@localhost:5432/mir
INSERT INTO api_keys (key_hash, label, is_admin)
VALUES (crypt('your-chosen-key', gen_salt('bf', 12)), 'my-app', false);
```

For an admin key set `is_admin = true`. Pass the plaintext in requests:

```bash
curl -X POST http://localhost:8000/api/v1/search/general \
  -H "X-API-Key: your-chosen-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "photography"}'
```

---

## Running Tests

```bash
.venv/bin/pytest tests/ -v
# Skip slow tests that load real ML models
.venv/bin/pytest tests/ -v -m "not slow"
```

---

## Project Structure

```
mir/
├── api/          — FastAPI app, endpoints, auth, schemas
├── db/           — SQLAlchemy models, migrations
├── ingestion/    — Tumblr crawler, image downloader, API client
├── processing/   — TextProcessor, ImageProcessor, NSFWClassifier, communities
├── search/       — General, account, tag, community search + Redis cache
├── workers/      — Celery app, task definitions, Beat schedule
└── config.py     — All settings via pydantic-settings

scripts/monitor.py  — Terminal dashboard
tests/              — E2E and unit tests
docker-compose.yml  — PostgreSQL, Qdrant, Redis
```
