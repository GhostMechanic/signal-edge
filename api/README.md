# Prediqt API

FastAPI wrapper over the Prediqt Python ML stack.

## Local dev

```bash
# From the Stock Bot folder:
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

Then open:

- **http://localhost:8000/docs** — interactive API explorer (Swagger UI)
- **http://localhost:8000/api/health** — liveness check

## Endpoints

### Health
- `GET /api/health` — `{status, service, version, timestamp}`

### Analytics
- `GET /api/analytics/summary` — Scoreboard: 3 honesty reads + sample sizes
- `GET /api/analytics/per-horizon` — 5 horizon cards (3D/1W/1M/1Q/1Y) with Target/Checkpoint/Expiration per horizon
- `GET /api/analytics/simulated-portfolio` — Hold + Take-Profit equity curves + stats

### Predictions
- `GET /api/predictions` — Paginated prediction log
  - Query params: `filter=all|scored|pending`, `days=N`, `sort=newest|oldest`, `limit=50`, `offset=0`

## Cache

Enriched analytics are cached in-memory for 5 minutes. A single Track Record
page load hits `/summary`, `/per-horizon`, `/simulated-portfolio`, and
`/predictions` — but the expensive target-hit OHLC scan only runs once.

## CORS

- `http://localhost:3000` (Next.js dev)
- `https://prediqt-web.vercel.app` (production)
- `https://prediqt-web-*.vercel.app` (preview deploys)

## Structure

- `api/main.py` — FastAPI app, CORS, cache, all endpoints
- `api/__init__.py` — makes `api` a Python package

No ML code lives here. Endpoints import from existing modules
(`prediction_logger_v2`, `target_hit_analyzer`, etc.) and wrap them.
