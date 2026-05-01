# ─── Prediqt API Dockerfile ──────────────────────────────────────────────────
# Ships the FastAPI wrapper (api/main.py) + every Python module it imports.
# Deployed via Render (free tier per render.yaml) or Fly.io. The Streamlit
# UI files (app.py, *_page.py) are excluded via .dockerignore.

FROM python:3.11-slim

# System deps for wheels that need compilation (pandas, numpy, scipy, lightgbm).
# build-essential brings gcc/g++ for any sdist that doesn't ship a wheel for
# linux/aarch64 or linux/amd64. curl is for the HEALTHCHECK below.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so layer caching kicks in — pip install is the
# slowest step and rarely changes between deploys.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo. .dockerignore controls what's excluded
# (Streamlit UI files, dev caches, secrets, the .git directory, tests,
# scripts, docs, the local .predictions/ SQLite store). What gets copied:
#
#   • api/                  — FastAPI route handlers
#   • All top-level .py     — model.py, data_fetcher.py, consensus_check.py,
#                              trade_decision.py, options_*.py, db.py, etc.
#   • .models/              — pre-trained joblib model files (one per ticker)
#   • supabase/migrations/  — SQL migrations (deployed to Supabase separately)
#
# Using `COPY . .` instead of an explicit allowlist because the import graph
# fans out to ~30+ modules and the previous "list every file" approach kept
# missing dependencies → uvicorn crashed on first import.
COPY . .

# Default port — Render and Fly.io both override with $PORT at runtime,
# but having a default makes local `docker run` work without env vars.
ENV PORT=8000
EXPOSE 8000

# Health check: Render reads this to know when the container is ready to
# serve traffic. /api/health is the lightweight liveness endpoint that
# doesn't touch Supabase or run any inference.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/api/health || exit 1

# Single-worker uvicorn is fine for this read-heavy API. The in-process
# caches (analytics, model loaders) would just duplicate across workers.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
