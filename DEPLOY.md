# Deploying the Prediqt API

The FastAPI wrapper in `api/main.py` can ship as a Docker container to
Fly.io, Render, Railway, or any other PaaS that runs Dockerfiles. This doc
covers the two most Prediqt-friendly options.

## 0. One-time prep

- Make sure `requirements.txt` is up to date (it is — `fastapi` + `uvicorn`
  are pinned as of the deployment prep commit).
- Copy the predictions log + target-hit cache into `data/` if they aren't
  already there. The Dockerfile ships a snapshot of `data/` so the live
  API has something to serve on first boot.
- Pick the FQDN you'll use (e.g. `prediqt-api.fly.dev`) and update
  `prediqt-web/.env.local` + Vercel's `NEXT_PUBLIC_API_URL` to match.

## 1. Fly.io (recommended — simpler, cheap)

```bash
# One-time
brew install flyctl                            # or: curl -L fly.io/install.sh | sh
fly auth signup                                # or fly auth login

# From Stock Bot/
fly launch --no-deploy                         # accept app name / region
fly volumes create prediqt_data --size 1       # persistent disk for data/
fly deploy                                     # build + ship
fly open                                       # see /docs in a browser
```

Set the custom-domain CORS allow-list if you ship to a non-vercel frontend:

```bash
fly secrets set CORS_EXTRA_ORIGINS="https://prediqt.com,https://app.prediqt.com"
```

Logs + metrics: `fly logs`, `fly status`.

## 2. Render (if you prefer a web dashboard)

1. Push the repo to GitHub.
2. On render.com, click "New → Blueprint" and point it at the repo. It'll
   pick up `render.yaml` and spin up the service.
3. In the Render dashboard, add any extra CORS origins under
   **Environment → `CORS_EXTRA_ORIGINS`**.
4. Watch the build; it auto-deploys on every push.

The `starter` plan ($7/mo) includes a persistent disk, which you want —
otherwise the predictions log resets on every deploy.

## 3. Wire up the frontend

Once the API is live at e.g. `https://prediqt-api.fly.dev`:

```bash
# In Vercel → Project Settings → Environment Variables:
NEXT_PUBLIC_API_URL=https://prediqt-api.fly.dev
```

Trigger a redeploy on Vercel (any push or the "Redeploy" button). The
`lib/api.ts` fetch helper picks up the env var at build time and points
every request at the new host.

## 4. Health check

```bash
curl https://prediqt-api.fly.dev/api/health
# → {"status":"ok","service":"prediqt-api","version":"0.2.0","timestamp":"..."}
```

If you get CORS errors in the browser, the Vercel preview URL is outside
`prediqt-web-{hash}.vercel.app`. Add the exact origin to `CORS_EXTRA_ORIGINS`
and redeploy the API.

## 5. Updating the prediction log

The simulated portfolio + calibration endpoints rely on live data. Two
strategies:

- **Push via redeploy** — commit updated `data/predictions_log.json`, push,
  new image ships. Simple, works until the log gets huge.
- **Volume-mounted updates** — SSH into the Fly VM (`fly ssh console`) and
  run whatever your prediction pipeline does. The `/app/data` volume
  persists across deploys.

The in-memory analytics cache TTL is 5 min (see `_ANALYTICS_TTL_SECONDS`
in `api/main.py`), so new data appears within that window.
