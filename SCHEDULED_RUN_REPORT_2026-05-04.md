# SignalEdge Daily Scan — Scheduled Run Report

**Date:** 2026-05-04
**Task:** `signalledge-daily-scan`
**Status:** Did not run — environment cannot execute the pipeline

## Outcome

The scheduled task could not be executed in the Cowork Linux sandbox. The pipeline did not run; no new ML predictions or data points were logged.

## Why

`batch_scanner.py` imports `yfinance` (and downstream uses `scikit-learn` and `xgboost`) and pulls market data over the network. None of these are available in the sandbox where the scheduled task ran:

1. **Python packages are missing in the sandbox.** The Linux sandbox's system Python 3.10 only has `numpy` and `pandas` preinstalled. `yfinance`, `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, and `shap` are not present.
2. **PyPI is unreachable.** Attempts to `pip install` returned `403 Forbidden` from the sandbox proxy at `localhost:3128`. The proxy allowlist only includes a small set of Anthropic domains; `pypi.org` and `files.pythonhosted.org` are blocked.
3. **The project's `.venv` is not usable here.** `Stock Bot/.venv/lib/python3.9/site-packages` does contain all required libraries, but the binaries are macOS arm64 Mach-O bundles built for Python 3.9, while the sandbox is aarch64 Linux running Python 3.10. The Python symlink in `.venv/bin/python3` is a broken symlink to `/Library/Developer/CommandLineTools/usr/bin/python3` (a path that only exists on the user's Mac).
4. **Yahoo Finance is also unreachable.** Even with packages installed, `query1.finance.yahoo.com` does not resolve from the sandbox, so `yfinance` data fetches would fail.

Direct attempt:
```
$ python3 batch_scanner.py --index all --deep-n 50 --data-period 5y
ModuleNotFoundError: No module named 'yfinance'
```

## Last successful run (for reference)

The most recent `pipeline_summary.json` on disk is from 2026-04-14 (about 3 weeks ago):

- pipeline: full
- universe_size: 30
- screened: 29
- deep_scanned: 5
- data_points_added: 25
- errors: 0

That run did not meet the success criteria in the task spec either (`deep_scanned > 30`, `data_points_added > 150`), suggesting either the run used a smaller universe than the task requested or earlier runs have also been impacted.

## Recommended fixes

To make this scheduled task succeed when run via Cowork, one of the following needs to change:

- **Run on the user's host directly, not in the Cowork sandbox.** The macOS `.venv` is fully provisioned. A scheduled task wrapper that shells out to `/Users/marcreda/Documents/Claude/Projects/Stock Bot/.venv/bin/python3 batch_scanner.py …` on the host (e.g. via `launchd` or a host-side runner) would have full network and dependency access.
- **Provision the sandbox.** Allow `pypi.org`, `files.pythonhosted.org`, and Yahoo Finance hosts (`query1.finance.yahoo.com`, `query2.finance.yahoo.com`, `fc.yahoo.com`) through the proxy, then add a one-time `pip install -r requirements.txt --break-system-packages` step to the task or a Linux-native venv at `Stock Bot/.venv-linux`.
- **Pre-bake a Linux venv in the project.** Commit an `.venv-linux/` (or similar) with Linux aarch64 wheels alongside the existing macOS `.venv`, and have the task pick the right one based on `uname`.

## Pipeline summary fields requested by the task

| Metric | Value |
|---|---|
| Stocks screened | 0 (pipeline did not run) |
| ML predictions logged | 0 |
| New data points added | 0 |
| Errors | 1 (ModuleNotFoundError: yfinance) |
| Most common failure | Missing Python dependencies in sandbox; PyPI blocked by proxy |
