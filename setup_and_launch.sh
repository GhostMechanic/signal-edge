#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Stock Prediction Bot — Setup & Launch Script (Mac/Linux)
# Double-click this file in Finder, or run: bash setup_and_launch.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║        📈 Stock Prediction Bot            ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 is not installed."
    echo "   Please install it from https://www.python.org/downloads/ and re-run."
    exit 1
fi

PYTHON=$(command -v python3)
echo "✅ Python found: $($PYTHON --version)"

# ── Create virtual environment (first run only) ───────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "📦 First-time setup — creating virtual environment…"
    $PYTHON -m venv "$VENV_DIR"
    echo "✅ Virtual environment created."
fi

# ── Activate virtual environment ─────────────────────────────────────────────
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated."

# ── Install / upgrade packages ────────────────────────────────────────────────
echo ""
echo "📦 Installing / checking dependencies…"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✅ All packages installed."

# ── Optional: check for libomp (needed by XGBoost/LightGBM on Mac) ───────────
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! brew list libomp &>/dev/null 2>&1; then
        echo ""
        echo "ℹ️  Tip: if you see libomp errors, install it with:"
        echo "   brew install libomp"
    fi
fi

# ── Launch the app ───────────────────────────────────────────────────────────
echo ""
echo "🚀 Launching Stock Prediction Bot…"
echo "   → Opening in your browser at http://localhost:8501"
echo "   → Press Ctrl+C to stop the app."
echo ""

streamlit run app.py \
    --server.headless false \
    --browser.gatherUsageStats false \
    --theme.base dark \
    --theme.primaryColor "#00c853" \
    --theme.backgroundColor "#0e1117" \
    --theme.secondaryBackgroundColor "#1c2333" \
    --theme.textColor "#e8ecf5"
