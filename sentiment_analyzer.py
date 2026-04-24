"""
sentiment_analyzer.py - Stock-specific news sentiment analysis module.

Provides Yahoo Finance news sentiment scoring, CNN Fear & Greed Index,
and optional NewsAPI integration. Designed to produce features for
stock prediction models.
"""

import os
import functools
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

# VADER sentiment — try to import, auto-install if possible, else fallback
_HAS_VADER = False
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
except ImportError:
    try:
        import subprocess
        import sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "vaderSentiment", "--break-system-packages"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _HAS_VADER = True
    except Exception:
        # Fallback: simple keyword-based sentiment (no external dependency)
        class SentimentIntensityAnalyzer:
            """Lightweight fallback when vaderSentiment is not available."""
            _POS = {"surge", "soar", "jump", "rally", "gain", "beat", "profit",
                    "bullish", "upgrade", "buy", "record", "growth", "boom",
                    "strong", "outperform", "positive", "high", "up", "rise",
                    "breakthrough", "win", "success"}
            _NEG = {"crash", "plunge", "drop", "fall", "loss", "miss", "bearish",
                    "downgrade", "sell", "decline", "fear", "risk", "weak",
                    "cut", "warn", "threat", "layoff", "recession", "low",
                    "slump", "fail", "worst", "negative", "down"}

            def polarity_scores(self, text: str) -> dict:
                words = set(text.lower().split())
                pos = len(words & self._POS)
                neg = len(words & self._NEG)
                total = pos + neg
                if total == 0:
                    return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
                compound = (pos - neg) / total
                return {"compound": compound, "pos": pos / total,
                        "neg": neg / total, "neu": 0.0}
        _HAS_VADER = False  # using fallback

try:
    import yfinance as yf
except ImportError:
    yf = None

# Streamlit caching (fall back to functools.lru_cache)
try:
    import streamlit as st
    _USE_ST_CACHE = True
except ImportError:
    _USE_ST_CACHE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Caching decorator helper
# ---------------------------------------------------------------------------

def _cached(ttl: int = 3600):
    """Return the appropriate caching decorator."""
    def decorator(func):
        if _USE_ST_CACHE:
            return st.cache_data(ttl=ttl)(func)
        else:
            return functools.lru_cache(maxsize=32)(func)
    return decorator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_vader = SentimentIntensityAnalyzer()


def _score_headline(text: str) -> float:
    """Return VADER compound sentiment score for *text* (-1 to +1)."""
    try:
        return _vader.polarity_scores(text)["compound"]
    except Exception:
        return 0.0


def _fetch_yahoo_news(symbol: str) -> List[dict]:
    """Fetch news articles from yfinance for *symbol*."""
    if yf is None:
        return []
    try:
        ticker = yf.Ticker(symbol)
        raw_news = ticker.news or []
        articles = []
        for item in raw_news:
            title = item.get("title", "")
            publisher = item.get("publisher", "Yahoo Finance")
            publish_ts = item.get("providerPublishTime", None)
            if publish_ts is not None:
                pub_date = datetime.utcfromtimestamp(publish_ts)
            else:
                pub_date = datetime.utcnow()
            articles.append({
                "title": title,
                "date": pub_date,
                "source": publisher,
                "sentiment_score": _score_headline(title),
            })
        return articles
    except Exception as exc:
        logger.warning("Yahoo Finance news fetch failed for %s: %s", symbol, exc)
        return []


def _fetch_newsapi(symbol: str, days_back: int = 30) -> List[dict]:
    """Fetch headlines from NewsAPI if NEWS_API_KEY is set."""
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return []

    try:
        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q": symbol,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 50,
            "apiKey": api_key,
        }
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for item in data.get("articles", []):
            title = item.get("title", "")
            source_name = (item.get("source") or {}).get("name", "NewsAPI")
            pub_str = item.get("publishedAt", "")
            try:
                pub_date = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                pub_date = datetime.utcnow()
            articles.append({
                "title": title,
                "date": pub_date,
                "source": source_name,
                "sentiment_score": _score_headline(title),
            })
        return articles
    except Exception as exc:
        logger.warning("NewsAPI fetch failed for %s: %s", symbol, exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news_sentiment(symbol: str, days_back: int = 30) -> dict:
    """Fetch and score news headlines for *symbol*.

    Returns a dictionary with the following keys:

    - **headlines**: list of dicts, each containing *title*, *date*,
      *source*, and *sentiment_score* (VADER compound, -1 to +1).
    - **sentiment_mean**: mean compound score across all headlines.
    - **sentiment_std**: standard deviation of compound scores.
    - **positive_ratio**: fraction of headlines with compound > 0.05.
    - **article_count**: total number of articles found.
    - **sentiment_recent_5d**: mean compound score of articles from the
      last 5 calendar days.
    - **sentiment_trend**: ``sentiment_recent_5d`` minus the mean of
      older articles. Positive values indicate improving sentiment.
    - **news_volume_trend**: ratio of recent 5-day article count to the
      average daily count over the full window, minus 1. Positive means
      above-average recent volume.
    """
    defaults = {
        "headlines": [],
        "sentiment_mean": 0.0,
        "sentiment_std": 0.0,
        "positive_ratio": 0.5,
        "article_count": 0,
        "sentiment_recent_5d": 0.0,
        "sentiment_trend": 0.0,
        "news_volume_trend": 0.0,
    }

    try:
        # Gather headlines from all sources
        yahoo_articles = _fetch_yahoo_news(symbol)
        newsapi_articles = _fetch_newsapi(symbol, days_back=days_back)

        # Merge and deduplicate by title
        seen_titles = set()
        all_articles: List[dict] = []
        for art in yahoo_articles + newsapi_articles:
            title_lower = art["title"].strip().lower()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                all_articles.append(art)

        if not all_articles:
            return defaults

        # Sort newest first
        all_articles.sort(key=lambda a: a["date"], reverse=True)

        scores = [a["sentiment_score"] for a in all_articles]
        n = len(scores)
        mean_score = sum(scores) / n
        variance = sum((s - mean_score) ** 2 for s in scores) / n
        std_score = variance ** 0.5
        positive_count = sum(1 for s in scores if s > 0.05)

        # Recent vs older split (5 days)
        cutoff = datetime.utcnow() - timedelta(days=5)
        recent = [a for a in all_articles if a["date"] >= cutoff]
        older = [a for a in all_articles if a["date"] < cutoff]

        recent_scores = [a["sentiment_score"] for a in recent]
        older_scores = [a["sentiment_score"] for a in older]

        recent_mean = (sum(recent_scores) / len(recent_scores)) if recent_scores else mean_score
        older_mean = (sum(older_scores) / len(older_scores)) if older_scores else mean_score

        # News volume trend
        total_days = max(days_back, 1)
        avg_daily = n / total_days
        recent_daily = len(recent) / 5.0
        volume_trend = (recent_daily / avg_daily - 1.0) if avg_daily > 0 else 0.0

        # Serialize dates for JSON-friendliness
        serialized = []
        for a in all_articles:
            serialized.append({
                "title": a["title"],
                "date": a["date"].isoformat(),
                "source": a["source"],
                "sentiment_score": round(a["sentiment_score"], 4),
            })

        return {
            "headlines": serialized,
            "sentiment_mean": round(mean_score, 4),
            "sentiment_std": round(std_score, 4),
            "positive_ratio": round(positive_count / n, 4),
            "article_count": n,
            "sentiment_recent_5d": round(recent_mean, 4),
            "sentiment_trend": round(recent_mean - older_mean, 4),
            "news_volume_trend": round(volume_trend, 4),
        }

    except Exception as exc:
        logger.error("fetch_news_sentiment failed for %s: %s", symbol, exc)
        return defaults


def fetch_fear_greed() -> dict:
    """Fetch the CNN Fear & Greed Index.

    Returns a dictionary with:

    - **score**: current index value (0-100).
    - **rating**: human-readable label, e.g. *Extreme Fear*, *Greed*.
    - **previous_close**: index value at previous market close.
    - **one_week_ago**: index value one week ago.
    - **one_month_ago**: index value one month ago.
    - **one_year_ago**: index value one year ago.

    All numeric values default to 50.0 (neutral) and *rating* defaults
    to ``"Neutral"`` when the data source is unavailable.
    """
    defaults = {
        "score": 50.0,
        "rating": "Neutral",
        "previous_close": 50.0,
        "one_week_ago": 50.0,
        "one_month_ago": 50.0,
        "one_year_ago": 50.0,
    }

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        fg = data.get("fear_and_greed", {})
        score = float(fg.get("score", 50.0))
        rating = fg.get("rating", "Neutral")

        prev_close = float(fg.get("previous_close", score))
        one_week = float(fg.get("previous_1_week", score))
        one_month = float(fg.get("previous_1_month", score))
        one_year = float(fg.get("previous_1_year", score))

        return {
            "score": round(score, 2),
            "rating": rating,
            "previous_close": round(prev_close, 2),
            "one_week_ago": round(one_week, 2),
            "one_month_ago": round(one_month, 2),
            "one_year_ago": round(one_year, 2),
        }

    except Exception as exc:
        logger.warning("fetch_fear_greed failed: %s", exc)
        return defaults


def get_sentiment_features(symbol: str) -> dict:
    """Return a flat dictionary of sentiment-derived features for *symbol*.

    Combines outputs of :func:`fetch_news_sentiment` and
    :func:`fetch_fear_greed` into a single dict suitable for use as
    model input features.

    Keys returned:

    - **news_sentiment_mean** (float): mean VADER compound score.
    - **news_sentiment_std** (float): std-dev of VADER scores.
    - **news_positive_ratio** (float, 0-1): fraction of positive articles.
    - **news_article_count** (int): total articles found.
    - **news_sentiment_trend** (float): recent 5d mean minus older mean.
    - **news_volume_trend** (float): recent volume vs average.
    - **fear_greed_score** (float, 0-1): current F&G index normalised.
    - **fear_greed_momentum** (float): (current - 1 week ago) / 100.
    """
    defaults = {
        "news_sentiment_mean": 0.0,
        "news_sentiment_std": 0.0,
        "news_positive_ratio": 0.5,
        "news_article_count": 0,
        "news_sentiment_trend": 0.0,
        "news_volume_trend": 0.0,
        "fear_greed_score": 0.5,
        "fear_greed_momentum": 0.0,
    }

    try:
        news = fetch_news_sentiment(symbol)
        fg = fetch_fear_greed()

        fg_score_norm = fg["score"] / 100.0
        fg_momentum = (fg["score"] - fg["one_week_ago"]) / 100.0

        return {
            "news_sentiment_mean": news["sentiment_mean"],
            "news_sentiment_std": news["sentiment_std"],
            "news_positive_ratio": news["positive_ratio"],
            "news_article_count": news["article_count"],
            "news_sentiment_trend": news["sentiment_trend"],
            "news_volume_trend": news["news_volume_trend"],
            "fear_greed_score": round(fg_score_norm, 4),
            "fear_greed_momentum": round(fg_momentum, 4),
        }

    except Exception as exc:
        logger.error("get_sentiment_features failed for %s: %s", symbol, exc)
        return defaults


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== Sentiment Analyzer Self-Test ===\n")

    symbol = "AAPL"
    print(f"Fetching news sentiment for {symbol}...")
    news_data = fetch_news_sentiment(symbol)
    print(f"  Articles found : {news_data['article_count']}")
    print(f"  Sentiment mean : {news_data['sentiment_mean']}")
    print(f"  Positive ratio : {news_data['positive_ratio']}")
    print(f"  Sentiment trend: {news_data['sentiment_trend']}")
    print()

    print("Fetching Fear & Greed Index...")
    fg_data = fetch_fear_greed()
    print(f"  Score  : {fg_data['score']}")
    print(f"  Rating : {fg_data['rating']}")
    print()

    print(f"Combined features for {symbol}:")
    features = get_sentiment_features(symbol)
    print(json.dumps(features, indent=2))
