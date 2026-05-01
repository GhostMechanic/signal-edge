"""
test_fetch_classification.py
----------------------------
Pin the data_fetcher.fetch_stock_data error classification.

The user got a "No data returned for 'CROC'. Check the ticker symbol."
error on a perfectly valid NASDAQ ticker (Crocs Inc) — the message
implied the user mis-typed when in fact Yahoo's chart endpoint had just
hiccuped. The fix:

  • retry once on empty Ticker.history
  • fall back to yf.download (different code path)
  • if everything still empty, classify via Ticker.info / fast_info:
      - Yahoo recognises symbol → TickerDataUnavailableError
      - Yahoo doesn't            → TickerNotFoundError

Run
---
    cd "Stock Bot"
    python3 -m unittest tests.test_fetch_classification -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import data_fetcher
from data_fetcher import (
    fetch_stock_data,
    TickerNotFoundError,
    TickerDataUnavailableError,
    _ticker_exists,
    _normalise_ohlcv,
    _fetch_via_stooq,
    _period_to_trading_days,
)


def _make_ohlcv(n: int = 30) -> pd.DataFrame:
    """Build a minimal but realistic OHLCV frame with a tz-aware index
    (yfinance always hands back UTC-aware timestamps)."""
    idx = pd.date_range("2026-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Open":   [100.0 + i * 0.1 for i in range(n)],
            "High":   [101.0 + i * 0.1 for i in range(n)],
            "Low":    [ 99.0 + i * 0.1 for i in range(n)],
            "Close":  [100.5 + i * 0.1 for i in range(n)],
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


class FetchSucceedsOnFirstTry(unittest.TestCase):

    def test_returns_normalised_frame_when_history_works(self):
        df = _make_ohlcv(20)
        with patch.object(data_fetcher.yf, "Ticker") as mock_ticker_cls:
            mock_ticker_cls.return_value.history.return_value = df
            out = fetch_stock_data("AAPL", period="1mo")
        self.assertEqual(len(out), 20)
        # tz must be stripped — downstream feature engineering relies on
        # naive datetimes.
        self.assertIsNone(out.index.tz)
        # Should not have called yf.download (path 3) when path 1 worked.


class FetchRetriesAndFallsBack(unittest.TestCase):

    def test_retries_then_succeeds_via_yf_download(self):
        """Path 1 returns empty twice, then yf.download saves the day —
        this is the actual CROC scenario (chart endpoint throttled,
        download endpoint not throttled)."""
        empty = pd.DataFrame()
        good = _make_ohlcv(15)

        with patch.object(data_fetcher.yf, "Ticker") as mock_ticker_cls, \
             patch.object(data_fetcher.yf, "download") as mock_download, \
             patch.object(data_fetcher.time, "sleep") as mock_sleep:
            mock_ticker_cls.return_value.history.return_value = empty
            mock_download.return_value = good

            out = fetch_stock_data("CROC", period="2y")

        self.assertEqual(len(out), 15)
        # history called twice (initial + 1 retry)
        self.assertEqual(
            mock_ticker_cls.return_value.history.call_count, 2,
            "history should be retried once before falling back",
        )
        # backoff sleep happened between attempts
        mock_sleep.assert_called_once()
        # download was tried as the fallback
        mock_download.assert_called_once()


class FetchClassifiesEmptyResultsCorrectly(unittest.TestCase):

    def test_unknown_ticker_raises_ticker_not_found(self):
        """All three paths come back empty, AND fast_info / info show no
        liveness signals → raise TickerNotFoundError so the UI says
        'check spelling.'"""
        empty = pd.DataFrame()
        with patch.object(data_fetcher.yf, "Ticker") as mock_ticker_cls, \
             patch.object(data_fetcher.yf, "download", return_value=empty), \
             patch.object(data_fetcher.time, "sleep"):
            mock_t = mock_ticker_cls.return_value
            mock_t.history.return_value = empty
            # No liveness signals — fast_info raises, info is empty
            type(mock_t).fast_info = MagicMock(
                side_effect=Exception("no fast info"),
            )
            type(mock_t).info = MagicMock(return_value={})
            # Properties aren't called the same way as methods; rebind
            # to a plain dict so the helper sees an empty info dict.
            mock_t.info = {}
            mock_t.fast_info = {}

            with self.assertRaises(TickerNotFoundError) as cm:
                fetch_stock_data("ZZZNOTREAL", period="1y")
        self.assertIn("ZZZNOTREAL", str(cm.exception))

    def test_valid_ticker_with_empty_feed_raises_data_unavailable(self):
        """All three paths come back empty BUT info/fast_info report a
        real security → raise TickerDataUnavailableError so the UI
        offers 'try again' instead of accusing the user of mistyping."""
        empty = pd.DataFrame()
        with patch.object(data_fetcher.yf, "Ticker") as mock_ticker_cls, \
             patch.object(data_fetcher.yf, "download", return_value=empty), \
             patch.object(data_fetcher.time, "sleep"):
            mock_t = mock_ticker_cls.return_value
            mock_t.history.return_value = empty
            # fast_info returns a real last_price → ticker exists
            mock_t.fast_info = {"last_price": 87.42, "previous_close": 86.10}
            mock_t.info = {
                "symbol": "CROC", "shortName": "Crocs, Inc.",
                "quoteType": "EQUITY", "regularMarketPrice": 87.42,
            }

            with self.assertRaises(TickerDataUnavailableError) as cm:
                fetch_stock_data("CROC", period="2y")
        self.assertIn("CROC", str(cm.exception))
        self.assertIn("transient", str(cm.exception).lower())


class TickerExistsHelper(unittest.TestCase):

    def test_recognises_via_fast_info(self):
        with patch.object(data_fetcher.yf, "Ticker") as mock_ticker_cls:
            mock_t = mock_ticker_cls.return_value
            mock_t.fast_info = {"last_price": 100.0}
            mock_t.info = {}
            self.assertTrue(_ticker_exists("AAPL"))

    def test_recognises_via_info_quote_type(self):
        with patch.object(data_fetcher.yf, "Ticker") as mock_ticker_cls:
            mock_t = mock_ticker_cls.return_value
            mock_t.fast_info = {}
            mock_t.info = {"quoteType": "EQUITY", "shortName": "Foo Corp"}
            self.assertTrue(_ticker_exists("FOO"))

    def test_returns_false_for_nothing(self):
        with patch.object(data_fetcher.yf, "Ticker") as mock_ticker_cls:
            mock_t = mock_ticker_cls.return_value
            mock_t.fast_info = {}
            mock_t.info = {}
            self.assertFalse(_ticker_exists("ZZZNOTREAL"))


class NormaliseOhlcvShape(unittest.TestCase):

    def test_strips_timezone(self):
        df = _make_ohlcv(5)
        out = _normalise_ohlcv(df)
        self.assertIsNone(out.index.tz)

    def test_handles_multiindex_columns(self):
        """yf.download in multi-symbol mode hands back MultiIndex
        columns. Our flattener has to cope even though we only ever
        ask for one symbol."""
        df = _make_ohlcv(5)
        df.columns = pd.MultiIndex.from_product([df.columns, ["AAPL"]])
        out = _normalise_ohlcv(df)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            self.assertIn(col, out.columns)


class StooqFallback(unittest.TestCase):
    """Path 4 in fetch_stock_data: when every yfinance code path
    returns empty, we hit Stooq (different vendor) before giving up.
    This is the actual production fix for the CROC outage — yfinance
    was getting empty payloads from Fly.io's IP because Yahoo's chart
    endpoint silently throttles cloud datacenter ranges. Stooq isn't
    affected by that throttle."""

    _STOOQ_CSV = (
        "Date,Open,High,Low,Close,Volume\n"
        "2026-04-28,86.50,87.20,86.10,86.95,1500000\n"
        "2026-04-29,86.95,88.10,86.80,87.85,1620000\n"
        "2026-04-30,87.85,88.50,87.20,87.42,1480000\n"
    )

    def _mock_response(self, status: int, body: str):
        m = MagicMock()
        m.status_code = status
        m.text = body
        return m

    def test_stooq_serves_data_when_yfinance_empty(self):
        """All three yfinance paths fail, Stooq returns CSV → we get
        a normalised frame with data_source='stooq' attached."""
        empty = pd.DataFrame()
        with patch.object(data_fetcher.yf, "Ticker") as mock_ticker_cls, \
             patch.object(data_fetcher.yf, "download", return_value=empty), \
             patch.object(data_fetcher.time, "sleep"), \
             patch("requests.get") as mock_get:
            mock_ticker_cls.return_value.history.return_value = empty
            mock_get.return_value = self._mock_response(200, self._STOOQ_CSV)

            out = fetch_stock_data("CROC", period="2y")

        self.assertEqual(len(out), 3)
        self.assertEqual(out.attrs.get("data_source"), "stooq")
        self.assertIn("Close", out.columns)
        # First request hit `croc.us` per Stooq's US convention
        called_url = mock_get.call_args.args[0]
        self.assertIn("s=croc.us", called_url)

    def test_stooq_no_data_response_falls_through(self):
        """Stooq's literal 'No data' body is treated as failure → we
        fall through to classification rather than returning a phantom
        success."""
        empty = pd.DataFrame()
        with patch.object(data_fetcher.yf, "Ticker") as mock_ticker_cls, \
             patch.object(data_fetcher.yf, "download", return_value=empty), \
             patch.object(data_fetcher.time, "sleep"), \
             patch("requests.get") as mock_get:
            mock_t = mock_ticker_cls.return_value
            mock_t.history.return_value = empty
            # Both Stooq candidates return "No data"
            mock_get.return_value = self._mock_response(200, "No data\n")
            # Ticker is real per fast_info → should classify as transient
            mock_t.fast_info = {"last_price": 87.42}
            mock_t.info = {"symbol": "CROC"}

            with self.assertRaises(TickerDataUnavailableError):
                fetch_stock_data("CROC", period="2y")

    def test_stooq_html_error_page_falls_through(self):
        """Stooq sometimes serves an HTML error page on bad symbols.
        The leading `<` should be detected and skipped, not parsed
        as CSV."""
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_response(
                200, "<!DOCTYPE html><html>error</html>",
            )
            out = _fetch_via_stooq("ZZZNOTREAL", period="1y")
        self.assertIsNone(out)

    def test_stooq_trims_to_period(self):
        """Stooq returns ALL history; we trim to ~1.5x the requested
        period to keep memory bounded for downstream feature
        engineering."""
        # 1000 days of synthetic history
        big_csv = "Date,Open,High,Low,Close,Volume\n"
        for i in range(1000):
            d = pd.Timestamp("2022-01-01") + pd.Timedelta(days=i)
            big_csv += f"{d.date()},100,101,99,100,1000000\n"
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_response(200, big_csv)
            out = _fetch_via_stooq("FOO", period="1y")
        self.assertIsNotNone(out)
        # 1y → 252 trading days × 1.5 = 378 row cap
        self.assertLessEqual(len(out), 378)


class PeriodToTradingDays(unittest.TestCase):

    def test_year_units(self):
        self.assertEqual(_period_to_trading_days("1y"), 252)
        self.assertEqual(_period_to_trading_days("2y"), 504)
        self.assertEqual(_period_to_trading_days("10y"), 2520)

    def test_month_units(self):
        self.assertEqual(_period_to_trading_days("1mo"), 21)
        self.assertEqual(_period_to_trading_days("6mo"), 126)

    def test_max_returns_none(self):
        self.assertIsNone(_period_to_trading_days("max"))

    def test_garbage_returns_none(self):
        self.assertIsNone(_period_to_trading_days("garbage"))
        self.assertIsNone(_period_to_trading_days(""))


if __name__ == "__main__":
    unittest.main()
