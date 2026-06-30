import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from util import get_price_history, lookup_price

DATA_DIR    = Path("data_quarterly_to_present_parquet")
# DATA_DIR    = Path("data_quarterly_parquet")

PRICE_CACHE = Path("price_cache")
PRICE_CACHE.mkdir(exist_ok=True)

METADATA_COLS = {"ticker", "sector", "industry", "fiscalDateEnding",
                 "filing_date_used", "target"}

SECTOR_ETF_MAP: dict[str, str] = {
    "basic_materials": "XLB",
    "communication_services": "XLC",
    "consumer_cyclical": "XLY",
    "consumer_defensive": "XLP",
    "energy": "XLE",
    "financial_services": "XLF",
    "healthcare": "XLV",
    "industrials": "XLI",
    "real_estate": "XLRE",
    "technology": "XLK",
    "utilities": "XLU"
}

def load_dataset(sectors: list[str] | None, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    files = ([data_dir / f"{s}.parquet" for s in sectors]
             if sectors else list(data_dir.glob("*.parquet")))
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing parquet(s): {missing}")
    df = pd.read_parquet(files)
    df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
    df["filing_date_used"] = pd.to_datetime(df["filing_date_used"])
    print(f"Dataset: {len(df):,} rows | {df['ticker'].nunique():,} tickers"
          if "ticker" in df.columns else f"Dataset: {len(df):,} rows (no ticker col)")
    return df

# ── Price helpers ──────────────────────────────────────────────────────────────
# Reuses util.get_price_history (disk cache + exponential-backoff retries) and
# util.lookup_price (forward-window close lookup with error printing).
# In-memory dict avoids repeated parquet reads within a single backtest run.

_hist_cache: dict[str, pd.DataFrame | None] = {}

def _get_hist(ticker: str) -> pd.DataFrame | None:
    if ticker not in _hist_cache:
        _hist_cache[ticker] = get_price_history(ticker)
    return _hist_cache[ticker]

def price_on(ticker: str, date: pd.Timestamp, window: int = 5) -> float | None:
    return lookup_price(_get_hist(ticker), date, window_days=window)

def fetch_rf_quarterly(start_date: str, end_date: str) -> pd.Series:
    """
    Returns quarterly risk-free reates derived from the 13-week T-bill yield (^IRX).
    Index is quarter period strings e.g. '2022Q1'. Falls back to empty Series (rf=0)
    """
    hist = get_price_history("^IRX")
    if hist is None:
        print("[WARN] Could not fetch ^IRX T-bill rates - Sharpe will use rf=0")
        return pd.Series(dtype=float)
    
    annual_rate = hist['Close'] / 100.0
    quarterly_annual = annual_rate.resample("QE").mean()
    rf_q = (1 + quarterly_annual) ** (1 / 4) - 1
    rf_q.index = rf_q.index.to_period("Q").astype(str)
    valid = pd.period_range(start=start_date, end=end_date, freq="Q").astype(str)
    return rf_q.reindex(valid)

def fit_cleanup(df_ref: pd.DataFrame, feat_cols: list[str]):
    X = df_ref[feat_cols].replace([np.inf, -np.inf], np.nan)
    lower   = X.quantile(0.01)
    upper   = X.quantile(0.99)
    medians = X.median()
    return lower, upper, medians

# ── Test-window inference ─────────────────────────────────────────────────────
def infer_test_window(df: pd.DataFrame,
                      test_months: int = 24) -> tuple[str, str]:
    """
    Derive the test-set date range from the dataset, mirroring the training
    pipeline split (last `test_months` months by filing_date_used).

    Returns (start_date, end_date) as 'YYYY-MM-DD' strings ready for run_backtest.
    """
    end_date   = df["filing_date_used"].max() - pd.DateOffset(month=3)
    start_date = datetime.strptime("2023-03-31", "%Y-%m-%d").date()

    return (
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

# ── Next-filing map (staggered mode) ──────────────────────────────────────────
def build_next_filing_map(df: pd.DataFrame) -> dict[str, list[pd.Timestamp]]:
    """
    For each ticker return its chronologically sorted list of filing_date_used
    timestamps.  Used in staggered mode to find the next filing after an entry.
    """
    return (
        df.groupby("ticker")["filing_date_used"]
        .apply(lambda s: sorted(s.tolist()))
        .to_dict()
    )

def next_filing_exit(
        ticker: str,
        entry_dt: pd.Timestamp,
        filing_map: dict,
        ) -> pd.Timestamp | None:

    next_qend = (entry_dt + pd.offsets.QuarterEnd(1)).normalize()

    future = [d for d in filing_map.get(ticker, []) if entry_dt < d]
    return min(min(future), next_qend) if future else next_qend