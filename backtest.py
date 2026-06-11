#!/usr/bin/env python3
"""
backtest.py — Quarterly portfolio backtest using a trained XGBoost ranking model.

Rebalance modes
---------------
fixed      Each calendar quarter, all stocks whose filing_date_used fell in that
           quarter are scored at the same time.  The entire cohort is entered at
           the quarter-end close and exited at the following quarter-end close.
           Clean portfolio math; one unambiguous return per quarter.

staggered  Each stock is entered at its own filing_date_used close and exited at
           the close on its NEXT filing_date_used (capped by --max-hold-days).
           More realistic: the position expires when fresh information arrives,
           not after an arbitrary fixed window.
           Per-position alpha vs SPY is computed over each stock's own window.

Usage
-----
python backtest.py --experiment healthcare_ic --sectors healthcare \\
                   --rebalance fixed --top-quantile 0.2 --bot-quantile 0.2 \\
                   --start 2021-01-01 --end 2024-06-30

python backtest.py --experiment healthcare_ic --sectors healthcare \\
                   --rebalance staggered --top-quantile 0.2 --max-hold-days 548
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from util import get_price_history, lookup_price
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
PRICE_CACHE = Path("price_cache")
DATA_DIR    = Path("data_quarterly_parquet")
PRICE_CACHE.mkdir(exist_ok=True)

METADATA_COLS = {"ticker", "sector", "industry", "fiscalDateEnding",
                 "filing_date_used", "target"}


# ── Data loading ───────────────────────────────────────────────────────────────
def load_model_and_meta(experiment_name: str):
    prefix    = f"final_{experiment_name}" if experiment_name else "final"
    reg_path  = RESULTS_DIR / f"{prefix}_reg.ubj"
    meta_path = RESULTS_DIR / f"{prefix}_meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)

    if not reg_path.exists():
        raise FileNotFoundError(
            f"Model not found: {reg_path}\n"
            "Run training.ipynb first to produce the .ubj file.")

    model = xgb.XGBRegressor()
    model.load_model(reg_path)
    sc = meta["scores"]
    print(f"Loaded  {reg_path.name}  "
          f"({len(meta['best_feats'])} feats | train IC={sc['test_ic']:.4f} "
          f"ICIR={sc['test_icir']:.4f})")
    return model, meta


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


# ── Preprocessing ──────────────────────────────────────────────────────────────
def fit_cleanup(df_ref: pd.DataFrame, feat_cols: list[str]):
    X = df_ref[feat_cols].replace([np.inf, -np.inf], np.nan)
    lower   = X.quantile(0.01)
    upper   = X.quantile(0.99)
    medians = X.median()
    sel     = VarianceThreshold(threshold=1e-5)
    sel.fit(X.fillna(medians))
    kept = X.columns[sel.get_support()].tolist()
    return lower, upper, medians, kept


def apply_cleanup(X: pd.DataFrame,
                  lower: pd.Series, upper: pd.Series,
                  medians: pd.Series, kept: list[str],
                  feats: list[str]) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=lower, upper=upper, axis=1)
    X = X.fillna(medians)
    cols = [c for c in feats if c in kept]
    return X.reindex(columns=cols, fill_value=0.0)


# ── Test-window inference ─────────────────────────────────────────────────────
def infer_test_window(df: pd.DataFrame,
                      test_months: int = 24) -> tuple[str, str]:
    """
    Derive the test-set date range from the dataset, mirroring the training
    pipeline split (last `test_months` months by filing_date_used).

    Returns (start_date, end_date) as 'YYYY-MM-DD' strings ready for run_backtest.
    """
    max_date   = df["filing_date_used"].max()
    start_date = max_date - pd.DateOffset(months=test_months)
    return start_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")


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


def next_filing_exit(ticker: str, entry_dt: pd.Timestamp,
                     filing_map: dict, max_hold_days: int) -> pd.Timestamp:
    """
    Return the filing_date_used that immediately follows entry_dt for this
    ticker.  Falls back to entry_dt + max_hold_days if no next filing exists
    within that cap (e.g. delisting, acquisition, reporting gap).
    """
    cap    = entry_dt + pd.Timedelta(days=max_hold_days)
    future = [d for d in filing_map.get(ticker, []) if d > entry_dt]
    return min(min(future), cap) if future else cap


# ── Core backtest ──────────────────────────────────────────────────────────────
def run_backtest(
    experiment_name: str,
    sectors: list[str] | None,
    top_q: float,
    bot_q: float,
    start_date: str,
    end_date: str,
    rebalance_mode: str,
    max_hold_days: int,
    min_stocks: int,
    output_dir: Path,
) -> pd.DataFrame:

    model, meta = load_model_and_meta(experiment_name)
    feats       = meta["best_feats"]

    df = load_dataset(sectors)
    if "ticker" not in df.columns:
        raise ValueError("'ticker' column required — regenerate parquets with get_metadata=True")

    feat_cols = [c for c in df.columns if c not in METADATA_COLS]

    # Fit cleanup on data before the backtest window (no leakage)
    GAP      = pd.Timedelta(days=456)
    pre_mask = df["filing_date_used"] < pd.Timestamp(start_date) - GAP

    if pre_mask.sum() == 0:
        raise ValueError("No pre-backtest data to fit cleanup — move start_date later or check GAP")
    elif pre_mask.sum() < 50:
        print(f"[WARN] Only {pre_mask.sum()} rows before cutoff — cleanup stats may be noisy")
        
    lower, upper, medians, kept = fit_cleanup(df[pre_mask], feat_cols)
    feats = [f for f in feats if f in kept]
    print(f"Features available for inference: {len(feats)}")

    # Staggered: build next-filing lookup over the ENTIRE dataset
    filing_map = build_next_filing_map(df) if rebalance_mode == "staggered" else {}

    quarters = pd.period_range(start=start_date, end=end_date, freq="Q")
    print(f"\nBacktest  {quarters[0]} → {quarters[-1]}  ({len(quarters)} Qs)  "
          f"mode={rebalance_mode}  long={top_q:.0%}  short={bot_q:.0%}\n")

    quarter_records: list[dict] = []
    trade_records:   list[dict] = []

    for i, q in enumerate(tqdm(quarters, desc="Quarters")):
        q_start = q.start_time
        q_end   = q.end_time

        batch = df[(df["filing_date_used"] >= q_start) &
                   (df["filing_date_used"] <= q_end)].copy()

        if len(batch) < min_stocks:
            tqdm.write(f"  {q}: {len(batch)} stocks < min {min_stocks} — skip")
            continue

        # ── Score & select legs ────────────────────────────────────────────────
        X_inf = apply_cleanup(batch[feat_cols].copy(), lower, upper, medians, kept, feats)
        batch = batch.reset_index(drop=True)
        batch["score"] = model.predict(X_inf)

        ranked = batch.sort_values("score", ascending=False).reset_index(drop=True)
        top_n  = max(1, int(np.ceil(len(ranked) * top_q)))
        bot_n  = max(1, int(np.ceil(len(ranked) * bot_q))) if bot_q > 0 else 0
        legs   = pd.concat([
            ranked.head(top_n).assign(direction="long"),
            ranked.tail(bot_n).assign(direction="short") if bot_n > 0 else pd.DataFrame()
        ], ignore_index=True)

        # ── Assign entry / exit dates per mode ────────────────────────────────
        if rebalance_mode == "fixed":
            # Shared rebalance dates: enter at q_end, exit at next quarter-end
            next_q_end = (quarters[i + 1].end_time
                          if i + 1 < len(quarters)
                          else q_end + pd.offsets.QuarterEnd(1))
            legs = legs.copy()
            legs["entry_dt"] = q_end
            legs["exit_dt"]  = next_q_end
        else:  # staggered
            legs = legs.copy()
            legs["entry_dt"] = legs["filing_date_used"]
            legs["exit_dt"]  = legs.apply(
                lambda r: next_filing_exit(
                    r["ticker"], r["filing_date_used"], filing_map, max_hold_days),
                axis=1,
            )

        # ── Fetch prices & compute per-position returns ────────────────────────
        rows = []
        n_skipped = 0
        for _, row in legs.iterrows():
            ticker   = row["ticker"]
            entry_dt = row["entry_dt"]
            exit_dt  = row["exit_dt"]
            p_in     = price_on(ticker, entry_dt)
            p_out    = price_on(ticker, exit_dt)
            if not p_in:
                tqdm.write(f"    skip {ticker}: no entry price on {entry_dt.date()}")
                n_skipped += 1
                continue
            if not p_out:
                tqdm.write(f"    skip {ticker}: no exit price on {exit_dt.date()}")
                n_skipped += 1
                continue
            if p_in <= 0:
                tqdm.write(f"    skip {ticker}: non-positive entry price {p_in}")
                n_skipped += 1
                continue

            raw_ret = (p_out - p_in) / p_in
            sign    = 1 if row["direction"] == "long" else -1

            # Per-position SPY alpha over the same window
            spy_in  = price_on("SPY", entry_dt)
            spy_out = price_on("SPY", exit_dt)
            spy_ret = (spy_out - spy_in) / spy_in if spy_in and spy_out else np.nan
            pos_alpha = (raw_ret - spy_ret) * sign if not np.isnan(spy_ret) else np.nan

            rows.append({
                "quarter":       str(q),
                "ticker":        ticker,
                "direction":     row["direction"],
                "entry_date":    entry_dt,
                "exit_date":     exit_dt,
                "hold_days":     (exit_dt - entry_dt).days,
                "entry_price":   round(p_in,  4),
                "exit_price":    round(p_out, 4),
                "raw_return":    raw_ret,
                "signed_return": raw_ret * sign,
                "spy_ret":       spy_ret,
                "pos_alpha":     pos_alpha,
                "score":         row["score"],
                "target":        row.get("target", np.nan),
            })
        trade_records.extend(rows)

        if not rows:
            tqdm.write(f"  {q}: no prices resolved — skip")
            continue

        qdf     = pd.DataFrame(rows)
        long_r  = qdf.loc[qdf.direction == "long",  "raw_return"].mean()
        short_r = qdf.loc[qdf.direction == "short", "raw_return"].mean() if bot_q > 0 else np.nan
        comb_r  = qdf["signed_return"].mean()
        # Mean position-level alpha (sign-adjusted, so short alpha = positive when short leg profits)
        alpha   = qdf["pos_alpha"].mean()

        # IC: predicted score vs actual raw return within this cohort
        ic_val, _ = spearmanr(qdf["score"], qdf["raw_return"])
        ic_val = float(ic_val) if not np.isnan(ic_val) else np.nan

        # SPY return for display (mean of per-position SPY windows)
        spy_r = qdf["spy_ret"].mean()

        quarter_records.append({
            "quarter": str(q), "q_start": q_start,
            "n_filed":   len(batch),
            "n_priced":  len(rows),
            "n_long":    (qdf.direction == "long").sum(),
            "n_short":   (qdf.direction == "short").sum(),
            "long_ret":     long_r,
            "short_ret":    short_r,
            "combined_ret": comb_r,
            "spy_ret":      spy_r,
            "alpha":        alpha,
            "ic":           ic_val,
        })
        skip_str = f"  [{n_skipped} skipped]" if n_skipped else ""
        tqdm.write(
            f"  {q}  n={len(batch):3d}→{len(rows):3d}{skip_str}  "
            f"long={long_r:+.2%}  "
            + (f"short={short_r:+.2%}  " if bot_q > 0 else "")
            + f"comb={comb_r:+.2%}  SPY≈{spy_r:+.2%}  "
              f"alpha={alpha:+.2%}  IC={ic_val:+.3f}"
        )

    if not quarter_records:
        print("\n[ERROR] No results — check dates, data, and model files.")
        return pd.DataFrame()

    results = pd.DataFrame(quarter_records)
    trades  = pd.DataFrame(trade_records)

    # ── Cumulative returns ─────────────────────────────────────────────────────
    results["cum_combined"] = (1 + results["combined_ret"].fillna(0)).cumprod() - 1
    results["cum_long"]     = (1 + results["long_ret"].fillna(0)).cumprod() - 1
    results["cum_spy"]      = (1 + results["spy_ret"].fillna(0)).cumprod() - 1

    # ── Summary stats ─────────────────────────────────────────────────────────
    def sharpe_ann(s: pd.Series) -> float:
        s = s.dropna()
        return s.mean() / s.std() * np.sqrt(4) if s.std() > 0 else np.nan

    def max_drawdown(cum: pd.Series) -> float:
        roll_max = (1 + cum).cummax()
        return float(((1 + cum) / roll_max - 1).min())

    print("\n─── Backtest Summary ────────────────────────────────────────────────")
    print(f"  Mode                  : {rebalance_mode}")
    print(f"  Quarters with results : {len(results)} / {len(quarters)}")
    print(f"  Mean quarterly IC     : {results['ic'].mean():+.4f}  "
          f"(std={results['ic'].std():.4f})")
    print(f"\n  Long leg")
    print(f"    Mean qtrly return   : {results['long_ret'].mean():+.2%}")
    print(f"    Annualised Sharpe   : {sharpe_ann(results['long_ret']):.3f}")
    print(f"    Win rate            : {(results['long_ret'] > 0).mean():.1%}")
    print(f"    Max drawdown        : {max_drawdown(results['cum_long']):.2%}")
    if bot_q > 0:
        print(f"\n  Short leg")
        print(f"    Mean qtrly return   : {results['short_ret'].mean():+.2%}")
        print(f"    Annualised Sharpe   : {sharpe_ann(results['short_ret']):.3f}")
    print(f"\n  Combined L/S")
    print(f"    Mean qtrly return   : {results['combined_ret'].mean():+.2%}")
    print(f"    Annualised Sharpe   : {sharpe_ann(results['combined_ret']):.3f}")
    print(f"    Total return        : {results['cum_combined'].iloc[-1]:+.2%}")
    print(f"    Max drawdown        : {max_drawdown(results['cum_combined']):.2%}")
    print(f"\n  SPY benchmark (per-position avg window)")
    print(f"    Mean qtrly return   : {results['spy_ret'].mean():+.2%}")
    print(f"    Total return        : {results['cum_spy'].iloc[-1]:+.2%}")
    print(f"\n  Mean position-level alpha : {results['alpha'].mean():+.2%}")
    if rebalance_mode == "staggered":
        print(f"  Mean hold period (days)   : {trades['hold_days'].mean():.0f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    tag    = f"{experiment_name}_{rebalance_mode}"
    q_path = output_dir / f"backtest_{tag}_quarterly.csv"
    t_path = output_dir / f"backtest_{tag}_trades.csv"
    results.to_csv(q_path, index=False)
    trades.to_csv(t_path, index=False)
    print(f"\nSaved → {q_path}")
    print(f"Saved → {t_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.suptitle(f"Backtest: {experiment_name} [{rebalance_mode}]  "
                 f"({quarters[0]} – {quarters[-1]})",
                 fontsize=13, fontweight="bold")

    # 1. Cumulative returns
    ax = axes[0]
    ax.plot(results["q_start"], results["cum_combined"] * 100,
            label="Strategy (combined)", color="steelblue", lw=2)
    ax.plot(results["q_start"], results["cum_long"] * 100,
            label="Long only", color="seagreen", lw=1.5, ls="--")
    ax.plot(results["q_start"], results["cum_spy"] * 100,
            label="SPY", color="gray", lw=1.5, ls=":")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Cumulative Return")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Quarterly returns bar chart
    ax2 = axes[1]
    width = pd.Timedelta(days=35)
    x = results["q_start"]
    ax2.bar(x, results["long_ret"] * 100, width=width,
            color="seagreen", alpha=0.7, label="Long")
    if bot_q > 0:
        ax2.bar(x, results["short_ret"] * 100, width=width,
                color="salmon", alpha=0.7, label="Short contrib",
                bottom=results["long_ret"] * 100)
    ax2.plot(x, results["spy_ret"] * 100, "o--", color="gray",
             markersize=4, label="SPY", lw=1)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Return (%)")
    ax2.set_title("Quarterly Returns")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. IC per quarter
    ax3 = axes[2]
    colors = ["seagreen" if v >= 0 else "salmon" for v in results["ic"]]
    ax3.bar(results["q_start"], results["ic"], width=width, color=colors, alpha=0.8)
    ax3.axhline(0, color="black", lw=0.5)
    ax3.axhline(results["ic"].mean(), color="steelblue", lw=1.5,
                ls="--", label=f"Mean IC={results['ic'].mean():+.3f}")
    ax3.set_ylabel("IC (Spearman)")
    ax3.set_title("Per-Quarter IC (score vs realised return)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    for ax_ in axes:
        ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        plt.setp(ax_.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / f"backtest_{tag}_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot   → {plot_path}")
    plt.show()

    return results


# ── Notebook usage ────────────────────────────────────────────────────────────
# Copy the cells below into a notebook.  `import backtest` must resolve —
# either run from the same directory or add it to sys.path.
#
# Cell 1 — parameters
# ──────────────────────────────────────────────────────────────────────────────
# from pathlib import Path
# import backtest
#
# EXPERIMENT    = "healthcare_ic"       # matches results/final_<NAME>_reg.ubj
# SECTORS       = ["healthcare"]        # None = all sectors
# TOP_QUANTILE  = 0.2
# BOT_QUANTILE  = 0.0                   # 0 = long-only
# REBALANCE     = "fixed"               # "fixed" | "staggered"
# MAX_HOLD_DAYS = 548                   # staggered cap only
# MIN_STOCKS    = 3
# OUTPUT_DIR    = Path("backtest_results")
#
# Cell 2 — derive test window from data (matches training pipeline split)
# ──────────────────────────────────────────────────────────────────────────────
# df = backtest.load_dataset(SECTORS)
# START, END = backtest.infer_test_window(df, test_months=24)
# print(f"Test window: {START}  →  {END}")
#
# Cell 3 — run
# ──────────────────────────────────────────────────────────────────────────────
# results = backtest.run_backtest(
#     experiment_name = EXPERIMENT,
#     sectors         = SECTORS,
#     top_q           = TOP_QUANTILE,
#     bot_q           = BOT_QUANTILE,
#     start_date      = START,
#     end_date        = END,
#     rebalance_mode  = REBALANCE,
#     max_hold_days   = MAX_HOLD_DAYS,
#     min_stocks      = MIN_STOCKS,
#     output_dir      = OUTPUT_DIR,
# )


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Quarterly fundamental backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Rebalance modes:\n"
            "  fixed      Enter entire cohort at quarter-end; exit at next quarter-end.\n"
            "  staggered  Enter each stock at filing_date_used; exit at its next filing\n"
            "             (capped by --max-hold-days). More realistic signal expiry.\n"
        ),
    )
    ap.add_argument("--experiment",    default="healthcare_ic",
                    help="Name matching results/final_<NAME>_reg.ubj")
    ap.add_argument("--sectors",       nargs="+", default=None,
                    help="Sector parquet names, e.g. --sectors healthcare technology")
    ap.add_argument("--top-quantile",  type=float, default=0.2,
                    help="Fraction of top-scored stocks to go long (default 0.2)")
    ap.add_argument("--bot-quantile",  type=float, default=0.0,
                    help="Fraction of bottom-scored stocks to short (default 0 = long-only)")
    ap.add_argument("--start",         default="2022-01-01",
                    help="Backtest start date YYYY-MM-DD")
    ap.add_argument("--end",           default="2024-12-31",
                    help="Backtest end date YYYY-MM-DD")
    ap.add_argument("--rebalance",     default="fixed", choices=["fixed", "staggered"],
                    help="Rebalancing mode (default: fixed)")
    ap.add_argument("--max-hold-days", type=int, default=548,
                    help="[staggered] Max hold in days if no next filing found (default 548 ≈ 18 mo)")
    ap.add_argument("--min-stocks",    type=int, default=3,
                    help="Skip a quarter if fewer stocks filed (default 3)")
    ap.add_argument("--output-dir",    default="backtest_results",
                    help="Directory for CSV and PNG outputs (default backtest_results)")
    args = ap.parse_args()

    run_backtest(
        experiment_name=args.experiment,
        sectors=args.sectors,
        top_q=args.top_quantile,
        bot_q=args.bot_quantile,
        start_date=args.start,
        end_date=args.end,
        rebalance_mode=args.rebalance,
        max_hold_days=args.max_hold_days,
        min_stocks=args.min_stocks,
        output_dir=Path(args.output_dir),
    )
