#!/usr/bin/env python3
"""
backtest_MLP.py — Quarterly portfolio backtest using a trained MLP ensemble.

Identical in structure and logic to backtest.py but loads MLP models saved by
training_MLP.py (joblib bundles in results_mlp/) instead of XGBoost .ubj files.

Key differences vs backtest.py
--------------------------------
- Models loaded via joblib (skorch NeuralNetRegressor + fitted StandardScaler)
  using load_final_models() from training_MLP.
- MLPEnsembleModel.predict() handles feature scaling internally; no apply_cleanup
  is called during inference.
- Incremental retraining (retrain_on_batch) is a no-op: skorch models do not
  support XGBoost-style warm-start continuation and retraining a full MLP each
  quarter during backtest would be prohibitively slow.
- RESULTS_DIR points to results_mlp/ (matching training_MLP.py).

Rebalance modes
---------------
fixed      Each calendar quarter, all stocks whose filing_date_used fell in that
           quarter are scored at the same time.  The entire batch is entered at
           the quarter-end close and exited at the following quarter-end close.
           Clean portfolio math; one unambiguous return per quarter.

staggered  Each stock is entered at its own filing_date_used close and exited at
           the close on its NEXT filing_date_used (capped by --max-hold-days).
           More realistic: the position expires when fresh information arrives,
           not after an arbitrary fixed window.
           Per-position alpha vs SPY is computed over each stock's own window.

Usage
-----
python backtest_MLP.py --experiment healthcare_mlp --sectors healthcare \\
                       --rebalance fixed --top-quantile 0.2 --bot-quantile 0.2 \\
                       --start 2021-01-01 --end 2024-06-30

python backtest_MLP.py --experiment healthcare_mlp --sectors healthcare \\
                       --rebalance staggered --top-quantile 0.2 --max-hold-days 548
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.notebook import tqdm

from training_util import *
from backtest_util import *
from training_MLP import MLPEnsembleModel, load_final_models

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results_mlp")

# ── Data loading ───────────────────────────────────────────────────────────────
def load_model_and_meta(experiment_name: str):
    """Load the MLP ensemble and its metadata from results_mlp/."""
    ensemble, meta = load_final_models(experiment_name)
    if ensemble is None:
        raise FileNotFoundError(
            f"No MLP models found for experiment '{experiment_name}' in {RESULTS_DIR}.\n"
            "Run training_MLP.py first to train and save the models."
        )
    sc = meta["scores"]
    print(f"Loaded MLP ensemble [{experiment_name}]  "
          f"({len(ensemble.models)} models | test IC={sc['test_ic']:.4f} "
          f"ICIR={sc['test_icir']:.4f})")
    return ensemble, meta

def retrain_on_batch_mlp(
    ensemble: MLPEnsembleModel,
    batch: pd.DataFrame,
    medians: pd.Series,
    max_epochs_finetune: int = 20,
) -> MLPEnsembleModel:
    for i, (model, scaler, feats) in enumerate(
        zip(ensemble.models, ensemble.scalers, ensemble.feature_lists)
    ):
        valid = batch.dropna(subset=["target"] + feats)
        if len(valid) < 50:
            print(f"[WARN] batch too short (len={len(valid)}, minimum: 50)")
            continue

        X_new = valid[feats].replace([np.inf, -np.inf], np.nan).fillna(medians.reindex(feats))
        y_new = rank_target_cross_sectionally(
            valid["target"].copy().set_axis(valid["fiscalDateEnding"].values)
        )

        X_sc  = scaler.transform(X_new).astype(np.float32)
        y_np  = y_new.values.astype(np.float32)

        orig_lr = model.lr
        model.set_params(warm_start=True,
                         max_epochs=max_epochs_finetune,
                         lr=orig_lr / 10)  # fine-tune at 1/10th of original training LR
        model.fit(X_sc, y_np)   # warm_start=True -> continues from current weights

        model.set_params(lr=orig_lr)  # restore original LR for future fine-tune steps
    return ensemble


# ── Core backtest ──────────────────────────────────────────────────────────────
def run_backtest(
    experiment_name: str,
    sectors: list[str] | None,
    top_q: float,
    bot_q: float,
    start_date: str,
    end_date: str,
    rebalance_mode: str,
    min_stocks: int,
    output_dir: Path,
    sector_etf: str | None = None,
) -> pd.DataFrame:

    model, meta = load_model_and_meta(experiment_name)

    df = load_dataset(sectors)
    if "ticker" not in df.columns:
        raise ValueError("'ticker' column required — regenerate parquets with get_metadata=True")

    feat_cols = [c for c in df.columns if c not in METADATA_COLS]

    # Collect all feature columns needed by the ensemble
    all_ensemble_feats = list({f for fl in model.feature_lists for f in fl})

    # Fit cleanup on data before the backtest window (used only for NaN/inf sanity)
    GAP      = pd.Timedelta(days=456)
    pre_mask = df["filing_date_used"] < pd.Timestamp(start_date) - GAP

    if pre_mask.sum() == 0:
        raise ValueError("No pre-backtest data to fit cleanup — move start_date later or check GAP")
    elif pre_mask.sum() < 50:
        print(f"[WARN] Only {pre_mask.sum()} rows before cutoff — cleanup stats may be noisy")

    _, _, medians = fit_cleanup(df[pre_mask], feat_cols)

    # Staggered: build next-filing lookup over the ENTIRE dataset
    filing_map = build_next_filing_map(df) if rebalance_mode == "staggered" else {}

    quarters = pd.period_range(start=start_date, end=end_date, freq="Q")
    print(f"\nBacktest  {quarters[0]} → {quarters[-1]}  ({len(quarters)} Qs)  "
          f"mode={rebalance_mode}  long={top_q:.0%}  short={bot_q:.0%}\n")

    if sector_etf is None and sectors and len(sectors) == 1:
        sector_etf = SECTOR_ETF_MAP.get(sectors[0].lower())
        if sector_etf is None:
            print(f"[WARN] No ETF mapping for sector '{sectors[0]}' - sector baseline disabled")
    elif sectors and len(sectors) > 1 and sector_etf is None:
        print("[WARN] Multiple sectors with no --sector-etf specified - sector baseline disabled")

    if sector_etf:
        print(f"Sector baseline ETF: {sector_etf}")

    quarter_records: list[dict] = []
    trade_records:   list[dict] = []

    prev_batch: pd.DataFrame | None = None

    for i, q in enumerate(tqdm(quarters[:-1], desc="Quarters")):
        q_start = q.start_time
        q_end   = q.end_time

        if rebalance_mode == "fixed" and prev_batch is not None:
            trainable_cutoff = q_start - pd.Timedelta(GAP_DAYS)

            matured_start = trainable_cutoff - pd.DateOffset(months=3)
            matured_batch = df[(df["filing_date_used"] > matured_start) &
                               (df["filing_date_used"] <= trainable_cutoff)].copy()
            
            if len(matured_batch) > 0:
                model = retrain_on_batch_mlp(model, matured_batch, medians)
            else:
                print(f"[retrain] No matured rows to retrain model on")

        batch = df[(df["filing_date_used"] >= q_start) &
                   (df["filing_date_used"] <= q_end)].copy()

        if len(batch) < min_stocks:
            tqdm.write(f"  {q}: {len(batch)} stocks < min {min_stocks} — skip")
            continue

        # ── Score & select legs ────────────────────────────────────────────────
        # MLPEnsembleModel handles scaling internally; pass raw feature values.
        # Replace inf with nan so scalers inside the ensemble are not disrupted.
        X_inf = batch[all_ensemble_feats].replace([np.inf, -np.inf], np.nan)
        # Fill remaining NaNs with column medians derived from pre-backtest data
        X_inf = X_inf.fillna(medians.reindex(X_inf.columns))
        batch = batch.reset_index(drop=True)
        X_inf = X_inf.reset_index(drop=True)
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
            next_q_end = quarters[i+1].end_time
            legs = legs.copy()
            legs["entry_dt"] = q_end
            legs["exit_dt"]  = next_q_end
        else:  # staggered
            legs = legs.copy()
            legs["entry_dt"] = legs["filing_date_used"]
            legs["exit_dt"]  = legs.apply(
                lambda r: next_filing_exit(
                            r["ticker"], r["filing_date_used"], filing_map
                        ),
                axis=1,
            )

        # ── Fetch prices & compute per-position returns ────────────────────────
        rows = []
        n_skipped = 0
        for _, leg in legs.iterrows():
            ticker   = leg["ticker"]
            entry_dt = leg["entry_dt"]
            exit_dt  = leg["exit_dt"]
            if exit_dt <= entry_dt:
                tqdm.write(f"    skip {ticker}: exit date {exit_dt.date()} <= entry {entry_dt.date()} (position still open)")
                n_skipped += 1
                continue
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
            sign    = 1 if leg["direction"] == "long" else -1

            # Per-position SPY alpha over the same window
            spy_in  = price_on("SPY", entry_dt)
            spy_out = price_on("SPY", exit_dt)
            spy_ret = (spy_out - spy_in) / spy_in if spy_in and spy_out else np.nan
            pos_alpha = (raw_ret - spy_ret) * sign if not np.isnan(spy_ret) else np.nan

            sector_in = price_on(sector_etf, entry_dt) if sector_etf else None
            sector_out = price_on(sector_etf, exit_dt) if sector_etf else None
            sector_ret = (sector_out - sector_in) / sector_in if sector_in and sector_out else np.nan
            sector_alpha = (raw_ret - sector_ret) * sign if not np.isnan(sector_ret) else np.nan

            rows.append({
                "quarter":       str(q),
                "ticker":        ticker,
                "direction":     leg["direction"],
                "entry_date":    entry_dt,
                "exit_date":     exit_dt,
                "hold_days":     (exit_dt - entry_dt).days,
                "entry_price":   round(p_in,  4),
                "exit_price":    round(p_out, 4),
                "raw_return":    raw_ret,
                "signed_return": raw_ret * sign,
                "spy_ret":       spy_ret,
                "sector_ret":    sector_ret,
                "sector_alpha":  sector_alpha,
                "pos_alpha":     pos_alpha,
                "score":         leg["score"],
                "target":        leg.get("target", np.nan),
            })
        trade_records.extend(rows)
        prev_batch = batch.copy()

        if not rows:
            tqdm.write(f"  {q}: no prices resolved — skip")
            continue

        qdf     = pd.DataFrame(rows)
        long_r  = qdf.loc[qdf.direction == "long",  "raw_return"].mean()
        short_r = qdf.loc[qdf.direction == "short", "raw_return"].mean() if bot_q > 0 else np.nan
        comb_r  = qdf["signed_return"].mean()
        alpha   = qdf["pos_alpha"].mean()

        # IC: predicted score vs actual raw return
        IC_VALID_THROUGH = pd.Period("2025Q2", freq="Q")

        cohort_ic = np.nan
        if "target" in batch.columns and q <= IC_VALID_THROUGH:
            cohort = batch[["score", "target"]].dropna()
            if len(cohort) >= 3:
                cohort_ic = spearmanr(cohort["score"], cohort["target"]).statistic
                cohort_ic = float(cohort_ic) if not np.isnan(cohort_ic) else np.nan
            else:
                print(f"[WARN] Batch size < 3, skipping Batch IC count for {q}")
        else:
            print(f"Quarter: {q} is after 2025Q2, can't derive Batch IC because target is unavailable (requires one year of future data)")

        spy_r = qdf["spy_ret"].mean()
        sector_r = qdf["sector_ret"].mean()
        sector_alpha = qdf["sector_alpha"].mean()

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
            "sector_ret":   sector_r,
            "sector_alpha": sector_alpha,
            "alpha":        alpha,
            "cohort_ic":    cohort_ic,
        })
        skip_str = f"  [{n_skipped} skipped]" if n_skipped else ""
        msg = (
            f"{q} "
            f"n={len(batch):3d}→{len(rows):3d}{skip_str} "
            f"long={long_r:+.2%} "
            f"{f'short={short_r:+.2%}' if bot_q > 0 else ''} "
            f"comb={comb_r:+.2%} "
            f"SPY≈{spy_r:+.2%} "
            f"{f'sector≈{sector_r:+.2%}' if sector_etf else ''} "
            f"alpha={alpha:+.2%} "
            f"IC(Batch)={cohort_ic:+.3f}"
        )
        tqdm.write(msg)

    if not quarter_records:
        print("\n[ERROR] No results — check dates, data, and model files.")
        return pd.DataFrame()

    results = pd.DataFrame(quarter_records)
    trades  = pd.DataFrame(trade_records)

    # ── Risk-free rates (^IRX T-bill) ──────────────────────────────────────────
    rf_map = fetch_rf_quarterly(start_date, df["filing_date_used"].max())
    results["rf_quarterly"] = results["quarter"].map(rf_map).fillna(0.0)
    if rf_map.empty:
        print("[WARN] rf_quarterly defaulting to 0 for all quarters")
    else:
        print(f"Risk-free rates loaded: mean={rf_map.mean():.3%}/qtr  "
              f"({rf_map.mean()*4:.2%} annualised)")

    # ── Summary stats ─────────────────────────────────────────────────────────
    def sharpe_ann(s: pd.Series) -> float:
        s = s.dropna()
        rf = results["rf_quarterly"].reindex(s.index).fillna(0.0)
        excess = s - rf
        return excess.mean() / excess.std() * np.sqrt(4) if excess.std() > 0 else np.nan

    def max_drawdown(cum: pd.Series) -> float:
        roll_max = (1 + cum).cummax()
        return float(((1 + cum) / roll_max - 1).min())

    # ── Cumulative returns ─────────────────────────────────────────────────────
    results["cum_combined"] = (1 + results["combined_ret"].fillna(0)).cumprod() - 1
    results["cum_long"]     = (1 + results["long_ret"].fillna(0)).cumprod() - 1
    results["cum_spy"]      = (1 + results["spy_ret"].fillna(0)).cumprod() - 1

    equity_strat = 1 + results["cum_combined"]
    equity_spy   = 1 + results["cum_spy"]

    results["drawdown"]     = equity_strat / equity_strat.cummax() - 1
    results["drawdown_spy"] = equity_spy / equity_spy.cummax() - 1

    if "sector_ret" in results.columns and results["sector_ret"].notna().any():
        results["cum_sector"] = (1 + results["sector_ret"].fillna(0)).cumprod() - 1
        equity_sector = 1 + results["cum_sector"]
        results["drawdown_sector"] = equity_sector / equity_sector.cummax() - 1

    ROLL = 4
    results["rolling_sharpe"] = results["combined_ret"].rolling(ROLL, min_periods=2).apply(
        sharpe_ann, raw=False
    )

    ic_col = "cohort_ic"
    ic_series = results[ic_col].where(
        results["quarter"].apply(lambda x: pd.Period(x, freq="Q") <= IC_VALID_THROUGH)
    )
    results["rolling_ic"] = ic_series.rolling(ROLL, min_periods=2).mean()

    ic_valid = results[results["quarter"].apply(lambda x: pd.Period(x, freq="Q") <= IC_VALID_THROUGH)]

    print("\n─── Backtest Summary (MLP) ──────────────────────────────────────────")
    print(f"  Mode                  : {rebalance_mode}")
    mean_rf_annual = results["rf_quarterly"].mean() * 4
    print(f"  Risk-free rate (avg)  : {mean_rf_annual:.2%}/yr  (^IRX T-bill)")
    print(f"  Quarters with results : {len(results)} / {len(quarters)}")
    print(f"  Mean Batch IC        : {ic_valid['cohort_ic'].mean():+.4f}  (std={ic_valid['cohort_ic'].std():.4f})")
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
    print(f"    Annualised Sharpe   : {sharpe_ann(results['spy_ret']):.3f}")
    print(f"    Total return        : {results['cum_spy'].iloc[-1]:+.2%}")
    print(f"    Max drawdown        : {max_drawdown(results['cum_spy']):.2%}")
    print(f"\n  Mean position-level alpha : {results['alpha'].mean():+.2%}")
    if sector_etf and "sector_ret" in results.columns:
        print(f"\n  Sector benchmark ({sector_etf})")
        print(f"    Mean qtrly return   : {results['sector_ret'].mean():+.2%}")
        print(f"    Total return        : {results['cum_sector'].iloc[-1]:+.2%}")
        print(f"\n  Mean sector-level alpha : {results['sector_alpha'].mean():+.2%}")
    if rebalance_mode == "staggered":
        print(f"  Mean hold period (days)   : {trades['hold_days'].mean():.0f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    tag    = f"{experiment_name}_{rebalance_mode}"
    q_path = output_dir / f"backtest_mlp_{tag}_quarterly.csv"
    t_path = output_dir / f"backtest_mlp_{tag}_trades.csv"
    results.to_csv(q_path, index=False)
    trades.to_csv(t_path, index=False)
    print(f"\nSaved → {q_path}")
    print(f"Saved → {t_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(6, 1, figsize=(12, 22), constrained_layout=True)

    fig.suptitle(
        f"Backtest (MLP): {experiment_name} [{rebalance_mode}] ({quarters[0]} - {quarters[-1]})",
        fontsize=13,
        fontweight="bold"
    )

    # 1. Cumulative returns
    ax = axes[0]

    final_combined = results["cum_combined"].iloc[-1] * 100
    final_spy = results["cum_spy"].iloc[-1] * 100

    ax.plot(results["q_start"], results["cum_combined"] * 100,
            label=f"Strategy  {final_combined:+.1f}%", color="steelblue", lw=2)
    if bot_q > 0:
        final_long = results["cum_long"].iloc[-1] * 100
        ax.plot(results["q_start"], results["cum_long"] * 100,
                label=f"Long only  {final_long:+.1f}%", color="seagreen", lw=1.5, ls="--")
    ax.plot(results["q_start"], results["cum_spy"] * 100,
            label=f"SPY  {final_spy:+.1f}%", color="gray", lw=1.5, ls=":")
    if "cum_sector" in results.columns and results["cum_sector"].notna().any():
        final_sector = results["cum_sector"].dropna().iloc[-1] * 100
        ax.plot(results["q_start"], results["cum_sector"] * 100,
                label=f"Sector ({sector_etf})  {final_sector:+.1f}%",
                color="darkorange", lw=1.5, ls='-.')
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
    if "sector_ret" in results.columns and results["sector_ret"].notna().any():
        ax2.plot(x, results["sector_ret"] * 100, "s-.", color="darkorange",
                 markersize=4, label=f"Sector ({sector_etf})", lw=1)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Return (%)")
    ax2.set_title("Quarterly Returns")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. IC per quarter
    ax3 = axes[2]
    colors = ["seagreen" if v >= 0 else "salmon" for v in ic_series.fillna(0)]
    ax3.bar(x, ic_series, width=width, color=colors, alpha=0.8)
    ax3.axhline(0, color="black", lw=0.5)
    ax3.axhline(ic_series.mean(), color="steelblue", lw=1.5,
                ls="--", label=f"Mean Batch IC={ic_series.mean():+.3f}")
    ax3.set_ylabel("IC (Spearman)")
    ax3.set_title("Per-Quarter IC (score vs realised return)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Drawdown
    ax4 = axes[3]
    ax4.fill_between(x, results["drawdown"] * 100, 0,
                    color="crimson", alpha=0.4)
    ax4.plot(x, results["drawdown"] * 100,
            color="crimson", lw=2, label="Strategy")
    ax4.plot(x, results["drawdown_spy"] * 100,
            color="darkorange", lw=2, label="SPY")
    if "drawdown_sector" in results.columns and results["drawdown_sector"].notna().any():
        ax4.plot(x, results["drawdown_sector"] * 100,
                color="seagreen", lw=2, label=f"Sector ({sector_etf})")
    ax4.axhline(0, color="black", lw=0.5)
    ax4.set_ylabel("Drawdown (%)")
    ax4.set_title("Drawdown comparison")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Rolling Sharpe
    ax5 = axes[4]
    ax5.plot(x, results["rolling_sharpe"],
             color="steelblue", lw=2, label=f"{ROLL}Q Rolling Sharpe")
    ax5.axhline(0, color="black", lw=0.5)
    ax5.set_ylabel("Sharpe (annualised)")
    ax5.set_title(f"Rolling {ROLL}-Quarter Sharpe Ratio")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # 6. Rolling IC
    ax6 = axes[5]
    ax6.plot(x, results["rolling_ic"],
             color="darkorchid", lw=2, label=f"{ROLL}Q Rolling Mean IC")
    ax6.axhline(0, color="black", lw=0.5)
    ax6.axhline(results[ic_col].mean(), color="gray", lw=1, ls=":",
                label=f"Overall mean={results[ic_col].mean():+.3f}")
    ax6.set_ylabel("IC (Spearman)")
    ax6.set_title(f"Rolling {ROLL}-Quarter Mean Batch IC")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    for ax_ in axes:
        ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        plt.setp(ax_.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    plot_path = output_dir / f"backtest_mlp_{tag}_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot   → {plot_path}")
    plt.show()

    return results


# ── Notebook usage ────────────────────────────────────────────────────────────
# Copy the cells below into a notebook.  `import backtest_MLP` must resolve —
# either run from the same directory or add it to sys.path.
#
# Cell 1 — parameters
# ──────────────────────────────────────────────────────────────────────────────
# from pathlib import Path
# import backtest_MLP
#
# EXPERIMENT    = "healthcare_mlp"       # matches results_mlp/final_<NAME>_mlp_*.joblib
# SECTORS       = ["healthcare"]         # None = all sectors
# TOP_QUANTILE  = 0.2
# BOT_QUANTILE  = 0.0                    # 0 = long-only
# REBALANCE     = "fixed"                # "fixed" | "staggered"
# MIN_STOCKS    = 3
# OUTPUT_DIR    = Path("backtest_results_mlp")
#
# Cell 2 — derive test window from data (matches training pipeline split)
# ──────────────────────────────────────────────────────────────────────────────
# df = backtest_MLP.load_dataset(SECTORS)
# START, END = backtest_MLP.infer_test_window(df, test_months=24)
# print(f"Test window: {START}  →  {END}")
#
# Cell 3 — run
# ──────────────────────────────────────────────────────────────────────────────
# results = backtest_MLP.run_backtest(
#     experiment_name = EXPERIMENT,
#     sectors         = SECTORS,
#     top_q           = TOP_QUANTILE,
#     bot_q           = BOT_QUANTILE,
#     start_date      = START,
#     end_date        = END,
#     rebalance_mode  = REBALANCE,
#     min_stocks      = MIN_STOCKS,
#     output_dir      = OUTPUT_DIR,
# )


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Quarterly fundamental backtest using MLP ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Rebalance modes:\n"
            "  fixed      Enter entire batch at quarter-end; exit at next quarter-end.\n"
            "  staggered  Enter each stock at filing_date_used; exit at its next filing\n"
            "             (capped by next quarter-end). More realistic signal expiry.\n"
        ),
    )
    ap.add_argument("--experiment",    default="healthcare_mlp",
                    help="Name matching results_mlp/final_<NAME>_mlp_*.joblib")
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
    ap.add_argument("--min-stocks",    type=int, default=3,
                    help="Skip a quarter if fewer stocks filed (default 3)")
    ap.add_argument("--output-dir",    default="backtest_results_mlp",
                    help="Directory for CSV and PNG outputs (default backtest_results_mlp)")
    ap.add_argument("--sector-etf",    default=None,
                    help="Sector ETF ticker for baseline, e.g. XLV. "
                         "Auto-resolved from SECTOR_ETF_MAP for single-sector runs.")
    args = ap.parse_args()

    run_backtest(
        experiment_name=args.experiment,
        sectors=args.sectors,
        top_q=args.top_quantile,
        bot_q=args.bot_quantile,
        start_date=args.start,
        end_date=args.end,
        rebalance_mode=args.rebalance,
        min_stocks=args.min_stocks,
        output_dir=Path(args.output_dir),
        sector_etf=args.sector_etf,
    )
