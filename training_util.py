import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import root_mean_squared_error, r2_score

def rank_target_cross_sectionally(y: pd.Series) -> pd.Series:
    """
    Rank target within each fiscal quarter (date bin).
    Each stock gets a 0->1 rank relative to peers reporting
    in the same quarter. Leak-free by construction.
    
    y must have a fiscalDateEnding DatetimeIndex.
    """
    bins = pd.PeriodIndex(y.index, freq='Q').asi8
    ranked = y.copy().astype(float)
    for b in np.unique(bins):
        mask = bins == b
        if mask.sum() < 2:
            continue
        ranked.iloc[mask] = rankdata(y.iloc[mask]) / mask.sum()
    return ranked

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = ["fiscalDateEnding", "filing_date_used"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in df.columns:
        if col not in date_cols:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    for col in ["ticker", "sector", "industry"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df

def export_parquets_for_training(
    data_dir: str = "data_quarterly_parquet",
    sectors: list | None = None,
    get_metadata: bool = False,
) -> pd.DataFrame:
    start = time.time()

    if sectors is None:
        sectors = [f for f in os.listdir(data_dir)]
    else:
        sectors = [f"{name}.parquet" for name in sectors]

    files   = [Path(data_dir) / sector for sector in sectors]
    dataset = pd.read_parquet(files)
    dataset["fiscalDateEnding"] = pd.to_datetime(dataset["fiscalDateEnding"])
    dataset = dataset.sort_values("fiscalDateEnding").reset_index(drop=True)

    if not get_metadata:
        dataset = dataset.drop(columns=["sector", "industry", "ticker"], errors="ignore")

    print(f"Dataset load time: {(time.time() - start):.2f}s")
    return dataset

def split_target_with_date_index(df: pd.DataFrame):
    """Drop target/filing_date_used; set fiscalDateEnding as index."""
    X = df.drop(columns=["target", "filing_date_used"], errors="ignore")
    X = X.set_index("fiscalDateEnding")
    y = df["target"].copy()
    y.index = X.index
    return X, y

def cleanup_base(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit on train, apply to test. Clips, fills NaN, drops near-zero-variance."""
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan)

    lower   = X_train.quantile(0.01)
    upper   = X_train.quantile(0.99)
    X_train = X_train.clip(lower=lower, upper=upper, axis=1)
    X_test  = X_test.clip(lower=lower, upper=upper, axis=1)

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test  = X_test.fillna(medians)

    selector = VarianceThreshold(threshold=1e-5)
    mask     = selector.fit(X_train).get_support()
    cols     = X_train.columns[mask]
    return X_train[cols], X_test[cols]

def assign_quarter_bins(dates: pd.DatetimeIndex) -> np.ndarray:
    return pd.PeriodIndex(dates, freq="Q").asi8

def compute_scorecard(reg_model, X, y_reg, split_name="val"):
    reg_preds = reg_model.predict(X)
    preds_s   = pd.Series(reg_preds, index=y_reg.index)
    bins      = pd.PeriodIndex(y_reg.index, freq="Q").asi8

    period_ics = []
    for b in np.unique(bins):
        mask = bins == b
        if mask.sum() < 5:
            continue
        ic, _ = spearmanr(y_reg.values[mask], preds_s.values[mask])
        if not np.isnan(ic):
            period_ics.append(ic)

    mean_ic = float(np.mean(period_ics)) if period_ics else 0.0
    std_ic  = float(np.std(period_ics))  if period_ics else 0.0
    icir    = mean_ic / std_ic if std_ic > 0 else 0.0
    global_ic, _ = spearmanr(y_reg, reg_preds)

    return {
        f"{split_name}_ic":        mean_ic,
        f"{split_name}_icir":      icir,
        f"{split_name}_ic_std":    std_ic,
        f"{split_name}_ic_global": global_ic,
        f"{split_name}_rmse":      root_mean_squared_error(y_reg, reg_preds),
        f"{split_name}_r2":        r2_score(y_reg, reg_preds),
    }

def ic_scorer(estimator, X, y):
    preds = estimator.predict(X)
    ic, _ = spearmanr(y, preds)
    return ic if not np.isnan(ic) else 0.0

def check_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            print(f"GPU detected: {result.stdout.strip()}")
            return True
    except Exception:
        pass
    print("No GPU detected — falling back to CPU")
    return False