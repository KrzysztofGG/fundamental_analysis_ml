#!/usr/bin/env python3
"""
training.py — Training pipeline for fundamental analysis XGBoost ensemble.

Usage in notebook
-----------------
from training import training_pipeline, export_parquets_for_training

df = export_parquets_for_training()
scores, ensemble, best_feats = training_pipeline(df, STUDY_STORAGE, False, "full_dataset_ic")
"""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import optuna
from scipy.stats import spearmanr, rankdata
from sklearn.base import clone
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from tqdm.notebook import tqdm

from training_util import *
from ensemble import EnsembleModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results_v2")
CACHE_DIR   = Path("cache")
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_TRIALS_FULL  = 100
TOP_K          = 100
RANDOM_STATE   = 42

GAP_DAYS     = 456   # 12-month return horizon + 3-month filing buffer (≥ 365+91)
QUARTER_DAYS = 91    # ~3 months per val slice

BASE_METHODS = ["shap", "rfe", "permutation"]

DEVICE = "cuda" if check_gpu() else "cpu"

def generate_parquets_from_csv(data_dir: str = "data_quarterly", max_missing_pct: float = 0.2):
    sectors = [f for f in os.listdir(data_dir) if not f.endswith(".json")]
    for sector in tqdm(sectors):
        df = _export_csv_sector(data_dir, sector)
        output_dir = Path(data_dir + "_parquet")
        output_dir.mkdir(parents=True, exist_ok=True)
        df = normalize_schema(df)
        metadata_cols = ["ticker", "sector", "industry", "fiscalDateEnding", "filing_date_used"]
        feature_cols  = [c for c in df.columns if c not in metadata_cols]
        mask = df[feature_cols].isnull().mean(axis=1) <= max_missing_pct
        print(f"{sector}: dropping {(~mask).sum()}/{len(df)} rows with >{max_missing_pct:.0%} missing data")
        df[mask].to_parquet(output_dir / f"{sector}.parquet", index=False)
        print(f"Generated data for {sector}")


def _export_csv_sector(data_dir: str, sector: str) -> pd.DataFrame:
    with open(os.path.join(data_dir, "metadata.json")) as f:
        metadata = json.load(f)
    frames = []
    sector_path = os.path.join(data_dir, sector)
    for file in os.listdir(sector_path):
        if not file.endswith(".csv"):
            continue
        df     = pd.read_csv(os.path.join(sector_path, file))
        ticker = file[: file.find(".")]
        meta   = metadata[ticker]
        meta["ticker"] = ticker
        df_meta = pd.DataFrame([meta] * len(df), index=df.index)
        frames.append(pd.concat([df, df_meta], axis=1))
    if frames:
        dataset = pd.concat(frames, ignore_index=True)
        dataset["fiscalDateEnding"] = pd.to_datetime(dataset["fiscalDateEnding"])
        return dataset
    return pd.DataFrame()

def method_shap(X_tr, y_tr, base_reg, top_k=TOP_K):
    model = clone(base_reg).fit(X_tr, y_tr)
    sv    = shap.TreeExplainer(model).shap_values(X_tr)
    idx   = np.argsort(np.abs(sv).mean(axis=0))[-top_k:]
    return X_tr.columns[idx].tolist()


def method_rfe(X_tr, y_tr, base_reg, top_k=TOP_K):
    rfe = RFE(clone(base_reg), n_features_to_select=top_k, step=50, verbose=0)
    rfe.fit(X_tr, y_tr)
    return X_tr.columns[rfe.support_].tolist()


def method_permutation(X_tr, y_tr, base_reg, top_k=TOP_K, n_repeats=5):
    model = clone(base_reg).fit(X_tr, y_tr)
    res   = permutation_importance(
        model, X_tr, y_tr, n_repeats=n_repeats,
        random_state=RANDOM_STATE, scoring=ic_scorer,
    )
    idx = np.argsort(res.importances_mean)[-top_k:]
    return X_tr.columns[idx].tolist()


def load_or_run_feature_selection(
    X_train, y_reg_train, base_reg,
    force: bool = False,
    experiment_name: str = "",
) -> dict[str, list[str]]:
    cache_name = "feature_selection"
    if experiment_name:
        cache_name += f"_{experiment_name}"
    cache_file = CACHE_DIR / f"{cache_name}.json"
    print(f"Cache file: {cache_file}")

    method_fns = {
        "shap":        lambda: method_shap(X_train, y_reg_train, base_reg),
        "rfe":         lambda: method_rfe(X_train, y_reg_train, base_reg),
        "permutation": lambda: method_permutation(X_train, y_reg_train, base_reg),
    }

    cached = {}
    if not force and cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        if cached:
            print("Loaded from cache:")
            for name, feats in cached.items():
                print(f"  {name:12s} -> {len(feats)} features")

    pending = {n: fn for n, fn in method_fns.items() if n not in cached}

    if not pending:
        print("All methods already cached, nothing to run.")
        return cached

    print(f"Running {len(pending)}/{len(method_fns)} method(s): {', '.join(pending)}")
    methods     = dict(cached)
    total_start = time.time()

    with tqdm(total=len(pending), desc="Feature selection", unit="method") as pbar:
        completed_times = []
        for method_name, fn in pending.items():
            pbar.set_postfix_str(f"running {method_name}...")
            t0      = time.time()
            feats   = fn()
            elapsed = time.time() - t0
            methods[method_name] = feats
            completed_times.append(elapsed)

            with open(cache_file, "w") as f:
                json.dump(methods, f, indent=2)

            remaining = len(pending) - len(completed_times)
            eta       = (sum(completed_times) / len(completed_times)) * remaining
            pbar.set_postfix_str(
                f"{method_name} done — {len(feats)} feats, took {elapsed:.0f}s, ETA {eta:.0f}s"
            )
            pbar.update(1)

    print(f"\nFeature selection complete in {(time.time() - total_start):.0f}s")
    for (name, feats), t in zip({n: methods[n] for n in pending}.items(), completed_times):
        print(f"  {name:<12}  {len(feats):>8} feats  {t:>7.1f}s")
    print(f"Saved to {cache_file}")
    return methods


def quick_cv_ic_gapped(
    feats: list[str],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    base_reg,
    gap_days: int = GAP_DAYS,
    train_years: int = 5,
    val_quarters: int = 4,
) -> tuple[float, float, float]:
    """Walk-forward IC CV with fixed-width training window and explicit gap."""
    dates     = X_tr.index.sort_values().unique()
    bin_index = assign_quarter_bins(dates)

    unique_bins = np.unique(bin_index)
    bin_start: dict[int, pd.Timestamp] = {}
    bin_end:   dict[int, pd.Timestamp] = {}
    for d, b in zip(dates, bin_index):
        if b not in bin_start or d < bin_start[b]:
            bin_start[b] = d
        if b not in bin_end or d > bin_end[b]:
            bin_end[b] = d

    first_val_start = dates[0] + pd.DateOffset(years=train_years) + pd.Timedelta(days=gap_days)
    val_bins = unique_bins[
        np.array([bin_start[b] for b in unique_bins]) >= first_val_start
    ]

    if len(val_bins) == 0:
        print("[ERROR] No valid val bins found — dataset too short for given train_years + gap_days")
        return 0.0, 0.0, 0.0

    grouped = [val_bins[i: i + val_quarters]
               for i in range(0, len(val_bins) - val_quarters + 1, val_quarters)]

    print(f"[INFO] {len(grouped)} val folds x {val_quarters}Q each, "
          f"derived from {len(val_bins)} eligible bins "
          f"({bin_start[val_bins[0]].strftime('%Y-%m-%d')} → "
          f"{bin_end[val_bins[-1]].strftime('%Y-%m-%d')})")

    fold_ics, fold_rmses = [], []

    for i, bin_group in enumerate(grouped):
        val_dates    = dates[np.isin(bin_index, bin_group)]
        val_cutoff   = val_dates.min()
        train_cutoff = val_cutoff - pd.Timedelta(days=gap_days)
        train_start  = train_cutoff - pd.DateOffset(years=train_years)
        train_dates  = dates[(dates >= train_start) & (dates < train_cutoff)]

        mask_tr  = X_tr.index.isin(train_dates)
        mask_val = X_tr.index.isin(val_dates)

        if mask_tr.sum() < 50 or mask_val.sum() < 5:
            print(f"[WARN] fold {i+1}: n_train={mask_tr.sum()} n_val={mask_val.sum()} — skipping")
            continue

        m     = clone(base_reg).fit(X_tr.loc[mask_tr, feats], y_tr.loc[mask_tr])
        preds = m.predict(X_tr.loc[mask_val, feats])
        ic, _ = spearmanr(y_tr.loc[mask_val].values, preds)

        if np.isnan(ic):
            print(f"[WARN] fold {i+1}: IC is NaN — skipping")
            continue

        fold_ics.append(ic)
        fold_rmses.append(float(np.sqrt(np.mean((y_tr.loc[mask_val].values - preds) ** 2))))

        q_label = " + ".join(str(pd.Period(ordinal=int(b), freq="Q")) for b in bin_group)
        print(f"    fold {i+1:2d} [{q_label}] "
              f"val {val_cutoff.strftime('%Y-%m-%d')} → {val_dates.max().strftime('%Y-%m-%d')} "
              f"| train {train_dates[0].strftime('%Y-%m-%d')} → {train_cutoff.strftime('%Y-%m-%d')} "
              f"| n_train={mask_tr.sum():5d} n_val={mask_val.sum():4d} "
              f"| IC={fold_ics[-1]:.4f} RMSE={fold_rmses[-1]:.4f}")

    if not fold_ics:
        return 0.0, 0.0, 0.0

    mean_ic, std_ic, mean_rmse = np.mean(fold_ics), np.std(fold_ics), np.mean(fold_rmses)
    print(f"\n[RESULT] {len(fold_ics)} folds | IC={mean_ic:.4f} ± {std_ic:.4f} | RMSE={mean_rmse:.4f}")
    return mean_ic, std_ic, mean_rmse

# ── Optuna objective ───────────────────────────────────────────────────────────
def make_objective(features, X_train, y_train, gap_days=GAP_DAYS):
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
            max_depth         = trial.suggest_int("max_depth", 3, 10),
            learning_rate     = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 20),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            gamma             = trial.suggest_float("gamma", 0.0, 5.0),
            tree_method       = "hist",
            device            = DEVICE,
            random_state      = RANDOM_STATE,
            verbosity         = 0,
        )
        trial_reg = xgb.XGBRegressor(**params)
        mean_ic, std_ic, mean_rmse = quick_cv_ic_gapped(
            features, X_train, y_train, base_reg=trial_reg, gap_days=gap_days,
        )
        icir = mean_ic / std_ic if std_ic > 0 else 0.0
        trial.set_user_attr("val_ic",     mean_ic)
        trial.set_user_attr("val_icir",   icir)
        trial.set_user_attr("val_ic_std", std_ic)
        trial.set_user_attr("val_rmse",   mean_rmse)
        return -mean_ic

    return objective


# ── Save / load ────────────────────────────────────────────────────────────────
def _ensemble_member_path(experiment_name: str, method_name: str) -> Path:
    prefix = f"final_{experiment_name}" if experiment_name else "final"
    return RESULTS_DIR / f"{prefix}_reg_{method_name}.ubj"

def _meta_path(experiment_name: str) -> Path:
    prefix = f"final_{experiment_name}" if experiment_name else "final"
    return RESULTS_DIR / f"{prefix}_meta.json"


def save_final_models(
    best_method: str,
    best_feats: list[str],
    scores: dict,
    best_params: dict,
    experiment_name: str = "",
    ensemble: EnsembleModel | None = None,
    ensemble_methods: list[str] | None = None,
    ensemble_params: dict | None = None,
):
    RESULTS_DIR.mkdir(exist_ok=True)
    meta_path = _meta_path(experiment_name)

    def _clean(v):
        return v.item() if hasattr(v, "item") else v

    if ensemble is not None:
        for method_name, model in zip(ensemble_methods, ensemble.models):
            model.save_model(_ensemble_member_path(experiment_name, method_name))

        meta = {
            "mode":             "ensemble",
            "ensemble_methods": ensemble_methods,
            "ensemble_weights": [_clean(w) for w in ensemble.weights],
            "ensemble_feats":   {m: f for m, f in zip(ensemble_methods, ensemble.feature_lists)},
            "ensemble_params":  {m: {k: _clean(v) for k, v in p.items()}
                                 for m, p in (ensemble_params or {}).items()},
            "best_method":      best_method,
            "best_feats":       best_feats,
            "best_params":      {k: _clean(v) for k, v in best_params.items()},
            "scores":           {k: _clean(v) for k, v in scores.items()},
        }
    else:
        raise ValueError("save_final_models requires an EnsembleModel")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Models saved → {meta_path.name}")


def load_final_models(study_name: str = ""):
    meta_path = _meta_path(study_name)

    if not meta_path.exists():
        label = f" [{study_name}]" if study_name else ""
        print(f"No saved models found{label}, will train from scratch.")
        return None, None

    with open(meta_path) as f:
        meta = json.load(f)

    label = f" [{study_name}]" if study_name else ""

    models, feats_list, weights = [], [], []
    for method_name in meta["ensemble_methods"]:
        m = xgb.XGBRegressor()
        m.load_model(_ensemble_member_path(study_name, method_name))
        models.append(m)
        feats_list.append(meta["ensemble_feats"][method_name])
        idx = meta["ensemble_methods"].index(method_name)
        weights.append(meta["ensemble_weights"][idx])

    ensemble = EnsembleModel(models, feats_list, weights)
    print(f"Loaded ensemble{label} ({len(models)} models, methods={meta['ensemble_methods']})")
    return ensemble, meta


# ── Main pipeline ──────────────────────────────────────────────────────────────
def training_pipeline(
    df: pd.DataFrame,
    optuna_storage_name: str,
    force_feature_selection: bool = False,
    experiment_name: str = "",
) -> tuple[dict, EnsembleModel, list[str]]:
    """
    Full training pipeline: data split → feature selection → per-method Optuna
    → ensemble of 3 XGBoost models weighted by CV IC → scorecard → save.

    Returns (scores, ensemble, best_feats_of_winning_method).
    """
    print("\n─── Load & Clean Data ────────────────────────────────────")

    TEST_MONTHS = 24
    GAP         = pd.Timedelta(days=GAP_DAYS)
    data_end    = df["fiscalDateEnding"].max()

    test_end     = data_end
    test_start   = data_end - pd.DateOffset(months=TEST_MONTHS)
    train_cutoff = test_start - GAP

    print(f"Train pool : up to  {train_cutoff.strftime('%Y-%m-%d')}")
    print(f"Gap        : {GAP.days}d  →  {test_start.strftime('%Y-%m-%d')}")
    print(f"Test       : {test_start.strftime('%Y-%m-%d')}  →  {test_end.strftime('%Y-%m-%d')}")
    print(f"Train rows : {(df['fiscalDateEnding'] <= train_cutoff).sum()}")
    print(f"Test rows  : {((df['fiscalDateEnding'] >= test_start) & (df['fiscalDateEnding'] <= test_end)).sum()}")

    train_df = df[df["fiscalDateEnding"] <= train_cutoff].copy()
    test_df  = df[(df["fiscalDateEnding"] >= test_start) &
                  (df["fiscalDateEnding"] <= test_end)].copy()

    X_train_raw, y_reg_train_raw = split_target_with_date_index(train_df)
    X_test_raw,  y_reg_test_raw  = split_target_with_date_index(test_df)

    y_reg_train = rank_target_cross_sectionally(y_reg_train_raw)
    y_reg_test  = rank_target_cross_sectionally(y_reg_test_raw)

    print(f"\nSplit sizes → train: {len(X_train_raw):,}  test: {len(X_test_raw):,}")

    X_train, X_test = cleanup_base(X_train_raw, X_test_raw)
    print(f"Features after variance threshold: {X_train.shape[1]}")

    base_reg = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", device=DEVICE,
        random_state=RANDOM_STATE, verbosity=0,
    )

    # ── Feature selection ─────────────────────────────────────────────────────
    print("\n─── Feature selection ────────────────────────────────────")
    methods = load_or_run_feature_selection(
        X_train, y_reg_train, base_reg,
        force_feature_selection, experiment_name,
    )

    print(f"\n  {'method':<12}  {'n_feats':>7}  {'mean_IC':>8}  {'std_IC':>7}  {'ICIR':>6}  {'RMSE':>7}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*7}")
    for name, feats in methods.items():
        mean_ic, std_ic, mean_rmse = quick_cv_ic_gapped(feats, X_train, y_reg_train, base_reg)
        icir = mean_ic / std_ic if std_ic > 0 else 0.0
        print(f"  {name:<12}  {len(feats):>7}  {mean_ic:>8.4f}  "
              f"{std_ic:>7.4f}  {icir:>6.3f}  {mean_rmse:>7.4f}")

    # ── Optuna pass — one study per method ───────────────────────────────────
    print("\n─── Optuna pass (all methods) ────────────────────────────")
    full_studies: dict[str, optuna.Study] = {}

    for name in tqdm(BASE_METHODS, desc="Methods", unit="method", position=0):
        feats      = methods[name]
        study_name = f"{experiment_name}_final_{name}"
        study = optuna.create_study(
            direction="minimize", study_name=study_name,
            storage=optuna_storage_name, load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )

        remaining = N_TRIALS_FULL - len(study.trials)
        if remaining <= 0:
            print(f"\n  {name}: already complete ({len(study.trials)} trials), skipping")
            full_studies[name] = study
            continue

        print(f"\n  {name}: {len(study.trials)} trials done, running {remaining} more")
        with tqdm(total=remaining, desc=f"  {name}", unit="trial",
                  position=1, leave=False) as bar:
            def _cb(study, trial, _bar=bar):
                _bar.set_postfix_str(
                    f"best={study.best_value:.4f}  "
                    f"IC={trial.user_attrs.get('val_ic', float('nan')):.4f}"
                )
                _bar.update(1)

            study.optimize(
                make_objective(feats, X_train, y_reg_train),
                n_trials=remaining, callbacks=[_cb], show_progress_bar=False,
            )

        t = study.best_trial
        print(f"  {name:<12}  best IC={t.user_attrs['val_ic']:.4f}  "
              f"RMSE={t.user_attrs['val_rmse']:.4f}")
        full_studies[name] = study

    assert X_train.index.max() <= train_cutoff, (
        f"Trainval bleeds into test gap: "
        f"{X_train.index.max().date()} > {train_cutoff.date()}"
    )

    # ── Train ensemble members, each with its own best_params ─────────────────
    print("\n─── Training ensemble models ─────────────────────────────")
    ensemble_models:  list[xgb.XGBRegressor] = []
    ensemble_feats:   list[list[str]]         = []
    ensemble_weights: list[float]             = []
    ensemble_params:  dict[str, dict]         = {}

    best_method = min(full_studies, key=lambda k: full_studies[k].best_value)
    best_feats  = methods[best_method]
    best_params = full_studies[best_method].best_params

    for name in BASE_METHODS:
        feats   = methods[name]
        params  = full_studies[name].best_params
        ic_val  = full_studies[name].best_trial.user_attrs.get("val_ic", 0.0)
        weight  = max(ic_val, 0.0)

        reg = xgb.XGBRegressor(
            **params, tree_method="hist", device=DEVICE,
            random_state=RANDOM_STATE, verbosity=0,
        )
        reg.fit(X_train[feats], y_reg_train)
        ensemble_models.append(reg)
        ensemble_feats.append(feats)
        ensemble_weights.append(weight)
        ensemble_params[name] = params
        print(f"  {name:<12}  {len(feats):>3} feats  CV IC={ic_val:.4f}  weight={weight:.4f}")

    if sum(ensemble_weights) == 0:
        print("[WARN] All weights zero — falling back to equal weights")
        ensemble_weights = [1.0] * len(BASE_METHODS)

    ensemble = EnsembleModel(ensemble_models, ensemble_feats, ensemble_weights)

    all_feats = list({f for fl in ensemble_feats for f in fl})
    scores = compute_scorecard(ensemble, X_test[all_feats], y_reg_test, split_name="test")

    print("\n─── Test set results ─────────────────────────────────────")
    print(f"  Ensemble : {BASE_METHODS}")
    print(f"  Weights  : {[f'{w:.3f}' for w in ensemble.weights]}")
    print(f"\n  Regression (ranking quality)")
    print(f"    Mean period IC : {scores['test_ic']:.4f}   ← primary metric, target >0.05")
    print(f"    ICIR           : {scores['test_icir']:.4f}   ← consistency, target >0.5")
    print(f"    IC std         : {scores['test_ic_std']:.4f}   ← lower = more stable")
    print(f"    RMSE           : {scores['test_rmse']:.4f}   ← secondary")
    print(f"    R²             : {scores['test_r2']:.4f}")

    save_final_models(
        best_method=best_method,
        best_feats=best_feats,
        scores=scores,
        best_params=best_params,
        experiment_name=experiment_name,
        ensemble=ensemble,
        ensemble_methods=BASE_METHODS,
        ensemble_params=ensemble_params,
    )

    return scores, ensemble, best_feats
