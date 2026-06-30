#!/usr/bin/env python3
"""
training_MLP.py — Training pipeline for fundamental analysis MLP ensemble (GPU).

Identical structure to training.py but uses a skorch NeuralNetRegressor
(PyTorch backend) instead of XGBoost. Key differences vs training.py:
  - skorch wraps a PyTorch nn.Module → GPU via device="cuda"
  - Features are StandardScaler-normalised inside every CV fold and at final fit
  - Feature selection: permutation (proxy MLP), mutual information, Lasso coefs
  - Optuna search space is MLP-specific (layers, width, lr, weight_decay, dropout)
  - Models are saved/loaded with joblib (skorch model + fitted scaler per member)

Usage in notebook
-----------------
df = export_parquets_for_training()
scores, ensemble, best_feats = training_pipeline_mlp(df, STUDY_STORAGE, False, "mlp_v1")
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from scipy.stats import spearmanr, rankdata
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, GradientNormClipping
from skorch.dataset import ValidSplit
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm

from training_util import *

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results_mlp")
RESULTS_DIR.mkdir(exist_ok=True)

BASE_METHODS = ["permutation", "mutual_info", "lasso"]

def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train, apply to test. Returns scaled frames + fitted scaler."""
    scaler  = StandardScaler()
    Xtr_sc  = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    Xte_sc  = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return Xtr_sc, Xte_sc, scaler

# ── PyTorch MLP module ─────────────────────────────────────────────────────────
class _MLPModule(nn.Module):
    """Configurable MLP used by skorch NeuralNetRegressor."""
    def __init__(self, n_input: int, hidden_layer_sizes: tuple, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        in_size = n_input
        for h in hidden_layer_sizes:
            layers += [
                nn.Linear(in_size, h), 
                nn.BatchNorm1d(h),
                nn.LeakyReLU(negative_slope=0.01)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(1)


def _make_skorch_net(n_input: int, hidden_layer_sizes: tuple, dropout: float,
                     lr: float, weight_decay: float, batch_size: int,
                     max_epochs: int,
                     use_early_stopping: bool = False,
                     use_warm_start: bool = False,
                     val_fraction: float = 0.15,
                     ) -> NeuralNetRegressor:
    """Build a skorch NeuralNetRegressor with the given hyperparameters."""
    
    if use_early_stopping:
        callbacks = [EarlyStopping(patience=15, monitor="valid_loss")]
        train_split = ValidSplit(cv=val_fraction, stratified=False)
    else:
        callbacks = []
        train_split=None

    callbacks.append(GradientNormClipping(gradient_clip_value=1.0))

    return NeuralNetRegressor(
        module=_MLPModule,
        module__n_input=n_input,
        module__hidden_layer_sizes=hidden_layer_sizes,
        module__dropout=dropout,
        lr=lr,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        device=DEVICE,
        callbacks=callbacks,
        train_split=train_split,
        verbose=0,
        iterator_train__shuffle=True,
        warm_start=use_warm_start,
    )


def _make_proxy_net(n_input: int) -> NeuralNetRegressor:
    """Small fast skorch net used only for feature selection."""
    return _make_skorch_net(
        n_input=n_input,
        hidden_layer_sizes=(64,),
        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=30,
    )

def method_permutation(X_tr: pd.DataFrame, y_tr: pd.Series,
                        top_k: int = TOP_K, n_repeats: int = 5) -> list[str]:
    """Permutation importance via a small proxy skorch MLP."""
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_tr).astype(np.float32)
    y_np   = y_tr.values.astype(np.float32)
    model  = _make_proxy_net(X_sc.shape[1]).fit(X_sc, y_np)
    res    = permutation_importance(
        model, X_sc, y_np, n_repeats=n_repeats,
        random_state=RANDOM_STATE, scoring=ic_scorer,
    )
    idx = np.argsort(res.importances_mean)[-top_k:]
    return X_tr.columns[idx].tolist()


def method_mutual_info(X_tr: pd.DataFrame, y_tr: pd.Series,
                        top_k: int = TOP_K) -> list[str]:
    """Mutual information regression — model-free, captures non-linear association."""
    scores = mutual_info_regression(X_tr.values, y_tr.values, random_state=RANDOM_STATE)
    idx    = np.argsort(scores)[-top_k:]
    return X_tr.columns[idx].tolist()


def method_lasso(X_tr: pd.DataFrame, y_tr: pd.Series,
                  top_k: int = TOP_K) -> list[str]:
    """LassoCV coefficient magnitude — selects linearly predictive features."""
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_tr)
    lasso   = LassoCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1, max_iter=2000)
    lasso.fit(X_sc, y_tr.values)
    scores  = np.abs(lasso.coef_)
    idx     = np.argsort(scores)[-top_k:]
    return X_tr.columns[idx].tolist()


def load_or_run_feature_selection_mlp(
    X_train: pd.DataFrame,
    y_reg_train: pd.Series,
    force: bool = False,
    experiment_name: str = "",
) -> dict[str, list[str]]:
    """
    Runs 3 feature selection methods at TOP_K=100:
      permutation — proxy skorch MLP permutation importance
      mutual_info — mutual information regression (model-free)
      lasso       — LassoCV coefficient magnitude
    Results are cached to avoid recomputation.
    """
    cache_name = "feature_selection_mlp"
    if experiment_name:
        cache_name += f"_{experiment_name}"
    cache_file = CACHE_DIR / f"{cache_name}.json"
    print(f"Cache file: {cache_file}")

    method_fns = {
        "permutation": lambda: method_permutation(X_train, y_reg_train),
        "mutual_info": lambda: method_mutual_info(X_train, y_reg_train),
        "lasso":       lambda: method_lasso(X_train, y_reg_train),
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

def quick_cv_ic_gapped_mlp(
    feats: list[str],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    net_params: dict,
    gap_days: int = GAP_DAYS,
    train_years: int = 5,
    val_quarters: int = 4,
    use_early_stopping: bool = False,
    use_warm_start: bool = False,
) -> tuple[float, float, float]:
    """
    Walk-forward IC CV using a skorch NeuralNetRegressor.
    net_params keys: n_layers, layer_size, dropout, lr, weight_decay, batch_size, max_epochs.
    StandardScaler is fit inside each fold on train only.
    """
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

        fold_X_tr  = X_tr.loc[mask_tr, feats]
        fold_X_val = X_tr.loc[mask_val, feats]
        fold_y_tr  = y_tr.loc[mask_tr].values.astype(np.float32)
        fold_y_val = y_tr.loc[mask_val].values.astype(np.float32)

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(fold_X_tr).astype(np.float32)
        X_val_sc = scaler.transform(fold_X_val).astype(np.float32)

        net = _make_skorch_net(
            n_input=len(feats),
            hidden_layer_sizes=tuple([net_params["layer_size"]] * net_params["n_layers"]),
            dropout=net_params["dropout"],
            lr=net_params["lr"],
            weight_decay=net_params["weight_decay"],
            batch_size=net_params["batch_size"],
            max_epochs=net_params["max_epochs"],
            use_early_stopping=use_early_stopping,
            use_warm_start=use_warm_start,
        )
        net.fit(X_tr_sc, fold_y_tr)
        preds = net.predict(X_val_sc)
        if np.std(preds) < 1e-6:
            print(f"[WARN] Model collapsed to constant: {preds[0]}")
            return -1.0, 0.0, 1.0
        ic, _ = spearmanr(fold_y_val, preds)

        if np.isnan(ic):
            print(f"[WARN] fold {i+1}: IC is NaN — skipping")
            continue

        fold_ics.append(ic)
        fold_rmses.append(float(np.sqrt(np.mean((fold_y_val - preds) ** 2))))

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


# ── Ensemble ───────────────────────────────────────────────────────────────────
class MLPEnsembleModel:
    """
    Weighted rank-average ensemble of skorch NeuralNetRegressors.
    Each member stores its own fitted StandardScaler so .predict(X) receives
    raw (unscaled) features and handles scaling + float32 casting internally.
    X must contain all columns needed by all sub-models.
    """
    def __init__(
        self,
        models: list[NeuralNetRegressor],
        scalers: list[StandardScaler],
        feature_lists: list[list[str]],
        weights: list[float],
    ):
        total = sum(weights)
        self.models        = models
        self.scalers       = scalers
        self.feature_lists = feature_lists
        self.weights       = [w / total for w in weights]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n   = len(X)
        agg = np.zeros(n)
        for model, scaler, feats, w in zip(
            self.models, self.scalers, self.feature_lists, self.weights
        ):
            X_sc  = scaler.transform(X[feats]).astype(np.float32)
            raw   = model.predict(X_sc)
            ranks = rankdata(raw) / n
            agg  += w * ranks
        return agg

# ── Optuna objective ───────────────────────────────────────────────────────────
def make_objective_mlp(features: list[str], X_train: pd.DataFrame,
                        y_train: pd.Series, gap_days: int = GAP_DAYS):
    def objective(trial):
        net_params = dict(
            n_layers     = trial.suggest_int("n_layers", 1, 3),
            layer_size   = trial.suggest_int("layer_size", 32, 256, log=True),
            dropout      = trial.suggest_float("dropout", 0.05, 0.4),
            lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            batch_size   = trial.suggest_categorical("batch_size", [128, 256, 512]),
            max_epochs   = trial.suggest_int("max_epochs", 50, 250),
        )

        mean_ic, std_ic, mean_rmse = quick_cv_ic_gapped_mlp(
            features, X_train, y_train, net_params=net_params, gap_days=gap_days,
            use_early_stopping=True, use_warm_start=False,
        )
        icir = mean_ic / std_ic if std_ic > 0 else 0.0
        trial.set_user_attr("val_ic",     mean_ic)
        trial.set_user_attr("val_icir",   icir)
        trial.set_user_attr("val_ic_std", std_ic)
        trial.set_user_attr("val_rmse",   mean_rmse)
        trial.set_user_attr("n_layers",   net_params["n_layers"])
        trial.set_user_attr("layer_size", net_params["layer_size"])
        return -mean_ic if not np.isnan(mean_ic) else 1.0

    return objective


# ── Save / load ────────────────────────────────────────────────────────────────
def _ensemble_member_path(experiment_name: str, method_name: str) -> Path:
    prefix = f"final_{experiment_name}" if experiment_name else "final"
    return RESULTS_DIR / f"{prefix}_mlp_{method_name}.joblib"

def _meta_path(experiment_name: str) -> Path:
    prefix = f"final_{experiment_name}" if experiment_name else "final"
    return RESULTS_DIR / f"{prefix}_meta.json"


def save_final_models(
    best_method: str,
    best_feats: list[str],
    scores: dict,
    best_params: dict,
    data_end: pd.Timestamp,
    experiment_name: str = "",
    ensemble: MLPEnsembleModel | None = None,
    ensemble_methods: list[str] | None = None,
    ensemble_params: dict | None = None,
):
    RESULTS_DIR.mkdir(exist_ok=True)
    meta_path = _meta_path(experiment_name)

    def _clean(v):
        return v.item() if hasattr(v, "item") else v

    if ensemble is None:
        raise ValueError("save_final_models requires an MLPEnsembleModel")

    for method_name, model, scaler in zip(
        ensemble_methods, ensemble.models, ensemble.scalers
    ):
        joblib.dump(
            {"model": model, "scaler": scaler},
            _ensemble_member_path(experiment_name, method_name),
        )

    meta = {
        "mode":             "ensemble_mlp",
        "ensemble_methods": ensemble_methods,
        "ensemble_weights": [_clean(w) for w in ensemble.weights],
        "ensemble_feats":   {m: f for m, f in zip(ensemble_methods, ensemble.feature_lists)},
        "ensemble_params":  {m: {k: _clean(v) for k, v in p.items()}
                             for m, p in (ensemble_params or {}).items()},
        "best_method":      best_method,
        "best_feats":       best_feats,
        "best_params":      {k: _clean(v) for k, v in best_params.items()},
        "scores":           {k: _clean(v) for k, v in scores.items()},
        "data_end": data_end.strftime("%Y-%m-%d"),
    }

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

    label   = f" [{study_name}]" if study_name else ""
    models, scalers, feats_list, weights = [], [], [], []

    for method_name in meta["ensemble_methods"]:
        bundle = joblib.load(_ensemble_member_path(study_name, method_name))
        models.append(bundle["model"])
        scalers.append(bundle["scaler"])
        feats_list.append(meta["ensemble_feats"][method_name])
        idx = meta["ensemble_methods"].index(method_name)
        weights.append(meta["ensemble_weights"][idx])

    ensemble = MLPEnsembleModel(models, scalers, feats_list, weights)
    print(f"Loaded MLP ensemble{label} ({len(models)} models, methods={meta['ensemble_methods']})")
    return ensemble, meta


# ── Main pipeline ──────────────────────────────────────────────────────────────
def training_pipeline_mlp(
    df: pd.DataFrame,
    optuna_storage_name: str,
    force_feature_selection: bool = False,
    experiment_name: str = "",
) -> tuple[dict, MLPEnsembleModel, list[str]]:
    """
    Full MLP training pipeline: data split → feature selection (permutation,
    mutual_info, lasso) → per-method Optuna with skorch/PyTorch search space
    → ensemble of 3 NeuralNetRegressors weighted by CV IC → scorecard → save.

    Returns (scores, ensemble, best_feats_of_winning_method).
    """
    print("\n─── Load & Clean Data ────────────────────────────────────")

    TEST_MONTHS = 24
    GAP         = pd.Timedelta(days=GAP_DAYS)
    # data_end    = df["fiscalDateEnding"].max()

    # test_end     = data_end
    # test_start   = data_end - pd.DateOffset(months=TEST_MONTHS)
    # train_cutoff = test_start - GAP


    # train_df = df[df["fiscalDateEnding"] <= train_cutoff].copy()
    # test_df  = df[(df["fiscalDateEnding"] >= test_start) &
    #               (df["fiscalDateEnding"] <= test_end)].copy()

    train_df, test_df, train_cutoff, test_start, test_end = derive_train_test_split(df, test_months=TEST_MONTHS, gap_days=GAP_DAYS)
    print(f"Train pool : up to  {train_cutoff.strftime('%Y-%m-%d')}")
    print(f"Gap        : {GAP.days}d  →  {test_start.strftime('%Y-%m-%d')}")
    print(f"Test       : {test_start.strftime('%Y-%m-%d')}  →  {test_end.strftime('%Y-%m-%d')}")
    print(f"Train rows : {(df['fiscalDateEnding'] <= train_cutoff).sum()}")
    print(f"Test rows  : {((df['fiscalDateEnding'] >= test_start) & (df['fiscalDateEnding'] <= test_end)).sum()}")

    X_train_raw, y_reg_train_raw = split_target_with_date_index(train_df)
    X_test_raw,  y_reg_test_raw  = split_target_with_date_index(test_df)

    y_reg_train = rank_target_cross_sectionally(y_reg_train_raw)
    y_reg_test  = rank_target_cross_sectionally(y_reg_test_raw)

    print(f"\nSplit sizes → train: {len(X_train_raw):,}  test: {len(X_test_raw):,}")

    X_train, X_test = cleanup_base(X_train_raw, X_test_raw)
    print(f"Features after variance threshold: {X_train.shape[1]}")

    # ── Feature selection ─────────────────────────────────────────────────────
    print("\n─── Feature selection ────────────────────────────────────")
    methods = load_or_run_feature_selection_mlp(
        X_train, y_reg_train,
        force_feature_selection, experiment_name,
    )

    # proxy_net_params = dict(n_layers=1, layer_size=64, dropout=0.0,
    #                         lr=1e-3, weight_decay=1e-4,
    #                         batch_size=256, max_epochs=30)
    # print(f"\n  {'method':<12}  {'n_feats':>7}  {'mean_IC':>8}  {'std_IC':>7}  {'ICIR':>6}  {'RMSE':>7}")
    # print(f"  {'-'*12}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*7}")
    # for name, feats in methods.items():
    #     mean_ic, std_ic, mean_rmse = quick_cv_ic_gapped_mlp(
    #         feats, X_train, y_reg_train, net_params=proxy_net_params,
    #     )
    #     icir = mean_ic / std_ic if std_ic > 0 else 0.0
    #     print(f"  {name:<12}  {len(feats):>7}  {mean_ic:>8.4f}  "
    #           f"{std_ic:>7.4f}  {icir:>6.3f}  {mean_rmse:>7.4f}")

    # ── Optuna pass — one study per method ───────────────────────────────────
    print("\n─── Optuna pass (all methods) ────────────────────────────")
    full_studies: dict[str, optuna.Study] = {}

    for name in tqdm(BASE_METHODS, desc="Methods", unit="method", position=0):
        feats      = methods[name]
        study_name = f"{experiment_name}_mlp_final_{name}"
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
                make_objective_mlp(feats, X_train, y_reg_train),
                n_trials=remaining, callbacks=[_cb], show_progress_bar=False,
            )

        t = study.best_trial
        print(f"  {name:<12}  best IC={t.user_attrs['val_ic']:.4f}  "
              f"RMSE={t.user_attrs['val_rmse']:.4f}  "
              f"layers={t.user_attrs.get('n_layers')}x{t.user_attrs.get('layer_size')}")
        full_studies[name] = study

    assert X_train.index.max() <= train_cutoff, (
        f"Trainval bleeds into test gap: "
        f"{X_train.index.max().date()} > {train_cutoff.date()}"
    )

    # ── Train ensemble members, each with its own best net_params + scaler ────
    print("\n─── Training ensemble models ─────────────────────────────")
    ensemble_models:  list[NeuralNetRegressor] = []
    ensemble_scalers: list[StandardScaler]     = []
    ensemble_feats:   list[list[str]]          = []
    ensemble_weights: list[float]              = []
    ensemble_params:  dict[str, dict]          = {}

    best_method = min(full_studies, key=lambda k: full_studies[k].best_value)
    best_feats  = methods[best_method]
    best_params = full_studies[best_method].best_params

    for name in BASE_METHODS:
        feats      = methods[name]
        params     = full_studies[name].best_params
        ic_val     = full_studies[name].best_trial.user_attrs.get("val_ic", 0.0)
        weight     = max(ic_val, 0.0)

        net = _make_skorch_net(
            n_input=len(feats),
            hidden_layer_sizes=tuple([params["layer_size"]] * params["n_layers"]),
            dropout=params["dropout"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            max_epochs=params["max_epochs"],
        )

        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X_train[feats]).astype(np.float32)
        y_np    = y_reg_train.values.astype(np.float32)
        net.fit(X_sc, y_np)

        ensemble_models.append(net)
        ensemble_scalers.append(scaler)
        ensemble_feats.append(feats)
        ensemble_weights.append(weight)
        ensemble_params[name] = params
        print(f"  {name:<12}  {len(feats):>3} feats  CV IC={ic_val:.4f}  weight={weight:.4f}  "
              f"arch={params['n_layers']}x{params['layer_size']}")

    if sum(ensemble_weights) == 0:
        print("[WARN] All weights zero — falling back to equal weights")
        ensemble_weights = [1.0] * len(BASE_METHODS)

    ensemble = MLPEnsembleModel(
        ensemble_models, ensemble_scalers, ensemble_feats, ensemble_weights
    )

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
        data_end=test_end,
        experiment_name=experiment_name,
        ensemble=ensemble,
        ensemble_methods=BASE_METHODS,
        ensemble_params=ensemble_params,
    )

    return scores, ensemble, best_feats
