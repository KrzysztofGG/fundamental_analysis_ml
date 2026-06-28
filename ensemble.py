import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import rankdata
from tqdm.notebook import tqdm

from training_util import rank_target_cross_sectionally

class EnsembleModel:
    """
    Weighted rank-average ensemble of XGBoost regressors.
    Each model scores independently; scores are percentile-ranked within the
    batch, then combined as a weighted average by CV IC.
    Exposes .predict(X) as a drop-in replacement for a single XGBRegressor.
    X passed to .predict() must contain all columns needed by all sub-models.
    """
    def __init__(self, models: list[xgb.XGBRegressor],
                 feature_lists: list[list[str]],
                 weights: list[float]):
        total = sum(weights)
        self.models        = models
        self.feature_lists = feature_lists
        self.weights       = [w / total for w in weights]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n   = len(X)
        agg = np.zeros(n)
        for model, feats, w in zip(self.models, self.feature_lists, self.weights):
            raw   = model.predict(X[feats])
            ranks = rankdata(raw) / n
            agg  += w * ranks
        return agg
    
    def retrain(self, batch: pd.DataFrame, lower: pd.Series,
                upper: pd.Series, medians: pd.Series):
        """Incrementally retrain each member on it's own feature subset."""
        for i, (model, feats) in enumerate(zip(self.models, self.feature_lists)):
            valid = batch.dropna(subset=["target"] + feats)
            if len(valid) < 5:
                tqdm.write(f"  [retrain] ensemble member {i}: only {len(valid)} rows - skipping")

            X_new = apply_cleanup(valid[feats].copy(), lower, upper, medians)
            y_new = valid["target"].copy()
            y_new.index = valid["fiscalDateEnding"].values
            y_new = rank_target_cross_sectionally(y_new)
            model.fit(X_new, y_new, xgb_model=model.get_booster())
            tqdm.write(f"  [retrain] ensemble member {i}: fit on {len(valid)} rows")

def apply_cleanup(X: pd.DataFrame,
                  lower: pd.Series, upper: pd.Series,
                  medians: pd.Series
                  ) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=lower, upper=upper, axis=1)
    X = X.fillna(medians)
    return X