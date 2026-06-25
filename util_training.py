import pandas as pd
import numpy as np
from scipy.stats import rankdata


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