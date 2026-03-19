import numpy as np
import pandas as pd
from typing import TypedDict


class CorrelationResult(TypedDict):
    numeric: dict[str, float]
    categorical: dict[str, float]


class TopFeatures(TypedDict):
    numeric: list[tuple[str, float]]
    categorical: list[tuple[str, float]]
    combined_slots: tuple[int, int]


def compute_correlations(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    corr_dict = {}
    X_num = X.select_dtypes(include=numerics)
    
    for col in X_num.columns:
        if X_num[col].nunique() > 1:
            corr = X_num[col].corr(y)
            if not np.isnan(corr):
                corr_dict[col] = float(corr)
    
    return corr_dict


def compute_categorical_scores(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    cat_score_dict = {}
    X_cat = X.select_dtypes(include=["object", "category", "bool"])
    
    for col in X_cat.columns:
        if X_cat[col].nunique() < 2:
            continue
        rates = (
            pd.DataFrame({"feat": X_cat[col].astype(str), "target": y})
            .groupby("feat")["target"]
            .mean()
        )
        score = float(rates.var())
        if not np.isnan(score):
            cat_score_dict[col] = score
    
    return cat_score_dict


def select_top_features(
    corr_dict: dict[str, float],
    cat_score_dict: dict[str, float],
    top_n: int = 10
) -> TopFeatures:
    top_cat = sorted(cat_score_dict.items(), key=lambda item: item[1], reverse=True)
    
    n_num = min(len(corr_dict), top_n)
    n_cat = min(len(cat_score_dict), top_n)
    
    if n_num + n_cat <= top_n:
        slots_num, slots_cat = n_num, n_cat
    else:
        slots_num = max(1, round(top_n * n_num / (n_num + n_cat)))
        slots_cat = top_n - slots_num
    
    top_numeric = sorted(
        corr_dict.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )[:slots_num]
    
    top_categoric = top_cat[:slots_cat]
    
    return {
        "numeric": top_numeric,
        "categorical": top_categoric,
        "combined_slots": (slots_num, slots_cat)
    }


def analyze_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 10
) -> tuple[dict[str, float], TopFeatures]:
    corr_dict = compute_correlations(X, y)
    cat_score_dict = compute_categorical_scores(X, y)
    top_features = select_top_features(corr_dict, cat_score_dict, top_n)
    
    return corr_dict, top_features
