import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, f1_score, accuracy_score
from typing import Any


class TrainingResult:
    def __init__(
        self,
        mean_score: float,
        std_score: float,
        fold_scores: list[float],
        feature_importance: dict[str, float] | None = None
    ):
        self.mean_score = mean_score
        self.std_score = std_score
        self.fold_scores = fold_scores
        self.feature_importance = feature_importance


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    is_classification: bool,
    cv_folds: int = 5,
    cv_random_state: int = 42
) -> TrainingResult:
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=cv_random_state)
    fold_scores = []
    feature_importances = None
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        if is_classification:
            fold_score = f1_score(y_val, preds, average='weighted')
            fold_acc = accuracy_score(y_val, preds)
            fold_scores.append(fold_score)
        else:
            fold_score = r2_score(y_val, preds)
            fold_scores.append(fold_score)
        
        feature_importances = _aggregate_importance(model, feature_importances)
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    feature_importance_dict = _finalize_importance(
        model, feature_importances, cv_folds, X
    )
    
    return TrainingResult(
        mean_score=mean_score,
        std_score=std_score,
        fold_scores=fold_scores,
        feature_importance=feature_importance_dict
    )


def _aggregate_importance(
    model: Any,
    current: np.ndarray | None
) -> np.ndarray | None:
    clf_step = model.named_steps.get('clf', model)
    if hasattr(clf_step, 'feature_importances_'):
        fi = clf_step.feature_importances_
        if current is None:
            return fi.copy()
        return current + fi
    return current


def _finalize_importance(
    model: Any,
    aggregated: np.ndarray | None,
    n_splits: int,
    X: pd.DataFrame
) -> dict[str, float] | None:
    if aggregated is None:
        return None
    
    clf_step = model.named_steps.get('clf', model)
    if not hasattr(clf_step, 'feature_importances_'):
        return None
    
    aggregated /= n_splits
    
    feature_names = _get_feature_names(model, X)
    
    importance_dict = {
        name: float(imp)
        for name, imp in sorted(
            zip(feature_names, aggregated),
            key=lambda x: x[1],
            reverse=True
        )
    }
    
    return importance_dict


def _get_feature_names(model: Any, X: pd.DataFrame) -> list[str]:
    prep_step = model.named_steps.get('prep', None)
    if prep_step is not None and hasattr(prep_step, 'get_feature_names_out'):
        try:
            return list(prep_step.get_feature_names_out())
        except Exception:
            pass
    return list(X.columns)
