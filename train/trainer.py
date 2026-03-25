import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from typing import Any


class TrainingResult:
    def __init__(
        self,
        mean_score: float,
        std_score: float,
        fold_scores: list[float],
        feature_importance: dict[str, float] | None = None,
        precision: float | None = None,
        recall: float | None = None,
        auc_roc: float | None = None
    ):
        self.mean_score = mean_score
        self.std_score = std_score
        self.fold_scores = fold_scores
        self.feature_importance = feature_importance
        self.precision = precision
        self.recall = recall
        self.auc_roc = auc_roc


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
    fold_precisions = []
    fold_recalls = []
    fold_aucs = []
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
            
            fold_precision = precision_score(y_val, preds, average='weighted', zero_division=0)
            fold_recall = recall_score(y_val, preds, average='weighted', zero_division=0)
            fold_precisions.append(fold_precision)
            fold_recalls.append(fold_recall)
            
            fold_auc = _compute_auc_roc(model, X_val, y_val)
            if fold_auc is not None:
                fold_aucs.append(fold_auc)
        else:
            fold_score = r2_score(y_val, preds)
            fold_scores.append(fold_score)
        
        feature_importances = _aggregate_importance(model, feature_importances)
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    feature_importance_dict = _finalize_importance(
        model, feature_importances, cv_folds, X
    )
    
    precision = np.mean(fold_precisions) if fold_precisions else None
    recall = np.mean(fold_recalls) if fold_recalls else None
    auc_roc = np.mean(fold_aucs) if fold_aucs else None
    
    return TrainingResult(
        mean_score=mean_score,
        std_score=std_score,
        fold_scores=fold_scores,
        feature_importance=feature_importance_dict,
        precision=precision,
        recall=recall,
        auc_roc=auc_roc
    )


def _compute_auc_roc(model: Any, X_val: pd.DataFrame, y_val: pd.Series) -> float | None:
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_val)
            if y_proba.shape[1] == 2:
                return roc_auc_score(y_val, y_proba[:, 1])
            elif y_proba.shape[1] > 2:
                return roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted')
        elif hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_val)
            return roc_auc_score(y_val, y_scores)
    except Exception:
        pass
    return None


def _aggregate_importance(
    model: Any,
    current: np.ndarray | None
) -> np.ndarray | None:
    try:
        if hasattr(model, 'named_steps'):
            clf_step = model.named_steps.get('clf', model)
        else:
            clf_step = model
    except Exception:
        return current
    
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
) -> dict[str, float]:
    if aggregated is None:
        return {}
    
    aggregated /= n_splits
    
    feature_names = _get_feature_names(model, X)
    
    if len(feature_names) != len(aggregated):
        feature_names = [f"feature_{i}" for i in range(len(aggregated))]
    
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
    if hasattr(model, 'named_steps'):
        prep_step = model.named_steps.get('prep', None)
        if prep_step is not None:
            try:
                if hasattr(prep_step, 'get_feature_names_out'):
                    return list(prep_step.get_feature_names_out())
            except Exception:
                pass
            try:
                if hasattr(prep_step, 'transformers_'):
                    feature_names = []
                    for name, transformer, columns in prep_step.transformers_:
                        if hasattr(transformer, 'get_feature_names_out'):
                            try:
                                fn = transformer.get_feature_names_out()
                                if hasattr(fn, 'tolist'):
                                    fn = fn.tolist()
                                feature_names.extend(fn)
                            except Exception:
                                feature_names.extend([f"{name}_{i}" for i in range(len(columns))])
                        else:
                            feature_names.extend([f"{name}_{i}" for i in range(len(columns))])
                    if feature_names:
                        return feature_names
            except Exception:
                pass
    return list(X.columns)
