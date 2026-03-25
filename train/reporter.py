import json
from typing import Any

from utils.config import get_paths


class EvaluationReport:
    def __init__(
        self,
        task_type: str,
        metric_name: str,
        score_mean: float,
        score_std: float,
        num_features: int,
        top_correlations_with_target: dict[str, float],
        feature_importance: dict[str, float],
        precision: float | None = None,
        recall: float | None = None,
        auc_roc: float | None = None
    ):
        self.task_type = task_type
        self.metric_name = metric_name
        self.score_mean = score_mean
        self.score_std = score_std
        self.num_features = num_features
        self.top_correlations_with_target = top_correlations_with_target
        self.feature_importance = feature_importance
        self.precision = precision
        self.recall = recall
        self.auc_roc = auc_roc

    def to_dict(self) -> dict[str, Any]:
        result = {
            "task_type": self.task_type,
            "metric_name": self.metric_name,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "num_features": self.num_features,
            "top_correlations_with_target": self.top_correlations_with_target,
            "feature_importance": self.feature_importance
        }
        if self.precision is not None:
            result["precision"] = self.precision
        if self.recall is not None:
            result["recall"] = self.recall
        if self.auc_roc is not None:
            result["auc_roc"] = self.auc_roc
        return result

    def save(self, path: str | None = None) -> None:
        if path is None:
            path = get_paths().evaluation_report
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def log_training_metrics(
    score_mean: float,
    score_std: float
) -> None:
    pass


def log_feature_importance(importance_dict: dict[str, float] | None) -> None:
    pass


def log_artifacts(
    report_path: str | None = None,
    features_path: str = "dynamic_features.py",
    glossary_path: str | None = None,
    plots_dir: str | None = None
) -> None:
    paths = get_paths()
    if report_path is None:
        report_path = paths.evaluation_report
    if glossary_path is None:
        glossary_path = paths.glossary
    if plots_dir is None:
        plots_dir = paths.output_dir
    pass


def create_report(
    task_type: str,
    metric_name: str,
    score_mean: float,
    score_std: float,
    num_features: int,
    top_correlations_with_target: dict[str, float],
    feature_importance: dict[str, float] | None,
    precision: float | None = None,
    recall: float | None = None,
    auc_roc: float | None = None
) -> EvaluationReport:
    return EvaluationReport(
        task_type=task_type,
        metric_name=metric_name,
        score_mean=score_mean,
        score_std=score_std,
        num_features=num_features,
        top_correlations_with_target=top_correlations_with_target,
        feature_importance=feature_importance or {},
        precision=precision,
        recall=recall,
        auc_roc=auc_roc
    )