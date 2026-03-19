import json
from typing import Any

import mlflow


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
        mlflow_run_id: str | None = None
    ):
        self.task_type = task_type
        self.metric_name = metric_name
        self.score_mean = score_mean
        self.score_std = score_std
        self.num_features = num_features
        self.top_correlations_with_target = top_correlations_with_target
        self.feature_importance = feature_importance
        self.mlflow_run_id = mlflow_run_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "metric_name": self.metric_name,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "num_features": self.num_features,
            "top_correlations_with_target": self.top_correlations_with_target,
            "feature_importance": self.feature_importance,
            "mlflow_run_id": self.mlflow_run_id
        }

    def save(self, path: str = "evaluation_report.json") -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def log_training_metrics(
    score_mean: float,
    score_std: float
) -> None:
    mlflow.log_metric("score_mean", score_mean)
    mlflow.log_metric("score_std", score_std)


def log_feature_importance(importance_dict: dict[str, float] | None) -> None:
    if importance_dict:
        mlflow.log_dict(importance_dict, "feature_importance.json")


def log_artifacts(
    report_path: str = "evaluation_report.json",
    features_path: str = "dynamic_features.py",
    glossary_path: str = "glossary.md",
    plots_dir: str = "evaluation_plots"
) -> None:
    mlflow.log_artifact(report_path)
    mlflow.log_artifact(features_path)
    mlflow.log_artifact(glossary_path)
    mlflow.log_artifacts(plots_dir, artifact_path="evaluation_plots")


def create_report(
    task_type: str,
    metric_name: str,
    score_mean: float,
    score_std: float,
    num_features: int,
    top_correlations_with_target: dict[str, float],
    feature_importance: dict[str, float] | None,
    mlflow_run_id: str | None
) -> EvaluationReport:
    return EvaluationReport(
        task_type=task_type,
        metric_name=metric_name,
        score_mean=score_mean,
        score_std=std_score,
        num_features=num_features,
        top_correlations_with_target=top_correlations_with_target,
        feature_importance=feature_importance or {},
        mlflow_run_id=mlflow_run_id
    )
