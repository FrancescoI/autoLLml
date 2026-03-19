from pydantic import BaseModel, Field
from typing import Optional


class StrategyOutput(BaseModel):
    business_strategy: str = Field(description="Generated business strategy text")
    model_selection: str = Field(description="Recommended model selection")

    def to_dict(self) -> dict:
        return self.model_dump()


class EvaluatorOutput(BaseModel):
    reflection_text: str = Field(description="Reflection text from evaluator")

    def to_dict(self) -> dict:
        return self.model_dump()


class CodeOutput(BaseModel):
    code: str = Field(description="Generated Python code")

    def to_dict(self) -> dict:
        return self.model_dump()


class IterationResult(BaseModel):
    iteration: int = Field(description="Iteration number")
    metric: Optional[float] = Field(default=None, description="Metric value (R2/F1)")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    def is_success(self) -> bool:
        return self.metric is not None

    def to_dict(self) -> dict:
        return self.model_dump()


class FeatureCorrelation(BaseModel):
    feature: str = Field(description="Feature name")
    correlation: float = Field(description="Pearson correlation value")

    def to_dict(self) -> dict:
        return self.model_dump()


class FeatureImportance(BaseModel):
    feature: str = Field(description="Feature name")
    importance: float = Field(description="Importance score")

    def to_dict(self) -> dict:
        return self.model_dump()


class EvaluationReport(BaseModel):
    task_type: str = Field(description="Task type: 'classification' or 'regression'")
    metric_name: str = Field(description="Metric name: 'F1_weighted' or 'R2'")
    score_mean: float = Field(description="Mean cross-validation score")
    score_std: float = Field(description="Standard deviation of cross-validation scores")
    num_features: int = Field(description="Number of features after engineering")
    top_correlations_with_target: dict[str, float] = Field(
        default_factory=dict,
        description="Top correlations with target variable"
    )
    feature_importance: dict[str, float] = Field(
        default_factory=dict,
        description="Feature importance scores"
    )
    mlflow_run_id: Optional[str] = Field(
        default=None,
        description="MLFlow run ID"
    )

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationReport":
        return cls(**data)


class TrainingResult(BaseModel):
    success: bool = Field(description="Whether training succeeded")
    metric: Optional[float] = Field(default=None, description="Achieved metric value")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    evaluation_report: Optional[EvaluationReport] = Field(
        default=None,
        description="Full evaluation report"
    )

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "metric": self.metric,
            "error": self.error,
            "evaluation_report": self.evaluation_report.to_dict() if self.evaluation_report else None
        }
