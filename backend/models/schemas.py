from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class TrainingStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class DatasetInfo(BaseModel):
    filename: str
    rows: int
    columns: int
    dtypes: dict[str, str]


class GlossaryContent(BaseModel):
    content: str


class TrainingParams(BaseModel):
    max_iterations: int = Field(default=5, ge=1, le=50)
    mlflow_experiment_name: Optional[str] = Field(default="AutoLLml_Experiments")
    mlflow_tracking_enabled: bool = Field(default=True)


class IterationSnapshot(BaseModel):
    iteration: int
    metric: Optional[float] = None
    error: Optional[str] = None
    status: str = "pending"
    timestamp: str


class TrainingStatusResponse(BaseModel):
    status: TrainingStatus
    current_iteration: int
    max_iterations: int
    iterations_completed: int
    best_metric: Optional[float] = None


class LogEntry(BaseModel):
    timestamp: str
    message: str
    level: str = "info"


class StrategyOutput(BaseModel):
    business_strategy: str
    model_selection: str


class ReflectionOutput(BaseModel):
    iteration: int
    reflection_text: str


class CodeOutput(BaseModel):
    iteration: int
    code_summary: str
    features_added: list[str]


class AgentOutputs(BaseModel):
    strategy: Optional[StrategyOutput] = None
    reflection: Optional[ReflectionOutput] = None
    code: Optional[CodeOutput] = None


class EvaluationReport(BaseModel):
    task_type: str
    metric_name: str
    score_mean: float
    score_std: float
    num_features: int
    top_correlations_with_target: dict[str, float]
    feature_importance: dict[str, float]
    mlflow_run_id: Optional[str] = None


class FeatureInfo(BaseModel):
    name: str
    category: str
    description: Optional[str] = None


class TrainingHistory(BaseModel):
    iterations: list[IterationSnapshot]
    best_iteration: Optional[int] = None
    best_metric: Optional[float] = None


class ErrorResponse(BaseModel):
    detail: str
