from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


class TrainingStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class IterationSnapshot(BaseModel):
    iteration: int = Field(description="Iteration number")
    metric: Optional[float] = Field(default=None, description="Achieved metric value")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="running", description="Status: running, completed, failed")
    reflection_text: Optional[str] = Field(default=None, description="LLM reflection text")
    business_strategy: Optional[str] = Field(default=None, description="Business strategy applied")


class SessionStateDTO(BaseModel):
    training_status: TrainingStatus = Field(default=TrainingStatus.IDLE)
    current_iteration: int = Field(default=0)
    max_iterations: int = Field(default=5, ge=1, le=50)
    iteration_history: list[IterationSnapshot] = Field(default_factory=list)
    glossary_content: Optional[str] = Field(default=None, description="Glossary markdown content")
    dataset_uploaded: bool = Field(default=False)
    dataset_filename: Optional[str] = Field(default=None)
    mlflow_experiment_name: Optional[str] = Field(default="AutoLLml_Experiments")
    mlflow_tracking_enabled: bool = Field(default=True)
    logs: list[str] = Field(default_factory=list)
    business_strategy: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)

    def add_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

    def add_iteration(self, snapshot: IterationSnapshot) -> None:
        self.iteration_history.append(snapshot)

    def update_iteration(self, iteration: int, **kwargs) -> None:
        for i, snap in enumerate(self.iteration_history):
            if snap.iteration == iteration:
                for key, value in kwargs.items():
                    if hasattr(snap, key):
                        setattr(snap, key, value)
                break

    def get_best_iteration(self) -> Optional[IterationSnapshot]:
        completed = [s for s in self.iteration_history if s.metric is not None]
        if not completed:
            return None
        return max(completed, key=lambda x: x.metric)

    def to_dict(self) -> dict:
        return {
            "training_status": self.training_status.value,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "iterations_completed": len([s for s in self.iteration_history if s.status == "completed"]),
            "best_metric": self.get_best_iteration().metric if self.get_best_iteration() else None,
            "dataset_uploaded": self.dataset_uploaded,
            "glossary_set": self.glossary_content is not None,
            "error": self.error_message,
        }


class RunConfigDTO(BaseModel):
    max_iterations: int = Field(default=5, ge=1, le=50)
    mlflow_experiment_name: str = Field(default="AutoLLml_Experiments")
    mlflow_tracking_enabled: bool = Field(default=True)
    enable_llm: bool = Field(default=True, description="Enable LLM-based feature engineering")
