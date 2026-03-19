from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional
from autogen_ext.models.openai import OpenAIChatCompletionClient


class CLIArgs(BaseModel):
    iterations: int = Field(default=5, ge=1, le=100, description="Maximum number of iterations")
    experiment: Optional[str] = Field(default=None, description="MLFlow experiment name")
    tracking_uri: Optional[str] = Field(default=None, description="MLFlow tracking URI")

    @field_validator("experiment", "tracking_uri")
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
        return v if v else None


class TrainCLIArgs(BaseModel):
    experiment: Optional[str] = Field(default=None, description="MLFlow experiment name")
    tracking_uri: Optional[str] = Field(default=None, description="MLFlow tracking URI")
    iter_num: int = Field(default=1, ge=1, description="Iteration number")
    no_mlflow: bool = Field(default=False, description="Disable MLFlow logging")

    @field_validator("experiment", "tracking_uri")
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
        return v if v else None


class StrategyInput(BaseModel):
    glossary: str = Field(description="Semantic glossary content")
    data_schema: str = Field(description="Data schema (dtypes)")
    data_sample: str = Field(description="Data sample (first rows)")

    @field_validator("glossary", "data_schema", "data_sample")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class EvaluatorInput(BaseModel):
    iter_num: int = Field(ge=1, description="Current iteration number")
    evaluation_report: dict = Field(description="Evaluation report dictionary")
    glossary: str = Field(description="Semantic glossary content")
    plot_dir: str = Field(default="evaluation_plots", description="Directory with evaluation plots")

    @field_validator("glossary")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Glossary cannot be empty")
        return v


class CodeGenerationInput(BaseModel):
    business_strategy: str = Field(description="Business feature strategy")
    reflection_text: str = Field(description="Reflection text from evaluator")
    last_code: str = Field(description="Previous code content")
    last_error: Optional[str] = Field(default=None, description="Last error message if any")

    @field_validator("business_strategy", "reflection_text", "last_code")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class CodeFixInput(BaseModel):
    error_message: str = Field(description="Error message to fix")
    previous_code: str = Field(description="Previous code content")

    @field_validator("error_message", "previous_code")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_client: OpenAIChatCompletionClient = Field(description="LLM model client")
    max_iterations: int = Field(default=5, ge=1, le=100, description="Maximum iterations")
    mlflow_experiment_name: Optional[str] = Field(default=None, description="MLFlow experiment name")
    mlflow_tracking_uri: Optional[str] = Field(default=None, description="MLFlow tracking URI")

    @field_validator("mlflow_experiment_name", "mlflow_tracking_uri")
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
        return v if v else None


class RunBaselineInput(BaseModel):
    iter_num: int = Field(ge=1, description="Current iteration number")
    mlflow_experiment_name: Optional[str] = Field(default=None, description="MLFlow experiment name")
    mlflow_tracking_uri: Optional[str] = Field(default=None, description="MLFlow tracking URI")
