import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo


@dataclass
class LLMConfig:
    model: str = "gpt-5.4-nano"
    temperature: float = 0.7
    reasoning_effort: str = "low"
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))

    def create_client(self) -> OpenAIChatCompletionClient:
        return OpenAIChatCompletionClient(
            model=self.model,
            api_key=self.api_key,
            model_info=ModelInfo(
                vision=False,
                function_calling=False,
                json_output=False,
                family="unknown",
                structured_output=False,
            ),
        )

    def to_dict(self) -> dict:
        return {
            "model_client": self.create_client(),
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }


@dataclass
class MLFlowConfig:
    experiment_name: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_EXPERIMENT_NAME", "AutoLLml_Experiments")
    )
    tracking_uri: str = field(default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI"))


llm_config = LLMConfig()
mlflow_config = MLFlowConfig()

llm_client = llm_config.create_client()


def get_llm_config() -> LLMConfig:
    return llm_config
