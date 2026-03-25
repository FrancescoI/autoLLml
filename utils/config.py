import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = load_config()


@dataclass
class LLMConfig:
    model: str
    temperature: float
    reasoning_effort: str

    @classmethod
    def from_config(cls, config: dict) -> "LLMConfig":
        llm = config.get("llm", {})
        return cls(
            model=llm.get("model", "gpt-4o-mini"),
            temperature=llm.get("temperature", 0.7),
            reasoning_effort=llm.get("reasoning_effort", "low"),
        )

    def create_client(self):
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_core.models import ModelInfo

        return OpenAIChatCompletionClient(
            model=self.model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_info=ModelInfo(
                vision=False,
                function_calling=False,
                json_output=False,
                family="unknown",
                structured_output=False,
            ),
        )


@dataclass
class PathsConfig:
    data: str
    glossary: str
    memory: str
    evaluation_report: str
    evaluation_report_md: str
    output_dir: str

    @classmethod
    def from_config(cls, config: dict) -> "PathsConfig":
        paths = config.get("paths", {})
        return cls(
            data=paths.get("data", "data/dataset.csv"),
            glossary=paths.get("glossary", "glossary.md"),
            memory=paths.get("memory", "memory.json"),
            evaluation_report=paths.get("evaluation_report", "evaluation_report.json"),
            evaluation_report_md=paths.get("evaluation_report_md", "evaluation_report.md"),
            output_dir=paths.get("output_dir", "evaluation_plots"),
        )


@dataclass
class AgentConfig:
    max_iterations: int
    max_error_retries: int

    @classmethod
    def from_config(cls, config: dict) -> "AgentConfig":
        agent = config.get("agent", {})
        return cls(
            max_iterations=agent.get("max_iterations", 5),
            max_error_retries=agent.get("max_error_retries", 3),
        )


@dataclass
class TrainingConfig:
    cv_folds: int
    cv_random_state: int

    @classmethod
    def from_config(cls, config: dict) -> "TrainingConfig":
        training = config.get("training", {})
        return cls(
            cv_folds=training.get("cv_folds", 5),
            cv_random_state=training.get("cv_random_state", 42),
        )


@dataclass
class AppConfig:
    llm: LLMConfig
    paths: PathsConfig
    agent: AgentConfig
    training: TrainingConfig

    @classmethod
    def from_config(cls, config: dict) -> "AppConfig":
        return cls(
            llm=LLMConfig.from_config(config),
            paths=PathsConfig.from_config(config),
            agent=AgentConfig.from_config(config),
            training=TrainingConfig.from_config(config),
        )


app_config = AppConfig.from_config(_config)

llm_config = app_config.llm
llm_client = llm_config.create_client()


def get_llm_config() -> LLMConfig:
    return llm_config


def get_paths() -> PathsConfig:
    return app_config.paths


def get_agent_config() -> AgentConfig:
    return app_config.agent


def get_training_config() -> TrainingConfig:
    return app_config.training