import os
from dotenv import load_dotenv

load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

llm_client = OpenAIChatCompletionClient(
    model="gpt-5.4-nano",
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_info=ModelInfo(
        vision=False,
        function_calling=False,
        json_output=False,
        family="unknown",
        structured_output=False,
    ),
)

llm_config = {
    "model_client": llm_client,
    "temperature": 0.7,
    "reasoning_effort": "low",
}

def get_llm_config():
    return llm_config.copy()

MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "AutoLLml_Experiments")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
