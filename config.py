import os
from dotenv import load_dotenv

load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

llm_client = OpenAIChatCompletionClient(
    model="gpt-5-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

llm_config = {
    "model_client": llm_client,
    "temperature": 0.7,
    "reasoning_effort": "low",
}

def get_llm_config():
    return llm_config.copy()
