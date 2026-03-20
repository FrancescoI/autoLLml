import json
import os
from pathlib import Path
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image
from autogen_ext.models.openai import OpenAIChatCompletionClient

from prompts import SYSTEM_PROMPT, get_reflection_prompt


class EvaluatorAgent:
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.agent = AssistantAgent(
            name="EvaluatorAgent",
            model_client=model_client,
            system_message=SYSTEM_PROMPT,
        )

    async def evaluate_and_reflect(
        self,
        iter_num: int,
        evaluation_report: dict,
        glossary: str,
        plot_paths: list[str] | None = None,
        feature_importance: dict | None = None
    ) -> str:
        plot_paths = plot_paths or []
        
        text_prompt = get_reflection_prompt(
            iter_num,
            json.dumps(evaluation_report, indent=2),
            glossary,
            json.dumps(feature_importance or {}, indent=2) if feature_importance else None
        )
        
        content: list[str | Image] = [text_prompt]
        
        for path in plot_paths[:10]:
            if os.path.isfile(path):
                content.append(Image.from_file(path))
        
        message = MultiModalMessage(content=content, source="user")
        
        response = await self.agent.run(task=message)
        
        return self._extract_text_from_response(response)

    def _extract_text_from_response(self, response) -> str:
        if hasattr(response, 'messages'):
            for msg in reversed(response.messages):
                if hasattr(msg, 'content'):
                    return str(msg.content)
        return str(response)
