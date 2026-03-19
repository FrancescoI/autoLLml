import json
from autogen_agentchat.agents import AssistantAgent
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
        glossary: str
    ) -> str:
        prompt = get_reflection_prompt(iter_num, json.dumps(evaluation_report, indent=2), glossary)
        
        response = await self.agent.run(task=prompt)
        
        return self._extract_text_from_response(response)

    def _extract_text_from_response(self, response) -> str:
        if hasattr(response, 'messages'):
            for msg in reversed(response.messages):
                if hasattr(msg, 'content'):
                    return str(msg.content)
        return str(response)
