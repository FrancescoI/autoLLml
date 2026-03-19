import re
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from prompts import SYSTEM_PROMPT, get_code_generation_prompt, get_error_fix_prompt


class CodeAgent:
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.agent = AssistantAgent(
            name="CodeAgent",
            model_client=model_client,
            system_message=SYSTEM_PROMPT,
        )

    async def generate_code(
        self,
        business_strategy: str,
        reflection_text: str,
        last_code: str,
        last_error: str | None = None
    ) -> str:
        prompt = get_code_generation_prompt(business_strategy, reflection_text, last_code, last_error)
        
        response = await self.agent.run(task=prompt)
        return self._clean_code_output(self._extract_text_from_response(response))

    async def fix_code_error(self, error_message: str, previous_code: str) -> str:
        prompt = get_error_fix_prompt(error_message, previous_code)
        
        response = await self.agent.run(task=prompt)
        return self._clean_code_output(self._extract_text_from_response(response))

    def _extract_text_from_response(self, response) -> str:
        if hasattr(response, 'messages'):
            for msg in reversed(response.messages):
                if hasattr(msg, 'content'):
                    return str(msg.content)
        return str(response)

    def _clean_code_output(self, raw_response: str) -> str:
        lines = raw_response.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith("```")]
        return "\n".join(cleaned_lines)
