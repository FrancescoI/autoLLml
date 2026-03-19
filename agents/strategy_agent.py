import json
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from prompts import SYSTEM_PROMPT, get_business_strategy_prompt


class StrategyAgent:
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.agent = AssistantAgent(
            name="StrategyAgent",
            model_client=model_client,
            system_message=SYSTEM_PROMPT,
        )

    async def generate_strategy(self, glossary: str, data_schema: str, data_sample: str) -> dict:
        prompt = get_business_strategy_prompt(glossary, data_schema, data_sample)
        
        response = await self.agent.run(task=prompt)
        response_text = self._extract_text_from_response(response)
        return self._parse_response(response_text)

    def _extract_text_from_response(self, response) -> str:
        if hasattr(response, 'messages'):
            for msg in reversed(response.messages):
                if hasattr(msg, 'content'):
                    return str(msg.content)
        return str(response)

    def _parse_response(self, response_text: str) -> dict:
        try:
            if not response_text.strip():
                return self._default_strategy()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
            
            strategy_data = json.loads(response_text)
            return {
                'business_strategy': strategy_data.get('business_strategy', ''),
                'model_selection': strategy_data.get('model_selection', '')
            }
        except json.JSONDecodeError:
            return self._default_strategy()

    def _default_strategy(self) -> dict:
        return {
            'business_strategy': 'Massimizza la monotonicità logica ed esplora interazioni non lineari basate sul glossario.',
            'model_selection': 'RandomForestClassifier: robusto a overfitting; GradientBoosting: alta precisione predittiva.'
        }
