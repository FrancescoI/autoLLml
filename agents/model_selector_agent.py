import json
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from prompts import MODEL_SELECTOR_SYSTEM_PROMPT, get_model_selection_prompt


class ModelSelectorAgent:
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.agent = AssistantAgent(
            name="ModelSelectorAgent",
            model_client=model_client,
            system_message=MODEL_SELECTOR_SYSTEM_PROMPT,
        )

    async def recommend_model(
        self,
        data_schema: str,
        data_sample: str,
        glossary: str,
        memory_context: str | None = None,
        feature_importance: dict | None = None,
    ) -> dict:
        prompt = get_model_selection_prompt(
            data_schema=data_schema,
            data_sample=data_sample,
            glossary=glossary,
            memory_context=memory_context,
            feature_importance=feature_importance,
        )

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
                return self._default_recommendation()

            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            data = json.loads(response_text.strip())
            return {
                'recommended_model': data.get('recommended_model', 'RandomForestClassifier'),
                'rationale': data.get('rationale', ''),
                'backup_model': data.get('backup_model', None)
            }
        except json.JSONDecodeError:
            return self._default_recommendation()

    def _default_recommendation(self) -> dict:
        return {
            'recommended_model': 'RandomForestClassifier',
            'rationale': 'Default: robusto e versatile per la maggior parte dei problemi.',
            'backup_model': 'GradientBoostingClassifier'
        }
