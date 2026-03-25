import json
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from prompts import PRUNING_SYSTEM_PROMPT, get_pruning_prompt


class PruningAgent:
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.agent = AssistantAgent(
            name="PruningAgent",
            model_client=model_client,
            system_message=PRUNING_SYSTEM_PROMPT,
        )

    async def analyze_and_prune(
        self,
        feature_importance: dict,
        correlations: dict | None = None,
        memory_context: str | None = None,
    ) -> dict:
        prompt = get_pruning_prompt(
            feature_importance=feature_importance,
            correlations=correlations,
            memory_context=memory_context,
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
                return self._default_pruning()

            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            data = json.loads(response_text.strip())
            return {
                'features_to_drop': data.get('features_to_drop', []),
                'rationale': data.get('rationale', {}),
                'suggestions': data.get('suggestions', [])
            }
        except json.JSONDecodeError:
            return self._default_pruning()

    def _default_pruning(self) -> dict:
        return {
            'features_to_drop': [],
            'rationale': {},
            'suggestions': []
        }

    def apply_auto_pruning(self, feature_importance: dict, correlations: dict | None = None) -> list[str]:
        to_drop = []

        for feat, importance in feature_importance.items():
            if importance < 0.01:
                to_drop.append(feat)

        if correlations:
            for (feat1, feat2), corr in correlations.items():
                if abs(corr) > 0.9:
                    imp1 = feature_importance.get(feat1, 0)
                    imp2 = feature_importance.get(feat2, 0)
                    if imp1 < imp2 and feat1 not in to_drop:
                        to_drop.append(feat1)
                    elif feat2 not in to_drop:
                        to_drop.append(feat2)

        return list(set(to_drop))
