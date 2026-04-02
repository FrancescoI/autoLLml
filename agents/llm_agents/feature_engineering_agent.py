import json
import re
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from prompts import SYSTEM_PROMPT, FEATURE_ENGINEERING_SYSTEM_PROMPT, get_code_generation_prompt, get_error_fix_prompt, get_pruning_prompt


class FeatureEngineeringAgent:
    def __init__(self, model_client: OpenAIChatCompletionClient, use_specialized_prompt: bool = True):
        system_msg = FEATURE_ENGINEERING_SYSTEM_PROMPT if use_specialized_prompt else SYSTEM_PROMPT
        self.agent = AssistantAgent(
            name="FeatureEngineeringAgent",
            model_client=model_client,
            system_message=system_msg,
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
        return self._parse_pruning_response(response_text)

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

    def _parse_pruning_response(self, response_text: str) -> dict:
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