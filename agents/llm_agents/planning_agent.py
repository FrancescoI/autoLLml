import json
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from prompts import SYSTEM_PROMPT, PLANNING_SYSTEM_PROMPT, get_business_strategy_prompt, get_iterative_strategy_prompt, get_model_selection_prompt


class PlanningAgent:
    def __init__(self, model_client: OpenAIChatCompletionClient, use_specialized_prompt: bool = True):
        system_msg = PLANNING_SYSTEM_PROMPT if use_specialized_prompt else SYSTEM_PROMPT
        self.agent = AssistantAgent(
            name="PlanningAgent",
            model_client=model_client,
            system_message=system_msg,
        )

    async def generate_strategy(self, glossary: str, data_schema: str, data_sample: str) -> dict:
        prompt = get_business_strategy_prompt(glossary, data_schema, data_sample)
        
        response = await self.agent.run(task=prompt)
        response_text = self._extract_text_from_response(response)
        return self._parse_strategy_response(response_text)

    async def generate_iterative_strategy(
        self,
        glossary: str,
        data_schema: str,
        data_sample: str,
        memory_context: str,
        last_iteration_results: dict | None = None,
        trend_context: str = "",
        strategy_context: str = ""
    ) -> dict:
        prompt = get_iterative_strategy_prompt(
            glossary=glossary,
            data_schema=data_schema,
            data_sample=data_sample,
            memory_context=memory_context,
            last_iteration_results=last_iteration_results,
            trend_context=trend_context,
            strategy_context=strategy_context
        )
        
        response = await self.agent.run(task=prompt)
        response_text = self._extract_text_from_response(response)
        return self._parse_iterative_strategy_response(response_text)

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
        return self._parse_model_response(response_text)

    def _extract_text_from_response(self, response) -> str:
        if hasattr(response, 'messages'):
            for msg in reversed(response.messages):
                if hasattr(msg, 'content'):
                    return str(msg.content)
        return str(response)

    def _parse_strategy_response(self, response_text: str) -> dict:
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

    def _parse_iterative_strategy_response(self, response_text: str) -> dict:
        try:
            if not response_text.strip():
                return self._default_strategy()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            data = json.loads(response_text.strip())
            return {
                'business_strategy': data.get('business_strategy', ''),
                'model_selection': data.get('model_selection', ''),
                'new_feature_ideas': data.get('new_feature_ideas', [])
            }
        except json.JSONDecodeError:
            return self._default_strategy()

    def _parse_model_response(self, response_text: str) -> dict:
        try:
            if not response_text.strip():
                return self._default_model_recommendation()

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
            return self._default_model_recommendation()

    def _default_strategy(self) -> dict:
        return {
            'business_strategy': 'Massimizza la monotonicità logica ed esplora interazioni non lineari basate sul glossario.',
            'model_selection': 'RandomForestClassifier: robusto a overfitting; GradientBoosting: alta precisione predittiva.',
            'new_feature_ideas': []
        }

    def _default_model_recommendation(self) -> dict:
        return {
            'recommended_model': 'RandomForestClassifier',
            'rationale': 'Default: robusto e versatile per la maggior parte dei problemi.',
            'backup_model': 'GradientBoostingClassifier'
        }