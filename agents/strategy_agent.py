import json
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from dto import StrategyInput, StrategyOutput

SYSTEM_PROMPT = """Sei un Senior Data Scientist esperto in Machine Learning e Artificial Intelligence, con una forte mentalità orientata al Business e al Domain Knowledge.

I TUOI PRINCIPI FONDAMENTALI:
1. Semantica prima della statistica: Ogni feature che crei deve avere un senso nel mondo reale legato allo specifico dominio del problema.
2. Clean Code & robustezza: Scrivi codice Python pulito, modulare e difensivo.
3. Collinearità e Pruning: Rimuovi le variabili superflue o che apportano solo rumore ridondante.
4. Niente Math-Bruteforcing: Rifiutati categoricamente di applicare trasformazioni matematiche brute.

IL TUO COMPITO COME STRATEGY AGENT:
Analizza il glossario semantico e lo schema dei dati per generare strategie di business applicabili tramite feature engineering.
Restituisci ESCLUSIVAMENTE un JSON valido con:
- "business_strategy": 3-5 strategie di business basate su feature crossing, ratio, o aggregazioni
- "model_selection": 2-3 modelli ML adatti al problema con motivazione"""


class StrategyAgent:
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.agent = AssistantAgent(
            name="StrategyAgent",
            model_client=model_client,
            system_message=SYSTEM_PROMPT,
        )

    async def generate_strategy(self, glossary: str, data_schema: str, data_sample: str) -> StrategyOutput:
        validated_input = StrategyInput(
            glossary=glossary,
            data_schema=data_schema,
            data_sample=data_sample
        )
        
        prompt = f"""
        Analizza i seguenti metadati di progetto:
        
        # GLOSSARIO SEMANTICO
        {validated_input.glossary}
        
        # SCHEMA E SAMPLE DATI
        {validated_input.data_schema}
        {validated_input.data_sample}
        
        Obiettivo: Genera strategie di business per feature engineering predittivo basato sul dominio.
        
        Restituisci ESCLUSIVAMENTE un JSON valido (senza blocchi markdown) con:
        - "business_strategy": Stringa con 3-5 strategie di business applicabili tramite feature crossing, ratio, o aggregazioni
        - "model_selection": Stringa con 2 o 3 modelli di ML adatti al problema, con motivazione
        """
        
        response = await self.agent.run(task=prompt)
        response_text = self._extract_text_from_response(response)
        return self._parse_response(response_text)

    def _extract_text_from_response(self, response) -> str:
        if hasattr(response, 'messages'):
            for msg in reversed(response.messages):
                if hasattr(msg, 'content'):
                    return str(msg.content)
        return str(response)

    def _parse_response(self, response_text: str) -> StrategyOutput:
        try:
            if not response_text.strip():
                return self._default_strategy()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
            
            strategy_data = json.loads(response_text)
            return StrategyOutput(
                business_strategy=strategy_data.get('business_strategy', ''),
                model_selection=strategy_data.get('model_selection', '')
            )
        except json.JSONDecodeError:
            return self._default_strategy()

    def _default_strategy(self) -> StrategyOutput:
        return StrategyOutput(
            business_strategy='Massimizza la monotonicità logica ed esplora interazioni non lineari basate sul glossario.',
            model_selection='RandomForestClassifier: robusto a overfitting; GradientBoosting: alta precisione predittiva.'
        )
