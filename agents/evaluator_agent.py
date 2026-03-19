import json
import os
import glob
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from dto import EvaluatorInput, EvaluatorOutput

SYSTEM_PROMPT = """Sei un Evaluator Agent specializzato nell'analisi di risultati ML e riflessione strategica.

IL TUO COMPITO:
1. Analizzare i risultati dell'iterazione corrente
2. Fornire riflessioni dettagliate sullo stato del modello
3. Suggerire miglioramenti basati su evidenze empiriche

USA I GRAFICI ALLEGATI per:
- Identificare feature con alta separabilità
- Rilevare outlier o sovrapposizioni
- Valutare la qualità delle distribuzioni

ATTENZIONE: Non proporre MAI trasformate elementari (logaritmi, exp, polinomi)."""


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
        plot_dir: str = "evaluation_plots"
    ) -> EvaluatorOutput:
        validated_input = EvaluatorInput(
            iter_num=iter_num,
            evaluation_report=evaluation_report,
            glossary=glossary,
            plot_dir=plot_dir
        )
        
        plot_paths = self._load_evaluation_plots(validated_input.plot_dir)
        
        prompt = self._build_prompt(
            validated_input.iter_num,
            validated_input.evaluation_report,
            validated_input.glossary,
            plot_paths
        )
        
        response = await self.agent.run(task=prompt)
        
        reflection_text = self._extract_text_from_response(response)
        return EvaluatorOutput(reflection_text=reflection_text)

    def _build_prompt(
        self,
        iter_num: int,
        evaluation_report: dict,
        glossary: str,
        plot_paths: list
    ) -> str:
        prompt = f"""
        Analizza i risultati dell'Iterazione {iter_num}:
        
        # REPORT DI VALUTAZIONE
        {json.dumps(evaluation_report, indent=2)}
        
        # GLOSSARIO
        {glossary}
        
        # GRAFICI ALLEGATI
        Sono allegati {len(plot_paths)} boxplot con le distribuzioni delle feature più correlate.
        Usa questi grafici per identificare separabilità e anomalie.
        
        TASK:
        1. Quali feature derivate hanno dimostrato alta importanza e qual è la loro logica fenomenologica?
        2. Quali variabili causano ridondanza e vanno eliminate?
        3. Quali nuove logiche di business andrebbero create per massimizzare il segnale?
        4. Seleziona un modello appropriato (logistica, Random Forest, o BoostedTree).
        
        Rispondi in modo tecnico e analitico (max 400 parole). NON scrivere codice Python.
        """
        return prompt

    def _load_evaluation_plots(self, plot_dir: str, max_plots: int = 10) -> list:
        if not os.path.isdir(plot_dir):
            return []
        pattern = os.path.join(plot_dir, "*.png")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        return files[:max_plots]

    def _extract_text_from_response(self, response) -> str:
        if hasattr(response, 'messages'):
            for msg in reversed(response.messages):
                if hasattr(msg, 'content'):
                    return str(msg.content)
        return str(response)
