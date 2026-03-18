import re
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

SYSTEM_PROMPT = """Sei un Code Generation Agent specializzato in feature engineering per ML.

REGOLE FONDAMENTALI:
1. Esegui pruning delle feature irrilevanti o ridondanti
2. Gestisci sempre i missing value in ottica di best practice
3. Codifica le variabili categoriche adeguatamente
4. COSTRUISCI FEATURE DERIVATE CHE RIFLETTONO I FENOMENI DI BUSINESS discussi
5. TASSATIVO: NON COSTRUIRE MAI trasformate elementari (np.log, polinomi, radici quadrate)
6. Scegli il modello di machine learning più adatto al problema

IL TUO OUTPUT:
- def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame
- def get_model()
- Restituisci SOLO codice Python raw, formattato correttamente"""


class CodeAgent:
    def __init__(self, model_client: OpenAIChatCompletionClient):
        self.agent = AssistantAgent(
            name="CodeAgent",
            model_client=model_client,
            system_message=SYSTEM_PROMPT,
            reflect_on_tool_use=True,
            tool_call_generator=None,
        )

    async def generate_code(
        self,
        business_strategy: str,
        reflection_text: str,
        last_code: str,
        last_error: str = None
    ) -> str:
        prompt = f"""
        Genera il nuovo file `dynamic_features.py`.
        
        # BUSINESS FEATURE STRATEGY (Ancoraggio Top-Down)
        {business_strategy}
        
        # RIFLESSIONE SUL RUN PRECEDENTE (Ancoraggio Bottom-Up)
        {reflection_text}
        
        # CODICE PRECEDENTE
        {last_code}
        """
        
        if last_error:
            prompt += f"""
        ATTENZIONE - CRITICAL FIX REQUIRED: L'esecuzione precedente ha generato questo errore:
        {last_error}
        Correggi il codice per gestire questo crash.
        """
        
        prompt += """
        REGOLE FONDAMENTALI:
        1. Esegui pruning delle feature irrilevanti
        2. Gestisci sempre i missing value (mediana per cont, 'Unknown' per cat)
        3. Codifica le variabili categoriche
        4. COSTRUISCI FEATURE DERIVATE CHE RIFLETTONO I FENOMENI DI BUSINESS
        5. NON COSTRUIRE MAI trasformate elementari (np.log, polinomi, radici quadrate)
        6. Scegli il modello ML più adatto
        7. Il codice deve contenere `def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:` e `def get_model():`
        
        Restituisci SOLO codice Python raw, formattato correttamente. NON includere testo discorsivo o blocchi markdown.
        """
        
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
