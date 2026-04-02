class CodeExecutionPipeline:
    """Gestisce la generazione, esecuzione e fix del codice per le feature."""

    def __init__(self, feature_engineering_agent):
        self.feature_engineering_agent = feature_engineering_agent

    async def generate_code(
        self,
        business_strategy: str,
        reflection_text: str,
        last_code: str,
        new_feature_ideas: list | None
    ) -> str:
        """Genera nuovo codice per le feature."""
        print("[*] Generazione codice con CodeAgent...")
        new_code = await self.feature_engineering_agent.generate_code(
            business_strategy,
            reflection_text,
            last_code,
            new_feature_ideas
        )

        if new_code.strip():
            with open("dynamic_features.py", "w", encoding="utf-8") as f:
                f.write(new_code)
        else:
            print("[!] CodeAgent ha restituito codice vuoto.")

        return new_code

    def load_last_code(self) -> str:
        """Carica l'ultimo codice dalle feature."""
        try:
            with open("dynamic_features.py", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""