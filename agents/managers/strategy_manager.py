class StrategyManager:
    """Gestisce la generazione e l'aggiornamento della strategia aziendale."""

    def __init__(self, planning_agent):
        self.planning_agent = planning_agent
        self.business_strategy: str | None = None

    async def generate_initial_strategy(self, glossary: str, data_schema: str, data_sample: str) -> str:
        """Genera la strategia iniziale."""
        print("[*] Generazione strategia iniziale con StrategyAgent...")
        strategy_result = await self.planning_agent.generate_strategy(
            glossary,
            data_schema,
            data_sample
        )
        self.business_strategy = strategy_result['business_strategy']
        print(f"[*] Strategia generata: {self.business_strategy[:100]}...")
        return self.business_strategy

    async def regenerate_with_context(
        self,
        glossary: str,
        data_schema: str,
        data_sample: str,
        memory_context: str,
        last_iter: dict,
        trend_context: str,
        strategy_context: str
    ) -> tuple[str, list]:
        """Rigenera la strategia con contesto memoria."""
        print("[*] Riesecuzione strategia con contesto memoria...")
        strategy_result = await self.planning_agent.generate_iterative_strategy(
            glossary,
            data_schema,
            data_sample,
            memory_context,
            last_iter,
            trend_context,
            strategy_context
        )
        self.business_strategy = strategy_result.get('business_strategy', self.business_strategy)
        new_feature_ideas = strategy_result.get('new_feature_ideas', [])
        if new_feature_ideas:
            print(f"[*] Nuove idee feature: {new_feature_ideas}")
        return self.business_strategy, new_feature_ideas

    def get_current_strategy(self) -> str | None:
        return self.business_strategy