class PruningAnalyzer:
    """Analizza e raccomanda il pruning delle feature."""

    def __init__(self, feature_engineering_agent):
        self.feature_engineering_agent = feature_engineering_agent

    async def analyze_and_prune(
        self,
        feature_importance: dict,
        correlations: dict | None,
        memory_context: str
    ) -> list[str]:
        """Analizza e restituisce le feature da rimuovere."""
        print("[*] Analisi pruning con PruningAgent...")
        pruning_result = await self.feature_engineering_agent.analyze_and_prune(
            feature_importance=feature_importance,
            correlations=correlations if correlations else None,
            memory_context=memory_context
        )
        features_to_drop = pruning_result.get('features_to_drop', [])
        if features_to_drop:
            print(f"[*] Feature da rimuovere: {features_to_drop}")
        return features_to_drop