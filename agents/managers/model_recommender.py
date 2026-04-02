class ModelRecommender:
    """Raccomanda e traccia il modello corrente."""

    def __init__(self, planning_agent):
        self.planning_agent = planning_agent
        self.current_model: str | None = None

    async def recommend_model(
        self,
        data_schema: str,
        data_sample: str,
        glossary: str,
        memory_context: str,
        feature_importance: dict | None
    ) -> str:
        """Raccomanda un modello basato sul contesto."""
        print("[*] Raccomandazione modello con ModelSelectorAgent...")
        model_rec = await self.planning_agent.recommend_model(
            data_schema=data_schema,
            data_sample=data_sample,
            glossary=glossary,
            memory_context=memory_context,
            feature_importance=feature_importance
        )
        self.current_model = model_rec['recommended_model']
        print(f"[*] Modello raccomandato: {self.current_model}")
        return self.current_model

    def get_current_model(self) -> str | None:
        return self.current_model