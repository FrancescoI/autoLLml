import os
import json
import glob
import re
from utils.memory_store import MemoryStore


class EvaluationPipeline:
    """Orchestra la valutazione, riflessione, estrazione feature e storage in memoria."""

    def __init__(self, evaluator_agent, memory_store: MemoryStore, paths):
        self.evaluator_agent = evaluator_agent
        self.memory_store = memory_store
        self.paths = paths

    async def evaluate_iteration(
        self,
        iter_num: int,
        report: dict,
        glossary: str,
        data_schema: str,
        data_sample: str,
        trend_context: str,
        successful_patterns: str,
        failed_patterns: str,
        business_strategy: str,
        current_model: str,
        features_to_drop: list[str]
    ) -> tuple[str, dict]:
        """Valuta l'iterazione e restituisce riflessione e feature importance."""
        plot_dir = os.path.join(self.paths.output_dir, f"iter_{iter_num}")
        plot_paths = []
        if os.path.isdir(plot_dir):
            plot_paths = sorted(glob.glob(os.path.join(plot_dir, "*.png")), key=os.path.getmtime, reverse=True)[:10]

        feature_importance = report.get("feature_importance", {})

        print(f"[*] Analisi risultati con EvaluatorAgent...")
        reflection_text = await self.evaluator_agent.evaluate_and_reflect(
            iter_num,
            report,
            glossary,
            plot_paths,
            feature_importance,
            trend_context,
            successful_patterns,
            failed_patterns
        )
        safe_to_print = reflection_text.encode('ascii', 'ignore').decode('ascii')
        print(f"\n--- RIFLESSIONE ITERAZIONE {iter_num} ---\n{safe_to_print[:500]}...\n")

        # Store in memory
        if report.get("score_mean") is not None:
            self._store_in_memory(
                iteration=iter_num,
                metric=report["score_mean"],
                reflection=reflection_text,
                feature_importance=feature_importance,
                features_to_drop=features_to_drop,
                business_strategy=business_strategy,
                model_used=current_model
            )

        return reflection_text, feature_importance

    def _store_in_memory(
        self,
        iteration: int,
        metric: float,
        reflection: str,
        feature_importance: dict,
        features_to_drop: list[str],
        business_strategy: str | None = None,
        model_used: str | None = None
    ):
        features_used = self.extract_implemented_features()

        self.memory_store.store(
            iteration=iteration,
            metric=metric,
            reflection=reflection,
            features_used=features_used,
            model_used=model_used or "Unknown",
            feature_importance=feature_importance,
            pruning_decisions=features_to_drop,
            business_strategy=business_strategy
        )
        print(f"[*] Dati iterazione {iteration} memorizzati.")

    def extract_implemented_features(self) -> list:
        try:
            with open("dynamic_features.py", "r", encoding="utf-8") as f:
                source = f.read()
            matches = re.findall(r'data\[[\'"](\w+)[\'"]\]\s*=', source)
            return [m for m in matches if m not in ['target', 'consumo_annuo', 'default_flag'] and m not in set()]
        except Exception:
            return []