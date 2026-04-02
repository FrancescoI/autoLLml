from utils.memory_store import MemoryStore


class TrendAnalyzer:
    """Analizza le tendenze delle prestazioni e determina l'early stopping."""

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def get_trend_context(self) -> str:
        """Restituisce il contesto della tendenza."""
        trend_info = self.memory_store.get_trend_info()
        if trend_info["trend"] == "insufficient_data":
            return ""

        trend = trend_info["trend"]
        values = trend_info.get("values", [])
        vals_str = " -> ".join([f"{v:.4f}" for v in values]) if values else ""

        if trend == "improving":
            return f"Tendenza: MIGLIORAMENTO ({vals_str})"
        elif trend == "declining":
            return f"Tendenza: PEGGIORAMENTO ({vals_str})"
        else:
            return f"Tendenza: STAGNO/PLATEAU ({vals_str})"

    def get_feature_patterns_context(self) -> tuple[str, str]:
        """Restituisce i pattern di successo e fallimento delle feature."""
        successful = self.memory_store.get_successful_patterns(limit=3)
        failed = self.memory_store.get_failed_patterns(limit=3)

        success_str = "\n".join([f"- {p['feature_name']}: {p.get('reason', 'N/A')}" for p in successful]) if successful else "Nessuna feature di successo registrata."
        fail_str = "\n".join([f"- {p['feature_name']}: {p.get('reason', 'N/A')}" for p in failed]) if failed else "Nessuna feature fallita registrata."

        return success_str, fail_str

    def should_stop_early(self, threshold: float = 0.01, window: int = 3) -> bool:
        """Determina se fermarsi presto basato sul miglioramento medio."""
        history = self.memory_store.data.get("metric_history", [])
        if len(history) < window + 1:
            return False

        recent = history[-window:]
        improvements = []
        for i in range(1, len(recent)):
            delta = recent[i]["metric"] - recent[i-1]["metric"]
            improvements.append(delta)

        avg_improvement = sum(improvements) / len(improvements)

        if avg_improvement < threshold:
            print(f"[*] Early stopping: miglioramento medio {avg_improvement:.4f} < soglia {threshold}")
            return True

        return False