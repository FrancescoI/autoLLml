import json
import os
from datetime import datetime
from typing import Any


MEMORY_FILE = "memory.json"


class MemoryAgent:
    def __init__(self):
        self.memory_file = MEMORY_FILE
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {"iterations": [], "patterns": [], "best_iteration": None}

    def _save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def store(
        self,
        iteration: int,
        metric: float | None,
        reflection: str,
        features_used: list[str],
        model_used: str,
        feature_importance: dict[str, float] | None = None,
        pruning_decisions: list[str] | None = None,
    ):
        entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "metric": metric,
            "reflection": reflection,
            "features_used": features_used,
            "model_used": model_used,
            "feature_importance": feature_importance or {},
            "pruning_decisions": pruning_decisions or [],
        }

        self.data["iterations"].append(entry)

        if metric is not None:
            if self.data["best_iteration"] is None or metric > self.data["best_iteration"]["metric"]:
                self.data["best_iteration"] = entry.copy()

        self._extract_patterns(entry)
        self._save_memory()

    def _extract_patterns(self, entry: dict):
        if entry.get("feature_importance"):
            top_features = sorted(
                entry["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            if top_features:
                pattern = {
                    "iteration": entry["iteration"],
                    "high_importance_features": [f[0] for f in top_features],
                    "metric": entry.get("metric"),
                }
                self.data["patterns"].append(pattern)

        if len(self.data["patterns"]) > 20:
            self.data["patterns"] = self.data["patterns"][-20:]

        self._save_memory()

    def get_context(self) -> str:
        if not self.data["iterations"]:
            return "Nessuna iterazione precedente memorizzata."

        context_parts = ["## CRONOLOGIA ITERAZIONI PRECEDENTI\n"]

        for entry in self.data["iterations"]:
            metric_str = f"{entry['metric']:.4f}" if entry.get("metric") else "N/A"
            context_parts.append(
                f"- **Iterazione {entry['iteration']}** (metric: {metric_str}): "
                f"Model: {entry.get('model_used', 'N/A')}, "
                f"Features: {', '.join(entry.get('features_used', [])[:5])}"
            )

        if self.data["patterns"]:
            context_parts.append("\n## PATTERN RICORRENTI\n")
            for p in self.data["patterns"][-5:]:
                context_parts.append(
                    f"- Iter {p['iteration']}: {p['high_importance_features']}"
                )

        if self.data["best_iteration"]:
            best = self.data["best_iteration"]
            context_parts.append(
                f"\n## MIGLIOR ITERAZIONE\n"
                f"- Iterazione {best['iteration']} con metrica {best.get('metric', 'N/A')}"
            )

        return "\n".join(context_parts)

    def get_best_iteration(self) -> dict | None:
        return self.data.get("best_iteration")

    def get_last_iteration(self) -> dict | None:
        if self.data["iterations"]:
            return self.data["iterations"][-1]
        return None
