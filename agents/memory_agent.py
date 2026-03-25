import json
import os
from datetime import datetime
from typing import Any, Union

from utils.config import get_paths


class MemoryAgent:
    def __init__(self):
        paths = get_paths()
        self.memory_file = paths.memory
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {
                "iterations": [],
                "patterns": [],
                "best_iteration": None,
                "successful_feature_patterns": [],
                "failed_feature_patterns": [],
                "strategy_effectiveness": [],
                "metric_history": []
            }

    def _save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def store(
        self,
        iteration: int,
        metric: Union[float, None],
        reflection: str,
        features_used: list[str],
        model_used: str,
        feature_importance: Union[dict[str, float], None] = None,
        pruning_decisions: Union[list[str], None] = None,
        business_strategy: Union[str, None] = None,
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
            "business_strategy": business_strategy or "",
        }

        self.data["iterations"].append(entry)

        if metric is not None:
            self.data["metric_history"].append({"iteration": iteration, "metric": metric})
            if len(self.data["metric_history"]) > 50:
                self.data["metric_history"] = self.data["metric_history"][-50:]
            
            if self.data["best_iteration"] is None or metric > self.data["best_iteration"]["metric"]:
                self.data["best_iteration"] = entry.copy()

        self._extract_patterns(entry)
        self._extract_strategy_effectiveness(entry)
        self._save_memory()

    def store_feature_outcome(
        self,
        feature_name: str,
        outcome: Union["success", "failure"],
        metric_delta: Union[float, None] = None,
        reason: Union[str, None] = None,
    ):
        pattern_entry = {
            "feature_name": feature_name,
            "outcome": outcome,
            "metric_delta": metric_delta,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        
        if outcome == "success":
            self.data["successful_feature_patterns"].append(pattern_entry)
            if len(self.data["successful_feature_patterns"]) > 50:
                self.data["successful_feature_patterns"] = self.data["successful_feature_patterns"][-50:]
        else:
            self.data["failed_feature_patterns"].append(pattern_entry)
            if len(self.data["failed_feature_patterns"]) > 50:
                self.data["failed_feature_patterns"] = self.data["failed_feature_patterns"][-50:]
        
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

    def _extract_strategy_effectiveness(self, entry: dict):
        if entry.get("business_strategy") and entry.get("metric") is not None:
            strategy_entry = {
                "iteration": entry["iteration"],
                "strategy": entry["business_strategy"],
                "metric": entry["metric"],
                "timestamp": datetime.now().isoformat(),
            }
            self.data["strategy_effectiveness"].append(strategy_entry)
            if len(self.data["strategy_effectiveness"]) > 30:
                self.data["strategy_effectiveness"] = self.data["strategy_effectiveness"][-30:]
            self._save_memory()

    def get_successful_patterns(self, limit: int = 10) -> list[dict]:
        return self.data.get("successful_feature_patterns", [])[-limit:]

    def get_failed_patterns(self, limit: int = 10) -> list[dict]:
        return self.data.get("failed_feature_patterns", [])[-limit:]

    def get_trend_info(self) -> dict:
        history = self.data.get("metric_history", [])
        if len(history) < 2:
            return {"trend": "insufficient_data", "values": []}
        
        recent = history[-3:] if len(history) >= 3 else history
        
        if len(recent) >= 3:
            if recent[-1]["metric"] > recent[-2]["metric"] > recent[-3]["metric"]:
                return {"trend": "improving", "values": [h["metric"] for h in recent]}
            elif recent[-1]["metric"] < recent[-2]["metric"] < recent[-3]["metric"]:
                return {"trend": "declining", "values": [h["metric"] for h in recent]}
            else:
                return {"trend": "plateau", "values": [h["metric"] for h in recent]}
        elif len(recent) == 2:
            delta = recent[-1]["metric"] - recent[-2]["metric"]
            return {"trend": "improving" if delta > 0 else "declining", "values": [h["metric"] for h in recent], "delta": delta}
        
        return {"trend": "insufficient_data", "values": []}

    def get_strategy_context(self) -> str:
        if not self.data.get("strategy_effectiveness"):
            return ""
        
        strategies = self.data["strategy_effectiveness"]
        best = max(strategies, key=lambda x: x.get("metric", 0)) if strategies else None
        
        if best:
            return f"Best strategy (iter {best['iteration']}): {best['strategy'][:200]}... with metric {best.get('metric', 'N/A')}"
        return ""

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

        trend_info = self.get_trend_info()
        if trend_info["trend"] != "insufficient_data":
            context_parts.append(f"\n## TENDENZA RECENTE\n")
            context_parts.append(f"- Trend: {trend_info['trend']}")
            if trend_info.get("values"):
                vals = [f"{v:.4f}" for v in trend_info["values"]]
                context_parts.append(f"- Valori: {' -> '.join(vals)}")

        if self.data["patterns"]:
            context_parts.append("\n## PATTERN RICORRENTI\n")
            for p in self.data["patterns"][-5:]:
                context_parts.append(
                    f"- Iter {p['iteration']}: {p['high_importance_features']}"
                )

        if self.data.get("successful_feature_patterns"):
            context_parts.append("\n## FEATURE DI SUCCESSO\n")
            for p in self.data["successful_feature_patterns"][-3:]:
                context_parts.append(f"- {p['feature_name']}: {p.get('reason', 'N/A')}")

        if self.data.get("failed_feature_patterns"):
            context_parts.append("\n## FEATURE FALLITE\n")
            for p in self.data["failed_feature_patterns"][-3:]:
                context_parts.append(f"- {p['feature_name']}: {p.get('reason', 'N/A')}")

        if self.data["best_iteration"]:
            best = self.data["best_iteration"]
            context_parts.append(
                f"\n## MIGLIOR ITERAZIONE\n"
                f"- Iterazione {best['iteration']} con metrica {best.get('metric', 'N/A')}"
            )

        return "\n".join(context_parts)

    def get_best_iteration(self) -> Union[dict, None]:
        return self.data.get("best_iteration")

    def get_last_iteration(self) -> Union[dict, None]:
        if self.data["iterations"]:
            return self.data["iterations"][-1]
        return None
