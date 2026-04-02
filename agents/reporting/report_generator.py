import os
import json
import datetime
import re


class ReportGenerator:
    """Genera e aggiorna i report Markdown."""

    def __init__(self, paths):
        self.paths = paths

    def update_report(
        self,
        iter_num: int,
        business_strategy: str | None,
        prev_metric: float | None,
        extract_implemented_features_func
    ):
        """Aggiorna il report Markdown."""
        md_path = self.paths.evaluation_report_md
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(self.paths.evaluation_report, "r", encoding="utf-8") as f:
                report = json.load(f)
        except Exception:
            return

        task_type = report.get("task_type", "unknown")
        metric_name = report.get("metric_name", "score")
        score_mean = report.get("score_mean", float("nan"))
        score_std = report.get("score_std", float("nan"))
        precision = report.get("precision")
        recall = report.get("recall")
        auc_roc = report.get("auc_roc")
        n_feat = report.get("num_features", "?")

        if prev_metric is not None:
            delta = score_mean - prev_metric
            delta_str = f"{delta:+.4f}  {'▲' if delta >= 0 else '▼'}"
        else:
            delta_str = "— (baseline)"

        top_corr = report.get("top_correlations_with_target", {})
        corr_md = "\n".join(f"  - `{k}`: {v:+.4f}" for k, v in list(top_corr.items())[:10]) or "  *(nessuna)*"

        fi = report.get("feature_importance", {})
        fi_md = "\n".join(f"  - `{k}`: {v:.4f}" for k, v in list(fi.items())[:10]) or "  *(non disponibile)*"

        impl_features = extract_implemented_features_func()
        feat_md = "\n".join(f"  - `{f}`" for f in impl_features) or "  *(baseline)*"

        strat_md = business_strategy.strip() if business_strategy else "*(non generata — run baseline)*"

        extra_metrics = ""
        if precision is not None:
            extra_metrics += f"| Precision (CV-5) | {precision:.4f} |\n"
        if recall is not None:
            extra_metrics += f"| Recall (CV-5) | {recall:.4f} |\n"
        if auc_roc is not None:
            extra_metrics += f"| AUC-ROC (CV-5) | {auc_roc:.4f} |\n"

        section = (
            f"\n---\n"
            f"## Run {iter_num}  —  {timestamp}\n\n"
            f"### Metriche\n"
            f"| Metrica | Valore |\n"
            f"|---------|--------|\n"
            f"| Task Type | {task_type.upper()} |\n"
            f"| {metric_name} Mean (CV-5) | **{score_mean:.4f}** ± {score_std:.4f} |\n"
            f"{extra_metrics}"
            f"| Δ vs run precedente | {delta_str} |\n"
            f"| Numero feature in input | {n_feat} |\n\n"
            f"### Top correlazioni con il target (Pearson)\n{corr_md}\n\n"
            f"### Feature importance (top 10)\n{fi_md}\n\n"
            f"### Feature implementate in questa run\n{feat_md}\n\n"
            f"### Business strategy applicata\n{strat_md}\n"
        )

        if not os.path.exists(md_path):
            header = (
                "# Evaluation Report — Cronologia delle Run\n\n"
                "> Generato automaticamente dall'OrchestratorAgent con Microsoft Agent Framework.\n"
            )
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(header)

        with open(md_path, "a", encoding="utf-8") as f:
            f.write(section)

        print(f"[*] evaluation_report.md aggiornato (Run {iter_num}).")