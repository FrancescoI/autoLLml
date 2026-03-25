import json
import os
import subprocess
import datetime
import re
import pandas as pd
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .strategy_agent import StrategyAgent
from .code_agent import CodeAgent
from .evaluator_agent import EvaluatorAgent
from .memory_agent import MemoryAgent
from .model_selector_agent import ModelSelectorAgent
from .pruning_agent import PruningAgent
from utils.config import get_paths, get_agent_config


class OrchestratorAgent:
    def __init__(
        self,
        model_client: OpenAIChatCompletionClient,
        max_iterations: int | None = None
    ):
        paths = get_paths()
        agent_config = get_agent_config()
        
        self.paths = paths
        self.max_iterations = max_iterations if max_iterations is not None else agent_config.max_iterations
        self.max_error_retries = agent_config.max_error_retries
        self.history = []
        self.business_strategy: str | None = None
        self.current_model: str | None = None
        
        with open(paths.glossary, "r", encoding="utf-8") as f:
            self.glossary = f.read()
        
        try:
            df = pd.read_csv(paths.data, encoding="latin-1")
            self.data_schema = str(df.dtypes.to_dict())
            self.data_sample = str(df.head(1).to_dict())
        except FileNotFoundError:
            print(f"[!] Assicurati di inserire il file in {paths.data} prima di avviare.")
            self.data_schema = "Dati non caricati."
            self.data_sample = "N/A"
        
        self.strategy_agent = StrategyAgent(model_client)
        self.code_agent = CodeAgent(model_client)
        self.evaluator_agent = EvaluatorAgent(model_client)
        self.memory_agent = MemoryAgent()
        self.model_selector = ModelSelectorAgent(model_client)
        self.pruning_agent = PruningAgent(model_client)
        
        print("[*] OrchestratorAgent inizializzato con tutti gli agenti")

    def _get_trend_context(self) -> str:
        trend_info = self.memory_agent.get_trend_info()
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

    def _get_feature_patterns_context(self) -> tuple[str, str]:
        successful = self.memory_agent.get_successful_patterns(limit=3)
        failed = self.memory_agent.get_failed_patterns(limit=3)
        
        success_str = "\n".join([f"- {p['feature_name']}: {p.get('reason', 'N/A')}" for p in successful]) if successful else "Nessuna feature di successo registrata."
        fail_str = "\n".join([f"- {p['feature_name']}: {p.get('reason', 'N/A')}" for p in failed]) if failed else "Nessuna feature fallita registrata."
        
        return success_str, fail_str

    def _should_stop_early(self, threshold: float = 0.01, window: int = 3) -> bool:
        history = self.memory_agent.data.get("metric_history", [])
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

    async def run_iteration(self, iter_num: int) -> dict:
        print(f"\n================ AVVIO ITERAZIONE {iter_num} ================")
        
        if iter_num == 1:
            return await self._run_baseline(iter_num)
        
        return await self._run_llm_iteration(iter_num)

    async def _run_baseline(self, iter_num: int) -> dict:
        print("[*] Prima run baseline (nessuna chiamata LLM). Esecuzione training loop...")
        
        cmd = ["python", "-m", "train", "--iter", str(iter_num)]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout = result.stdout.strip()
        
        iteration_data = {"iteration": iter_num, "metric": None, "error": None}
        
        if "SUCCESS_METRIC" in stdout:
            metric_val = float(stdout.split("SUCCESS_METRIC: ")[1].split("\n")[0])
            iteration_data["metric"] = metric_val
            print(f"[+] Iterazione {iter_num} - Successo. Metrica (F1): {metric_val:.4f}")
            self._update_report(iter_num, None, None)
        else:
            iteration_data["error"] = stdout[-1000:]
            print(f"[!] Iterazione {iter_num} - Fallita")
        
        self.history.append(iteration_data)
        return iteration_data

    async def _run_llm_iteration(self, iter_num: int) -> dict:
        memory_context = self.memory_agent.get_context()
        
        if self.business_strategy is None:
            print("[*] Generazione strategia iniziale con StrategyAgent...")
            strategy_result = await self.strategy_agent.generate_strategy(
                self.glossary,
                self.data_schema,
                self.data_sample
            )
            self.business_strategy = strategy_result['business_strategy']
            print(f"[*] Strategia generata: {self.business_strategy[:100]}...")
        else:
            print("[*] Riesecuzione strategia con contesto memoria...")
            last_iter = self.memory_agent.get_last_iteration()
            trend_context = self._get_trend_context()
            strategy_context = self.memory_agent.get_strategy_context()
            strategy_result = await self.strategy_agent.generate_iterative_strategy(
                self.glossary,
                self.data_schema,
                self.data_sample,
                memory_context,
                last_iter,
                trend_context,
                strategy_context
            )
            self.business_strategy = strategy_result.get('business_strategy', self.business_strategy)
            new_feature_ideas = strategy_result.get('new_feature_ideas', [])
            if new_feature_ideas:
                print(f"[*] Nuove idee feature: {new_feature_ideas}")
        
        with open(self.paths.evaluation_report, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        plot_dir = os.path.join(self.paths.output_dir, f"iter_{iter_num}")
        plot_paths = []
        if os.path.isdir(plot_dir):
            import glob
            plot_paths = sorted(glob.glob(os.path.join(plot_dir, "*.png")), key=os.path.getmtime, reverse=True)[:10]
        
        feature_importance = report.get("feature_importance", {})
        
        print("[*] Raccomandazione modello con ModelSelectorAgent...")
        model_rec = await self.model_selector.recommend_model(
            data_schema=self.data_schema,
            data_sample=self.data_sample,
            glossary=self.glossary,
            memory_context=memory_context,
            feature_importance=feature_importance if feature_importance else None
        )
        self.current_model = model_rec['recommended_model']
        print(f"[*] Modello raccomandato: {self.current_model}")
        
        print("[*] Analisi pruning con PruningAgent...")
        correlations = report.get("correlations", {})
        pruning_result = await self.pruning_agent.analyze_and_prune(
            feature_importance=feature_importance,
            correlations=correlations if correlations else None,
            memory_context=memory_context
        )
        features_to_drop = pruning_result.get('features_to_drop', [])
        if features_to_drop:
            print(f"[*] Feature da rimuovere: {features_to_drop}")
        
        print(f"[*] Analisi risultati con EvaluatorAgent...")
        trend_context = self._get_trend_context()
        successful_patterns, failed_patterns = self._get_feature_patterns_context()
        reflection_text = await self.evaluator_agent.evaluate_and_reflect(
            iter_num,
            report,
            self.glossary,
            plot_paths,
            feature_importance,
            trend_context,
            successful_patterns,
            failed_patterns
        )
        safe_to_print = reflection_text.encode('ascii', 'ignore').decode('ascii')
        print(f"\n--- RIFLESSIONE ITERAZIONE {iter_num} ---\n{safe_to_print[:500]}...\n")
        
        last_code = self._load_last_code()
        
        print("[*] Generazione codice con CodeAgent...")
        new_code = await self.code_agent.generate_code(
            self.business_strategy,
            reflection_text,
            last_code,
            None
        )
        
        if new_code.strip():
            with open("dynamic_features.py", "w", encoding="utf-8") as f:
                f.write(new_code)
        else:
            print("[!] CodeAgent ha restituito codice vuoto.")
        
        retries = 0
        current_code = new_code
        final_metric = None
        final_error = None
        
        while retries < self.max_error_retries:
            print(f"[*] Esecuzione training loop... (attempt {retries + 1})")
            
            cmd = ["python", "-m", "train", "--iter", str(iter_num)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            stdout = result.stdout.strip()
            
            if "SUCCESS_METRIC" in stdout:
                metric_val = float(stdout.split("SUCCESS_METRIC: ")[1].split("\n")[0])
                final_metric = metric_val
                final_error = None
                print(f"[+] Training completato. Metrica (F1): {metric_val:.4f}")
                break
            else:
                retries += 1
                if retries >= self.max_error_retries:
                    final_error = stdout[-1000:]
                    print(f"[!] Training fallito dopo {self.max_error_retries} tentativi")
                    break
                
                error_msg = stdout[-1000:]
                print(f"[!] Training fallito. Retry {retries}/{self.max_error_retries} con fix errore...")
                print(f"[*] Errore: {error_msg[:200]}...")
                
                current_code = await self.code_agent.fix_code_error(
                    error_message=error_msg,
                    previous_code=current_code
                )
                
                if current_code.strip():
                    with open("dynamic_features.py", "w", encoding="utf-8") as f:
                        f.write(current_code)
                else:
                    print("[!] CodeAgent ha restituito codice vuoto.")
        
        iteration_data = {"iteration": iter_num, "metric": final_metric, "error": final_error}
        
        if final_metric is not None:
            self._store_in_memory(
                iteration=iter_num,
                metric=final_metric,
                reflection=reflection_text,
                feature_importance=feature_importance,
                features_to_drop=features_to_drop,
                business_strategy=self.business_strategy
            )
            self._update_report(iter_num, self.business_strategy, self.history[-1].get('metric') if self.history else None)
        elif final_error:
            print(f"[!] Iterazione {iter_num} - Fallita")
        
        self.history.append(iteration_data)
        return iteration_data

    def _store_in_memory(
        self,
        iteration: int,
        metric: float,
        reflection: str,
        feature_importance: dict,
        features_to_drop: list[str],
        business_strategy: str | None = None
    ):
        features_used = self._extract_implemented_features()
        
        self.memory_agent.store(
            iteration=iteration,
            metric=metric,
            reflection=reflection,
            features_used=features_used,
            model_used=self.current_model or "Unknown",
            feature_importance=feature_importance,
            pruning_decisions=features_to_drop,
            business_strategy=business_strategy
        )
        print(f"[*] Dati iterazione {iteration} memorizzati.")

    def _load_last_code(self) -> str:
        try:
            with open("dynamic_features.py", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def _extract_implemented_features(self) -> list:
        try:
            with open("dynamic_features.py", "r", encoding="utf-8") as f:
                source = f.read()
            matches = re.findall(r'data\[[\'"](\w+)[\'"]\]\s*=', source)
            return [m for m in matches if m not in ['target', 'consumo_annuo', 'default_flag'] and m not in set()]
        except Exception:
            return []

    def _update_report(
        self,
        iter_num: int,
        business_strategy: str | None,
        prev_metric: float | None
    ):
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
        
        impl_features = self._extract_implemented_features()
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

    async def optimize(self):
        for i in range(1, self.max_iterations + 1):
            await self.run_iteration(i)
            
            if i >= 4 and self._should_stop_early(threshold=0.01, window=3):
                print(f"[*] Early stopping attivato dopo iterazione {i}. Il modello non sta migliorando significativamente.")
                break
        
        valid_runs = [exp for exp in self.history if exp['metric'] is not None]
        if valid_runs:
            best_exp = max(valid_runs, key=lambda x: x['metric'])
            print(f"\n[+] Ottimizzazione conclusa. Miglior iterazione: {best_exp['iteration']} (R2: {best_exp['metric']:.4f})")
        else:
            print("\n[!] Ottimizzazione conclusa senza metriche valide.")
