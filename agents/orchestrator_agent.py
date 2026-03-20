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

import mlflow

MAX_ERROR_RETRIES = 3

class OrchestratorAgent:
    def __init__(
        self,
        model_client: OpenAIChatCompletionClient,
        max_iterations: int = 5,
        mlflow_experiment_name: str | None = None,
        mlflow_tracking_uri: str | None = None
    ):
        self.max_iterations = max_iterations
        self.history = []
        self.business_strategy: str | None = None
        self.mlflow_experiment_name = mlflow_experiment_name or "AutoLLml_Experiments"
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        
        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)
        
        with open("glossary.md", "r", encoding="utf-8") as f:
            self.glossary = f.read()
        
        try:
            df = pd.read_csv("data/dataset.csv", encoding="latin-1")
            self.data_schema = str(df.dtypes.to_dict())
            self.data_sample = str(df.head(1).to_dict())
        except FileNotFoundError:
            print("[!] Assicurati di inserire il file in data/dataset.csv prima di avviare.")
            self.data_schema = "Dati non caricati."
            self.data_sample = "N/A"
        
        self.strategy_agent = StrategyAgent(model_client)
        self.code_agent = CodeAgent(model_client)
        self.evaluator_agent = EvaluatorAgent(model_client)
        
        print("[*] OrchestratorAgent inizializzato")
        print(f"[*] MLFlow Experiment: {self.mlflow_experiment_name}")
        print(f"[*] MLFlow Tracking URI: {mlflow.get_tracking_uri()}")

    async def run_iteration(self, iter_num: int) -> dict:
        print(f"\n================ AVVIO ITERAZIONE {iter_num} ================")
        
        if iter_num == 1:
            return await self._run_baseline(iter_num)
        
        return await self._run_llm_iteration(iter_num)

    async def _run_baseline(self, iter_num: int) -> dict:
        print("[*] Prima run baseline (nessuna chiamata LLM). Esecuzione training loop...")
        
        cmd = ["python", "-m", "train", "--iter", str(iter_num)]
        if self.mlflow_experiment_name:
            cmd.extend(["--experiment", self.mlflow_experiment_name])
        if self.mlflow_tracking_uri:
            cmd.extend(["--tracking-uri", self.mlflow_tracking_uri])
        
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
        if self.business_strategy is None:
            print("[*] Generazione strategia di business con StrategyAgent...")
            strategy_result = await self.strategy_agent.generate_strategy(
                self.glossary,
                self.data_schema,
                self.data_sample
            )
            self.business_strategy = strategy_result['business_strategy']
            print(f"[*] Strategia generata: {self.business_strategy[:100]}...")
        
        with open("evaluation_report.json", "r", encoding="utf-8") as f:
            report = json.load(f)
        
        # Load plot paths for this iteration
        plot_dir = os.path.join("evaluation_plots", f"iter_{iter_num}")
        plot_paths = []
        if os.path.isdir(plot_dir):
            import glob
            plot_paths = sorted(glob.glob(os.path.join(plot_dir, "*.png")), key=os.path.getmtime, reverse=True)[:10]
        
        feature_importance = report.get("feature_importance", {})
        
        print(f"[*] Analisi risultati con EvaluatorAgent...")
        reflection_text = await self.evaluator_agent.evaluate_and_reflect(
            iter_num,
            report,
            self.glossary,
            plot_paths,
            feature_importance
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
        
        # Try training with retries (error-fix only, no MLFlow logging)
        retries = 0
        current_code = new_code
        final_metric = None
        final_error = None
        
        while retries < MAX_ERROR_RETRIES:
            print(f"[*] Esecuzione training loop... (attempt {retries + 1})")
            
            cmd = ["python", "-m", "train", "--iter", str(iter_num), "--no-mlflow"]
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
                if retries >= MAX_ERROR_RETRIES:
                    final_error = stdout[-1000:]
                    print(f"[!] Training fallito dopo {MAX_ERROR_RETRIES} tentativi")
                    break
                
                error_msg = stdout[-1000:]
                print(f"[!] Training fallito. Retry {retries}/{MAX_ERROR_RETRIES} con fix errore...")
                print(f"[*] Errore: {error_msg[:200]}...")
                
                # Error-fix only (no strategy/reflection)
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
            self._update_report(iter_num, self.business_strategy, self.history[-1].get('metric'))
        elif final_error:
            print(f"[!] Iterazione {iter_num} - Fallita")
        
        self.history.append(iteration_data)
        return iteration_data

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
        md_path = "evaluation_report.md"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            with open("evaluation_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)
        except Exception:
            return
        
        task_type = report.get("task_type", "unknown")
        metric_name = report.get("metric_name", "score")
        score_mean = report.get("score_mean", float("nan"))
        score_std = report.get("score_std", float("nan"))
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
        
        section = (
            f"\n---\n"
            f"## Run {iter_num}  —  {timestamp}\n\n"
            f"### Metriche\n"
            f"| Metrica | Valore |\n"
            f"|---------|--------|\n"
            f"| Task Type | {task_type.upper()} |\n"
            f"| {metric_name} Mean (CV-5) | **{score_mean:.4f}** ± {score_std:.4f} |\n"
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
        
        valid_runs = [exp for exp in self.history if exp['metric'] is not None]
        if valid_runs:
            best_exp = max(valid_runs, key=lambda x: x['metric'])
            print(f"\n[+] Ottimizzazione conclusa. Miglior iterazione: {best_exp['iteration']} (R2: {best_exp['metric']:.4f})")
        else:
            print("\n[!] Ottimizzazione conclusa senza metriche valide.")
