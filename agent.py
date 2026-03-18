import os
import json
import subprocess
import glob
import re
import datetime
import pandas as pd
from llm_client import generate_response, generate_response_with_images
import prompts

class BusinessAwareFeatureAgent:
    def __init__(self, max_iterations=5):
        self.max_iterations = max_iterations
        self.history = []
        
        with open("glossary.md", "r", encoding="utf-8") as f:
            self.glossary = f.read()
            
        try:
            df = pd.read_csv("data/dataset.csv", encoding="latin-1")
            self.data_schema = str(df.dtypes.to_dict())
            self.data_sample = str(df.head(3).to_dict())
        except FileNotFoundError:
            print("[!] Assicurati di inserire il file in data/dataset.csv prima di avviare.")
            self.data_schema = "Dati non caricati."
            self.data_sample = "N/A"
        
        self.business_strategy = None  # Generata dopo la prima run
        print("[*] Inizializzazione Agente...")

    def _clean_code_output(self, raw_response: str) -> str:
        """Rimuove eventuali blocchi markdown dal codice generato."""
        lines = raw_response.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith("```")]
        return "\n".join(cleaned_lines)

    def _generate_business_and_data_strategy(self):
        prompt = prompts.get_business_strategy_prompt(
            glossary=self.glossary,
            data_schema=self.data_schema,
            data_sample=self.data_sample
        )
        response_text = generate_response(prompt)
        
        try:
            if not response_text.strip():
                raise json.JSONDecodeError("Risposta vuota dall'API.", "", 0)
                
            # Pulisce eventuali rimasugli di markdown json
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
                
            strategy_data = json.loads(response_text)
            business_strategy = strategy_data.get('business_strategy', 'Strategia non generata.')
            
            return business_strategy
            
        except json.JSONDecodeError as e:
            print(f"[!] Errore parsing strategia JSON: {e}")
            return "Massimizza la monotonicità logica ed esplora interazioni non lineari basate sul glossario."

    def _load_evaluation_plots(self, max_plots: int = 10) -> list:
        """
        Scansiona la cartella evaluation_plots/ e restituisce i path
        dei plot piu' recenti (fino a max_plots).
        """
        plot_dir = "evaluation_plots"
        if not os.path.isdir(plot_dir):
            print('|--- Nessun grafico disponibile ---|')
            return []
        pattern = os.path.join(plot_dir, "*.png")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        return files[:max_plots]

    def _generate_reflection(self, iter_num, evaluation_report):
        prompt = prompts.get_reflection_prompt(
            iter_num=iter_num,
            evaluation_report=json.dumps(evaluation_report, indent=2),
            glossary=self.glossary
        )
        plot_paths = self._load_evaluation_plots()
        if plot_paths:
            print(f"[*] Invio {len(plot_paths)} plot di distribuzione all'LLM per la riflessione...")
            reflection_text = generate_response_with_images(prompt, plot_paths)
        else:
            print(f"Ignorando i plot per la riflessione...")
            reflection_text = generate_response(prompt)
        safe_to_print = reflection_text.encode('ascii', 'ignore').decode('ascii')
        print(f"\n--- RIFLESSIONE ITERAZIONE {iter_num} ---\n{safe_to_print}\n")
        return reflection_text

    def _generate_code(self, reflection_text, last_code, last_error=None):
        prompt = prompts.get_code_generation_prompt(
            business_strategy=self.business_strategy,
            reflection_text=reflection_text,
            last_code=last_code,
            last_error=last_error
        )
        raw_code = generate_response(prompt)
        if not raw_code.strip():
            return ""
        return self._clean_code_output(raw_code)

    def _extract_implemented_features(self) -> list:
        """Estrae i nomi delle feature assegnate a 'data[...]' in dynamic_features.py."""
        try:
            with open("dynamic_features.py", "r", encoding="utf-8") as f:
                source = f.read()
            # Cerca pattern: data["feature_name"] = ...
            matches = re.findall(r'data\[[\'"](\w+)[\'"]\]\s*=', source)
            # Rimuovi colonne target candidate e duplicati
            from dynamic_features import TARGET_CANDIDATES
            seen = set(TARGET_CANDIDATES)
            return [m for m in matches if m not in seen and not seen.add(m)]
        except Exception:
            return []

    def _update_evaluation_report_md(
        self,
        iter_num: int,
        report: dict,
        business_strategy: str | None,
        prev_metric: float | None
    ):
        """Aggiunge una sezione Run alla cronologia in evaluation_report.md."""
        md_path = "evaluation_report.md"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ---- Metriche principali ----
        r2_mean = report.get("r2_score_mean", float("nan"))
        r2_std  = report.get("r2_score_std",  float("nan"))
        n_feat   = report.get("num_features", "?")

        # Delta vs run precedente
        if prev_metric is not None:
            delta = r2_mean - prev_metric
            delta_str = f"{delta:+.4f}  {'▲' if delta >= 0 else '▼'}"
        else:
            delta_str = "— (baseline)"

        # ---- Top correlazioni ----
        top_corr = report.get("top_correlations_with_target", {})
        corr_md = "\n".join(
            f"  - `{k}`: {v:+.4f}" for k, v in list(top_corr.items())[:10]
        ) or "  *(nessuna)*"

        # ---- Feature importance ----
        fi = report.get("feature_importance", {})
        fi_md = "\n".join(
            f"  - `{k}`: {v:.4f}" for k, v in list(fi.items())[:10]
        ) or "  *(non disponibile per questo modello)*"

        # ---- Feature implementate ----
        impl_features = self._extract_implemented_features()
        feat_md = "\n".join(f"  - `{f}`" for f in impl_features) or "  *(baseline — nessuna derivata)*"

        # ---- Business strategy ----
        strat_md = business_strategy.strip() if business_strategy else "*(non generata — run baseline)*"

        section = (
            f"\n---\n"
            f"## Run {iter_num}  —  {timestamp}\n\n"
            f"### Metriche\n"
            f"| Metrica | Valore |\n"
            f"|---------|--------|\n"
            f"| R2 Mean (CV-5) | **{r2_mean:.4f}** ± {r2_std:.4f} |\n"
            f"| Δ vs run precedente | {delta_str} |\n"
            f"| Numero feature in input | {n_feat} |\n\n"
            f"### Top correlazioni con il target (Pearson)\n{corr_md}\n\n"
            f"### Feature importance (top 10)\n{fi_md}\n\n"
            f"### Feature implementate in questa run\n{feat_md}\n\n"
            f"### Business strategy applicata\n{strat_md}\n"
        )

        # Header solo alla prima run (file non ancora esistente)
        if not os.path.exists(md_path):
            header = (
                "# Evaluation Report — Cronologia delle Run\n\n"
                "> Generato automaticamente da `agent.py`. "
                "Ogni sezione corrisponde a una run del training loop.\n"
            )
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(header)

        with open(md_path, "a", encoding="utf-8") as f:
            f.write(section)

        print(f"[*] evaluation_report.md aggiornato (Run {iter_num}).")

    def run_iteration(self, iter_num):
        print(f"\n================ AVVIO ITERAZIONE {iter_num} ================")

        # Iterazione 1: run baseline senza nessuna chiamata LLM
        if iter_num == 1:
            print("[*] Prima run baseline (nessuna chiamata LLM). Esecuzione training loop...")
            result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
            stdout = result.stdout.strip()
            iteration_data = {"iteration": iter_num, "metric": None, "error": None}
            if "SUCCESS_METRIC" in stdout:
                metric_val = float(stdout.split("SUCCESS_METRIC: ")[1].split("\n")[0])
                iteration_data["metric"] = metric_val
                print(f"[+] Iterazione {iter_num} - Successo. Metrica (R2): {metric_val:.4f}")
                try:
                    with open("evaluation_report.json", "r", encoding="utf-8") as f:
                        report = json.load(f)
                    self._update_evaluation_report_md(iter_num, report, None, None)
                except Exception as e:
                    print(f"[!] Impossibile aggiornare evaluation_report.md: {e}")
            else:
                iteration_data["error"] = stdout[-1000:]
                print(f"[!] Iterazione {iter_num} - Fallita con errore:\n{iteration_data['error']}")
            self.history.append(iteration_data)
            return

        # Dalla seconda iterazione in poi: genera business strategy (una volta sola) e avvia LLM loop
        if self.business_strategy is None:
            print("[*] Generazione strategia di business (post prima run)...")
            self.business_strategy = self._generate_business_and_data_strategy()

        # Recupera codice precedente
        last_code = ""
        with open("dynamic_features.py", "r", encoding="utf-8") as f:
            last_code = f.read()

        last_error = self.history[-1].get('error') if self.history else None

        # Reflection step (sempre disponibile dalla iter 2 in poi)
        with open("evaluation_report.json", "r", encoding="utf-8") as f:
            report = json.load(f)
        reflection_text = self._generate_reflection(iter_num, report)

        # Generazione e iniezione codice
        new_code = self._generate_code(reflection_text, last_code, last_error)
        if new_code.strip():
            with open("dynamic_features.py", "w", encoding="utf-8") as f:
                f.write(new_code)
        else:
            print("[!] API LLM ha restituito codice vuoto. Utilizzo della baseline/logica precedente.")
            
        # Training loop run
        print("[*] Esecuzione training loop...")
        result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
        stdout = result.stdout.strip()
        
        iteration_data = {"iteration": iter_num, "metric": None, "error": None}

        if "SUCCESS_METRIC" in stdout:
            metric_val = float(stdout.split("SUCCESS_METRIC: ")[1].split("\n")[0])
            iteration_data["metric"] = metric_val
            print(f"[+] Iterazione {iter_num} - Successo. Metrica (R2): {metric_val:.4f}")
            prev_metric = self.history[-1].get("metric") if self.history else None
            try:
                with open("evaluation_report.json", "r", encoding="utf-8") as f:
                    fresh_report = json.load(f)
                self._update_evaluation_report_md(iter_num, fresh_report, self.business_strategy, prev_metric)
            except Exception as e:
                print(f"[!] Impossibile aggiornare evaluation_report.md: {e}")
        else:
            iteration_data["error"] = stdout[-1000:]
            print(f"[!] Iterazione {iter_num} - Fallita con errore:\n{iteration_data['error']}")

        self.history.append(iteration_data)

    def optimize(self):
        for i in range(1, self.max_iterations + 1):
            self.run_iteration(i)
            
        valid_runs = [exp for exp in self.history if exp['metric'] is not None]
        if valid_runs:
            best_exp = max(valid_runs, key=lambda x: x['metric'])
            print(f"\n[+] Ottimizzazione conclusa. Miglior iterazione: {best_exp['iteration']} (R2: {best_exp['metric']:.4f})")
        else:
            print("\n[!] Ottimizzazione conclusa, ma nessuna iterazione ha prodotto una metrica valida.")

if __name__ == "__main__":
    agent = BusinessAwareFeatureAgent(max_iterations=5)
    agent.optimize()