import subprocess


class IterationExecutor:
    """Gestisce l'esecuzione dei subprocess per il training loop, parsing delle metriche e gestione errori."""

    def __init__(self, max_error_retries: int):
        self.max_error_retries = max_error_retries

    async def execute_baseline(self, iter_num: int) -> dict:
        """Esegue la run baseline senza LLM."""
        print("[*] Prima run baseline (nessuna chiamata LLM). Esecuzione training loop...")

        cmd = ["python", "-m", "train", "--iter", str(iter_num)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout = result.stdout.strip()

        iteration_data = {"iteration": iter_num, "metric": None, "error": None}

        if "SUCCESS_METRIC" in stdout:
            metric_val = float(stdout.split("SUCCESS_METRIC: ")[1].split("\n")[0])
            iteration_data["metric"] = metric_val
            print(f"[+] Iterazione {iter_num} - Successo. Metrica (F1): {metric_val:.4f}")
        else:
            iteration_data["error"] = stdout[-1000:]
            print(f"[!] Iterazione {iter_num} - Fallita")

        return iteration_data

    async def execute_with_code_generation(
        self,
        iter_num: int,
        code: str,
        feature_engineering_agent
    ) -> dict:
        """Esegue il training con generazione e fix di codice."""
        if code.strip():
            with open("dynamic_features.py", "w", encoding="utf-8") as f:
                f.write(code)
        else:
            print("[!] Codice vuoto restituito.")

        retries = 0
        current_code = code
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

                current_code = await feature_engineering_agent.fix_code_error(
                    error_message=error_msg,
                    previous_code=current_code
                )

                if current_code.strip():
                    with open("dynamic_features.py", "w", encoding="utf-8") as f:
                        f.write(current_code)
                else:
                    print("[!] Codice vuoto restituito.")

        return {"iteration": iter_num, "metric": final_metric, "error": final_error}