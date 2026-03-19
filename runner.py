import asyncio
import threading
import queue
import json
import os
import subprocess
import re
from datetime import datetime
from typing import Optional, Callable
from pathlib import Path

import pandas as pd

from config import llm_client
from agents.orchestrator_agent import OrchestratorAgent
from dto import IterationSnapshot, TrainingStatus


class TrainingRunner:
    def __init__(
        self,
        max_iterations: int = 5,
        mlflow_experiment_name: str = "AutoLLml_Experiments",
        mlflow_tracking_enabled: bool = True,
        enable_llm: bool = True,
    ):
        self.max_iterations = max_iterations
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_tracking_enabled = mlflow_tracking_enabled
        self.enable_llm = enable_llm
        self._stop_event = threading.Event()
        self._log_queue: queue.Queue = queue.Queue()
        self._current_iteration = 0
        self._history: list[IterationSnapshot] = []
        self._business_strategy: Optional[str] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._stop_event.set()

    def get_logs(self) -> list[str]:
        logs = []
        while not self._log_queue.empty():
            try:
                logs.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        return logs

    def get_history(self) -> list[IterationSnapshot]:
        return self._history.copy()

    def get_business_strategy(self) -> Optional[str]:
        return self._business_strategy

    def _log(self, message: str) -> None:
        self._log_queue.put(message)

    def _parse_evaluation_report(self) -> dict:
        try:
            with open("evaluation_report.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _extract_features(self) -> list[str]:
        try:
            with open("dynamic_features.py", "r", encoding="utf-8") as f:
                source = f.read()
            matches = re.findall(r'data\[[\'"](\w+)[\'"]\]\s*=', source)
            TARGET_CANDIDATES = ['default_flag', 'consumo_annuo', 'target']
            return [m for m in matches if m not in TARGET_CANDIDATES]
        except Exception:
            return []

    def run(self) -> None:
        self._running = True
        self._stop_event.clear()
        self._history = []
        self._current_iteration = 0

        try:
            asyncio.run(self._run_async())
        except Exception as e:
            self._log(f"[ERROR] Training failed: {str(e)}")
        finally:
            self._running = False

    async def _run_async(self) -> None:
        self._log("[*] Initializing OrchestratorAgent...")
        
        try:
            orchestrator = OrchestratorAgent(
                llm_client,
                self.max_iterations,
                mlflow_experiment_name=self.mlflow_experiment_name if self.mlflow_tracking_enabled else None,
                mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI") if self.mlflow_tracking_enabled else None,
            )
        except Exception as e:
            self._log(f"[ERROR] Failed to initialize orchestrator: {str(e)}")
            return

        for i in range(1, self.max_iterations + 1):
            if self._stop_event.is_set():
                self._log("[*] Training stopped by user")
                break

            self._current_iteration = i
            self._log(f"\n{'='*60}")
            self._log(f"AVVIO ITERAZIONE {i}")
            self._log(f"{'='*60}")

            snapshot = IterationSnapshot(
                iteration=i,
                status="running",
                timestamp=datetime.now(),
            )
            self._history.append(snapshot)

            try:
                result = await orchestrator.run_iteration(i)

                snapshot.status = "completed" if result.is_success() else "failed"
                snapshot.metric = result.metric
                snapshot.error = result.error

                if result.is_success():
                    self._log(f"[+] Iterazione {i} completata. Metrica: {result.metric:.4f}")
                    report = self._parse_evaluation_report()
                    self._log(f"[*] Features: {len(report.get('feature_importance', {}))}")
                    self._log(f"[*] Task: {report.get('task_type', 'unknown').upper()}")
                else:
                    self._log(f"[!] Iterazione {i} fallita: {result.error[:200] if result.error else 'Unknown error'}")

            except Exception as e:
                snapshot.status = "failed"
                snapshot.error = str(e)
                self._log(f"[ERROR] Iteration {i} exception: {str(e)}")

            self._business_strategy = orchestrator.business_strategy

        self._log(f"\n{'='*60}")
        best = max([s for s in self._history if s.metric is not None], key=lambda x: x.metric, default=None)
        if best:
            self._log(f"[+] Training completato. Miglior iterazione: {best.iteration} (Metric: {best.metric:.4f})")
        else:
            self._log("[!] Training completato senza metriche valide")
        self._log(f"{'='*60}")


def start_training_thread(
    max_iterations: int,
    mlflow_experiment_name: str,
    mlflow_tracking_enabled: bool,
) -> TrainingRunner:
    runner = TrainingRunner(
        max_iterations=max_iterations,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_tracking_enabled=mlflow_tracking_enabled,
    )
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    return runner
