import asyncio
import threading
import queue
import json
import os
import re
from datetime import datetime
from typing import Optional, Callable, AsyncGenerator
from pathlib import Path

sys_path = str(Path(__file__).parent.parent.parent)
if sys_path not in os.sys.path:
    os.sys.path.insert(0, sys_path)


class TrainingService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self._stop_event = threading.Event()
        self._log_queue: queue.Queue = queue.Queue()
        self._current_iteration = 0
        self._max_iterations = 5
        self._history: list = []
        self._business_strategy: Optional[str] = None
        self._running = False
        self._status = "idle"
        self._agent_outputs: dict = {}
        self._current_log_count = 0

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._running

    def get_logs(self) -> list:
        logs = []
        while not self._log_queue.empty():
            try:
                logs.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        return logs

    def get_new_logs_count(self) -> int:
        return self._log_queue.qsize()

    def get_history(self) -> list:
        return self._history.copy()

    def get_business_strategy(self) -> Optional[str]:
        return self._business_strategy

    def get_agent_outputs(self) -> dict:
        return self._agent_outputs.copy()

    def _log(self, message: str, level: str = "info") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_queue.put({
            "timestamp": timestamp,
            "message": message,
            "level": level
        })

    def _parse_evaluation_report(self) -> dict:
        try:
            with open("evaluation_report.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _extract_features(self) -> list:
        try:
            with open("dynamic_features.py", "r", encoding="utf-8") as f:
                source = f.read()
            matches = re.findall(r'data\[[\'"](\w+)[\'"]\]\s*=', source)
            TARGET_CANDIDATES = ['default_flag', 'consumo_annuo', 'target', 'target_col']
            return [m for m in matches if m not in TARGET_CANDIDATES]
        except Exception:
            return []

    def stop(self) -> None:
        self._stop_event.set()

    def reset(self) -> None:
        self._stop_event.clear()
        self._history = []
        self._current_iteration = 0
        self._business_strategy = None
        self._agent_outputs = {}
        self._status = "idle"
        self._running = False
        while not self._log_queue.empty():
            try:
                self._log_queue.get_nowait()
            except queue.Empty:
                break

    def run(
        self,
        max_iterations: int,
        mlflow_experiment_name: str = "AutoLLml_Experiments",
        mlflow_tracking_enabled: bool = True
    ) -> None:
        self._max_iterations = max_iterations
        self._running = True
        self._status = "running"
        self._stop_event.clear()
        self._history = []
        self._current_iteration = 0

        try:
            asyncio.run(self._run_async(
                max_iterations,
                mlflow_experiment_name,
                mlflow_tracking_enabled
            ))
        except Exception as e:
            self._log(f"[ERROR] Training failed: {str(e)}", "error")
            self._status = "failed"
        finally:
            self._running = False
            if self._status == "running":
                self._status = "completed"

    async def _run_async(
        self,
        max_iterations: int,
        mlflow_experiment_name: str,
        mlflow_tracking_enabled: bool
    ) -> None:
        self._log("[*] Initializing OrchestratorAgent...")
        
        try:
            from config import llm_client
            from agents.orchestrator_agent import OrchestratorAgent
            
            orchestrator = OrchestratorAgent(
                llm_client,
                max_iterations,
                mlflow_experiment_name=mlflow_experiment_name if mlflow_tracking_enabled else None,
                mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI") if mlflow_tracking_enabled else None,
            )
        except Exception as e:
            self._log(f"[ERROR] Failed to initialize orchestrator: {str(e)}", "error")
            return

        for i in range(1, max_iterations + 1):
            if self._stop_event.is_set():
                self._log("[*] Training stopped by user", "warning")
                self._status = "stopped"
                break

            self._current_iteration = i
            self._log(f"\n{'='*60}")
            self._log(f"AVVIO ITERAZIONE {i}")
            self._log(f"{'='*60}")

            iteration_data = {
                "iteration": i,
                "metric": None,
                "error": None,
                "status": "running",
                "timestamp": datetime.now().isoformat()
            }
            self._history.append(iteration_data)

            try:
                result = await orchestrator.run_iteration(i)

                iteration_data["status"] = "completed" if result.get("metric") else "failed"
                iteration_data["metric"] = result.get("metric")
                iteration_data["error"] = result.get("error")

                if result.get("metric"):
                    self._log(f"[+] Iterazione {i} completata. Metrica: {result['metric']:.4f}", "success")
                    report = self._parse_evaluation_report()
                    self._log(f"[*] Features: {len(report.get('feature_importance', {}))}")
                    self._log(f"[*] Task: {report.get('task_type', 'unknown').upper()}")
                else:
                    self._log(f"[!] Iterazione {i} fallita: {result.get('error', 'Unknown error')[:200]}", "error")

            except Exception as e:
                iteration_data["status"] = "failed"
                iteration_data["error"] = str(e)
                self._log(f"[ERROR] Iteration {i} exception: {str(e)}", "error")

            self._business_strategy = orchestrator.business_strategy

        best = max(
            [s for s in self._history if s.get("metric") is not None],
            key=lambda x: x["metric"],
            default=None
        )
        if best:
            self._log(f"\n{'='*60}", "info")
            self._log(f"[+] Training completato. Miglior iterazione: {best['iteration']} (Metric: {best['metric']:.4f})", "success")
        else:
            self._log("[!] Training completato senza metriche valide", "warning")
        self._log(f"{'='*60}", "info")


training_service = TrainingService()
