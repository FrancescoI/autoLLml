import os
import json
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

EXPERIMENT_NAME = "AutoLLml_Experiments"

def setup_mlflow(experiment_name: str = None, tracking_uri: str = None) -> str:
    """Initialize MLFlow experiment.
    
    Args:
        experiment_name: Name of the experiment (default: AutoLLml_Experiments)
        tracking_uri: Optional MLFlow tracking server URI
        
    Returns:
        The experiment name
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    exp_name = experiment_name or EXPERIMENT_NAME
    mlflow.set_experiment(exp_name)
    
    return exp_name


def get_or_create_experiment(experiment_name: str = None) -> str:
    """Get or create an MLFlow experiment."""
    exp_name = experiment_name or EXPERIMENT_NAME
    
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        mlflow.create_experiment(exp_name)
    
    mlflow.set_experiment(exp_name)
    return exp_name


def log_iteration_run(
    iter_num: int,
    report: dict,
    business_strategy: str = None,
    model = None,
    parent_run_id: str = None,
    plots_dir: str = "evaluation_plots",
    dynamic_features_path: str = "dynamic_features.py",
    glossary_path: str = "glossary.md"
) -> str:
    """Log a single iteration run to MLFlow.
    
    Args:
        iter_num: Iteration number
        report: Evaluation report dictionary
        business_strategy: Business strategy string
        model: Trained sklearn model
        parent_run_id: Optional parent run ID for nesting
        plots_dir: Path to evaluation plots directory
        dynamic_features_path: Path to feature engineering code
        glossary_path: Path to glossary file
        
    Returns:
        The MLFlow run ID
    """
    run_name = f"iteration_{iter_num}"
    
    with mlflow.start_run(run_name=run_name, nested=parent_run_id is not None) as run:
        run_id = run.info.run_id
        
        mlflow.set_tag("iteration", iter_num)
        mlflow.set_tag("task_type", report.get("task_type", "unknown"))
        mlflow.set_tag("metric_name", report.get("metric_name", "score"))
        
        if parent_run_id:
            mlflow.set_tag("parent_run_id", parent_run_id)
        
        mlflow.log_param("iteration", iter_num)
        mlflow.log_param("task_type", report.get("task_type", "unknown"))
        mlflow.log_param("metric_name", report.get("metric_name", "score"))
        mlflow.log_param("num_features", report.get("num_features", 0))
        
        if business_strategy:
            strategy_short = business_strategy[:500] if len(business_strategy) > 500 else business_strategy
            mlflow.log_param("business_strategy", strategy_short)
        
        mlflow.log_metric("score_mean", report.get("score_mean", 0))
        mlflow.log_metric("score_std", report.get("score_std", 0))
        
        top_corr = report.get("top_correlations_with_target", {})
        if top_corr:
            for feat, corr in list(top_corr.items())[:5]:
                mlflow.log_metric(f"corr_{feat}", corr)
        
        if model is not None:
            try:
                mlflow.sklearn.log_model(
                    model,
                    f"model_iter_{iter_num}",
                    registered_model_name=f"AutoLLml_iter_{iter_num}"
                )
            except Exception as e:
                print(f"[!] Could not log model to MLFlow: {e}")
        
        if os.path.exists("evaluation_report.json"):
            try:
                with open("evaluation_report.json", "r") as f:
                    full_report = json.load(f)
                mlflow.log_dict(full_report, "evaluation_report.json")
            except Exception:
                pass
        
        if os.path.exists(dynamic_features_path):
            mlflow.log_artifact(dynamic_features_path)
        
        if os.path.exists(glossary_path):
            mlflow.log_artifact(glossary_path)
        
        if os.path.exists(plots_dir):
            mlflow.log_artifacts(plots_dir, artifact_path="evaluation_plots")
    
    return run_id


def log_optimization_summary(
    history: list,
    best_iter: int,
    best_score: float,
    task_type: str = "classification"
) -> str:
    """Log the optimization summary as a parent run.
    
    Args:
        history: List of iteration dictionaries
        best_iter: Best iteration number
        best_score: Best score achieved
        task_type: Task type string
        
    Returns:
        The parent run ID
    """
    with mlflow.start_run(run_name="optimization_summary", nested=False) as run:
        run_id = run.info.run_id
        
        mlflow.set_tag("run_type", "optimization_summary")
        mlflow.set_tag("task_type", task_type)
        
        mlflow.log_param("total_iterations", len(history))
        mlflow.log_param("best_iteration", best_iter)
        mlflow.log_param("task_type", task_type)
        mlflow.log_metric("best_score", best_score)
        
        for h in history:
            iter_num = h.get("iteration", 0)
            metric = h.get("metric")
            if metric is not None:
                mlflow.log_metric(f"iter_{iter_num}_score", metric)
        
        return run_id


def get_best_model_info(experiment_name: str = None) -> dict:
    """Get information about the best model from MLFlow.
    
    Returns:
        Dictionary with best run info or empty dict if no runs found
    """
    exp_name = experiment_name or EXPERIMENT_NAME
    exp = mlflow.get_experiment_by_name(exp_name)
    
    if exp is None:
        return {}
    
    client = MlflowClient()
    runs = client.search_runs(exp.experiment_id, order_by=["metrics.score_mean DESC"], max_results=1)
    
    if not runs:
        return {}
    
    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "run_name": best_run.info.run_name,
        "score_mean": best_run.data.metrics.get("score_mean"),
        "score_std": best_run.data.metrics.get("score_std"),
        "num_features": best_run.data.params.get("num_features"),
        "task_type": best_run.data.params.get("task_type"),
    }


def print_mlflow_uri():
    """Print the MLFlow tracking URI."""
    uri = mlflow.get_tracking_uri()
    print(f"[*] MLFlow Tracking URI: {uri}")
    print(f"[*] To view experiments, run: mlflow ui --port 5000")
    if uri == "file:///mlruns":
        print(f"[*] Local mlruns directory: ./mlruns")


def register_best_model(model_name: str = "AutoLLml_Best", experiment_name: str = None) -> str:
    """Register the best model from the experiment to the MLFlow Model Registry.
    
    Args:
        model_name: Name for the registered model
        experiment_name: Name of the experiment to search
        
    Returns:
        Model version or None if no model found
    """
    exp_name = experiment_name or EXPERIMENT_NAME
    exp = mlflow.get_experiment_by_name(exp_name)
    
    if exp is None:
        print("[!] No experiment found")
        return None
    
    client = MlflowClient()
    runs = client.search_runs(exp.experiment_id, order_by=["metrics.score_mean DESC"], max_results=1)
    
    if not runs:
        print("[!] No runs found")
        return None
    
    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    
    try:
        model_version = mlflow.register_model(model_uri, model_name)
        print(f"[+] Registered model '{model_name}' version {model_version.version}")
        return model_version
    except Exception as e:
        print(f"[!] Could not register model: {e}")
        return None


class ReproducibleBestModel:
    """Save and restore best model for reproducibility.
    
    Keeps best dynamic_features.py and metrics in local best_run/ directory
    (not committed to git).
    """
    
    def __init__(self, experiment_name: str = None, best_dir: str = "best_run"):
        self.experiment_name = experiment_name or EXPERIMENT_NAME
        self.best_dir = best_dir
    
    def save(self):
        """Save current dynamic_features.py + metrics to best_run/ (local only)."""
        os.makedirs(self.best_dir, exist_ok=True)
        
        if os.path.exists("dynamic_features.py"):
            shutil.copy2("dynamic_features.py", f"{self.best_dir}/dynamic_features.py")
            print(f"[*] Saved: dynamic_features.py -> {self.best_dir}/")
        
        if os.path.exists("evaluation_report.json"):
            shutil.copy2("evaluation_report.json", f"{self.best_dir}/evaluation_report.json")
            print(f"[*] Saved: evaluation_report.json -> {self.best_dir}/")
        
        best_info = get_best_model_info(self.experiment_name)
        if best_info:
            best_info_path = f"{self.best_dir}/best_run_info.json"
            with open(best_info_path, "w") as f:
                json.dump(best_info, f, indent=2)
            print(f"[*] Saved: best_run_info.json -> {self.best_dir}/")
            print(f"[*] Best model score: {best_info.get('score_mean', 'N/A'):.4f}")
        
        print(f"\n[+] Best run saved to: {self.best_dir}/")
        print("[*] This directory is local-only (not committed to git)")
    
    def restore(self):
        """Copy best_run/dynamic_features.py to working directory."""
        src = f"{self.best_dir}/dynamic_features.py"
        if os.path.exists(src):
            shutil.copy2(src, "dynamic_features.py")
            print(f"[*] Restored: {self.best_dir}/dynamic_features.py -> ./dynamic_features.py")
        else:
            print(f"[!] No saved model found in {self.best_dir}/")
            print("[*] Run experiments first, then save best run")
    
    def load_info(self) -> dict:
        """Load best run info from saved directory."""
        info_path = f"{self.best_dir}/best_run_info.json"
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                return json.load(f)
        return {}
    
    def exists(self) -> bool:
        """Check if a saved best run exists."""
        return os.path.exists(f"{self.best_dir}/dynamic_features.py")
