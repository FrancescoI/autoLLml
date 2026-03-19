import asyncio
import argparse
import sys
import os

from utils.config import llm_client
from agents.orchestrator_agent import OrchestratorAgent


async def main(
    max_iterations: int = 5,
    mlflow_experiment_name: str = None,
    mlflow_tracking_uri: str = None
):
    print("=" * 60)
    print("   AutoML Agent con Microsoft Agent Framework")
    print("   + MLFlow Experiment Tracking")
    print("=" * 60)
    print()
    
    orchestrator = OrchestratorAgent(
        llm_client,
        max_iterations,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_tracking_uri=mlflow_tracking_uri
    )
    
    await orchestrator.optimize()
    
    print()
    print("=" * 60)
    print("   Ottimizzazione completata")
    print("=" * 60)
    print()
    print(f"[*] Per visualizzare gli esperimenti MLFlow:")
    print(f"    mlflow ui --port 5000")
    print(f"[*] Oppure apri: http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML Agent con Microsoft Agent Framework")
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Numero massimo di iterazioni (default: 5)"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME"),
        help="Nome esperimento MLFlow (default: AutoLLml_Experiments)"
    )
    parser.add_argument(
        "--tracking-uri", "-t",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLFlow tracking URI (es. http://localhost:5000)"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(
            max_iterations=args.iterations,
            mlflow_experiment_name=args.experiment,
            mlflow_tracking_uri=args.tracking_uri
        ))
    except KeyboardInterrupt:
        print("\n[!] Ottimizzazione interrotta dall'utente.")
        sys.exit(0)
