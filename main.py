import asyncio
import argparse
import sys
import os
import platform

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from utils.config import llm_client
from agents.orchestrator_agent import OrchestratorAgent
from utils.telemetry import setup_telemetry

setup_telemetry()


async def main(
    max_iterations: int = 5
):
    print("=" * 60)
    print("   AutoML Agent con Microsoft Agent Framework")
    print("=" * 60)
    print()
    
    orchestrator = OrchestratorAgent(
        llm_client,
        max_iterations
    )
    
    await orchestrator.optimize()
    
    print()
    print("=" * 60)
    print("   Ottimizzazione completata")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML Agent con Microsoft Agent Framework")
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Numero massimo di iterazioni (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(
            max_iterations=args.iterations
        ))
    except KeyboardInterrupt:
        print("\n[!] Ottimizzazione interrotta dall'utente.")
        sys.exit(0)
