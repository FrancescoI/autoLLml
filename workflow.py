import asyncio
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.orchestrator_agent import OrchestratorAgent


class AutoMLWorkflow:
    def __init__(
        self,
        model_client: OpenAIChatCompletionClient,
        max_iterations: int = 5
    ):
        self.orchestrator = OrchestratorAgent(model_client, max_iterations)
        
        self.team = RoundRobinGroupChat(
            participants=[
                self.orchestrator.agent,
            ],
            max_turns=1,
        )

    async def run(self):
        print("[*] Avvio workflow AutoML con Microsoft Agent Framework")
        print("[*] Agenti disponibili:")
        print("    - StrategyAgent: genera strategie di business")
        print("    - CodeAgent: genera codice feature engineering")
        print("    - EvaluatorAgent: analizza risultati e riflette")
        print("    - OrchestratorAgent: coordina il workflow")
        print()
        
        await self.orchestrator.optimize()


async def run_workflow(max_iterations: int = 5):
    from config import llm_client
    
    workflow = AutoMLWorkflow(llm_client, max_iterations)
    await workflow.run()


if __name__ == "__main__":
    import sys
    
    max_iterations = 5
    if len(sys.argv) > 1:
        try:
            max_iterations = int(sys.argv[1])
        except ValueError:
            print(f"[!] Uso non valido. Sintassi: python workflow.py [max_iterations]")
            sys.exit(1)
    
    asyncio.run(run_workflow(max_iterations))
