import json
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .llm_agents.planning_agent import PlanningAgent
from .llm_agents.feature_engineering_agent import FeatureEngineeringAgent
from .llm_agents.evaluator_agent import EvaluatorAgent
from .components.data_context_provider import DataContextProvider
from .components.iteration_executor import IterationExecutor
from .pipelines.evaluation_pipeline import EvaluationPipeline
from .managers.strategy_manager import StrategyManager
from .managers.model_recommender import ModelRecommender
from .managers.trend_analyzer import TrendAnalyzer
from .reporting.report_generator import ReportGenerator
from .pipelines.pruning_analyzer import PruningAnalyzer
from .pipelines.code_execution_pipeline import CodeExecutionPipeline
from utils.memory_store import MemoryStore
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
        self.history = []
        
        # Data layer
        self.data_provider = DataContextProvider(paths)
        
        # Core execution
        self.executor = IterationExecutor(agent_config.max_error_retries)
        
        # Analysis & memory
        self.memory_store = MemoryStore()
        self.evaluator_pipeline = EvaluationPipeline(
            EvaluatorAgent(model_client), self.memory_store, paths
        )
        self.trend_analyzer = TrendAnalyzer(self.memory_store)
        
        # Strategy & modeling
        self.strategy_manager = StrategyManager(PlanningAgent(model_client))
        self.model_recommender = ModelRecommender(PlanningAgent(model_client))
        self.pruning_analyzer = PruningAnalyzer(FeatureEngineeringAgent(model_client))
        
        # Code generation
        self.code_pipeline = CodeExecutionPipeline(FeatureEngineeringAgent(model_client))
        
        # Reporting
        self.report_generator = ReportGenerator(paths)
        
        print("[*] OrchestratorAgent inizializzato")

    def _get_trend_context(self) -> str:
        return self.trend_analyzer.get_trend_context()

    def _get_feature_patterns_context(self) -> tuple[str, str]:
        return self.trend_analyzer.get_feature_patterns_context()

    def _should_stop_early(self, threshold: float = 0.01, window: int = 3) -> bool:
        return self.trend_analyzer.should_stop_early(threshold, window)

    async def run_iteration(self, iter_num: int) -> dict:
        print(f"\n================ AVVIO ITERAZIONE {iter_num} ================")
        
        if iter_num == 1:
            iteration_data = await self.executor.execute_baseline(iter_num)
            if iteration_data["metric"] is not None:
                self.report_generator.update_report(
                    iter_num, None, None, self.evaluator_pipeline.extract_implemented_features
                )
        else:
            iteration_data = await self._run_llm_iteration(iter_num)
        
        self.history.append(iteration_data)
        return iteration_data



    async def _run_llm_iteration(self, iter_num: int) -> dict:
        context = self.data_provider.get_context()
        memory_context = self.memory_store.get_context()
        
        # Strategy generation
        if self.strategy_manager.get_current_strategy() is None:
            await self.strategy_manager.generate_initial_strategy(
                context["glossary"], context["data_schema"], context["data_sample"]
            )
        else:
            last_iter = self.memory_store.get_last_iteration()
            trend_context = self._get_trend_context()
            strategy_context = self.memory_store.get_strategy_context()
            await self.strategy_manager.regenerate_with_context(
                context["glossary"], context["data_schema"], context["data_sample"],
                memory_context, last_iter, trend_context, strategy_context
            )
        
        with open(self.paths.evaluation_report, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        feature_importance = report.get("feature_importance", {})
        
        # Model recommendation
        await self.model_recommender.recommend_model(
            data_schema=context["data_schema"],
            data_sample=context["data_sample"],
            glossary=context["glossary"],
            memory_context=memory_context,
            feature_importance=feature_importance
        )
        
        # Pruning analysis
        correlations = report.get("correlations", {})
        features_to_drop = await self.pruning_analyzer.analyze_and_prune(
            feature_importance, correlations, memory_context
        )
        
        # Evaluation and reflection
        trend_context = self._get_trend_context()
        successful_patterns, failed_patterns = self._get_feature_patterns_context()
        reflection_text, feature_importance = await self.evaluator_pipeline.evaluate_iteration(
            iter_num, report, context["glossary"], context["data_schema"], context["data_sample"],
            trend_context, successful_patterns, failed_patterns,
            self.strategy_manager.get_current_strategy(), self.model_recommender.get_current_model(), features_to_drop
        )
        
        # Code generation
        last_code = self.code_pipeline.load_last_code()
        new_code = await self.code_pipeline.generate_code(
            self.strategy_manager.get_current_strategy(),
            reflection_text,
            last_code,
            None  # new_feature_ideas not used here
        )
        
        # Execute with retries
        iteration_data = await self.executor.execute_with_code_generation(
            iter_num, new_code, self.code_pipeline.feature_engineering_agent
        )
        
        if iteration_data["metric"] is not None:
            self.report_generator.update_report(
                iter_num, self.strategy_manager.get_current_strategy(),
                self.history[-1].get('metric') if self.history else None,
                self.evaluator_pipeline.extract_implemented_features
            )
        elif iteration_data["error"]:
            print(f"[!] Iterazione {iter_num} - Fallita")
        
        return iteration_data



    async def optimize(self):
        for i in range(1, self.max_iterations + 1):
            await self.run_iteration(i)
            
            if i >= 4 and self.trend_analyzer.should_stop_early(threshold=0.01, window=3):
                print(f"[*] Early stopping attivato dopo iterazione {i}. Il modello non sta migliorando significativamente.")
                break
        
        valid_runs = [exp for exp in self.history if exp['metric'] is not None]
        if valid_runs:
            best_exp = max(valid_runs, key=lambda x: x['metric'])
            print(f"\n[+] Ottimizzazione conclusa. Miglior iterazione: {best_exp['iteration']} (R2: {best_exp['metric']:.4f})")
        else:
            print("\n[!] Ottimizzazione conclusa senza metriche valide.")
