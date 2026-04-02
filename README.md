# AutoLLml

LLM-powered Automated Machine Learning system that iteratively improves ML model performance through AI-driven feature engineering.

## Overview

AutoLLml uses GPT-5-family models to generate business-aware derived features and optimize ML pipelines. The agent iteratively analyzes results, reflects on performance, and generates new feature engineering code to maximize predictive power.

## Features

- **Business-First Feature Engineering**: Features must have real-world semantic meaning based on domain knowledge and data understanding
- **Iterative Improvement**: LLM analyzes previous results and bivariate (Xs vs Y) relationships to generate better features each round
- **Multi-Modal Analysis**: LLM uses multi-modal capability to analyze feature distribution plots
- **Automatic Pruning**: LLM detects noisy/redundant features
- **Code Generation**: LLM re-writes dynamic_feature.py each round, for both feature engineering step and model selection
- **Error Recovery**: If code crashes, LLM receives error message to fix in the next iteration
- **Memory Management**: LLM stores successful and unsuccessful patterns in a structured format each round

## Architecture

Built with **Microsoft Agent Framework (AutoGen)**, featuring a consolidated multi-agent system:

```
┌─────────────────────────────────────────────────────────────────┐
│                     OrchestratorAgent                            │
│  Coordinates the workflow between agents, managers, pipelines  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│PlanningAgent  │ │FeatureEngAgent│ │EvaluatorAgent │
│Strategy &     │ │Code + Pruning │ │Analysis &     │
│Model Selection│ │               │ │Reflection     │
└───────────────┘ └───────────────┘ └───────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────┐
│         Managers: Strategy, Model, Trend            │
├─────────────────────────────────────────────────────┤
│         Pipelines: Code Exec, Eval, Pruning         │
├─────────────────────────────────────────────────────┤
│         Components: Data Context, Iteration Exec    │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                 MemoryStore                         │
│         (Utility for persistent storage)            │
└─────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.10+
- OpenAI API key (GPT-5)

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Basic Usage

```bash
python main.py
```

### Custom Iterations

```bash
python main.py --iterations 10
```

### Training Only

```bash
python -m train --iter 1
```

## Project Structure

```
automl/
├── main.py                    # Entry point 
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── config.py              # LLM configuration (dataclasses)
│   ├── llm_client.py          # OpenAI client wrapper
│   └── telemetry.py           # OpenTelemetry telemetry
├── prompts/                   # LLM prompt templates
│   └── __init__.py
├── agents/                    # AutoGen agents and components
│   ├── __init__.py
│   ├── orchestrator_agent.py   # Workflow coordinator
│   ├── components/
│   │   ├── data_context_provider.py  # Data loading and context
│   │   └── iteration_executor.py     # Subprocess execution and retries
│   ├── llm_agents/
│   │   ├── planning_agent.py         # Business strategy and model selection
│   │   ├── feature_engineering_agent.py # Code generation and pruning
│   │   └── evaluator_agent.py        # Results analysis and reflection
│   ├── managers/
│   │   ├── strategy_manager.py       # Strategy state management
│   │   ├── model_recommender.py      # ML model recommendations
│   │   └── trend_analyzer.py         # Trend analysis and early stopping
│   ├── pipelines/
│   │   ├── code_execution_pipeline.py # Code generation pipeline
│   │   ├── evaluation_pipeline.py    # Reflection orchestration
│   │   └── pruning_analyzer.py       # Feature pruning decisions
│   └── reporting/
│       └── report_generator.py       # Report generation
├── train/                     # Training pipeline
│   ├── __init__.py
│   ├── __main__.py            # Module entry point
│   ├── cli.py                  # CLI interface
│   ├── main.py                 # Training orchestrator
│   ├── data_loader.py          # Data loading & validation
│   ├── feature_analyzer.py     # Correlation analysis
│   ├── plot_generator.py       # Plot generation
│   ├── trainer.py              # Cross-validation & metrics
│   └── reporter.py             # Report generation
├── scripts/                   # Utility scripts
│   └── reset_codebase.py       # Reset to baseline
├── dynamic_features.py         # Generated feature engineering (overwritten each iteration)
├── glossary.md               # Data dictionary & domain knowledge
├── best_run.py               # Best run save/restore utility
└── data/
    └── dataset.csv              # Input dataset
```

## How It Works

1. **Baseline Run**: First iteration runs without LLM to establish a baseline metric using LogisticRegression on raw features
2. **Strategy Generation**: PlanningAgent analyzes glossary and data schema to generate business-focused feature strategies and recommends optimal ML models
3. **Pruning Analysis**: PruningAnalyzer identifies redundant or noisy features to remove based on importance and correlations
4. **Reflection**: EvaluatorAgent analyzes evaluation results, distribution plots, and trend context to provide actionable insights
5. **Code Generation**: FeatureEngineeringAgent generates new feature engineering code incorporating strategy, reflection, and pruning decisions
6. **Execution**: IterationExecutor runs the training pipeline in a subprocess, with retry logic for errors
7. **Reporting**: ReportGenerator updates evaluation reports and MemoryStore saves iteration data for future learning
8. **Early Stopping**: TrendAnalyzer checks for convergence; stops if improvement < 1% over 3 consecutive iterations
9. **Iteration**: Process repeats up to max_iterations or until early stopping

## Configuration

Edit `config.yaml` to change LLM settings:
- Model selection (eg: gpt-5.4-mini-2026-03-17)
- Temperature
- Reasoning effort

Edit `glossary.md` to add domain-specific knowledge about your dataset.

## Output

- `evaluation_report.json`: Latest evaluation metrics (F1/R2, correlations, feature importance)
- `evaluation_report.md`: Run history with all iterations
- `evaluation_plots/`: Feature distribution visualizations
- `best_run/`: Best model artifacts (local only, not in git)

## Reproducibility

### Save Best Run

```bash
python -c "from best_run import save_best"
```

This saves to `best_run/`:
- `dynamic_features.py` - best feature engineering code
- `evaluation_report.json` - metrics

### Restore Best Run

```bash
python -c "from best_run import restore_best"
```

### Reset Codebase

Reset `dynamic_features.py` to baseline for new experimentation:

```bash
python scripts/reset_codebase.py
```

### Workflow

```bash
# 1. Run experiments
python main.py --iterations 5

# 2. Save best run (local only, not in git)
python -c "from best_run import save_best"

# 3. Reset codebase for new experiment
python scripts/reset_codebase.py

# 4. Restore best run when needed
python -c "from best_run import restore_best"
```

## License

MIT
