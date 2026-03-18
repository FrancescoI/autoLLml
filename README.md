# AutoLLml

LLM-powered Automated Machine Learning system that iteratively improves ML model performance through AI-driven feature engineering.

## Overview

AutoLLml uses GPT-5 to generate business-aware derived features and optimize ML pipelines. The agent iteratively analyzes results, reflects on performance, and generates new feature engineering code to maximize predictive power.

## Features

- **Business-First Feature Engineering**: Features must have real-world semantic meaning based on domain knowledge
- **Iterative Improvement**: LLM analyzes previous results + plots to generate better features each round
- **Multi-Modal Analysis**: Uses GPT-5 Vision to analyze feature distribution plots
- **Automatic Pruning**: Removes noisy/redundant features
- **Error Recovery**: If code crashes, LLM receives error message to fix in next iteration

## Architecture

Built with **Microsoft Agent Framework (AutoGen)**, featuring a multi-agent system:

```
┌─────────────────────────────────────────────────────────────────┐
│                     OrchestratorAgent                            │
│  Coordinates the workflow between specialized agents            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│StrategyAgent  │ │  CodeAgent    │ │EvaluatorAgent │
│Generates      │ │Generates      │ │Analyzes       │
│business       │ │feature        │ │results and    │
│strategy       │ │engineering    │ │reflects       │
│               │ │code           │ │               │
└───────────────┘ └───────────────┘ └───────────────┘
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

### Original Agent (Backward Compatible)

```bash
python agent.py
```

## Project Structure

```
automl/
├── main.py                    # Entry point (AutoGen version)
├── agent.py                   # Original entry point
├── workflow.py                # AutoGen workflow orchestration
├── config.py                 # LLM configuration
├── agents/                   # AutoGen agents
│   ├── __init__.py
│   ├── strategy_agent.py      # Business strategy generation
│   ├── code_agent.py          # Feature engineering code
│   ├── evaluator_agent.py     # Results analysis
│   └── orchestrator_agent.py  # Workflow coordinator
├── train.py                   # Training pipeline
├── dynamic_features.py        # Generated feature engineering (overwritten each iteration)
├── prompts.py                 # LLM prompt templates
├── glossary.md                # Data dictionary & domain knowledge
└── data/
    └── dataset.csv             # Input dataset
```

## How It Works

1. **Baseline Run**: First iteration runs without LLM to establish a baseline metric
2. **Strategy Generation**: StrategyAgent analyzes the glossary and data schema to generate business-focused feature strategies
3. **Training & Evaluation**: Train pipeline runs 5-fold cross-validation and generates distribution plots
4. **Reflection**: EvaluatorAgent analyzes results and plots to provide insights
5. **Code Generation**: CodeAgent generates new feature engineering code based on strategy + reflection
6. **Iteration**: Process repeats up to max_iterations

## Configuration

Edit `config.py` to change:
- Model selection (`gpt-5`, `gpt-4`, etc.)
- Temperature
- Reasoning effort

Edit `glossary.md` to add domain-specific knowledge about your dataset.

## Output

- `evaluation_report.json`: Latest evaluation metrics (R2, correlations, feature importance)
- `evaluation_report.md`: Run history with all iterations
- `evaluation_plots/`: Feature distribution visualizations

## License

MIT
