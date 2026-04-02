# AutoLLml - Architectural Blueprint

## 1. Overview

AutoLLml is an LLM-powered Automated Machine Learning system that iteratively improves ML model performance through AI-driven feature engineering.

- **Purpose**: Automate feature engineering with domain-aware, business-first approach
- **Goal**: Maximize predictive performance through iterative AI-driven optimization
- **Framework**: Microsoft Agent Framework (AutoGen)
- **Core Value**: Semantic feature engineering based on real-world business logic, not statistical brute-forcing

---

## 2. System Architecture

### 2.1 Agent Layer

The system uses four specialized LLM-powered agents coordinated by an orchestrator:

| Agent | Responsibility |
|-------|----------------|
| **OrchestratorAgent** | Workflow coordinator, manages iteration loop, handles errors, updates reports |
| **PlanningAgent** | Generates business-focused feature strategies and recommends optimal ML models |
| **FeatureEngineeringAgent** | Generates Python code for feature engineering, applies pruning, and builds model pipeline |
| **EvaluatorAgent** | Analyzes results, plots, and feature importance to provide actionable reflection |

### 2.2 Manager Layer

| Manager | Responsibility |
|---------|----------------|
| **StrategyManager** | Maintains business strategy state, triggers initial and iterative strategy regeneration |
| **ModelRecommender** | Tracks and recommends optimal ML models (LogisticRegression, RandomForest, XGBoost/LightGBM) |
| **TrendAnalyzer** | Analyzes metric trends (improving/stagnating/declining), determines early stopping conditions |

### 2.3 Pipeline Layer

| Pipeline | Responsibility |
|----------|----------------|
| **CodeExecutionPipeline** | Generates feature engineering code, executes via subprocess, loads/saves `dynamic_features.py` |
| **EvaluationPipeline** | Orchestrates reflection, stores iteration data in memory, extracts feature patterns |
| **PruningAnalyzer** | Analyzes feature importance and correlations to recommend features for removal |

### 2.4 Component Layer

| Component | Responsibility |
|-----------|----------------|
| **DataContextProvider** | Loads glossary (business domain knowledge), data schema, and sample data for LLM context |
| **IterationExecutor** | Manages subprocess execution for training, metric parsing, and error retry logic (up to 3 retries) |

### 2.5 Supporting Services

| Service | Responsibility |
|---------|----------------|
| **MemoryStore** | Persistent JSON storage tracking iterations, feature patterns, best metrics, strategy effectiveness |
| **ReportGenerator** | Generates markdown reports with metrics, feature importance, correlations |

### 2.2 Training Pipeline

Modular pipeline for model training and evaluation:

1. **Data Loader** - Load dataset, detect task type, prepare features
2. **Feature Analyzer** - Compute correlations, identify top features
3. **Trainer** - Cross-validation (5-fold), compute metrics
4. **Plot Generator** - Generate distribution visualizations
5. **Reporter** - Output evaluation metrics and artifacts

---

## 3. Core Principles

- **Semantic over statistical**: Every feature must have real-world business meaning rooted in domain knowledge. Features should map to observable business phenomena (risk indices, propensity scores, efficiency ratios), not mathematical transformations.

- **No math-bruteforcing**: Raw transformations like `log()`, `exp()`, polynomial features, or square roots on single columns are explicitly forbidden. All derived features must combine multiple columns or apply true operational logic.

- **Pruning**: Remove redundant, collinear, or noisy features that add no discriminative power.

- **Clean code**: Robust handling of missing values, proper data types, defensive programming with pandas and scikit-learn.

---

## 4. Iteration Flow

The optimization runs for a configurable number of iterations (default: 5).

### Phase 1: Baseline (Iteration 1)

- Runs training pipeline without LLM invocation
- Uses default baseline model (LogisticRegression)
- Establishes baseline metric (F1 for classification, R2 for regression)
- No strategy generation or reflection in this phase

### Phase 2: LLM-Driven Iterations (Iteration 2+)

1. **Strategy Generation**
   - PlanningAgent reads glossary.md and data schema
   - Considers trend context (improving/stagnating/declining) from memory
   - Generates 3-5 business feature strategies (not raw transforms)
   - Suggests 2-3 appropriate ML models with rationale

2. **Model Selection**
   - ModelRecommender selects optimal model based on data characteristics and memory

3. **Pruning Analysis**
   - PruningAnalyzer identifies features to remove (importance < 0.01 or correlation > 0.9)

4. **Training & Evaluation**
   - Execute training pipeline with generated code
   - Run 5-fold cross-validation
   - Generate feature distribution plots (violin plots, bar charts)
   - Compute feature importance from model

5. **Reflection**
   - EvaluatorAgent receives: evaluation report, feature importance, plots, trend context
   - Analyzes separability in violin plots, monotonic trends in bar charts
   - Performs trend analysis: is model improving, stagnating, or declining?
   - Conducts counterfactual reasoning: what alternative approaches could work?
   - Performs ablation analysis: which features can be removed while maintaining predictive power?
   - Identifies: high-importance features, redundant features, new business logic opportunities
   - Outputs actionable feedback for next iteration

6. **Code Generation**
   - FeatureEngineeringAgent receives: business strategy, reflection, pruning, previous code
   - Generates new `dynamic_features.py`
   - Includes: derived features, feature pruning, model selection

7. **Retry Loop** (on code errors)
   - IterationExecutor handles retries up to 3 attempts

8. **Report Update**
   - Append iteration results to evaluation_report.md
   - Include metrics, correlations, feature importance, business strategy applied

### Phase 3: Early Stopping (Iteration 4+)

- After iteration 4, Orchestrator checks if improvement is below 1% over 3 consecutive iterations
- If improvement < threshold: stop optimization early (convergence detected)
- Prevents wasted computation on non-improving runs

---

## 5. Retry Logic Details

The retry mechanism handles code execution failures gracefully.

### Trigger Conditions
- Training exits with non-zero code
- stdout does not contain "SUCCESS_METRIC"
- Python exceptions during feature engineering or model training

### Retry Flow

```
MAX_ERROR_RETRIES = 3

for attempt in range(1, MAX_ERROR_RETRIES + 1):
    execute training pipeline
    
    if SUCCESS_METRIC found:
        capture metric value
        break loop (success)
    else:
        if attempt == MAX_ERROR_RETRIES:
            log final error, move to next iteration
        else:
            extract error message (last 1000 chars from stdout)
            call FeatureEngineeringAgent.fix_code_error(error_message, current_code)
            write fixed code to dynamic_features.py
            continue to next attempt
```

### Error Fix Behavior
- **Minimal change**: Only fix what caused the error
- **No strategy change**: Do NOT regenerate business strategy or reflection
- **Focus**: Handle specific errors (division by zero, NaN propagation, type mismatches)
- **Preservation**: Keep all existing feature engineering logic intact

### Failure Handling
- After 3 failed attempts: log error to iteration history, continue to next iteration
- Metrics set to None for failed iterations
- System does not abort - continues with remaining iterations

---

## 6. Data Flow

### Input Files

| File | Purpose |
|------|---------|
| `data/dataset.csv` | Input dataset (CSV format) |
| `glossary.md` | Domain knowledge and data dictionary |
| `dynamic_features.py` | Generated feature engineering code (overwritten each iteration) |

### Output Artifacts

| File/Directory | Content |
|---------------|---------|
| `evaluation_report.json` | Latest metrics per iteration (task_type, score_mean, score_std, feature_importance, correlations) |
| `evaluation_report.md` | Cumulative run history with all iterations |
| `evaluation_plots/iter_N/` | Feature distribution visualizations for iteration N |
| `memory.json` | Persistent iteration history and patterns |
| `best_run/` | Local directory (not git-tracked) for saving best model artifacts |

---

## 7. Configuration

### LLM Configuration (config.yaml)

- **model**: GPT model identifier (e.g., gpt-5.4-mini-2026-03-17)
- **temperature**: Sampling temperature for generation
- **reasoning_effort**: Reasoning budget (low/medium/high)

### Runtime Parameters

- **max_iterations**: Number of optimization cycles (default: 5)
- **cv_folds**: Cross-validation folds (default: 5)
- **cv_random_state**: Random seed for reproducibility (default: 42)

### Best Run Management

- Save: `python best_run.py save` → copies dynamic_features.py + evaluation_report.json to best_run/
- Restore: `python best_run.py restore` → overwrites working directory with saved best run
- Note: best_run/ is local-only and excluded from git

---

## 8. Execution Commands

```bash
# Run full optimization (5 iterations by default)
python main.py

# Custom number of iterations
python main.py --iterations 10

# Training only (baseline run)
python -m train --iter 1

# Save best model locally
python best_run.py save

# Restore best model
python best_run.py restore

# Reset to baseline (clears reports, resets dynamic_features.py)
python scripts/reset_codebase.py
```