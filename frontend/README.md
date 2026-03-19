# AutoML Dashboard - Custom Web UI

A custom web dashboard for AutoML experimentation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the backend (serves both API and HTML UI)
python -m uvicorn backend.main:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

## Features

### Config Page
- Upload CSV dataset
- Edit glossary (Markdown)
- Set training parameters (iterations, MLFlow)

### Training Page
- Start/Stop training
- Real-time logs
- Iteration history with metrics

### Results Page
- Metrics dashboard (score, features)
- Feature importance chart
- Correlation charts
- Plots gallery
- Generated code viewer

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/dataset/upload` | Upload CSV |
| GET | `/api/dataset/info` | Get dataset info |
| GET | `/api/glossary/` | Get glossary |
| POST | `/api/glossary/save` | Save glossary |
| POST | `/api/training/start` | Start training |
| POST | `/api/training/stop` | Stop training |
| GET | `/api/training/status` | Get training status |
| GET | `/api/training/logs` | Get logs |
| GET | `/api/results/latest` | Get latest report |
| GET | `/api/results/plots` | List plots |
| GET | `/api/results/features` | Get features |
| GET | `/api/results/code` | Get generated code |
