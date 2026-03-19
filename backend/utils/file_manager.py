import os
from pathlib import Path
from typing import Optional
import pandas as pd


BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "evaluation_plots"
GLOSSARY_PATH = BASE_DIR / "glossary.md"
DATASET_PATH = DATA_DIR / "dataset.csv"
REPORT_PATH = BASE_DIR / "evaluation_report.json"
DYNAMIC_FEATURES_PATH = BASE_DIR / "dynamic_features.py"


def ensure_directories():
    DATA_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def get_dataset_info() -> Optional[dict]:
    try:
        if DATASET_PATH.exists():
            df = pd.read_csv(DATASET_PATH, encoding="latin-1")
            return {
                "filename": DATASET_PATH.name,
                "rows": len(df),
                "columns": len(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict()
            }
    except Exception:
        pass
    return None


def get_dataset_preview(rows: int = 10) -> Optional[list[dict]]:
    try:
        if DATASET_PATH.exists():
            df = pd.read_csv(DATASET_PATH, encoding="latin-1")
            return df.head(rows).to_dict(orient="records")
    except Exception:
        pass
    return None


def get_glossary() -> str:
    try:
        if GLOSSARY_PATH.exists():
            return GLOSSARY_PATH.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""


def save_glossary(content: str) -> bool:
    try:
        GLOSSARY_PATH.write_text(content, encoding="utf-8")
        return True
    except Exception:
        return False


def get_evaluation_report() -> Optional[dict]:
    try:
        if REPORT_PATH.exists():
            import json
            return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def get_plots() -> list[dict]:
    try:
        if PLOTS_DIR.exists():
            plots = []
            for p in sorted(PLOTS_DIR.glob("*.png"), key=os.path.getmtime, reverse=True):
                plots.append({
                    "name": p.name,
                    "path": str(p.relative_to(BASE_DIR)),
                    "modified": os.path.getmtime(p)
                })
            return plots
    except Exception:
        pass
    return []


def get_dynamic_features_code() -> str:
    try:
        if DYNAMIC_FEATURES_PATH.exists():
            return DYNAMIC_FEATURES_PATH.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""
