import os
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent


BASELINE_DYNAMIC_FEATURES = '''import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Sequence, Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


TARGET_CANDIDATES = ['default_flag', 'consumo_annuo', 'target']


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    target_col = None
    for t in TARGET_CANDIDATES:
        if t in df.columns:
            target_col = df[t].copy()
            break
    
    return df


def get_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), ['annual_income', 'tot_outstanding_debt', 'credit_lines_count', 'delinquency_30d_freq']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['industry_sector']),
        ]
    )
    
    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('clf', model)
    ])

    return pipeline
'''


def reset_codebase():
    """Reset codebase to baseline."""
    
    items = [
        "evaluation_report.json",
        "evaluation_report.md",
        "evaluation_plots",
        "traces.jsonl",
        "memory.json",
        "__pycache__",
        "agents/__pycache__",
    ]
    
    for item in items:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
            print(f"[*] Deleted: {item}")
    
    output_path = ROOT_DIR / "dynamic_features.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(BASELINE_DYNAMIC_FEATURES)
    print("[*] Reset: dynamic_features.py to baseline (LogisticRegression)")
    
    print("\n[+] Codebase reset complete")
    print("[*] To start new experiment: python main.py --iterations 5")


if __name__ == "__main__":
    reset_codebase()
