import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Sequence, Tuple

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression




# ------------------------------
# Funzione principale di Feature Engineering
# ------------------------------
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    return df


# ------------------------------
# Modello ML
# ------------------------------
def get_model():

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", model)
    ])

    return pipeline
