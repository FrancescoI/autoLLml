import numpy as np
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
