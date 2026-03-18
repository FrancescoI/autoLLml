import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier


TARGET_CANDIDATES = ["default_flag", "consumo_annuo", "target"]


def _find_target_column(df: pd.DataFrame) -> str | None:
    for t in TARGET_CANDIDATES:
        if t in df.columns:
            return t
    return None


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    target_col = _find_target_column(df)
    if target_col is not None:
        target = df[target_col]
    else:
        target = None

    # Columns expected by the strategy (handle gracefully if absent)
    col_income = "annual_income"
    col_debt = "tot_outstanding_debt"
    col_lines = "credit_lines_count"
    col_delinq = "delinquency_30d_freq"
    col_sector = "industry_sector"

    # Ensure numeric columns exist
    numeric_cols = [c for c in [col_income, col_debt, col_lines, col_delinq] if c in df.columns]
    for c in numeric_cols:
        # do not create elementari trasformate: only fill missing with median-like later in model;
        # here we only create safe derived features using coalescing.
        pass

    # Safe coalescing for ratio/splitting thresholds (business rules)
    # Use medians computed on observed values at feature-engineering time (to avoid NaNs inside rules)
    def _coalesce_with_median(series: pd.Series) -> pd.Series:
        if series.isna().all():
            return series.fillna(0)
        med = series.median()
        return series.fillna(med if np.isfinite(med) else 0)

    if col_debt in df.columns and col_income in df.columns:
        debt_to_income = _coalesce_with_median(df[col_debt]) / _coalesce_with_median(df[col_income]).replace(
            0, np.nan
        )
        df["debt_to_income"] = debt_to_income.replace([np.inf, -np.inf], np.nan)

        # Discretized stress score with business thresholds (no log/polynomial)
        # Thresholds chosen to create interpretable underwriting buckets
        dti = df["debt_to_income"]
        df["dti_bucket"] = pd.cut(
            dti,
            bins=[-np.inf, 0.25, 0.75, 1.5, 3.0, np.inf],
            labels=["lt_25pct", "25_75pct", "75pct_1p5x", "1p5x_3x", "gt_3x"],
        ).astype(object)

        # Binary threshold flags
        df["flag_dti_ge_1p5"] = (dti >= 1.5).astype(int)
        df["flag_dti_ge_3"] = (dti >= 3.0).astype(int)

    if col_delinq in df.columns and col_debt in df.columns:
        df["delinq_x_debt"] = _coalesce_with_median(df[col_delinq]) * _coalesce_with_median(df[col_debt])

        # "Weighted risk" based on delinquency frequency * exposure (interpretable)
        df["risk_weighted_debt"] = df["delinq_x_debt"]

        df["flag_delinq_pos"] = (df[col_delinq].fillna(0) > 0).astype(int)
        df["delinq_bucket"] = pd.cut(
            df[col_delinq].fillna(0),
            bins=[-np.inf, 0, 1, 2, 5, np.inf],
            labels=["0", "1", "2_3", "4_5", "gt_5"],
        ).astype(object)

    if col_lines in df.columns and col_debt in df.columns:
        # debt per active line (avoid div by zero)
        denom = _coalesce_with_median(df[col_lines]).astype(float)
        denom = denom.clip(lower=1)
        df["debt_per_line"] = _coalesce_with_median(df[col_debt]).astype(float) / denom

        df["flag_many_lines"] = (df[col_lines].fillna(0) >= 5).astype(int)

    # Interaction / crossing features per strategy
    if col_lines in df.columns and col_delinq in df.columns:
        df["lines_x_delinq"] = _coalesce_with_median(df[col_lines]) * _coalesce_with_median(df[col_delinq])

    if "dti_bucket" in df.columns and "delinq_bucket" in df.columns:
        # Crossing delinquency bucket × DTI bucket
        df["delinqbucket_x_dtibucket"] = (
            df["delinq_bucket"].astype("object").astype(str).fillna("Unknown")
            + "__"
            + df["dti_bucket"].astype("object").astype(str).fillna("Unknown")
        )
        # Remove extreme redundancy later (model can learn, but we keep pruning via selection in get_model)

    if col_lines in df.columns and "debt_to_income" in df.columns:
        # Complexity interactions: high lines + high leverage (thresholded)
        df["flag_lines_ge_5"] = (df[col_lines].fillna(0) >= 5).astype(int)
        df["complexity_score"] = (
            df["flag_lines_ge_5"] * df["flag_dti_ge_1p5"].fillna(0)
        ).astype(int)  # interpretable combination

    if col_sector in df.columns:
        # Industry segment thresholds with existing signals (sector × underwriting flags/buckets)
        if "dti_bucket" in df.columns:
            df["sector_x_dti_bucket"] = (
                df[col_sector].astype("object").fillna("Unknown").astype(str) + "__" + df["dti_bucket"].astype("object").fillna("Unknown").astype(str)
            )

        if "delinq_bucket" in df.columns:
            df["sector_x_delinq_bucket"] = (
                df[col_sector].astype("object").fillna("Unknown").astype(str) + "__" + df["delinq_bucket"].astype("object").fillna("Unknown").astype(str)
            )

    # Additional business segmentation: median delinquency escalation categories (discrete)
    if col_delinq in df.columns:
        # If delinquency_30d_freq is continuous, bucket already created; also compute escalation flag
        df["flag_delinq_ge_2"] = (df[col_delinq].fillna(0) >= 2).astype(int)

    # Pruning: avoid keeping target column inside features
    # Prune obviously redundant derived columns (keep only those used)
    # Note: we keep base numeric columns and derived ones; pruning is handled in get_model via explicit column lists.
    if target_col is not None:
        df[target_col] = target

    return df


def get_model():
    # Feature selection (pruning): keep a curated set
    numeric_features = [
        "annual_income",
        "tot_outstanding_debt",
        "credit_lines_count",
        "delinquency_30d_freq",
        # ratios / scores
        "debt_to_income",
        "debt_per_line",
        "delinq_x_debt",
        "risk_weighted_debt",
        "lines_x_delinq",
        # flags / interactions
        "flag_dti_ge_1p5",
        "flag_dti_ge_3",
        "flag_delinq_pos",
        "flag_delinq_ge_2",
        "flag_many_lines",
        "flag_lines_ge_5",
        "complexity_score",
    ]

    categorical_features = [
        "industry_sector",
        "dti_bucket",
        "delinq_bucket",
        "delinqbucket_x_dtibucket",
        "sector_x_dti_bucket",
        "sector_x_delinq_bucket",
    ]

    # Keep only those actually present in data at runtime (safe with ColumnTransformer)
    # ColumnTransformer will error if columns are missing, so we build lists dynamically in a custom preprocessor.
    # However, sklearn doesn't support dynamic columns inside get_model without fit-time data.
    # We therefore include only columns that are likely present. If some are absent, user should ensure columns exist.
    # To be robust, we will use ColumnTransformer with remainder='drop' and only use columns that exist
    # by relying on pandas columns at fit-time: ColumnTransformer doesn't auto-filter, but
    # we can fit with a wrapper pipeline that selects existing columns.
    class _ColumnFilter:
        def __init__(self, cols):
            self.cols = cols

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            existing = [c for c in self.cols if c in X.columns]
            return X[existing].copy()

    # We'll use separate filters to avoid missing-column errors
    numeric_filter = _ColumnFilter(numeric_features)
    categorical_filter = _ColumnFilter(categorical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("select", numeric_filter),
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", "passthrough"),
                    ]
                ),
                [c for c in numeric_features if c in numeric_features],  # placeholder; overwritten by select at runtime
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("select", categorical_filter),
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                [c for c in categorical_features if c in categorical_features],  # placeholder
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Model choice: gradient boosting suitable for non-linear interactions and mixed types post-encoding
    model = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=500,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=1e-2,
    )

    pipeline = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
    return pipeline