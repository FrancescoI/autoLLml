import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def have(c: str) -> bool:
        return c in df.columns

    def to_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
        a = to_num(a)
        b = to_num(b)
        denom = b.replace(0, np.nan)
        out = a / denom
        return out.replace([np.inf, -np.inf], np.nan)

    # Coerce base numerics
    for c in ["annual_income", "tot_outstanding_debt", "credit_lines_count", "delinquency_30d_freq"]:
        if have(c):
            df[c] = to_num(df[c])

    # Categorical
    if have("industry_sector"):
        df["industry_sector"] = df["industry_sector"].astype("object")

    # ----------------------------
    # 1) Stress finanziario (DTI) + band + crossing
    # ----------------------------
    if have("tot_outstanding_debt") and have("annual_income"):
        df["debt_to_income"] = safe_divide(df["tot_outstanding_debt"], df["annual_income"])

        dti = df["debt_to_income"]
        if dti.notna().sum() >= 20:
            qs = dti.quantile([0.2, 0.4, 0.6, 0.8]).values
            df["debt_to_income_band"] = pd.cut(
                dti,
                bins=[-np.inf, qs[0], qs[1], qs[2], qs[3], np.inf],
                labels=["very_low", "low", "medium", "high", "very_high"],
                include_lowest=True,
            ).astype("object")
        else:
            med = dti.median(skipna=True)
            df["debt_to_income_band"] = np.where(dti.isna(), np.nan, np.where(dti <= med, "low", "high")).astype("object")

    # ----------------------------
    # 2) Capacità di rimborso vs rischio (lines_to_debt, income_per_line) + interactions
    # ----------------------------
    if have("tot_outstanding_debt") and have("credit_lines_count"):
        df["lines_to_debt"] = safe_divide(df["tot_outstanding_debt"], df["credit_lines_count"])

    if have("annual_income") and have("credit_lines_count"):
        df["income_per_line"] = safe_divide(df["annual_income"], df["credit_lines_count"])

    if have("delinquency_30d_freq"):
        if have("debt_to_income"):
            df["risk_score_raw"] = (df["delinquency_30d_freq"] * df["debt_to_income"]).clip(-1e9, 1e9)

        if have("debt_to_income_band"):
            band_weight = {"very_low": 0.2, "low": 0.4, "medium": 0.7, "high": 1.0, "very_high": 1.3}
            df["dti_band_weight"] = df["debt_to_income_band"].map(band_weight)
            df["delinq_x_dti_band"] = df["delinquency_30d_freq"] * df["dti_band_weight"]

        if have("lines_to_debt"):
            med_del = df["delinquency_30d_freq"].median(skipna=True)
            med_ltd = df["lines_to_debt"].median(skipna=True)
            df["delinq_x_lines_to_debt"] = df["delinquency_30d_freq"] * df["lines_to_debt"]
            df["high_delinq_high_lines_exposure"] = ((df["delinquency_30d_freq"] >= med_del) & (df["lines_to_debt"] >= med_ltd)).astype("int8")

        if have("income_per_line"):
            med_del = df["delinquency_30d_freq"].median(skipna=True)
            med_income_line = df["income_per_line"].median(skipna=True)
            df["delinq_x_income_per_line"] = df["delinquency_30d_freq"] * df["income_per_line"]
            df["high_delinq_low_income_per_line"] = ((df["delinquency_30d_freq"] >= med_del) & (df["income_per_line"] <= med_income_line)).astype("int8")

    # ----------------------------
    # 3) Cross categorico con settore (baseline sector-specific)
    # ----------------------------
    if have("industry_sector") and have("delinquency_30d_freq"):
        med_sector_delinq = df.groupby("industry_sector", dropna=True)["delinquency_30d_freq"].transform("median")
        df["delinq_x_sector_baseline"] = df["delinquency_30d_freq"] * med_sector_delinq

        if have("debt_to_income"):
            med_sector_dti = df.groupby("industry_sector", dropna=True)["debt_to_income"].transform("median")
            df["dti_x_sector_baseline"] = df["debt_to_income"] * med_sector_dti

    # ----------------------------
    # 4) Combinazione meccanica + ulteriori ratio utili
    # ----------------------------
    if have("delinquency_30d_freq") and have("tot_outstanding_debt"):
        df["delinquency_per_debt"] = safe_divide(df["delinquency_30d_freq"], df["tot_outstanding_debt"])

    # ----------------------------
    # 5) Interazioni su conteggio linee e morosità
    # ----------------------------
    if have("credit_lines_count") and have("delinquency_30d_freq"):
        df["lines_x_delinquency"] = df["credit_lines_count"] * df["delinquency_30d_freq"]
        df["delinquency_per_line"] = safe_divide(df["delinquency_30d_freq"], df["credit_lines_count"])

        med_del = df["delinquency_30d_freq"].median(skipna=True)
        med_lines = df["credit_lines_count"].median(skipna=True)
        df["high_delinq_high_lines_x"] = ((df["delinquency_30d_freq"] >= med_del) & (df["credit_lines_count"] >= med_lines)).astype("int8")
        df["high_delinq_low_lines_persist"] = ((df["delinquency_30d_freq"] >= med_del) & (df["credit_lines_count"] <= med_lines)).astype("int8")

    # ----------------------------
    # Stabilizzazione clip (no log/polinomi/radici)
    # ----------------------------
    clip_cols = [
        "debt_to_income",
        "lines_to_debt",
        "income_per_line",
        "risk_score_raw",
        "dti_band_weight",
        "delinq_x_dti_band",
        "delinq_x_lines_to_debt",
        "delinq_x_income_per_line",
        "delinq_x_sector_baseline",
        "dti_x_sector_baseline",
        "lines_x_delinquency",
        "delinquency_per_line",
        "delinquency_per_debt",
    ]
    for c in clip_cols:
        if c in df.columns:
            df[c] = df[c].clip(-1e9, 1e9)

    return df


def get_model():
    # NOTE: lista attesa di feature engineered + originali.
    # Se alcune colonne non sono presenti nel training data, garantire coerenza a monte.
    numeric_features = [
        "annual_income",
        "tot_outstanding_debt",
        "credit_lines_count",
        "delinquency_30d_freq",
        "debt_to_income",
        "lines_to_debt",
        "income_per_line",
        "risk_score_raw",
        "dti_band_weight",
        "delinq_x_dti_band",
        "delinq_x_lines_to_debt",
        "delinq_x_income_per_line",
        "delinq_x_sector_baseline",
        "dti_x_sector_baseline",
        "lines_x_delinquency",
        "delinquency_per_line",
        "delinquency_per_debt",
        "high_delinq_high_lines_exposure",
        "high_delinq_low_income_per_line",
        "high_delinq_high_lines_x",
        "high_delinq_low_lines_persist",
    ]
    categorical_features = ["industry_sector", "debt_to_income_band"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        random_state=42,
        learning_rate=0.05,
        max_depth=5,
        max_iter=900,
        l2_regularization=0.0,
        max_leaf_nodes=63,
    )

    return Pipeline(steps=[("prep", preprocessor), ("clf", model)])