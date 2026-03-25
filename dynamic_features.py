import numpy as np
import pandas as pd
from typing import List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier


TARGET_CANDIDATES = ["default_flag", "consumo_annuo", "target"]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _cat(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].astype("object").fillna("Unknown").replace([np.inf, -np.inf], "Unknown")
    return pd.Series("Unknown", index=df.index, dtype="object")


def _bucket(series: pd.Series, q: int = 4) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(0, index=series.index, dtype="int64")
    try:
        x = s.rank(method="first")
        b = pd.qcut(x, q=min(q, int(x.nunique())), labels=False, duplicates="drop")
        return b.fillna(0).astype("int64")
    except Exception:
        return pd.Series(0, index=series.index, dtype="int64")


def _safe_group_rate(df: pd.DataFrame, keys: List[str], target: pd.Series, prefix: str) -> None:
    existing = [k for k in keys if k in df.columns]
    if not existing:
        return

    grp_key = df[existing].fillna("Unknown").astype("object")
    df[f"{prefix}_group_size"] = grp_key.groupby(existing, dropna=False)[existing[0]].transform("size").astype("float64")

    t = pd.to_numeric(target, errors="coerce")
    if t.notna().nunique() > 1:
        tmp = grp_key.copy()
        tmp["_target_"] = t.fillna(0)
        rate_map = tmp.groupby(existing, dropna=False)["_target_"].mean()
        df[f"{prefix}_historical_default_rate"] = grp_key.join(rate_map.rename("rate"), on=existing)["rate"].astype("float64")
    else:
        df[f"{prefix}_historical_default_rate"] = np.nan


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    target_col = None
    for t in TARGET_CANDIDATES:
        if t in df.columns:
            target_col = df[t].copy()
            break

    df = df.replace([np.inf, -np.inf], np.nan)

    annual_income = _num(df, "annual_income")
    tot_outstanding_debt = _num(df, "tot_outstanding_debt")
    delinquency_30d_freq = _num(df, "delinquency_30d_freq")
    credit_lines_count = _num(df, "credit_lines_count")
    customer_tenure_months = _num(df, "customer_tenure_months")
    support_tickets_count = _num(df, "support_tickets_count")
    marketing_email_opens = _num(df, "marketing_email_opens")

    industry_sector = _cat(df, "industry_sector")
    branch_code = _cat(df, "branch_code")
    account_manager_id = _cat(df, "account_manager_id")

    # Consolidamento blocco finanziario
    df["debt_sustainability_index"] = annual_income - tot_outstanding_debt
    df["debt_burden_gap"] = tot_outstanding_debt - annual_income
    df["financial_stress_normalized"] = tot_outstanding_debt + credit_lines_count - annual_income
    df["multi_line_pressure"] = tot_outstanding_debt + (credit_lines_count * annual_income)
    df["leverage_complexity_pressure"] = tot_outstanding_debt + credit_lines_count + delinquency_30d_freq

    # Fragilità comportamentale e relazionale
    df["support_marketing_disalignment"] = support_tickets_count - marketing_email_opens
    df["engagement_gap"] = marketing_email_opens - support_tickets_count
    df["relationship_deterioration"] = support_tickets_count + delinquency_30d_freq - customer_tenure_months
    df["early_fragility_signal"] = delinquency_30d_freq + customer_tenure_months
    df["service_intensity"] = support_tickets_count + marketing_email_opens

    # Rischio di ecosistema commerciale
    if "default_flag" in df.columns:
        target_ref = target_col if target_col is not None else df["default_flag"]
    else:
        target_ref = target_col if target_col is not None else pd.Series(np.nan, index=df.index)

    _safe_group_rate(df, ["industry_sector"], target_ref, "industry")
    _safe_group_rate(df, ["branch_code"], target_ref, "branch")
    _safe_group_rate(df, ["account_manager_id"], target_ref, "manager")
    _safe_group_rate(df, ["industry_sector", "branch_code"], target_ref, "industry_branch")
    _safe_group_rate(df, ["industry_sector", "account_manager_id"], target_ref, "industry_manager")

    # Crossings a maggiore semantica
    df["sector_financial_profile"] = industry_sector.astype(str) + "__" + _bucket(df["financial_stress_normalized"]).astype(str)
    df["branch_service_profile"] = branch_code.astype(str) + "__" + _bucket(df["support_marketing_disalignment"]).astype(str)
    df["manager_fragility_profile"] = account_manager_id.astype(str) + "__" + _bucket(df["relationship_deterioration"]).astype(str)
    df["sector_fragility_profile"] = industry_sector.astype(str) + "__" + _bucket(df["early_fragility_signal"]).astype(str)

    # Vulnerabilità combinata ma non troppo ridondante
    df["composite_risk_core"] = (
        _bucket(tot_outstanding_debt) +
        _bucket(-annual_income.fillna(0)) +
        _bucket(credit_lines_count) +
        _bucket(support_tickets_count)
    ).astype("int64")

    df["ecosystem_risk_bundle"] = (
        _bucket(df["industry_historical_default_rate"]) +
        _bucket(df["branch_historical_default_rate"]) +
        _bucket(df["manager_historical_default_rate"])
    ).astype("int64")

    # Pruning aggressivo di segnali deboli/rumorosi
    prune_cols = [
        "marketing_email_opens",
        "support_tickets_count",
        "delinquency_30d_freq",
    ]
    for c in prune_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    if target_col is not None:
        df["default_flag"] = target_col

    return df


def get_model():
    numeric_features = [
        "annual_income",
        "tot_outstanding_debt",
        "credit_lines_count",
        "customer_tenure_months",
        "debt_sustainability_index",
        "debt_burden_gap",
        "financial_stress_normalized",
        "multi_line_pressure",
        "leverage_complexity_pressure",
        "support_marketing_disalignment",
        "engagement_gap",
        "relationship_deterioration",
        "early_fragility_signal",
        "service_intensity",
        "industry_group_size",
        "industry_historical_default_rate",
        "branch_group_size",
        "branch_historical_default_rate",
        "manager_group_size",
        "manager_historical_default_rate",
        "industry_branch_group_size",
        "industry_branch_historical_default_rate",
        "industry_manager_group_size",
        "industry_manager_historical_default_rate",
        "composite_risk_core",
        "ecosystem_risk_bundle",
    ]

    categorical_features = [
        "industry_sector",
        "branch_code",
        "account_manager_id",
        "sector_financial_profile",
        "branch_service_profile",
        "manager_fragility_profile",
        "sector_fragility_profile",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    model = GradientBoostingClassifier(random_state=42)

    return Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", model),
    ])