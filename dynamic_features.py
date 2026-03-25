import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Sequence, Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier


TARGET_CANDIDATES = ['default_flag', 'consumo_annuo', 'target']


def _safe_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def _safe_divide(a: pd.Series, b: pd.Series, default: float = 0.0) -> pd.Series:
    a = _safe_numeric(a)
    b = _safe_numeric(b)
    out = pd.Series(default, index=a.index, dtype='float64')
    mask = b.notna() & (b != 0)
    out.loc[mask] = a.loc[mask] / b.loc[mask]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _safe_product(a: pd.Series, b: pd.Series) -> pd.Series:
    a = _safe_numeric(a)
    b = _safe_numeric(b)
    out = a * b
    return out.replace([np.inf, -np.inf], np.nan)


def _safe_sum(*series_list: pd.Series) -> pd.Series:
    if not series_list:
        return pd.Series(dtype='float64')
    first = series_list[0]
    if isinstance(first, (int, float, np.integer, np.floating)):
        return pd.Series(first)
    idx = first.index
    total = pd.Series(0.0, index=idx, dtype='float64')
    for s in series_list:
        if isinstance(s, (int, float, np.integer, np.floating)):
            s = pd.Series(s, index=idx, dtype='float64')
        total = total.add(_safe_numeric(s), fill_value=0.0)
    return total.replace([np.inf, -np.inf], np.nan)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Preserva la colonna target
    target_col = None
    for t in TARGET_CANDIDATES:
        if t in df.columns:
            target_col = df[t].copy()  # COPIA i valori, non il nome!
            break

    # Normalizza inf/nan
    df = df.replace([np.inf, -np.inf], np.nan)

    # Colonne attese
    annual_income = _safe_numeric(df['annual_income']) if 'annual_income' in df.columns else pd.Series(np.nan, index=df.index)
    total_debt = _safe_numeric(df['tot_outstanding_debt']) if 'tot_outstanding_debt' in df.columns else pd.Series(np.nan, index=df.index)
    delinquency = _safe_numeric(df['delinquency_30d_freq']) if 'delinquency_30d_freq' in df.columns else pd.Series(np.nan, index=df.index)
    credit_lines = _safe_numeric(df['credit_lines_count']) if 'credit_lines_count' in df.columns else pd.Series(np.nan, index=df.index)
    tenure = _safe_numeric(df['customer_tenure_months']) if 'customer_tenure_months' in df.columns else pd.Series(np.nan, index=df.index)
    tickets = _safe_numeric(df['support_tickets_count']) if 'support_tickets_count' in df.columns else pd.Series(np.nan, index=df.index)
    emails = _safe_numeric(df['marketing_email_opens']) if 'marketing_email_opens' in df.columns else pd.Series(np.nan, index=df.index)

    # 1) Rischio di sovraindebitamento
    df['debt_burden_ratio'] = _safe_divide(total_debt, annual_income)
    df['debt_capacity_gap'] = _safe_sum(total_debt, -annual_income)
    df['debt_pressure_index'] = _safe_product(df['debt_burden_ratio'], _safe_sum(pd.Series(1.0, index=df.index), delinquency))

    # 2) Pressione creditizia e fragilità di comportamento
    df['credit_stress_index'] = _safe_product(delinquency, credit_lines)
    df['active_line_delinquency_burden'] = _safe_sum(_safe_product(delinquency, credit_lines), total_debt)
    df['delinquency_per_line'] = _safe_divide(delinquency, credit_lines)

    # 3) Vitalità della relazione cliente-banca
    df['relationship_health_index'] = _safe_sum(
        _safe_divide(emails, _safe_sum(pd.Series(1.0, index=df.index), tenure)),
        -_safe_divide(tickets, _safe_sum(pd.Series(1.0, index=df.index), tenure))
    )
    df['engagement_vs_attrition'] = _safe_sum(emails, -tickets)
    df['tenure_adjusted_relationship_frict'] = _safe_product(tickets, _safe_divide(pd.Series(1.0, index=df.index), _safe_sum(pd.Series(1.0, index=df.index), tenure)))

    # 4) Rischio contestuale per segmento operativo
    if 'industry_sector' in df.columns:
        industry = df['industry_sector'].astype('string').fillna('Unknown')
        df['industry_sector_x_debt_burden'] = industry + '|' + df['debt_burden_ratio'].fillna(-1).round(4).astype(str)
        df['industry_sector_x_delinquency'] = industry + '|' + delinquency.fillna(-1).round(4).astype(str)
        df['industry_sector_x_credit_stress'] = industry + '|' + df['credit_stress_index'].fillna(-1).round(4).astype(str)

    if 'branch_code' in df.columns:
        branch = df['branch_code'].astype('string').fillna('Unknown')
        df['branch_code_x_delinquency'] = branch + '|' + delinquency.fillna(-1).round(4).astype(str)
        df['branch_code_x_tickets'] = branch + '|' + tickets.fillna(-1).round(4).astype(str)

    if 'account_manager_id' in df.columns:
        manager = df['account_manager_id'].astype('string').fillna('Unknown')
        df['account_manager_id_x_delinquency'] = manager + '|' + delinquency.fillna(-1).round(4).astype(str)
        df['account_manager_id_x_debt_burden'] = manager + '|' + df['debt_burden_ratio'].fillna(-1).round(4).astype(str)

    # 5) Intensità di servizio rispetto alla maturità del cliente
    df['service_intensity_per_tenure'] = _safe_divide(_safe_sum(tickets, emails), _safe_sum(pd.Series(1.0, index=df.index), tenure))
    df['service_frict_to_maturity'] = _safe_sum(_safe_product(tickets, _safe_divide(pd.Series(1.0, index=df.index), _safe_sum(pd.Series(1.0, index=df.index), tenure))), -_safe_product(emails, _safe_divide(pd.Series(1.0, index=df.index), _safe_sum(pd.Series(1.0, index=df.index), tenure))))
    df['recent_customer_service_load'] = _safe_sum(tickets, _safe_divide(emails, _safe_sum(pd.Series(1.0, index=df.index), tenure)))

    # Pruning feature irrilevanti individuate: rimuoviamo le originali candidate al pruning
    prune_cols = ['credit_lines_count', 'customer_tenure_months', 'marketing_email_opens', 'support_tickets_count']
    for c in prune_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Riproduci la colonna target
    if target_col is not None:
        df["default_flag"] = target_col

    return df


def get_model():
    categorical_features = [
        c for c in ['industry_sector', 'branch_code', 'account_manager_id']
        if c is not None
    ]

    numerical_features = [
        'annual_income',
        'tot_outstanding_debt',
        'delinquency_30d_freq',
        'debt_burden_ratio',
        'debt_capacity_gap',
        'debt_pressure_index',
        'credit_stress_index',
        'active_line_delinquency_burden',
        'delinquency_per_line',
        'relationship_health_index',
        'engagement_vs_attrition',
        'tenure_adjusted_relationship_frict',
        'service_intensity_per_tenure',
        'service_frict_to_maturity',
        'recent_customer_service_load',
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                ]),
                numerical_features
            ),
            (
                'cat',
                Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                ]),
                categorical_features
            ),
        ],
        remainder='drop'
    )

    model = HistGradientBoostingClassifier(
        random_state=42,
        max_depth=None,
        learning_rate=0.1,
        max_iter=200
    )

    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('clf', model)
    ])

    return pipeline