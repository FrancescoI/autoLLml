import pandas as pd
import sys

TARGET_CANDIDATES = ['default_flag', 'consumo_annuo', 'target']


def extract_target_name(glossary_path: str = "glossary.md") -> str:
    with open(glossary_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Target" in line or "target" in line.lower():
            for sub_line in lines[i+1:i+5]:
                if sub_line.strip().startswith("-"):
                    return sub_line.split("`")[1] if "`" in sub_line else sub_line.split(":")[0].strip("- ")
    return "target"


def is_classification_task(target_values: pd.Series) -> bool:
    unique_vals = target_values.dropna().unique()
    if len(unique_vals) == 2:
        return True
    if set(unique_vals).issubset({0, 1, True, False}):
        return True
    for name in TARGET_CANDIDATES:
        if 'flag' in name.lower() or 'default' in name.lower():
            return True
    return len(unique_vals) < 10 and all(v == int(v) for v in unique_vals if v is not None)


def load_dataset(path: str = "data/dataset.csv", encoding: str = "latin-1") -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=encoding)
    except FileNotFoundError:
        print(f"ERROR_DATA: File {path} non trovato.")
        sys.exit(1)


def validate_target(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        print(f"ERROR_DATA: Colonna target '{target_col}' non trovata nel dataset.")
        sys.exit(1)


def prepare_features(
    df: pd.DataFrame,
    target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    from dynamic_features import apply_feature_engineering
    
    try:
        df_engineered = apply_feature_engineering(df.copy())
    except Exception as e:
        print(f"ERROR_FE: Errore durante l'esecuzione di apply_feature_engineering:\n{e}")
        sys.exit(1)
    
    X = df_engineered.drop(columns=[target_col])
    y = df_engineered[target_col]
    
    return X, y


def detect_task_info(df: pd.DataFrame, target_col: str) -> tuple[str, str]:
    y_raw = df[target_col]
    is_classification = is_classification_task(y_raw)
    
    task_type = "classification" if is_classification else "regression"
    metric_name = "F1_weighted" if is_classification else "R2"
    
    return task_type, metric_name
