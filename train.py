import pandas as pd
import json
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, f1_score, accuracy_score, roc_auc_score
from dynamic_features import apply_feature_engineering, get_model

import mlflow
import mlflow.sklearn

TARGET_CANDIDATES = ['default_flag', 'consumo_annuo', 'target']

def extract_target_name(glossary_path="glossary.md"):
    with open(glossary_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Target" in line or "target" in line.lower():
            for sub_line in lines[i+1:i+5]:
                if sub_line.strip().startswith("-"):
                    return sub_line.split("`")[1] if "`" in sub_line else sub_line.split(":")[0].strip("- ")
    return "target"

def is_classification_task(target_values) -> bool:
    unique_vals = target_values.dropna().unique()
    if len(unique_vals) == 2:
        return True
    if set(unique_vals).issubset({0, 1, True, False}):
        return True
    for name in TARGET_CANDIDATES:
        if 'flag' in name.lower() or 'default' in name.lower():
            return True
    return len(unique_vals) < 10 and all(v == int(v) for v in unique_vals if v is not None)

def setup_mlflow_experiment(experiment_name: str = None, tracking_uri: str = None):
    """Setup MLFlow experiment."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    exp_name = experiment_name or "AutoLLml_Experiments"
    mlflow.set_experiment(exp_name)
    
    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=True,
        log_models=True,
        disable_for_unsupported_versions=True
    )
    
    return exp_name

def main(mlflow_experiment_name: str = None, mlflow_tracking_uri: str = None, iter_num: int = 1):
    exp_name = setup_mlflow_experiment(mlflow_experiment_name, mlflow_tracking_uri)
    
    with mlflow.start_run(run_name=f"iteration_{iter_num}") as run:
        run_id = run.info.run_id
        print(f"[*] MLFlow Run ID: {run_id}")
        
        mlflow.set_tag("iteration", iter_num)
        mlflow.log_param("iteration", iter_num)
        mlflow.log_param("experiment_name", exp_name)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("cv_random_state", 42)
        
        try:
            df = pd.read_csv("data/dataset.csv", encoding="latin-1")
        except FileNotFoundError:
            print("ERROR_DATA: File data/dataset.csv non trovato.")
            mlflow.set_tag("error", "data_file_not_found")
            sys.exit(1)
            
        TARGET_COL = extract_target_name()
        mlflow.log_param("target_column", TARGET_COL)
        
        if TARGET_COL not in df.columns:
            print(f"ERROR_DATA: Colonna target '{TARGET_COL}' non trovata nel dataset.")
            mlflow.set_tag("error", "target_column_not_found")
            sys.exit(1)
        
        y_raw = df[TARGET_COL]
        IS_CLASSIFICATION = is_classification_task(y_raw)
        
        task_type = "classification" if IS_CLASSIFICATION else "regression"
        metric_name = "F1_weighted" if IS_CLASSIFICATION else "R2"
        
        mlflow.set_tag("task_type", task_type)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("metric_name", metric_name)
        
        print(f"[*] Task rilevato: {'CLASSIFICAZIONE' if IS_CLASSIFICATION else 'REGRESSIONE'}")
        
        try:
            df_engineered = apply_feature_engineering(df.copy())
        except Exception as e:
            print(f"ERROR_FE: Errore durante l'esecuzione di apply_feature_engineering:\n{e}")
            mlflow.set_tag("error", "feature_engineering_failed")
            mlflow.log_text(str(e), "error_message.txt")
            sys.exit(1)
            
        X = df_engineered.drop(columns=[TARGET_COL])
        y = df_engineered[TARGET_COL]
        
        mlflow.log_param("num_features_raw", len(df.columns) - 1)
        mlflow.log_param("num_features_engineered", len(X.columns))
        
        model = get_model()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        mlflow.log_param("model_type", type(model.named_steps.get('clf', model)).__name__)
        
        TOP_N = 10
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        corr_dict = {}
        X_num = X.select_dtypes(include=numerics)
        for col in X_num.columns:
            if X_num[col].nunique() > 1:
                corr = X_num[col].corr(y)
                if not np.isnan(corr):
                    corr_dict[col] = float(corr)

        top_corr_dict = {
            k: v for k, v in
            sorted(corr_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:TOP_N]
        }

        cat_score_dict = {}
        X_cat = X.select_dtypes(include=["object", "category", "bool"])
        for col in X_cat.columns:
            if X_cat[col].nunique() < 2:
                continue
            rates = (
                pd.DataFrame({"feat": X_cat[col].astype(str), "target": y})
                .groupby("feat")["target"]
                .mean()
            )
            score = float(rates.var())
            if not np.isnan(score):
                cat_score_dict[col] = score

        top_cat = sorted(cat_score_dict.items(), key=lambda item: item[1], reverse=True)

        n_num = min(len(corr_dict), TOP_N)
        n_cat = min(len(cat_score_dict), TOP_N)
        if n_num + n_cat <= TOP_N:
            slots_num, slots_cat = n_num, n_cat
        else:
            slots_num = max(1, round(TOP_N * n_num / (n_num + n_cat)))
            slots_cat = TOP_N - slots_num

        top_numeric = sorted(corr_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:slots_num]
        top_categoric = top_cat[:slots_cat]

        plot_dir = "evaluation_plots"
        os.makedirs(plot_dir, exist_ok=True)

        print(f"[*] Generando {len(top_numeric)} plot numerici e {len(top_categoric)} plot categorici...")

        for feat, corr_val in top_numeric:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(
                x=y, y=X[feat],
                hue=y,
                palette="Set2",
                inner="quartile",
                cut=0,
                ax=ax,
                legend=False
            )
            ax.set_ylabel(feat)
            ax.set_title(f"{feat}  vs  {TARGET_COL}", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel(f"{TARGET_COL}  (Pearson r = {corr_val:+.3f})", fontsize=11)
            sns.despine(ax=ax)
            plt.tight_layout()
            safe_name = "".join([c if c.isalnum() else "_" for c in feat])
            plt.savefig(os.path.join(plot_dir, f"{safe_name}_vs_target.png"), dpi=120)
            plt.close()

        for feat, score_val in top_categoric:
            tmp = pd.DataFrame({"feat": X_cat[feat].astype(str), "target": y})
            stats = (
                tmp.groupby("feat")["target"]
                .agg(rate="mean", count="count")
                .reset_index()
                .sort_values("rate", ascending=True)
            )
            fig, ax = plt.subplots(figsize=(10, max(4, len(stats) * 0.55 + 1)))
            bars = ax.barh(
                stats["feat"], stats["rate"],
                color=plt.cm.coolwarm_r(stats["rate"].values),
                edgecolor="white", linewidth=0.5
            )
            for bar, (_, row) in zip(bars, stats.iterrows()):
                ax.text(
                    bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"n={int(row['count'])}",
                    va="center", ha="left", fontsize=9, color="#555555"
                )
            ax.set_xlim(0, min(1.0, stats["rate"].max() * 1.25))
            ax.set_xlabel(f"Tasso di {TARGET_COL}  (var rate = {score_val:.4f})", fontsize=11)
            ax.set_ylabel(feat)
            ax.set_title(f"{feat}  —  Tasso di {TARGET_COL} per categoria", fontsize=13, fontweight="bold", pad=12)
            sns.despine(ax=ax)
            plt.tight_layout()
            safe_name = "".join([c if c.isalnum() else "_" for c in feat])
            plt.savefig(os.path.join(plot_dir, f"{safe_name}_vs_target.png"), dpi=120)
            plt.close()
        
        fold_scores = []
        feature_importances = None
        
        try:
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                
                if IS_CLASSIFICATION:
                    fold_score = f1_score(y_val, preds, average='weighted')
                    fold_acc = accuracy_score(y_val, preds)
                    fold_scores.append(fold_score)
                    mlflow.log_metric(f"fold_{fold_idx}_f1", fold_score)
                    mlflow.log_metric(f"fold_{fold_idx}_accuracy", fold_acc)
                else:
                    fold_score = r2_score(y_val, preds)
                    fold_scores.append(fold_score)
                    mlflow.log_metric(f"fold_{fold_idx}_r2", fold_score)
                
                clf_step = model.named_steps.get('clf', model)
                if hasattr(clf_step, 'feature_importances_'):
                    fi = clf_step.feature_importances_
                    if feature_importances is None:
                        feature_importances = fi.copy()
                    else:
                        feature_importances += fi
                    
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            mlflow.log_metric("score_mean", mean_score)
            mlflow.log_metric("score_std", std_score)
            
            importance_dict = {}
            clf_step = model.named_steps.get('clf', model)
            if feature_importances is not None and hasattr(clf_step, 'feature_importances_'):
                feature_importances /= cv.get_n_splits()
                prep_step = model.named_steps.get('prep', None)
                if prep_step is not None and hasattr(prep_step, 'get_feature_names_out'):
                    try:
                        feature_names = list(prep_step.get_feature_names_out())
                    except Exception:
                        feature_names = [f"feature_{i}" for i in range(len(feature_importances))]
                else:
                    feature_names = list(X.columns)
                importance_dict = {
                    name: float(imp)
                    for name, imp in sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
                }
                
                mlflow.log_dict(importance_dict, "feature_importance.json")

            report = {
                "task_type": task_type,
                "metric_name": metric_name,
                "score_mean": mean_score,
                "score_std": std_score,
                "num_features": len(X.columns),
                "top_correlations_with_target": top_corr_dict,
                "feature_importance": importance_dict,
                "mlflow_run_id": run_id
            }
            
            with open("evaluation_report.json", "w") as f:
                json.dump(report, f, indent=2)
            
            mlflow.log_artifact("evaluation_report.json")
            mlflow.log_artifact("dynamic_features.py")
            mlflow.log_artifact("glossary.md")
            mlflow.log_artifacts(plot_dir, artifact_path="evaluation_plots")
            
            print(f"SUCCESS_METRIC: {mean_score:.4f}")
            
        except Exception as e:
            print(f"ERROR_MODEL: Fallimento durante il training loop:\n{e}")
            mlflow.set_tag("error", "training_failed")
            mlflow.log_text(str(e), "error_message.txt")
            sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoML Training with MLFlow tracking")
    parser.add_argument("--experiment", type=str, default=None, help="MLFlow experiment name")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLFlow tracking URI")
    parser.add_argument("--iter", type=int, default=1, help="Iteration number")
    args = parser.parse_args()
    
    main(
        mlflow_experiment_name=args.experiment,
        mlflow_tracking_uri=args.tracking_uri,
        iter_num=args.iter
    )
