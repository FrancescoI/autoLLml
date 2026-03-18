import pandas as pd
import json
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from dynamic_features import apply_feature_engineering, get_model

def extract_target_name(glossary_path="glossary.md"):
    # Estrazione euristica del target dal glossario
    with open(glossary_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Target" in line or "target" in line.lower():
            for sub_line in lines[i+1:i+5]:
                if sub_line.strip().startswith("-"):
                    return sub_line.split("`")[1] if "`" in sub_line else sub_line.split(":")[0].strip("- ")
    return "target" # Fallback

def main():
    try:
        df = pd.read_csv("data/dataset.csv", encoding="latin-1")
    except FileNotFoundError:
        print("ERROR_DATA: File data/dataset.csv non trovato.")
        sys.exit(1)
        
    TARGET_COL = extract_target_name()
    if TARGET_COL not in df.columns:
        print(f"ERROR_DATA: Colonna target '{TARGET_COL}' non trovata nel dataset.")
        sys.exit(1)
    
    try:
        df_engineered = apply_feature_engineering(df.copy())
    except Exception as e:
        print(f"ERROR_FE: Errore durante l'esecuzione di apply_feature_engineering:\n{e}")
        sys.exit(1)
        
    X = df_engineered.drop(columns=[TARGET_COL])
    y = df_engineered[TARGET_COL]
    
    model = get_model()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # -------------------------------------------------------------
    # 1. Ranking univariato e Plotting adattivo (numerico vs categorico)
    # -------------------------------------------------------------
    TOP_N = 10
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    # -- Stream A: feature numeriche → ranking per |Pearson corr| --
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

    # -- Stream B: feature categoriche/bool → ranking per varianza del target rate --
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

    # -- Distribuzione dei TOP_N slot tra i due pool --
    n_num = min(len(corr_dict), TOP_N)
    n_cat = min(len(cat_score_dict), TOP_N)
    if n_num + n_cat <= TOP_N:
        slots_num, slots_cat = n_num, n_cat
    else:
        # Proporzione bilanciata, priorità ai numerici
        slots_num = max(1, round(TOP_N * n_num / (n_num + n_cat)))
        slots_cat = TOP_N - slots_num

    top_numeric = sorted(corr_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:slots_num]
    top_categoric = top_cat[:slots_cat]

    # Creazione directory plots
    plot_dir = "evaluation_plots"
    os.makedirs(plot_dir, exist_ok=True)

    print(f"[*] Generando {len(top_numeric)} plot numerici e {len(top_categoric)} plot categorici...")

    # -- Plot numerici: violin plot (distribuzione per classe) --
    for feat, corr_val in top_numeric:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            x=y, y=X[feat],
            palette="Set2",
            inner="quartile",
            cut=0,
            ax=ax
        )
        ax.set_xlabel(TARGET_COL)
        ax.set_ylabel(feat)
        ax.set_title(
            f"{feat}  vs  {TARGET_COL}",
            fontsize=13, fontweight="bold", pad=12
        )
        ax.set_xlabel(f"{TARGET_COL}  (Pearson r = {corr_val:+.3f})", fontsize=11)
        sns.despine(ax=ax)
        plt.tight_layout()
        safe_name = "".join([c if c.isalnum() else "_" for c in feat])
        plt.savefig(os.path.join(plot_dir, f"{safe_name}_vs_target.png"), dpi=120)
        plt.close()

    # -- Plot categorici: bar chart orizzontale del target rate per categoria --
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
        # Annotazioni n= per ogni barra
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
        ax.set_title(
            f"{feat}  —  Tasso di {TARGET_COL} per categoria",
            fontsize=13, fontweight="bold", pad=12
        )
        sns.despine(ax=ax)
        plt.tight_layout()
        safe_name = "".join([c if c.isalnum() else "_" for c in feat])
        plt.savefig(os.path.join(plot_dir, f"{safe_name}_vs_target.png"), dpi=120)
        plt.close()
    # -------------------------------------------------------------
    
    fold_aucs = []
    feature_importances = None  # Inizializzato lazily dopo il primo fit (dimensione post-preprocessing)
    
    try:
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            fold_aucs.append(r2_score(y_val, preds))
            
            # Estrazione feature importance cumulativa (supporto Pipeline)
            clf_step = model.named_steps.get('clf', model)
            if hasattr(clf_step, 'feature_importances_'):
                fi = clf_step.feature_importances_
                if feature_importances is None:
                    feature_importances = fi.copy()
                else:
                    feature_importances += fi
                
        mean_auc = np.mean(fold_aucs)
        
        # Costruzione del dizionario delle importanze (media sui fold)
        importance_dict = {}
        clf_step = model.named_steps.get('clf', model)
        if feature_importances is not None and hasattr(clf_step, 'feature_importances_'):
            feature_importances /= cv.get_n_splits()
            # Recupera i nomi delle feature dopo il preprocessing (ColumnTransformer)
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

        report = {
            "r2_score_mean": mean_auc,
            "r2_score_std": np.std(fold_aucs),
            "num_features": len(X.columns),
            "top_correlations_with_target": top_corr_dict,
            "feature_importance": importance_dict
        }
        
        with open("evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"SUCCESS_METRIC: {mean_auc:.4f}")
        
    except Exception as e:
        print(f"ERROR_MODEL: Fallimento durante il training loop:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()