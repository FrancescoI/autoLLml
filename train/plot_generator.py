import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.config import get_paths


def ensure_plot_dir(plot_dir: str | None = None, iter_num: int | None = None) -> str:
    if plot_dir is None:
        plot_dir = get_paths().output_dir
    if iter_num is not None:
        plot_dir = os.path.join(plot_dir, f"iter_{iter_num}")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def _sanitize_filename(name: str) -> str:
    return "".join([c if c.isalnum() else "_" for c in name])


def get_latest_plot_paths(base_dir: str | None = None, iter_num: int | None = None, max_plots: int = 10) -> list[str]:
    if base_dir is None:
        base_dir = get_paths().output_dir
    if iter_num is not None:
        plot_dir = os.path.join(base_dir, f"iter_{iter_num}")
    else:
        plot_dir = base_dir
    
    if not os.path.isdir(plot_dir):
        return []
    
    import glob
    files = sorted(glob.glob(os.path.join(plot_dir, "*.png")), key=os.path.getmtime, reverse=True)
    return files[:max_plots]


def plot_numeric_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_numeric: list[tuple[str, float]],
    target_col: str,
    plot_dir: str
) -> list[str]:
    plot_paths = []
    
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
        ax.set_title(f"{feat}  vs  {target_col}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(f"{target_col}  (Pearson r = {corr_val:+.3f})", fontsize=11)
        sns.despine(ax=ax)
        plt.tight_layout()
        
        safe_name = _sanitize_filename(feat)
        path = os.path.join(plot_dir, f"{safe_name}_vs_target.png")
        plt.savefig(path, dpi=120)
        plt.close()
        plot_paths.append(path)
    
    return plot_paths


def plot_categorical_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_categoric: list[tuple[str, float]],
    target_col: str,
    plot_dir: str
) -> list[str]:
    X_cat = X.select_dtypes(include=["object", "category", "bool"])
    plot_paths = []
    
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
        ax.set_xlabel(f"Tasso di {target_col}  ( var rate = {score_val:.4f})", fontsize=11)
        ax.set_ylabel(feat)
        ax.set_title(f"{feat}  —  Tasso di {target_col} per categoria", fontsize=13, fontweight="bold", pad=12)
        sns.despine(ax=ax)
        plt.tight_layout()
        
        safe_name = _sanitize_filename(feat)
        path = os.path.join(plot_dir, f"{safe_name}_vs_target.png")
        plt.savefig(path, dpi=120)
        plt.close()
        plot_paths.append(path)
    
    return plot_paths


def generate_plots(
    X: pd.DataFrame,
    y: pd.Series,
    top_numeric: list[tuple[str, float]],
    top_categoric: list[tuple[str, float]],
    target_col: str,
    plot_dir: str | None = None,
    iter_num: int | None = None
) -> list[str]:
    plot_dir = ensure_plot_dir(plot_dir, iter_num)
    
    print(f"[*] Generando {len(top_numeric)} plot numerici e {len(top_categoric)} plot categorici...")
    
    numeric_paths = plot_numeric_features(X, y, top_numeric, target_col, plot_dir)
    categorical_paths = plot_categorical_features(X, y, top_categoric, target_col, plot_dir)
    
    return numeric_paths + categorical_paths
