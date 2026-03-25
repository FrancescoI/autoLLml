from .data_loader import (
    extract_target_name,
    load_dataset,
    validate_target,
    prepare_features,
    detect_task_info,
)

from .feature_analyzer import analyze_features

from .plot_generator import (
    ensure_plot_dir,
    generate_plots,
)

from .trainer import cross_validate

from .reporter import (
    log_training_metrics,
    log_feature_importance,
    log_artifacts,
)

from utils.config import get_paths


def run_training(iter_num: int) -> tuple[float, list[str]]:
    df = load_dataset()
    target_col = extract_target_name()
    
    validate_target(df, target_col)
    
    task_type, metric_name = detect_task_info(df, target_col)
    
    is_classification = task_type == "classification"
    print(f"[*] Task rilevato: {'CLASSIFICAZIONE' if is_classification else 'REGRESSIONE'}")
    
    X, y = prepare_features(df, target_col)
    
    model = _get_model()
    
    corr_dict, top_features = analyze_features(X, y, top_n=10)
    
    paths = get_paths()
    plot_paths = generate_plots(
        X, y,
        top_features["numeric"],
        top_features["categorical"],
        target_col,
        paths.output_dir,
        iter_num
    )
    
    training_result = cross_validate(X, y, model, is_classification)
    
    log_training_metrics(training_result.mean_score, training_result.std_score)
    log_feature_importance(training_result.feature_importance)
    
    report_data = {
        "task_type": task_type,
        "metric_name": metric_name,
        "score_mean": training_result.mean_score,
        "score_std": training_result.std_score,
        "precision": training_result.precision,
        "recall": training_result.recall,
        "auc_roc": training_result.auc_roc,
        "num_features": len(X.columns),
        "top_correlations_with_target": dict(top_features["numeric"]),
        "feature_importance": training_result.feature_importance or {},
    }
    
    _save_report(report_data)
    log_artifacts()
    
    print(f"SUCCESS_METRIC: {training_result.mean_score:.4f}")
    
    import json as json_mod
    print(f"PLOT_PATHS:{json_mod.dumps(plot_paths)}")
    
    return training_result.mean_score, plot_paths


def _get_model():
    from dynamic_features import get_model
    return get_model()


def _save_report(report_data: dict) -> None:
    import json
    paths = get_paths()
    with open(paths.evaluation_report, "w") as f:
        json.dump(report_data, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoML Training")
    parser.add_argument("--iter", type=int, default=1, help="Iteration number")
    args = parser.parse_args()
    
    run_training(iter_num=args.iter)


if __name__ == "__main__":
    main()