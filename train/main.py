import mlflow
import mlflow.sklearn

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


def setup_mlflow(experiment_name: str = None, tracking_uri: str = None) -> str:
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


def run_training(
    iter_num: int,
    experiment_name: str = None,
    tracking_uri: str = None,
    enable_mlflow: bool = True
) -> float:
    exp_name = setup_mlflow(experiment_name, tracking_uri) if enable_mlflow else None
    
    run_id = None
    
    with mlflow.start_run(run_name=f"iteration_{iter_num}") as run:
        run_id = run.info.run_id
        print(f"[*] MLFlow Run ID: {run_id}")
        
        mlflow.set_tag("iteration", iter_num)
        mlflow.log_param("iteration", iter_num)
        mlflow.log_param("experiment_name", exp_name or "disabled")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("cv_random_state", 42)
        
        df = load_dataset()
        target_col = extract_target_name()
        mlflow.log_param("target_column", target_col)
        
        validate_target(df, target_col)
        
        task_type, metric_name = detect_task_info(df, target_col)
        
        mlflow.set_tag("task_type", task_type)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("metric_name", metric_name)
        
        is_classification = task_type == "classification"
        print(f"[*] Task rilevato: {'CLASSIFICAZIONE' if is_classification else 'REGRESSIONE'}")
        
        X, y = prepare_features(df, target_col)
        
        mlflow.log_param("num_features_raw", len(df.columns) - 1)
        mlflow.log_param("num_features_engineered", len(X.columns))
        
        model = _get_model()
        mlflow.log_param("model_type", type(model.named_steps.get('clf', model)).__name__)
        
        corr_dict, top_features = analyze_features(X, y, top_n=10)
        
        plot_dir = ensure_plot_dir("evaluation_plots")
        generate_plots(
            X, y,
            top_features["numeric"],
            top_features["categorical"],
            target_col,
            plot_dir
        )
        
        training_result = cross_validate(X, y, model, is_classification)
        
        log_training_metrics(training_result.mean_score, training_result.std_score)
        log_feature_importance(training_result.feature_importance)
        
        report_data = {
            "task_type": task_type,
            "metric_name": metric_name,
            "score_mean": training_result.mean_score,
            "score_std": training_result.std_score,
            "num_features": len(X.columns),
            "top_correlations_with_target": dict(top_features["numeric"]),
            "feature_importance": training_result.feature_importance or {},
            "mlflow_run_id": run_id
        }
        
        _save_report(report_data)
        log_artifacts()
        
        print(f"SUCCESS_METRIC: {training_result.mean_score:.4f}")
        return training_result.mean_score


def _get_model():
    from dynamic_features import get_model
    return get_model()


def _save_report(report_data: dict) -> None:
    import json
    with open("evaluation_report.json", "w") as f:
        json.dump(report_data, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoML Training with MLFlow tracking")
    parser.add_argument("--experiment", type=str, default=None, help="MLFlow experiment name")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLFlow tracking URI")
    parser.add_argument("--iter", type=int, default=1, help="Iteration number")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLFlow logging")
    args = parser.parse_args()
    
    run_training(
        iter_num=args.iter,
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri,
        enable_mlflow=not args.no_mlflow
    )


if __name__ == "__main__":
    main()
