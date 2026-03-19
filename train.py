import argparse

from train import run_training


def main():
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
