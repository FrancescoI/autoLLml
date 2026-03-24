import argparse

from .main import run_training


def main():
    parser = argparse.ArgumentParser(description="AutoML Training")
    parser.add_argument("--iter", type=int, default=1, help="Iteration number")
    args = parser.parse_args()
    
    run_training(iter_num=args.iter)


if __name__ == "__main__":
    main()
