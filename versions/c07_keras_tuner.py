# ==========================================================
# Classifier 7 Versions - Hyperparameter Tuning (KerasTuner)
# ==========================================================

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import setup_keras_environment, load_and_prepare_data, keras_tuner


def iter1(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 1 (Fast Search)...")
    acc, prec, rec, f1, prob, model = keras_tuner(
        x_train, y_train, x_test, y_test, iter_num=1,
        max_trials=3,
        epochs=5,
        description="Fast Search: 3 combinations, 5 epochs each.",
    )

    print(f"\n [- -] Iteration 1 (Fast Search): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter2(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 2 (Deep Search)...")
    acc, prec, rec, f1, prob, model = keras_tuner(
        x_train, y_train, x_test, y_test, iter_num=2,
        max_trials=10,
        epochs=12,
        description="Deep Search: Testing 10 combinations with 12 epochs each to find SOTA parameters.",
    )

    print(f"\n [- -] Iteration 2 (Deep Search): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model


def main(target_iter):
    print("# =======================================================")
    print(f"# RUNNING CLASSIFIER 7 VERSIONS (KERAS KERAS TUNER) - Mode: {target_iter}")
    print("# =======================================================\n")
    # Load and Prepare data
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    if target_iter in [1, "all"] : iter1(x_train, y_train, x_test, y_test)
    if target_iter in [2, "all"] : iter2(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    setup_keras_environment()

    parser = argparse.ArgumentParser(description="Run specific iterations of the KerasTuner classifier.")
    parser.add_argument(
        "--iter",
        type=str,
        default="2",  # Latest iteration as default
        choices=["1", "2", "all"],
        help="Choose which iteration to run: '1', '2' or 'all' (latest as default).",
    )

    args = parser.parse_args()
    run_iter = int(args.iter) if args.iter.isdigit() else args.iter

    main(run_iter)