# ================================
# Classifier 2 Versions - Deep ANN
# ================================

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import setup_keras_environment, load_and_prepare_data, deep_ann


def iter1(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 1 (Baseline)...")
    acc, prec, rec, f1, prob, model = deep_ann(x_train, y_train, x_test, y_test)

    print(f"\n [- -] Iteration 1 (Baseline) : Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter2(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 2 (Optimized)...")
    acc, prec, rec, f1, prob, model = deep_ann(
        x_train, y_train, x_test, y_test, iter_num=2,
        layer1_neurons=512,
        layer2_neurons=128,
        description="Expanded capacity of dense layers (512 and 128) to better map the 49 complex characters.",
    )

    print(f"\n [- -] Iteration 2 (Optimized): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model


def main(target_iter):
    print("# ====================================================")
    print(f"# RUNNING CLASSIFIER 2 VERSIONS (DEEP ANN) - Mode: {target_iter}")
    print("# ====================================================\n")
    # Load and Prepare data
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    if target_iter in [1, "all"] : iter1(x_train, y_train, x_test, y_test)
    if target_iter in [2, "all"] : iter2(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    setup_keras_environment()

    parser = argparse.ArgumentParser(description="Run specific iterations of the Deep ANN classifier.")
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