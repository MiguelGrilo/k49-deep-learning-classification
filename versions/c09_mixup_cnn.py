# ==================================================
# Classifier 9 Versions - MixUp Data Augmentation
# ==================================================

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import setup_keras_environment, load_and_prepare_data, mixup_cnn


def iter1(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 1 (40 Epochs)...")
    acc, prec, rec, f1, prob, model = mixup_cnn(
        x_train, y_train, x_test, y_test, iter_num=1,
        epochs=40,
        batch_size=256,
        description="Baseline CNN trained with MixUp Data Augmentation (40 epochs).",
    )

    print(f"\n [- -] Iteration 1 (40 Epochs) : Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter2(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 2 (80 Epochs)...")
    acc, prec, rec, f1, prob, model = mixup_cnn(
        x_train, y_train, x_test, y_test, iter_num=2,
        epochs=80,
        batch_size=256,
        description="Extended epochs to 80. MixUp networks benefit heavily from longer training.",
    )

    print(f"\n [- -] Iteration 2 (80 Epochs): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model


def main(target_iter):
    print("# ==============================================================")
    print(f"# RUNNING CLASSIFIER 9 VERSIONS (MIXUP AUGMENTATION) - Mode: {target_iter}")
    print("# ==============================================================\n")
    # Load and Prepare data
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    if target_iter in [1, "all"] : iter1(x_train, y_train, x_test, y_test)
    if target_iter in [2, "all"] : iter2(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    setup_keras_environment()

    parser = argparse.ArgumentParser(description="Run specific iterations of the MixUp CNN classifier.")
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