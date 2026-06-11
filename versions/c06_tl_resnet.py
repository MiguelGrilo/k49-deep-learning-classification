# ======================================================
# Classifier 6 Versions - Transfer Learning (ResNet50V2)
# ======================================================

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import setup_keras_environment, load_and_prepare_data, tl_resnet


def iter1(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 1 (Frozen)...")
    acc, prec, rec, f1, prob, model = tl_resnet(
        x_train, y_train, x_test, y_test, iter_num=1,
        description="Baseline Transfer Learning (Base weights completely frozen).",
    )

    print(f"\n [- -] Iteration 1 (Frozen) : Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter2(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 2 (Fine-Tuning)...")
    acc, prec, rec, f1, prob, model = tl_resnet(
        x_train, y_train, x_test, y_test, iter_num=2,
        epochs=10,
        batch_size=128,
        fine_tune=True,
        fine_tune_epochs=15,
        fine_tune_lr=1e-5,
        description="Applied Fine-Tuning: Unfroze base model after warmup to adapt features to K49.",
    )

    print(f"\n [- -] Iteration 2 (Fine-Tuning): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter3(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 3 (SOTA Setup)...")
    acc, prec, rec, f1, prob, model = tl_resnet(
        x_train, y_train, x_test, y_test, iter_num=3,
        epochs=15,
        batch_size=128,
        img_size=72,
        use_augmentation=True,
        pooling_type="max",
        fine_tune=True,
        fine_tune_epochs=20,
        fine_tune_lr=1e-5,
        description="SOTA Fine-Tuning: Upscaling to 72x72, Data Augmentation, and Global MaxPooling.",
    )

    print(f"\n [- -] Iteration 3 (SOTA Setup): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model


def main(target_iter):
    print("# =============================================================")
    print(f"# RUNNING CLASSIFIER 6 VERSIONS (TRANSFER LEARNING) - Mode: {target_iter}")
    print("# =============================================================\n")
    # Load and Prepare data
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    if target_iter in [1, "all"] : iter1(x_train, y_train, x_test, y_test)
    if target_iter in [2, "all"] : iter2(x_train, y_train, x_test, y_test)
    if target_iter in [3, "all"] : iter3(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    setup_keras_environment()

    parser = argparse.ArgumentParser(description="Run specific iterations of the Transfer Learning classifier.")
    parser.add_argument(
        "--iter",
        type=str,
        default="3",  # Latest iteration as default
        choices=["1", "2", "3", "all"],
        help="Choose which iteration to run: '1', '2', '3' or 'all' (latest as default).",
    )

    args = parser.parse_args()
    run_iter = int(args.iter) if args.iter.isdigit() else args.iter

    main(run_iter)