# ====================================
# Classifier 3 Versions - Advanced CNN
# ====================================

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import setup_keras_environment, load_and_prepare_data, advanced_cnn


def iter1(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 1 (Baseline)...")
    acc, prec, rec, f1, prob, model = advanced_cnn(
        x_train, y_train, x_test, y_test, iter_num=1,
        description="Baseline CNN parameters."
    )

    print(f"\n [- -] Iteration 1 (Baseline) : Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter2(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 2 (Dense=256)...")
    acc, prec, rec, f1, prob, model = advanced_cnn(
        x_train, y_train, x_test, y_test, iter_num=2,
        dense_neurons=256,
        description="Increased the final dense layer to 256 neurons for better feature combination.",
    )

    print(f"\n [- -] Iteration 2 (Dense=256): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter3(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 3 (Deep+Aug+Reg)...")
    acc, prec, rec, f1, prob, model = advanced_cnn(
        x_train, y_train, x_test, y_test, iter_num=3,
        num_conv_blocks=3,
        use_data_augmentation=True,
        aug_rotation=0.05,
        use_batch_norm=True,
        use_l2_reg=True,
        dropout_rate=0.4,
        epochs=40,
        early_stop_patience=5,
        description="Added depth (3 blocks), Data Augmentation (0.05 rot), Batch Norm, L2 Reg, and Dropout (0.4).",
    )

    print(f"\n [- -] Iteration 3 (Deep+Aug+Reg): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter4(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 4 (LR Scheduler)...")
    acc, prec, rec, f1, prob, model = advanced_cnn(
        x_train, y_train, x_test, y_test, iter_num=4,
        num_conv_blocks=3,
        use_data_augmentation=True,
        aug_rotation=0.03,
        use_batch_norm=True,
        use_l2_reg=True,
        use_lr_scheduler=True,
        dropout_rate=0.4,
        epochs=50,
        early_stop_patience=8,
        description="Refined augmentation (0.03 rot), increased patience to 8, and added Dynamic LR Scheduler.",
    )

    print(f"\n [- -] Iteration 4 (LR Scheduler): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter5(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 5 (150 Epochs - SOTA)...")
    acc, prec, rec, f1, prob, model = advanced_cnn(
        x_train, y_train, x_test, y_test, iter_num=5,
        num_conv_blocks=3,
        use_data_augmentation=True,
        aug_rotation=0.03,
        use_batch_norm=True,
        use_l2_reg=True,
        use_lr_scheduler=True,
        dropout_rate=0.4,
        epochs=150,
        early_stop_patience=8,
        description="Pushed limits to 150 epochs allowing full generalization of Data Augmentation.",
    )

    print(f"\n [- -] Iteration 5 (150 Epochs): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model


def main(target_iter):
    print("# ========================================================")
    print(f"# RUNNING CLASSIFIER 3 VERSIONS (ADVANCED CNN) - Mode: {target_iter}")
    print("# ========================================================\n")
    # Load and Prepare data
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    if target_iter in [1, "all"] : iter1(x_train, y_train, x_test, y_test)
    if target_iter in [2, "all"] : iter2(x_train, y_train, x_test, y_test)
    if target_iter in [3, "all"] : iter3(x_train, y_train, x_test, y_test)
    if target_iter in [4, "all"] : iter4(x_train, y_train, x_test, y_test)
    if target_iter in [5, "all"] : iter5(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    setup_keras_environment()

    parser = argparse.ArgumentParser(description="Run specific iterations of the Advanced CNN classifier.")
    parser.add_argument(
        "--iter",
        type=str,
        default="5",  # Latest iteration as default
        choices=["1", "2", "3", "4", "5", "all"],
        help="Choose which iteration to run: '1', '2', '3', '4', '5' or 'all' (latest as default).",
    )

    args = parser.parse_args()
    run_iter = int(args.iter) if args.iter.isdigit() else args.iter

    main(run_iter)