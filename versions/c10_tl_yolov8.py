# ======================================================
# Classifier 10 Versions - YOLOv8 (Image Classification)
# ======================================================

import os
if "LD_LIBRARY_PATH" in os.environ:
    del os.environ["LD_LIBRARY_PATH"]
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"

import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import setup_pytorch_environment, load_and_prepare_data, tl_yolo


def iter1(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 1 (10 Epochs)...")
    acc, prec, rec, f1, prob, model = tl_yolo(
        x_train, y_train, x_test, y_test, iter_num=1,
        epochs=10,
        batch_size=256,
        img_size=32,
        description="Baseline YOLOv8 Nano for Image Classification.",
    )

    print(f"\n [- -] Iteration 1 (10 Epochs): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter2(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 2 (25 Epochs)...")
    acc, prec, rec, f1, prob, model = tl_yolo(
        x_train, y_train, x_test, y_test, iter_num=2,
        epochs=25,
        batch_size=256,
        img_size=32,
        description="Extended epochs to allow Ultralytics internal data augmentation to shine.",
    )

    print(f"\n [- -] Iteration 2 (25 Epochs): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter3(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 3 (128px)...")
    acc, prec, rec, f1, prob, model = tl_yolo(
        x_train, y_train, x_test, y_test, iter_num=3,
        epochs=15,
        batch_size=128,
        img_size=128,
        description="Upscaling to 128x128 so YOLO's deep layers have actual features to look at.",
    )

    print(f"\n [- -] Iteration 3 (128px): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter4(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 4 (50 Epochs, 224px)...")
    acc, prec, rec, f1, prob, model = tl_yolo(
        x_train, y_train, x_test, y_test, iter_num=4,
        epochs=50,
        batch_size=64,
        img_size=224,
        description="Overkill Mode: Native 224x224 resolution and 50 epochs for maximum SOTA stabilization.",
    )

    print(f"\n [- -] Iteration 4 (50 Epochs, 224px): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model


def main(target_iter):
    print("# =======================================================")
    print(f"# RUNNING CLASSIFIER 10 VERSIONS (YOLOv8-cls) - Mode: {target_iter}")
    print("# =======================================================\n")
    # Load and Prepare data
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    if target_iter in [1, "all"] : iter1(x_train, y_train, x_test, y_test)
    if target_iter in [2, "all"] : iter2(x_train, y_train, x_test, y_test)
    if target_iter in [3, "all"] : iter3(x_train, y_train, x_test, y_test)
    if target_iter in [4, "all"] : iter4(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    setup_pytorch_environment()

    parser = argparse.ArgumentParser(description="Run specific iterations of the YOLOv8 classifier.")
    parser.add_argument(
        "--iter",
        type=str,
        default="4",  # Latest iteration as default
        choices=["1", "2", "3", "4", "all"],
        help="Choose which iteration to run: '1', '2', '3', '4' or 'all' (latest as default).",
    )

    args = parser.parse_args()
    run_iter = int(args.iter) if args.iter.isdigit() else args.iter

    main(run_iter)