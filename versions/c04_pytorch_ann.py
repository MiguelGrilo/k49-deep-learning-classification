# ===================================
# Classifier 4 Versions - PyTorch ANN
# ===================================

import os
if "LD_LIBRARY_PATH" in os.environ:
    del os.environ["LD_LIBRARY_PATH"]
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"

import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import setup_pytorch_environment, load_and_prepare_data, pytorch_ann


def iter1(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 1 (Baseline)...")
    acc, prec, rec, f1, prob, model = pytorch_ann(
        x_train, y_train, x_test, y_test, iter_num=1,
        description="Baseline PyTorch ANN parameters."
    )

    print(f"\n [- -] Iteration 1 (Baseline) : Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model

def iter2(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 2 (Adam Opt)...")
    acc, prec, rec, f1, prob, model = pytorch_ann(
        x_train, y_train, x_test, y_test, iter_num=2,
        hidden_neurons=256,
        epochs=15,
        optimizer_name="adam",
        learning_rate=0.001,
        description="Replaced SGD with Adam optimizer, increased hidden neurons (256), and bumped epochs (15) to match Keras.",
    )

    print(f"\n [- -] Iteration 2 (Adam Opt): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, prob, model


def main(target_iter):
    print("# =======================================================")
    print(f"# RUNNING CLASSIFIER 4 VERSIONS (PYTORCH ANN) - Mode: {target_iter}")
    print("# =======================================================\n")
    # Load and Prepare data
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    if target_iter in [1, "all"] : iter1(x_train, y_train, x_test, y_test)
    if target_iter in [2, "all"] : iter2(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    setup_pytorch_environment()

    parser = argparse.ArgumentParser(description="Run specific iterations of the PyTorch ANN classifier.")
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