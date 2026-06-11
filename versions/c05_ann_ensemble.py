# =========================================
# Classifier 5 Versions - Ensemble Learning
# =========================================

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import setup_keras_environment, load_and_prepare_data, deep_ann, evaluate_ensemble


def iter1(x_train, y_train, x_test, y_test):
    print("\n [-] Iteration 1 (Ensemble)...")
    print("\n [- -] Phase 1: Training Ensemble Members...\n")
    # Member A
    print(" [- - -] Training Member A...")
    acc_a, prec_a, rec_a, f1_a, prob_a, model_a = deep_ann(
        x_train, y_train, x_test, y_test, iter_num="A",
        layer1_neurons=512,
        layer2_neurons=128,
        epochs=30,
        description="Ensemble Member A",
    )
    # Member B
    print(" [- - -] Training Member B...")
    acc_b, prec_b, rec_b, f1_b, prob_b, model_b = deep_ann(
        x_train, y_train, x_test, y_test, iter_num="B",
        layer1_neurons=512,
        layer2_neurons=128,
        epochs=30,
        dropout_rate=0.4,
        description="Ensemble Member B (Higher Dropout)",
    )
    # Member C
    print(" [- - -] Training Member C...")
    acc_c, prec_c, rec_c, f1_c, prob_c, model_c = deep_ann(
        x_train, y_train, x_test, y_test, iter_num="C",
        layer1_neurons=400,
        layer2_neurons=200,
        epochs=30,
        description="Ensemble Member C (Different layer capacities)",
    )

    print("\n [- -] Phase 2: Combining Predictions...\n")
    models_list = [model_a, model_b, model_c]
    acc, prec, rec, f1, ensemble_preds = evaluate_ensemble(
        models_list,
        x_test, y_test, iter_num=1,
        description="Averaging Softmax probabilities of Models A, B, and C",
    )

    print(f"\n [- -] Member A : Acc: {acc_a * 100:.2f}% | F1: {f1_a * 100:.2f}%")
    print(f" [- -] Member B : Acc: {acc_b * 100:.2f}% | F1: {f1_b * 100:.2f}%")
    print(f" [- -] Member C : Acc: {acc_c * 100:.2f}% | F1: {f1_c * 100:.2f}%")

    print(f"\n [- -] Iteration 1 (Ensemble): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, ensemble_preds, None

def main(target_iter):
    print("# =============================================================")
    print(f"# RUNNING CLASSIFIER 5 VERSIONS (ENSEMBLE LEARNING) - Mode: {target_iter}")
    print("# =============================================================\n")
    # Load and Prepare data
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    if target_iter in [1, "all"] : iter1(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    setup_keras_environment()

    parser = argparse.ArgumentParser(description="Run specific iterations of the Ensemble Learning classifier.")
    parser.add_argument(
        "--iter",
        type=str,
        default="1",  # Latest iteration as default
        choices=["1", "all"],
        help="Choose which iteration to run: '1' or 'all' (latest as default).",
    )

    args = parser.parse_args()
    run_iter = int(args.iter) if args.iter.isdigit() else args.iter

    main(run_iter)