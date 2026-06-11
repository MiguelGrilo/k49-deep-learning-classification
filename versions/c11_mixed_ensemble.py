# ==================================================
# Classifier 11 Versions - Mixed Ensemble Learning
# ==================================================

import os
import sys
import argparse
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import setup_gpu_memory, load_and_prepare_data
from functions.paths import WEIGHTS_DIR


def iter1(
        x_train, y_train, x_test, y_test,
        prob_3=None,
        prob_6=None,
        prob_9=None
):
    print("\n [-] Iteration 1 (Mixed Ensemble)...")

    try:
        if prob_3 is None:
            prob_3 = np.load(os.path.join(WEIGHTS_DIR, "ADVANCED_CNN_prob.npy"))
            print(" [- -] Loaded Advanced CNN probabilities from SSD.")
        if prob_6 is None:
            prob_6 = np.load(os.path.join(WEIGHTS_DIR, "TL_RESNET50V2_prob.npy"))
            print(" [- -] Loaded ResNet50V2 probabilities from SSD.")
        if prob_9 is None:
            prob_9 = np.load(os.path.join(WEIGHTS_DIR, "MIXUP_CNN_prob.npy"))
            print(" [- -] Loaded MixUp CNN probabilities from SSD.")

    except FileNotFoundError:
        print("\n [!] ERROR: Probabilities not found in memory OR on the SSD!")
        print(" [INFO] You must train Models 3, 6, and 9 at least once before running the Ensemble.")
        dummy_prob = np.zeros((len(y_test), 49))
        return 0, 0, 0, 0, dummy_prob, None

    print("\n [- -] Phase 1: Receiving cached predictions from Models 3, 6, and 9...")
    print(" [- -] Phase 2: Ensemble Prediction (Averaging Probabilities)...")

    # ENSEMBLE FUSION (Instant math without retraining anything!)
    averaged_probabilities = (prob_3 + prob_6 + prob_9) / 3.0
    final_predictions = np.argmax(averaged_probabilities, axis=1)

    acc = balanced_accuracy_score(y_test, final_predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, final_predictions, average="macro", zero_division=0
    )
    print(f"\n [- -] Iteration 1 (Ensemble): Acc: {acc * 100:.2f}% | Prec: {prec * 100:.2f}% | Rec: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%\n\n")
    return acc, prec, rec, f1, averaged_probabilities, None


def main(target_iter):
    print("# ===========================================================")
    print(f"# RUNNING CLASSIFIER 11 VERSIONS (MIXED ENSEMBLE) - Mode: {target_iter}")
    print("# ===========================================================\n")
    # Load and Prepare Data
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    if target_iter in [1, "all"] : iter1(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    setup_gpu_memory()

    parser = argparse.ArgumentParser(description="Run mixed ensemble classifier.")
    parser.add_argument(
        "--iter",
        type=str,
        default="1",
        choices=["1", "all"],
        help="Run ensemble pipeline (only iteration 1 available).",
    )

    args = parser.parse_args()
    run_iter = int(args.iter) if args.iter.isdigit() else args.iter

    main(run_iter)