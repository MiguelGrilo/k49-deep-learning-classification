import os
import numpy as np
from functions.paths import WEIGHTS_DIR
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support


# =====================
# KERAS TRAINING HELPER
# =====================
def keras_classifier(
    name,
    train_func,
    train_kwargs,
    y_test,
):
    # Lazy Imports
    import tensorflow as tf

    """
    Checks if a model and its predictions are already saved to disk.
    If yes, it instantly loads the predictions and skips training.
    If no, it trains the model, saves the prediction cache, and returns it to main.py.
    """

    weights_path = os.path.join(WEIGHTS_DIR, f"{name}.keras")
    prob_path = os.path.join(WEIGHTS_DIR, f"{name}_prob.npy")

    if os.path.exists(weights_path) and os.path.exists(prob_path):
        print(f"     [WEIGHTS FOUND] Found {name}. Skipping training...")

        prob = np.load(prob_path)
        pred = np.argmax(prob, axis=1)

        acc = balanced_accuracy_score(y_test, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="macro", zero_division=0
        )
        return acc, prec, rec, f1, prob, None

    else:
        print(f"     [TRAINING] {name}...")

        acc, prec, rec, f1, prob, model = train_func(**train_kwargs)
        np.save(prob_path, prob)
        return acc, prec, rec, f1, prob, model